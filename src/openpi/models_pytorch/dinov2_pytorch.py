from functools import partial
from typing import Literal
import os
import sys

import pytest
import torch
from torch import nn

# Ensure dinov2 package is importable by adding src/dinov2 to path if needed
# This allows using 'from dinov2.*' imports which matches DINOv2's internal pattern
_dinov2_path = None
if "dinov2" not in sys.modules:
    # Try to find src/dinov2 relative to this file
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # This file is at: src/openpi/models_pytorch/dinov2_pytorch.py
    # So src/dinov2 would be at: src/dinov2
    src_dir = os.path.join(current_file_dir, "..", "..")
    dinov2_src_dir = os.path.join(src_dir, "dinov2")
    if os.path.exists(dinov2_src_dir):
        _dinov2_path = os.path.abspath(dinov2_src_dir)
        if _dinov2_path not in sys.path:
            sys.path.insert(0, _dinov2_path)

from dinov2.hub.backbones import dinov2_vits14_reg
from dinov2.layers import NestedTensorBlock, MemEffAttention, Mlp

import openpi.models.gemma as _gemma


class DinoWithExpertModel(nn.Module):
    """DINOv2 vision encoder with DINOv2-style transformer for actions.
    
    Language input is accepted for compatibility but not processed (treated as dummy).
    """

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # DINOv2 vision encoder (ViT-S/14 with registers)
        self.dino = dinov2_vits14_reg(pretrained=True)
        dino_embed_dim = self.dino.embed_dim

        # Project DINOv2 outputs to target width
        self.image_proj = nn.Linear(dino_embed_dim, vlm_config.width)

        # DINOv2-style transformer for processing actions
        # Use same architecture as DINOv2 but for sequence processing
        embed_dim = action_expert_config.width
        depth = action_expert_config.depth
        num_heads = action_expert_config.num_heads
        mlp_ratio = action_expert_config.mlp_dim / action_expert_config.width
        
        # Create transformer blocks using DINOv2's NestedTensorBlock architecture
        # NestedTensorBlock extends Block and works with regular tensors too
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        blocks_list = [
            NestedTensorBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                attn_class=MemEffAttention,
                init_values=None,
            )
            for _ in range(depth)
        ]
        self.action_transformer_blocks = nn.ModuleList(blocks_list)
        self.action_norm = norm_layer(embed_dim)

        # Store config for reference
        self.vlm_config = vlm_config
        self.action_expert_config = action_expert_config
        self.use_adarms = use_adarms
        self.embed_dim = embed_dim

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        """Convert model to specified precision, keeping certain params in float32."""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep certain parameters in float32 for numerical stability
        params_to_keep_float32 = [
            "image_proj",
            "action_norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        """Embed images using DINOv2 and project to target width.
        
        Args:
            image: Image tensor of shape [B, C, H, W] or [B, H, W, C]
            
        Returns:
            Image embeddings of shape [B, num_tokens, embed_dim]
        """
        # Ensure [B, C, H, W] format
        if image.ndim != 4:
            raise ValueError(f"Expected image of shape [B, C, H, W], got {image.shape}")
        if image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)

        # Convert image to match DINOv2 model dtype
        # Check the dtype of the first DINOv2 parameter to determine model dtype
        dino_dtype = next(self.dino.parameters()).dtype
        # Convert to float32 first for stability, then to model dtype if needed
        image = image.to(dtype=torch.float32)
        if dino_dtype != torch.float32:
            image = image.to(dtype=dino_dtype)

        # Forward through DINOv2
        features = self.dino.forward_features(image)
        
        # Extract tokens: cls token, register tokens, and patch tokens
        cls_token = features["x_norm_clstoken"].unsqueeze(1)  # [B, 1, embed_dim]
        reg_tokens = features["x_norm_regtokens"]  # [B, num_reg_tokens, embed_dim]
        patch_tokens = features["x_norm_patchtokens"]  # [B, num_patches, embed_dim]
        
        # Concatenate all tokens: [cls, registers, patches]
        all_tokens = torch.cat([cls_token, reg_tokens, patch_tokens], dim=1)  # [B, 1+num_reg+num_patches, embed_dim]
        
        # Convert to float32 for image_proj (which is kept in float32)
        all_tokens = all_tokens.to(dtype=torch.float32)
        image_embeds = self.image_proj(all_tokens)  # [B, num_tokens, vlm_config.width]
        
        return image_embeds

    def embed_language_tokens(self, tokens: torch.Tensor):
        """Dummy method for language tokens - accepts input but does not process it.
        
        This method exists for interface compatibility. Language tokens are not processed
        in the DINOv2-only version.
        
        Args:
            tokens: Language token tensor (ignored)
            
        Returns:
            None or dummy tensor (for compatibility)
        """
        # Return None to indicate language is not processed
        # The caller should handle this appropriately
        return None

    def _process_with_transformer(self, x, attention_mask=None):
        """Process embeddings through DINOv2-style transformer blocks.
        
        Args:
            x: Input embeddings [B, seq_len, embed_dim]
            attention_mask: Optional attention mask (currently not used in DINOv2 blocks)
            
        Returns:
            Processed embeddings [B, seq_len, embed_dim]
        """
        # Convert input to float32 first for stability, then to block dtype if needed
        x = x.to(dtype=torch.float32)
        
        # Check dtype of transformer blocks (they should be bfloat16 if precision is bfloat16)
        block_dtype = next(self.action_transformer_blocks[0].parameters()).dtype
        if block_dtype != torch.float32:
            x = x.to(dtype=block_dtype)
        
        # Process through transformer blocks
        for block in self.action_transformer_blocks:
            x = block(x)
        
        # Convert to float32 for action_norm (which is kept in float32)
        x = x.to(dtype=torch.float32)
        # Final normalization
        x = self.action_norm(x)
        
        return x

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        """Forward pass matching PaliGemmaWithExpertModel interface.
        
        Args:
            attention_mask: 4D attention mask (for compatibility, not fully used with DINOv2 blocks)
            position_ids: Position IDs (for compatibility, not used)
            past_key_values: Cached key-value pairs (for compatibility, not used with DINOv2 blocks)
            inputs_embeds: List of [prefix_embeds, suffix_embeds] where:
                - prefix_embeds: Image embeddings (from embed_image) or None
                - suffix_embeds: Action embeddings or None
            use_cache: Whether to cache key-value pairs (for compatibility, not used)
            adarms_cond: List of [prefix_adarms_cond, suffix_adarms_cond] (for compatibility, not used)
            
        Returns:
            Tuple of ([prefix_output, suffix_output], prefix_past_key_values)
            Note: prefix_past_key_values is always None since DINOv2 blocks don't support KV caching
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Case 1: Only prefix (images) - process and return
        if inputs_embeds[1] is None:
            prefix_output = self._process_with_transformer(inputs_embeds[0], attention_mask)
            suffix_output = None
            # DINOv2 blocks don't support KV caching, return None
            prefix_past_key_values = None
            return [prefix_output, suffix_output], prefix_past_key_values

        # Case 2: Only suffix (actions) - process and return
        elif inputs_embeds[0] is None:
            suffix_output = self._process_with_transformer(inputs_embeds[1], attention_mask)
            prefix_output = None
            prefix_past_key_values = None
            return [prefix_output, suffix_output], prefix_past_key_values

        # Case 3: Both prefix and suffix - concatenate and process together
        else:
            # Concatenate prefix and suffix embeddings
            all_embeds = torch.cat([inputs_embeds[0], inputs_embeds[1]], dim=1)
            
            # Process through transformer
            hidden_states = self._process_with_transformer(all_embeds, attention_mask)
            
            # Split back into prefix and suffix outputs
            prefix_len = inputs_embeds[0].shape[1]
            prefix_output = hidden_states[:, :prefix_len]
            suffix_output = hidden_states[:, prefix_len:]
            
            # No past_key_values returned (DINOv2 blocks don't support caching)
            prefix_past_key_values = None
            
            return [prefix_output, suffix_output], prefix_past_key_values
