from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers.models.auto import CONFIG_MAPPING

from dinov2.hub.backbones import dinov2_vits14_reg

import openpi.models.gemma as _gemma


class DinoWithExpertModel(nn.Module):
    """DINOv2 vision encoder with Gemma action expert, matching PaliGemmaWithExpertModel interface."""

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

        # Project DINOv2 outputs to Gemma width
        self.image_proj = nn.Linear(dino_embed_dim, vlm_config.width)

        # Action expert (Gemma) - same as PaliGemmaWithExpertModel
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        # Gemma model for processing both prefix (image embeddings) and suffix (action embeddings)
        # We use the same Gemma model for both since DINOv2 doesn't have a language model
        self.gemma_model = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_model.model.embed_tokens = None

        # Store config for reference
        self.vlm_config = vlm_config
        self.action_expert_config = action_expert_config
        self.use_adarms = use_adarms

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
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        """Embed images using DINOv2 and project to Gemma width.
        
        Args:
            image: Image tensor of shape [B, C, H, W] or [B, H, W, C]
            
        Returns:
            Image embeddings of shape [B, num_tokens, embed_dim] where embed_dim = vlm_config.width
        """
        # Ensure [B, C, H, W] format
        if image.ndim != 4:
            raise ValueError(f"Expected image of shape [B, C, H, W], got {image.shape}")
        if image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)

        # Forward through DINOv2
        # DINOv2 expects [B, C, H, W] format which we've already ensured above
        features = self.dino.forward_features(image)
        
        # Extract tokens: cls token, register tokens, and patch tokens
        # x_norm_clstoken is [B, embed_dim], we need to add sequence dimension
        cls_token = features["x_norm_clstoken"].unsqueeze(1)  # [B, 1, embed_dim]
        reg_tokens = features["x_norm_regtokens"]  # [B, num_reg_tokens, embed_dim]
        patch_tokens = features["x_norm_patchtokens"]  # [B, num_patches, embed_dim]
        
        # Concatenate all tokens: [cls, registers, patches]
        all_tokens = torch.cat([cls_token, reg_tokens, patch_tokens], dim=1)  # [B, 1+num_reg+num_patches, embed_dim]
        
        # Project to Gemma width
        image_embeds = self.image_proj(all_tokens)  # [B, num_tokens, vlm_config.width]
        
        return image_embeds

    def embed_language_tokens(self, tokens: torch.Tensor):
        """Placeholder for language token embedding (not used in DINOv2 version).
        
        This method exists for interface compatibility but should not be called.
        DINOv2 version only processes images, not language.
        """
        raise NotImplementedError(
            "DinoWithExpertModel does not support language tokens. "
            "Use embed_image() for image-only processing."
        )

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
            attention_mask: 4D attention mask
            position_ids: Position IDs for tokens
            past_key_values: Cached key-value pairs from previous forward pass
            inputs_embeds: List of [prefix_embeds, suffix_embeds] where:
                - prefix_embeds: Image embeddings (from embed_image) or None
                - suffix_embeds: Action embeddings or None
            use_cache: Whether to cache key-value pairs
            adarms_cond: List of [prefix_adarms_cond, suffix_adarms_cond] or None
            
        Returns:
            Tuple of ([prefix_output, suffix_output], prefix_past_key_values)
            where prefix_past_key_values is the cached KV pairs for prefix tokens
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Case 1: Only prefix (images) - cache key-values for later use
        if inputs_embeds[1] is None:
            prefix_output = self.gemma_model.model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
            return [prefix_output, suffix_output], prefix_past_key_values

        # Case 2: Only suffix (actions) - use cached prefix key-values
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_model.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
            return [prefix_output, suffix_output], prefix_past_key_values

        # Case 3: Both prefix and suffix - process together
        else:
            # Concatenate prefix and suffix embeddings
            all_embeds = torch.cat([inputs_embeds[0], inputs_embeds[1]], dim=1)
            
            # Forward through Gemma model
            outputs = self.gemma_model.model.forward(
                inputs_embeds=all_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,  # Only suffix uses adarms
            )
            
            hidden_states = outputs.last_hidden_state
            
            # Split back into prefix and suffix outputs
            prefix_len = inputs_embeds[0].shape[1]
            prefix_output = hidden_states[:, :prefix_len]
            suffix_output = hidden_states[:, prefix_len:]
            
            # No past_key_values returned when processing both together
            prefix_past_key_values = None
            
            return [prefix_output, suffix_output], prefix_past_key_values

