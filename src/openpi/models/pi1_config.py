import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models_pytorch.pi1_pytorch import PI1Pytorch


@dataclasses.dataclass(frozen=True)
class ActionExpertConfig:
    """Configuration for action expert transformer architecture.
    
    PI1 uses DINOv2-style transformer blocks, so these are just architectural parameters.
    """
    width: int  # Embedding dimension
    depth: int  # Number of transformer layers
    num_heads: int  # Number of attention heads
    mlp_dim: int  # MLP hidden dimension


@dataclasses.dataclass(frozen=True)
class Pi1Config(_model.BaseModelConfig):
    """Configuration for PI1 model that uses DINOv2 for vision encoding instead of PaliGemma.
    
    PI1 uses DINOv2 for vision and DINOv2-style transformer blocks for action processing.
    No Gemma models are used.
    """
    
    dtype: str = "float32"
    # Action expert architecture parameters (DINOv2-style transformer)
    # These use DINOv2's Block architecture, so they should follow DINOv2 conventions
    # Default values match dinov2_vits14_reg (ViT-Small) dimensions:
    # - embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0
    # Note: DINOv2 uses mlp_ratio=4.0, so mlp_dim = width * 4
    action_expert_width: int = 384  # embed_dim - matches dinov2_vits14_reg (ViT-Small)
    action_expert_depth: int = 12  # depth - matches dinov2_vits14_reg (ViT-Small)
    action_expert_num_heads: int = 6  # num_heads - matches dinov2_vits14_reg (ViT-Small)
    action_expert_mlp_dim: int = 1536  # mlp_dim - width * 4 = 384 * 4 = 1536 for DINOv2 convention

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi15 (PI1 with pi05-style modifications) has two differences from Pi1:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # PI1 uses same ModelType enum values as PI0 since they share the same structure
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike):
        """Create a PI1 model instance.
        
        Note: PI1 is PyTorch-only and doesn't have a JAX implementation.
        This method exists for interface compatibility but raises an error.
        Use PI1Pytorch directly from openpi.models_pytorch.pi1_pytorch for PyTorch training.
        """
        # PI1 is PyTorch-only, so we raise an error if someone tries to create it via JAX
        raise NotImplementedError(
            "PI1 is a PyTorch-only model. Use PI1Pytorch directly from openpi.models_pytorch.pi1_pytorch"
        )

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                # PI1 doesn't use language tokens, but we keep the interface for compatibility
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config.
        
        Note: PI1 uses DINOv2 (which is frozen) and only trains the action expert.
        Since PI1 doesn't use Gemma/LoRA, this always returns nnx.Nothing (no freezing).
        """
        # PI1 doesn't use Gemma or LoRA, so no special freezing logic needed
        return nnx.Nothing
    
    def get_action_expert_config(self) -> ActionExpertConfig:
        """Get action expert config object."""
        return ActionExpertConfig(
            width=self.action_expert_width,
            depth=self.action_expert_depth,
            num_heads=self.action_expert_num_heads,
            mlp_dim=self.action_expert_mlp_dim,
        )

