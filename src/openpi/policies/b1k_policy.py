import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.configs.robots.base_config import RobotConfig


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/image_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(23),
        "prompt": "do something",
    }


def extract_state_from_proprio(proprio_data, robot_config: RobotConfig) -> np.ndarray:
    """Extract state from proprioception data based on robot configuration.

    We assume perfect correlation for the two gripper fingers.

    Args:
        proprio_data: Raw proprioception data
        robot_config: RobotConfig instance containing robot configuration

    Returns:
        Extracted state array
    """
    state = []
    for proprio in robot_config.proprio:
        if proprio.is_eef:
            # Sum the gripper finger positions to get a single width value
            state.append(proprio_data[..., proprio.indices].sum(axis=-1, keepdims=True))
        else:
            state.append(proprio_data[..., proprio.indices])
    return np.concatenate(state, axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1KInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Robot configuration object
    robot_config: RobotConfig = dataclasses.field(default=None)

    def __call__(self, data: dict) -> dict:
        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data, self.robot_config)
        if "actions" in data:
            action = data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        images, image_masks = [], []
        for camera_id in self.robot_config.observations:
            images.append(_parse_image(data[f"observation/{camera_id}"]))
            image_masks.append(np.True_)
        while len(images) < 3:
            images.append(np.zeros_like(images[0]))
            image_masks.append(np.False_)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = tuple(images)
                image_masks = tuple(image_masks)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = tuple(images)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class B1KOutputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
