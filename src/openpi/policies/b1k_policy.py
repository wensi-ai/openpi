import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(21),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        proprio_data = data["observation/state"]
        # extract joint position
        base_qvel = proprio_data[246:249] # 3
        trunk_qpos = proprio_data[238:242] # 4
        arm_left_qpos = proprio_data[158:165] #  7
        arm_right_qpos = proprio_data[198:205] #  7
        left_gripper_width = proprio_data[194:196].sum(axis=-1, keepdims=True) # 1
        right_gripper_width = proprio_data[234:236].sum(axis=-1, keepdims=True) # 1
        state = np.concatenate([
            base_qvel,
            trunk_qpos,
            arm_left_qpos,
            arm_right_qpos,
            left_gripper_width,
            right_gripper_width,
        ])
        state = transforms.pad_to_dim(state, self.action_dim)
        if "actions" in data:
            action =  data["actions"]
            action = transforms.pad_to_dim(action, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, wrist_image_left, wrist_image_right)
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
class B1kOutputs(transforms.DataTransformFn):
    action_dim: int = 23
    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
