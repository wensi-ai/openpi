from openpi.training import config
import numpy as np
from openpi.policies import policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.policies.b1k_policy import extract_state_from_proprio
from omnigibson.learning.datas import BehaviorLeRobotDataset
from openpi_client.image_tools import resize_with_pad

# example = b1k_policy.make_b1k_example()
# print("\n=== Example Contents ===")
# print("-" * 50)
# for key, value in example.items():
#     print(f"\n📌 Key: {key}")
#     print(f"   Type: {type(value).__name__}")
#     if isinstance(value, np.ndarray):
#         print(f"   Dtype: {value.dtype}")
#         print(f"   Shape: {value.shape}")
#     else:
#         print(f"   Value: {value}")
# print("-" * 50 + "\n")

checkpoint_dir = "/home/svl/Research/libs/openpi/outputs/checkpoints/pi0_b1k/openpi/49999"
policy = policy_config.create_trained_policy(
    config.get_config("pi0_b1k"), checkpoint_dir
)
openpi_policy = B1KPolicyWrapper(policy, control_mode="receeding_horizon", action_horizon=1)


ds = BehaviorLeRobotDataset(
    repo_id="behavior-1k/2025-challenge-demos",
    root="/scr/behavior/2025-challenge-demos",
    tasks=["turning_on_radio"],
    modalities=["rgb"],
    local_only=True,
    shuffle=False,
)

def get_action(idx: int):
    data = ds[idx]
    gt_action  = data["action"]
    example = {
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": data["observation.images.rgb.head"].permute(1, 2, 0).numpy(),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": data["observation.images.rgb.left_wrist"].permute(1, 2, 0).numpy(),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": data["observation.images.rgb.right_wrist"].permute(1, 2, 0).numpy(),
        "robot_r1::proprio": data["observation.state"],
    }
    return openpi_policy.act(example), gt_action

breakpoint()