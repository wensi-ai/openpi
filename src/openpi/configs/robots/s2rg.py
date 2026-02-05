from .base_config import ObservationConfig, StateActionConfig, RobotConfig, register_robot


# A1 Robot Configuration
# Single-arm robot with external and wrist cameras
Franka = RobotConfig(
    name="robot",
    robot_type="Franka",
    observations={
        "image_0": ObservationConfig(
            name="external",
            obs_key="external::external_camera_1::rgb",
            dataset_key="observation.rgb.external",
            resolution=[240, 416]
        ),
        "image_1": ObservationConfig(
            name="wrist",
            obs_key="robot::robot:camera_link:Camera:0::rgb",
            dataset_key="observation.rgb.wrist",
            resolution=[240, 416]
        ),
    },
    action_dim=8,
    action=[
        StateActionConfig(name="arm", indices=list(range(7)), needs_delta_comp=True),
        StateActionConfig(name="gripper", indices=[7], is_eef=True),
    ],
    proprio=[
        StateActionConfig(name="arm_qpos", indices=list(range(7))),
        StateActionConfig(name="gripper_qpos", indices=list(range(22, 24)), is_eef=True),
    ],
)


# Register robots in the global registry
register_robot("s2rg/franka", Franka)