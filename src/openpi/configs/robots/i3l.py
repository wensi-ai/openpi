from .base_config import ObservationConfig, StateActionConfig, RobotConfig, register_robot


# SimA1 Robot Configuration
# Single-arm robot with external and wrist cameras
SimA1 = RobotConfig(
    name="robot",
    robot_type="A1",
    observations={
        "image_0": ObservationConfig(
            name="external",
            obs_key="external::external_camera::rgb",
            dataset_key="observation.rgb.external",
            resolution=[240, 240]
        ),
        "image_1": ObservationConfig(
            name="wrist",
            obs_key="external::wrist_camera::rgb",
            dataset_key="observation.rgb.wrist",
            resolution=[240, 240]
        ),
    },
    action_key="action",
    action_dim=7,
    action=[
        StateActionConfig(name="arm", indices=list(range(6)), needs_delta_comp=True),
        StateActionConfig(name="gripper", indices=[6], is_eef=True),
    ],
    proprio=[
        StateActionConfig(name="arm_qpos", indices=list(range(6))),
        StateActionConfig(name="gripper_qpos", indices=list(range(48, 50)), is_eef=True),
    ],
)


# SimR1Pro Robot Configuration
# Dual-arm mobile manipulator with base, torso, and multiple camera views
SimR1Pro = RobotConfig(
    name="robot",
    robot_type="R1Pro",
    observations={
        "image_0": ObservationConfig(
            name="head",
            obs_key="robot::robot:zed_link:Camera:0::rgb",
            dataset_key="observation.rgb.head",
            resolution=[240, 240]
        ),
        "image_1": ObservationConfig(
            name="left_wrist",
            obs_key="robot::robot:left_realsense_link:Camera:0::rgb",
            dataset_key="observation.rgb.left_wrist",
            resolution=[240, 240]
        ),
        "image_2": ObservationConfig(
            name="right_wrist",
            obs_key="robot::robot:right_realsense_link:Camera:0::rgb",
            dataset_key="observation.rgb.right_wrist",
            resolution=[240, 240]
        ),
    },
    action_key="action",
    action_dim=23,
    action=[
        StateActionConfig(name="base", indices=list(range(3))),
        StateActionConfig(name="torso", indices=list(range(3, 7)), needs_delta_comp=True),
        StateActionConfig(name="left_arm", indices=list(range(7, 14)), needs_delta_comp=True),
        StateActionConfig(name="left_gripper", indices=[14], is_eef=True),
        StateActionConfig(name="right_arm", indices=list(range(15, 22)), needs_delta_comp=True),
        StateActionConfig(name="right_gripper", indices=[22], is_eef=True),
    ],
    proprio=[
        StateActionConfig(name="base_qvel", indices=list(range(253, 256))),
        StateActionConfig(name="trunk_qpos", indices=list(range(236, 240))),
        StateActionConfig(name="left_arm_qpos", indices=list(range(158, 165))),
        StateActionConfig(name="left_gripper_qpos", indices=list(range(193, 195)), is_eef=True),
        StateActionConfig(name="right_arm_qpos", indices=list(range(197, 204))),
        StateActionConfig(name="right_gripper_qpos", indices=list(range(232, 234)), is_eef=True),
    ],
)


# RealR1Pro Robot Configuration
# Dual-arm mobile manipulator with base, torso, and multiple camera views
RealR1Pro = RobotConfig(
    name="robot",
    robot_type="RealR1Pro",
    observations={
        "image_0": ObservationConfig(
            name="head",
            obs_key="robot::robot:zed_link:Camera:0::rgb",
            dataset_key="observation.rgb.head",
            resolution=[240, 240]
        ),
        "image_1": ObservationConfig(
            name="left_wrist",
            obs_key="robot::robot:left_realsense_link:Camera:0::rgb",
            dataset_key="observation.rgb.left_wrist",
            resolution=[240, 240]
        ),
        "image_2": ObservationConfig(
            name="right_wrist",
            obs_key="robot::robot:right_realsense_link:Camera:0::rgb",
            dataset_key="observation.rgb.right_wrist",
            resolution=[240, 240]
        ),
    },
    action_key="action",
    action_dim=23,
    action=[
        StateActionConfig(name="base", indices=list(range(3))),
        StateActionConfig(name="torso", indices=list(range(3, 7)), needs_delta_comp=True),
        StateActionConfig(name="left_arm", indices=list(range(7, 14)), needs_delta_comp=True),
        StateActionConfig(name="left_gripper", indices=[14], is_eef=True),
        StateActionConfig(name="right_arm", indices=list(range(15, 22)), needs_delta_comp=True),
        StateActionConfig(name="right_gripper", indices=[22], is_eef=True),
    ],
    proprio=[
        StateActionConfig(name="base_qpos", indices=list(range(0, 3))),
        StateActionConfig(name="trunk_qpos", indices=list(range(3, 7))),
        StateActionConfig(name="left_arm_qpos", indices=list(range(7, 14))),
        StateActionConfig(name="left_gripper_qpos", indices=list(range(14, 15)), is_eef=True),
        StateActionConfig(name="right_arm_qpos", indices=list(range(15, 22))),
        StateActionConfig(name="right_gripper_qpos", indices=list(range(22, 23)), is_eef=True),
        StateActionConfig(name="base_qvel", indices=list(range(23, 26))),
    ],
)


# Register robots in the global registry
register_robot("i3l/SimA1", SimA1)
register_robot("i3l/SimR1Pro", SimR1Pro)
register_robot("i3l/RealR1Pro", RealR1Pro)
