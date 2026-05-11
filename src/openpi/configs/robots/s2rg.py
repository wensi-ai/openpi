from .base_config import ObservationConfig, StateActionConfig, RobotConfig, register_robot


# Franka Robot Configuration
# Single-arm robot with external and wrist cameras
SimFranka = RobotConfig(
    name="robot",
    robot_type="Franka",
    observations={
        "image_0": ObservationConfig(
            name="external",
            obs_key="external::external_camera_1::rgb",
            dataset_key="observation.sim.rgb.external_1",
            resolution=[240, 416]
        ),
        "image_1": ObservationConfig(
            name="wrist",
            obs_key="robot::robot:camera_link:Camera:0::rgb",
            dataset_key="observation.sim.rgb.wrist",
            resolution=[240, 416]
        ),
    },
    action_key="action_qpos",
    action_dim=8,
    action=[
        StateActionConfig(name="arm", indices=list(range(7)), needs_delta_comp=True),
        StateActionConfig(name="gripper", indices=[7], is_eef=True),
    ],
    proprio=[
        StateActionConfig(name="arm_qpos", indices=list(range(7))),
        StateActionConfig(name="gripper_qpos", indices=list(range(7, 9)), is_eef=True),
    ],
)


RealFranka = RobotConfig(
    name="robot",
    robot_type="Franka",
    observations={
        "image_0": ObservationConfig(
            name="external",
            obs_key="external::external_camera_1::rgb",
            dataset_key="observation.real.rgb.external_1",
            resolution=[240, 416]
        ),
        "image_1": ObservationConfig(
            name="wrist",
            obs_key="robot::robot:camera_link:Camera:0::rgb",
            dataset_key="observation.real.rgb.wrist",
            resolution=[240, 416]
        ),
    },
    action_key="action_qpos",
    action_dim=8,
    action=[
        StateActionConfig(name="arm", indices=list(range(7)), needs_delta_comp=True),
        StateActionConfig(name="gripper", indices=[7], is_eef=True),
    ],
    proprio=[
        StateActionConfig(name="arm_qpos", indices=list(range(7))),
        StateActionConfig(name="gripper_qpos", indices=list(range(7, 9)), is_eef=True),
    ],
)


# Dual-arm mobile manipulator with base, torso, and multiple camera views
SimR1Pro = RobotConfig(
    name="robot",
    robot_type="R1Pro",
    observations={
        "image_0": ObservationConfig(
            name="head",
            obs_key="external::zed_camera::rgb",
            dataset_key="observation.sim.rgb.head",
            resolution=[240, 416]
        ),
        "image_1": ObservationConfig(
            name="left_wrist",
            obs_key="external::left_realsense_camera::rgb",
            dataset_key="observation.sim.rgb.left_wrist",
            resolution=[240, 416]
        ),
        "image_2": ObservationConfig(
            name="right_wrist",
            obs_key="external::right_realsense_camera::rgb",
            dataset_key="observation.sim.rgb.right_wrist",
            resolution=[240, 416]
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
        StateActionConfig(name="trunk_qpos", indices=list(range(28, 32))),
        StateActionConfig(name="left_arm_qpos", indices=list(range(32, 39))),
        StateActionConfig(name="left_gripper_qpos", indices=list(range(39, 41)), is_eef=True),
        StateActionConfig(name="right_arm_qpos", indices=list(range(41, 48))),
        StateActionConfig(name="right_gripper_qpos", indices=list(range(48, 50)), is_eef=True),
    ],
)


S2rgR1Pro = RobotConfig(
    name="robot",
    robot_type="R1Pro",
    observations={
        "image_0": ObservationConfig(
            name="head",
            obs_key="external::zed_camera::rgb",
            dataset_key="observation.real.rgb.head",
            resolution=[240, 416]
        ),
        "image_1": ObservationConfig(
            name="left_wrist",
            obs_key="external::left_realsense_camera::rgb",
            dataset_key="observation.real.rgb.left_wrist",
            resolution=[240, 416]
        ),
        "image_2": ObservationConfig(
            name="right_wrist",
            obs_key="external::right_realsense_camera::rgb",
            dataset_key="observation.real.rgb.right_wrist",
            resolution=[240, 416]
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
        StateActionConfig(name="trunk_qpos", indices=list(range(28, 32))),
        StateActionConfig(name="left_arm_qpos", indices=list(range(32, 39))),
        StateActionConfig(name="left_gripper_qpos", indices=list(range(39, 41)), is_eef=True),
        StateActionConfig(name="right_arm_qpos", indices=list(range(41, 48))),
        StateActionConfig(name="right_gripper_qpos", indices=list(range(48, 50)), is_eef=True),
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
            dataset_key="observation.real.rgb.head",
            resolution=[240, 416]
        ),
        "image_1": ObservationConfig(
            name="left_wrist",
            obs_key="robot::robot:left_realsense_link:Camera:0::rgb",
            dataset_key="observation.real.rgb.left_wrist",
            resolution=[240, 416]
        ),
        "image_2": ObservationConfig(
            name="right_wrist",
            obs_key="robot::robot:right_realsense_link:Camera:0::rgb",
            dataset_key="observation.real.rgb.right_wrist",
            resolution=[240, 416]
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
        StateActionConfig(name="trunk_qpos", indices=list(range(3, 7))),
        StateActionConfig(name="left_arm_qpos", indices=list(range(7, 14))),
        StateActionConfig(name="left_gripper_qpos", indices=list(range(14, 15)), is_eef=True),
        StateActionConfig(name="right_arm_qpos", indices=list(range(15, 22))),
        StateActionConfig(name="right_gripper_qpos", indices=list(range(22, 23)), is_eef=True),
    ],
)


# Register robots in the global registry
register_robot("s2rg/sim_franka", SimFranka)
register_robot("s2rg/real_franka", RealFranka)
register_robot("s2rg/sim_r1pro", SimR1Pro)
register_robot("s2rg/s2rg_r1pro", S2rgR1Pro)
register_robot("s2rg/real_r1pro", RealR1Pro)