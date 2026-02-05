## BEHAVIOR-1K

This tutorial provides instructions for using Pi models with the [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) repository.

**Last updated:** Feb 4th, 2026  
**BEHAVIOR-1K version:** 3.7.2


> **Note:** The default branch `behavior` is under active development. We recommend forking the repository or creating a new branch for your development work to avoid breaking changes.


### Installation

OpenPi uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
cd $OPENPI_DIR
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Collect Data with BEHAVIOR

> **Note:** As of v3.7.2, BEHAVIOR-1K currently only supports saving data to HDF5 format. For v3.8.0, we are working on data wrappers that can directly save trajectories in LeRobot format.

1. Collect data in HDF5 format.

2. Use `hdf2lerobot.py` to convert the dataset into LeRobot format. 

### Finetune OpenPi

#### 1. Register robot and task

OpenPi uses Python-based configuration files to register robots and tasks.

**Register a robot:**

Create or edit a robot configuration file in `src/openpi/configs/robots/`. For example, to register a BEHAVIOR-1K robot:

```python
# src/openpi/configs/robots/behavior.py
from .base_config import ObservationConfig, StateActionConfig, RobotConfig, register_robot

# Define your robot configuration
MyRobot = RobotConfig(
    name="robot",
    robot_type="MyRobotType",
    observations={
        "image_0": ObservationConfig(
            name="external",
            obs_key="external::external_camera::rgb",
            dataset_key="observation.rgb.external",
            resolution=[240, 240]
        ),
        # Add more camera views as needed
    },
    action_dim=7,  # Total action dimensions
    action=[
        StateActionConfig(name="arm", indices=list(range(6)), needs_delta_comp=True),
        StateActionConfig(name="gripper", indices=[6], is_eef=True),
    ],
    proprio=[
        StateActionConfig(name="arm_qpos", indices=list(range(6))),
        StateActionConfig(name="gripper_qpos", indices=list(range(6, 8)), is_eef=True),
    ],
)

# Register the robot with format: "bucket/robot_name"
register_robot("behavior/MyRobot", MyRobot)
```

**Register tasks:**

Create or edit a task configuration file in `src/openpi/configs/tasks/`. For example:

```python
# src/openpi/configs/tasks/behavior.py
from . import TASK_REGISTRY

# Define task prompts
TASKS = {
    "task_name_1": "Task instruction/prompt for task 1.",
    "task_name_2": "Task instruction/prompt for task 2.",
}

# Register in global registry
TASK_REGISTRY["behavior"] = TASKS
```

See `src/openpi/configs/robots/i3l.py` and `src/openpi/configs/tasks/i3l.py` for reference examples.

#### 2. Configure model and training settings

Training configurations are defined in `src/openpi/training/config.py`. The `pi0_b1k` config (lines 734-752) provides a reference for BEHAVIOR-1K training.

**Key configuration components:**

```python
TrainConfig(
    name="pi0_b1k",
    # Model configuration
    model=pi0_config.Pi0Config(action_horizon=50),
    
    # Data configuration
    data=LeRobotB1KDataConfig(
        repo_id="iiil/books",  # Your LeRobot dataset repo ID
        base_config=DataConfig(
            prompt_from_task=True,
            episodes_index=list(range(100)),  # Training episodes
            dataset_root="/path/to/your/dataset",  # Local dataset path
        ),
        robot_config_name="i3l/RealR1Pro",  # Must match your registered robot
    ),
    
    # Checkpoint settings
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    num_train_steps=50_000,
    assets_base_dir="./outputs/assets",
    checkpoint_base_dir="./outputs/checkpoints",
)
```

**To create your own config:**

1. Copy the `pi0_b1k` config in `src/openpi/training/config.py`
2. Update the following fields:
   - `name`: Unique identifier for your config
   - `repo_id`: Your LeRobot dataset repository ID
   - `dataset_root`: Local path to your dataset
   - `robot_config_name`: Reference to your registered robot (format: `"bucket/robot_name"`)
   - `episodes_index`: List of episode indices to use for training
   - Optionally configure validation with `val_repo_id` and `val_episodes_index`

#### 3. Compute normalization statistics

Before running training, we need to compute normalization statistics for the training data. Change line 98 of `compute_norm_stats.py` to specify the task name you want (or `None` to include all tasks), then run the script:

```bash
uv run scripts/compute_norm_stats.py --config-name $CONFIG_NAME
```

Replace `$CONFIG_NAME` with your training config name (e.g., `pi0_b1k`). This will create `norm_stats.json` under `assets/$CONFIG_NAME/$REPO_ID`.

#### 4. Finetune OpenPi

Start training with your config:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_val.py $CONFIG_NAME \
    --exp_name="openpi_$(date +%Y%m%d_%H%M%S)" \
    --overwrite \
    --batch_size=64 \
    --num_train_steps=50000 \
    --weight_loader.params_path=gs://openpi-assets/checkpoints/pi0_base/params
```

Replace `$CONFIG_NAME` with your training config name (e.g., `pi0_b1k`). You can override any config parameters via command-line arguments.

**Common parameters to adjust:**
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--num_train_steps`: Total training steps
- `--data.repo_id`: Override dataset repo ID
- `--data.robot_config_name`: Override robot configuration
- `--val_log_interval`: How often to log validation metrics (default: 2500)


### Evaluation

After finetuning, you can run evaluation by following these steps:

#### 1. Deploy finetuned checkpoint

```bash
source .venv/bin/activate
uv run scripts/serve_b1k.py --robot $ROBOT_TAG --task_name $TASK_TAG policy:checkpoint --policy.config $MODEL_TAG --policy.dir $PATH_TO_CKPT
```

**Example:**

```bash
uv run scripts/serve_b1k.py --robot i3l/RealR1Pro --task_name i3l/books policy:checkpoint --policy.config pi0_b1k --policy.dir outputs/checkpoints/pi0_b1k/openpi/49999_books
```

This opens a policy server listening on `0.0.0.0:8000`. You can then run your robot client to send observations and receive actions.