"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created. May also be a
    # tuple/list of repo ids; the data loader will load each as a separate
    # LeRobotDataset and combine them via torch ConcatDataset for multi-task
    # training. When using multiple repos, set assets.asset_id explicitly so
    # there is a single normalization-stats path on disk.
    repo_id: str | tuple[str, ...] | list[str] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Optional per-repo episode subset. Keys are LeRobot repo ids (matching
    # entries in `repo_id`); values are the explicit episode indices to load.
    # Repos absent from the dict (or all repos when this is None) are loaded
    # in full. Applied uniformly by both training and norm-stats compute.
    episode_filters: dict[str, list[int]] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id. May be a single string or a tuple/list of strings
    # for multi-task training (see DataConfig.repo_id).
    repo_id: str | tuple[str, ...] = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        # asset_id defaults to repo_id only for single-string repos; tuples
        # of repo ids must specify AssetsConfig(asset_id=...) so the merged
        # norm_stats live at one well-defined path.
        if isinstance(repo_id, (tuple, list)):
            asset_id = self.assets.asset_id
            if asset_id is None:
                raise ValueError(
                    "Multi-repo configs (repo_id is a tuple/list) must set "
                    "assets=AssetsConfig(asset_id=...) explicitly so a single "
                    "norm_stats file can be located on disk."
                )
        else:
            asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotMolmospacesDroidDataConfig(DataConfigFactory):
    """Like LeRobotDROIDDataConfig but built for molmospaces drawer sim data.

    The molmospaces datagen pipeline only records one exocentric camera, so
    we drop exterior_image_2_left from the repack (DroidInputs zero-fills the
    third image slot anyway). Skipping that field saves ~33% of the image
    writes and per-episode mp4 encoding cost during conversion.
    """

    # If set, binarize the gripper observation (observation/gripper_position)
    # to {0.0, 1.0} at this threshold (>= threshold -> 1.0) before it enters
    # the state vector. Applied identically at training, norm-stat, and
    # inference time. None disables binarization (default).
    binarize_gripper_threshold: float | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        gripper_transforms = (
            [_transforms.BinarizeGripper(threshold=self.binarize_gripper_threshold)]
            if self.binarize_gripper_threshold is not None
            else []
        )
        data_transforms = _transforms.Group(
            inputs=[*gripper_transforms, droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    TrainConfig(
        # Fine-tune pi05-droid on the molmospaces drawer-open simulation
        # dataset produced by scripts/datagen/data_gen.sh. The LeRobot dataset
        # is built by examples/molmospaces/convert_molmodata_to_lerobot.py.
        #
        # We compute fresh normalization stats (no assets override) because
        # the sim observation/action distributions differ from real DROID
        # (joint velocities from sim qpos diffs, gripper obs scaled by
        # 0.824033 to match pi_policy.py, etc.).
        name="pi05_droid_renderscale",
        project_name="renderscale-pi05",
        # LoRA fine-tune so training fits on 2xL40S (48GB each). Full pi05
        # fine-tuning OOMs at batch_size=4 on this hardware. Same pattern as
        # pi0_libero_low_mem_finetune.
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/drawer_open",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same as pi05_droid_renderscale (drawer) but with action_horizon=32.
        # Used to serve the run_drawer_h32_3376eps_40k_20260501_223926 checkpoint.
        name="pi05_droid_renderscale_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/drawer_open",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same recipe as pi05_droid_renderscale, but trained on the fridge
        # open/close molmospaces dataset (430 trajectories per shard x 4 shards).
        name="pi05_droid_renderscale_fridge",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/fridge_open",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same recipe as pi05_droid_renderscale_fridge but trained on the
        # SIM2REAL-GENERATED fridge dataset at
        # /viscam/projects/egodex/renderscale/lerobot_data_generated/renderscale/fridge_open_generated
        # (1000 trajectories converted from molmodata_generated/fridge — varied-prompt run).
        # Set HF_LEROBOT_HOME=/viscam/projects/egodex/renderscale/lerobot_data_generated
        # when running compute_norm_stats.py and train.py with this config.
        name="pi05_droid_renderscale_fridge_generated",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/fridge_open_generated",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # fridge_m sim+gen share the same h5 (identical actions). To reuse
        # norm stats, symlink:
        #   mkdir -p assets/pi05_droid_renderscale_fridge_m_h32_gen/fridge_m_gen
        #   ln -sf assets/pi05_droid_renderscale_fridge_m_h32/fridge_m_sim/norm_stats.json \
        #          assets/pi05_droid_renderscale_fridge_m_h32_gen/fridge_m_gen/norm_stats.json
        # then HF_LEROBOT_HOME=/scr/ravenh/fridge_m/lerobot_gen train.py
        #   pi05_droid_renderscale_fridge_m_h32_gen ...
        name="pi05_droid_renderscale_fridge_m_h32_gen",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="fridge_m_gen",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # fridge_7 sim+gen share the same h5 (and therefore identical actions),
        # so norm stats are computed once from sim and SYMLINKED into the gen
        # asset dir. To reuse:
        #   mkdir -p assets/pi05_droid_renderscale_fridge_7_h32_gen/fridge_7_gen
        #   ln -sf assets/pi05_droid_renderscale_fridge_7_h32/fridge_7_sim/norm_stats.json \
        #          assets/pi05_droid_renderscale_fridge_7_h32_gen/fridge_7_gen/norm_stats.json
        # then HF_LEROBOT_HOME=/scr/ravenh/fridge_7/lerobot_gen train.py
        #   pi05_droid_renderscale_fridge_7_h32_gen ...
        name="pi05_droid_renderscale_fridge_7_h32_gen",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="fridge_7_gen",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # fridge_7 (Fridge_7_upside_down) sim run @ ~719 trajs, chunked pipeline
        # 2026-05-10. Set HF_LEROBOT_HOME=/scr/ravenh/fridge_7/lerobot_sim for
        # both compute_norm_stats.py and train.py to find the local dataset.
        name="pi05_droid_renderscale_fridge_7_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="fridge_7_sim",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # fridge_m (ithor_minimal_fridge_m) sim run, combined from two agents:
        # viscam3 (chunks 001-007, ~348 trajs) + viscam5 (chunks 099, 100-104, ~245 trajs)
        # → 592 trajs total. Set HF_LEROBOT_HOME=/scr/ravenh/fridge_m/lerobot_sim
        # for both compute_norm_stats.py and train.py to find the local dataset.
        name="pi05_droid_renderscale_fridge_m_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="fridge_m_sim",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # fridge_m_0.7_pd sim run from sibling Agent_fridge_m_0.7_pd on viscam4.
        # Wider robot placement rotation range (0.7 vs base 0.2) + sysid PD gains.
        # HF subfolder: fridge_m_0.7_pd_sim/. Local dataset will be downloaded to
        # /scr/ravenh/fridge_m_0.7_pd/lerobot_sim/fridge_m_0.7_pd_sim/.
        name="pi05_droid_renderscale_fridge_m_0_7_pd_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="fridge_m_0.7_pd_sim",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # fridge_m_0.7_pd GEN variant — same h5 actions as sim, but with
        # sim2real-rendered exo + wrist mp4s. Built from
        # Ravenh97/generated_video:fridge_m_0.7_pd/ (gen mp4s) + sim h5s.
        # Local dataset at /scr/ravenh/fridge_m_0_7_pd_train/lerobot_home/fridge_m_0.7_pd_gen/.
        name="pi05_droid_renderscale_fridge_m_0_7_pd_h32_gen",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="fridge_m_0.7_pd_gen",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # GEN variant of pi05_droid_renderscale_fridge_7_m_h32 — joint training on
        # fridge_7_gen + fridge_m_gen. Reuses the SIM-derived norm stats via
        # asset_id="renderscale/fridge_7_m_sim" (sim and gen share identical h5
        # actions, so the stats are identical). To set up:
        #   mkdir -p assets/pi05_droid_renderscale_fridge_7_m_h32_gen/renderscale/fridge_7_m_sim
        #   ln -sf assets/pi05_droid_renderscale_fridge_7_m_h32/renderscale/fridge_7_m_sim/norm_stats.json \
        #          assets/pi05_droid_renderscale_fridge_7_m_h32_gen/renderscale/fridge_7_m_sim/norm_stats.json
        # Stage data: mkdir /scr/ravenh/fridge_7_m_gen && ln -s
        #   /scr/ravenh/fridge_7/lerobot_gen/fridge_7_gen,
        #   /scr/ravenh/fridge_m/lerobot_gen/fridge_m_gen
        # into it, then HF_LEROBOT_HOME=/scr/ravenh/fridge_7_m_gen train.py.
        name="pi05_droid_renderscale_fridge_7_m_h32_gen",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=(
                "fridge_7_gen",
                "fridge_m_gen",
            ),
            assets=AssetsConfig(asset_id="renderscale/fridge_7_m_sim"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # 3drawers GEN variant. Pulls HF Ravenh97/lerobot_data subfolder
        # 3drawers_gen/. Set HF_LEROBOT_HOME=/scr/ravenh/3drawers_gen/lerobot_gen
        # locally. Compute fresh norm stats (won't bother symlinking from sim
        # since user explicitly asked to compute them).
        name="pi05_droid_renderscale_3drawers_h32_gen",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="3drawers_gen",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # 3drawers sim run pulled from HF Ravenh97/lerobot_data subfolder
        # 3drawers_sim/. Set HF_LEROBOT_HOME=/viscam/projects/egodex/renderscale/lerobot_data_3drawers
        # for both compute_norm_stats.py and train.py. Norm stats at
        # assets/<config>/3drawers_sim/norm_stats.json.
        name="pi05_droid_renderscale_3drawers_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="3drawers_sim",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,  # default; pipeline.sh overrides to --batch_size=48
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # 3drawers_v2 sim — extended 3drawers sim run built from
        # Ravenh97/sim_data:3drawers/sim_chunks/ (17 chunks). LeRobot dataset
        # pushed to Ravenh97/lerobot_data:3drawers_v2/. Local at
        # /scr/ravenh/3drawers_v2_train/lerobot_home/3drawers_v2/.
        name="pi05_droid_renderscale_3drawers_v2_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="3drawers_v2",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,  # default; launch overrides to --batch_size=72 on 6x H200
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # 3drawers_v2 GEN variant — same per-step actions+qpos as 3drawers_v2,
        # but RGB streams come from Wan2.1 1.3B sim2real generation
        # (Ravenh97/generated_video:3drawers/gen_chunks/). LeRobot dataset
        # pushed to Ravenh97/lerobot_data:3drawers_v2_gen/. Local at
        # /scr/ravenh/3drawers_v2_gen/lerobot_home/3drawers_v2_gen/.
        name="pi05_droid_renderscale_3drawers_v2_h32_gen",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="3drawers_v2_gen",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,  # default; launch overrides to --batch_size=48 on 4x L40S
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # fridge_m_2house sim — 2-house variant of fridge_m sim with cam_rand
        # augmentation. 90 chunks across 2 houses (house_1: 44 chunks 001-049,
        # house_2: 46 chunks 051-098). ~990 episodes, 15 fps. LeRobot dataset
        # pushed to Ravenh97/lerobot_data:fridge_m_2house/. Local at
        # /scr/ravenh/fridge_m_2house_train/lerobot_home/fridge_m_2house/.
        name="pi05_droid_renderscale_fridge_m_2house_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="fridge_m_2house",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 18/GPU on 4x L40S; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # fridge_m_v2 sim — extended fridge_m sim run with cam_rand augmentation
        # (camera_extrinsics_pos_noise=0.05, rot_noise=5deg). 36 chunks across
        # 4 datagen workers (chunks 001-010 + 101-111 + 201-208 + 301-307).
        # 1272 episodes, 380,821 frames, 15 fps. LeRobot dataset pushed to
        # Ravenh97/lerobot_data:fridge_m_v2/. Local at
        # /scr/ravenh/fridge_m_v2_train/lerobot_home/fridge_m_v2/.
        name="pi05_droid_renderscale_fridge_m_v2_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="fridge_m_v2",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # fridge_m_v2 GEN variant — same per-step actions+qpos as fridge_m_v2 sim,
        # but RGB streams come from Wan2.1 sim2real generation
        # (Ravenh97/generated_video:fridge_m_v2/gen_chunks/). Reuses the sim h5
        # files (qpos/actions identical) with the GEN mp4s. LeRobot dataset
        # pushed to Ravenh97/lerobot_data:fridge_m_v2_gen/. Local at
        # /scr/ravenh/fridge_m_v2_gen_train/lerobot_home/fridge_m_v2_gen/.
        name="pi05_droid_renderscale_fridge_m_v2_h32_gen",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="fridge_m_v2_gen",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Combined fridge_m_v2_gen + fridge_m_gen multi-task (ConcatDataset).
        # fridge_m_v2_gen: 1272 eps / 380,821 frames (HF Ravenh97/lerobot_data:fridge_m_v2_gen).
        # fridge_m_gen:    666 eps / 198,756 frames (HF Ravenh97/lerobot_data:fridge_m_gen).
        # Per-frame uniform sampling -> fridge_m_v2_gen dominates 65%/35% by frame count.
        # Norm stats land at assets/<cfg>/fridge_m_v2_gen_plus_m_gen/norm_stats.json.
        name="pi05_droid_renderscale_fridge_m_v2_gen_plus_m_gen_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("fridge_m_v2_gen", "fridge_m_gen"),
            assets=AssetsConfig(asset_id="fridge_m_v2_gen_plus_m_gen"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Combined dresser_gen + 3drawers_v2_gen multi-task (ConcatDataset).
        # dresser_gen:      to-be-converted; ~1250 trajs / 76 chunks gen pipeline 2026-05-18.
        # 3drawers_v2_gen:  ~965 eps (HF Ravenh97/lerobot_data:3drawers_v2_gen).
        # Norm stats land at assets/<cfg>/dresser_gen_plus_3drawers_v2_gen/norm_stats.json.
        name="pi05_droid_renderscale_dresser_gen_plus_3drawers_v2_gen_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("dresser_gen", "3drawers_v2_gen"),
            assets=AssetsConfig(asset_id="dresser_gen_plus_3drawers_v2_gen"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same as pi05_droid_renderscale_dresser_gen_plus_3drawers_v2_gen_h32 but
        # with the gripper observation binarized to {0,1} at threshold 0.5
        # (binarize_gripper_threshold). Distinct asset_id so norm stats are
        # recomputed against the binarized state distribution.
        name="pi05_droid_renderscale_dresser_gen_plus_3drawers_v2_gen_h32_binarize",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("dresser_gen", "3drawers_v2_gen"),
            assets=AssetsConfig(asset_id="dresser_gen_plus_3drawers_v2_gen_binarize"),
            base_config=DataConfig(prompt_from_task=True),
            binarize_gripper_threshold=0.5,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Combined dresser_dr + 3drawers_v2_dr multi-task (ConcatDataset), gripper
        # observation binarized to {0,1} at threshold 0.5.
        # dresser_dr:      all 1207 eps (HF Ravenh97/lerobot_data:dresser_dr, dr_count=1).
        # 3drawers_v2_dr:  first 1000 eps of 5450 via DataConfig.episode_filters.
        # 3drawers_v2_dr parquets use the datasets 4.x 'List' feature -> needs the
        # List->Sequence monkey-patch when loading (see datasets_list_compat).
        # Norm stats land at assets/<cfg>/dresser_dr_plus_3drawers_v2_dr_first1k_binarize/norm_stats.json.
        name="pi05_droid_renderscale_dresser_dr_plus_3drawers_v2_dr_first1k_h32_binarize",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("dresser_dr", "3drawers_v2_dr"),
            assets=AssetsConfig(asset_id="dresser_dr_plus_3drawers_v2_dr_first1k_binarize"),
            base_config=DataConfig(
                prompt_from_task=True,
                episode_filters={"3drawers_v2_dr": list(range(1000))},
            ),
            binarize_gripper_threshold=0.5,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Combined fridge_m_2house + fridge_m_v2 multi-task (ConcatDataset).
        # fridge_m_2house: all 990 eps (HF Ravenh97/lerobot_data:fridge_m_2house).
        # fridge_m_v2:     first 500 eps of 1272 via DataConfig.episode_filters.
        # Gripper observation NOT binarized (continuous).
        # Norm stats land at assets/<cfg>/fridge_m_2house_plus_v2_first500/norm_stats.json.
        name="pi05_droid_renderscale_fridge_m_2house_plus_v2_first500_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("fridge_m_2house", "fridge_m_v2"),
            assets=AssetsConfig(asset_id="fridge_m_2house_plus_v2_first500"),
            base_config=DataConfig(
                prompt_from_task=True,
                episode_filters={"fridge_m_v2": list(range(500))},
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Combined fridge_m_v2_gen + fridge_m_0.7_pd_gen multi-task (ConcatDataset).
        # fridge_m_v2_gen:     1272 eps / 380,821 frames (HF Ravenh97/lerobot_data:fridge_m_v2_gen).
        # fridge_m_0.7_pd_gen: 1018 eps / 304,491 frames (HF Ravenh97/lerobot_data:fridge_m_0.7_pd_gen).
        # Per-frame uniform sampling -> fridge_m_v2_gen 56% / fridge_m_0.7_pd_gen 44% by frame count.
        # Norm stats land at assets/<cfg>/fridge_m_v2_gen_plus_0_7_pd_gen/norm_stats.json.
        name="pi05_droid_renderscale_fridge_m_v2_gen_plus_0_7_pd_gen_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("fridge_m_v2_gen", "fridge_m_0.7_pd_gen"),
            assets=AssetsConfig(asset_id="fridge_m_v2_gen_plus_0_7_pd_gen"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Combined fridge_m_sim + fridge_m_v2 multi-task (ConcatDataset).
        # fridge_m_sim: 666 eps / 198,756 frames (HF Ravenh97/lerobot_data:fridge_m_sim).
        # fridge_m_v2:  1272 eps / 380,821 frames (HF Ravenh97/lerobot_data:fridge_m_v2).
        # Per-frame uniform sampling -> fridge_m_v2 dominates 65%/35% by frame count.
        # Norm stats land at assets/<cfg>/fridge_m_sim_plus_v2/norm_stats.json.
        name="pi05_droid_renderscale_fridge_m_sim_plus_v2_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("fridge_m_sim", "fridge_m_v2"),
            assets=AssetsConfig(asset_id="fridge_m_sim_plus_v2"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Combined fridge_m_0.7_pd_sim + fridge_m_v2 multi-task (ConcatDataset).
        # fridge_m_0.7_pd_sim: 1018 eps / 304,491 frames (HF Ravenh97/lerobot_data:fridge_m_0.7_pd_sim).
        # fridge_m_v2:         1272 eps / 380,821 frames (HF Ravenh97/lerobot_data:fridge_m_v2).
        # Per-frame uniform sampling -> fridge_m_v2 56% / fridge_m_0.7_pd_sim 44% by frame count.
        # Norm stats land at assets/<cfg>/fridge_m_0_7_pd_sim_plus_v2/norm_stats.json.
        name="pi05_droid_renderscale_fridge_m_0_7_pd_sim_plus_v2_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("fridge_m_v2", "fridge_m_0.7_pd_sim"),
            assets=AssetsConfig(asset_id="fridge_m_0_7_pd_sim_plus_v2"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=48,  # default: 12/GPU on 4x L40S; train.py CLI override to 64/72 if GPU memory permits
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Combined fridge_m_2house + first-500-eps fridge_m_v2 multi-task (ConcatDataset).
        # fridge_m_2house:    990 eps / 296,092 frames (HF Ravenh97/lerobot_data:fridge_m_2house).
        # fridge_m_v2 (0:500): ~149,694 frames (first 500 of 1272 eps, via episode_filters).
        # Per-frame uniform sampling -> fridge_m_2house ~66% / fridge_m_v2-500 ~34% by frame count.
        # Norm stats land at assets/<cfg>/fridge_m_2house_plus_v2_500/norm_stats.json.
        name="pi05_droid_renderscale_fridge_m_2house_plus_v2_500_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("fridge_m_2house", "fridge_m_v2"),
            assets=AssetsConfig(asset_id="fridge_m_2house_plus_v2_500"),
            base_config=DataConfig(
                prompt_from_task=True,
                episode_filters={"fridge_m_v2": list(range(500))},
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=48,  # default: 12/GPU on 4x L40S; train.py CLI override to 64/72 if GPU memory permits
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Combined fridge_m_2house_dr (all) + fridge_m_v2_dr (first 500 eps)
        # multi-task (ConcatDataset). Both are DR-rendered (domain-randomized
        # RGB) datasets pulled from HF Ravenh97/lerobot_data.
        # fridge_m_2house_dr:    990 eps / 296,092 frames (full).
        # fridge_m_v2_dr (0:500): first 500 of 6360 eps via episode_filters.
        # Both are DR-pushed (datasets 4.x 'List' feature) — wrappers in
        # /scr/ravenh/fridge_m_2house_dr_train/ apply the List->Sequence alias
        # before invoking openpi scripts.
        # Norm stats land at assets/<cfg>/fridge_m_2house_dr_plus_v2_dr_500/norm_stats.json.
        name="pi05_droid_renderscale_fridge_m_2house_dr_plus_v2_dr_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("fridge_m_2house_dr", "fridge_m_v2_dr"),
            assets=AssetsConfig(asset_id="fridge_m_2house_dr_plus_v2_dr_500"),
            base_config=DataConfig(
                prompt_from_task=True,
                episode_filters={
                    "fridge_m_v2_dr": list(range(500)),
                },
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 18/GPU on 4x L40S; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # 3drawers_v2 DR variant — sim-rendered RGB with domain randomization
        # (texture_swap + bind_untextured, DR_COUNT=5). 5450 episodes,
        # 1,467,350 frames, 15 fps. LeRobot dataset pushed to
        # Ravenh97/lerobot_data:3drawers_v2_dr/. Local at
        # /scr/ravenh/3drawers_v2_dr_train/lerobot_home/3drawers_v2_dr/.
        name="pi05_droid_renderscale_3drawers_v2_h32_dr",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="3drawers_v2_dr",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; also divisible by 4 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Combined fridge_m_0.7_pd_dr + fridge_m_v2_dr multi-task (ConcatDataset),
        # each capped at first 1000 episodes via DataConfig.episode_filters
        # (200 trajs × 5 DR variants = 1000 episodes, so this is the first 200
        # trajs of each source dataset, all DR variants).
        # Both are DR-pushed (datasets 4.x 'List' feature) — wrappers in
        # /scr/ravenh/fridge_m_0_7_pd_dr_plus_v2_dr_train/ apply the
        # List->Sequence alias before invoking openpi scripts.
        # Norm stats land at assets/<cfg>/fridge_m_0_7_pd_dr_plus_v2_dr_first1k/norm_stats.json.
        name="pi05_droid_renderscale_fridge_m_0_7_pd_dr_plus_v2_dr_first1k_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("fridge_m_0.7_pd_dr", "fridge_m_v2_dr"),
            assets=AssetsConfig(asset_id="fridge_m_0_7_pd_dr_plus_v2_dr_first1k"),
            base_config=DataConfig(
                prompt_from_task=True,
                episode_filters={
                    "fridge_m_0.7_pd_dr": list(range(1000)),
                    "fridge_m_v2_dr": list(range(1000)),
                },
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=72,  # 12/GPU on 6x H200; divisible by 4/6/8 GPUs (norm stats compatibility)
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # fridge_m_0.7_pd DR variant capped at first 1000 episodes via
        # DataConfig.episode_filters. Pulls HF Ravenh97/lerobot_data:fridge_m_0.7_pd_dr/.
        # Local: /scr/ravenh/fridge_m_0_7_pd_dr_train/lerobot_home/fridge_m_0.7_pd_dr/.
        name="pi05_droid_renderscale_fridge_m_0_7_pd_dr_first1k_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="fridge_m_0.7_pd_dr",
            base_config=DataConfig(
                prompt_from_task=True,
                episode_filters={
                    "fridge_m_0.7_pd_dr": list(range(1000)),
                },
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,  # default; launch overrides to --batch_size=72 on 6x H200
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Joint training on fridge_7_sim + fridge_m_sim pulled from HF repo
        # Ravenh97/lerobot_data (subfolders fridge_7_sim/ and fridge_m_sim/).
        # Set HF_LEROBOT_HOME=/viscam/projects/egodex/renderscale/lerobot_data_fridge_7_m
        # so LeRobot finds both subdirs locally. Norm stats land under
        # assets/<config>/renderscale/fridge_7_m_sim/.
        name="pi05_droid_renderscale_fridge_7_m_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=(
                "fridge_7_sim",
                "fridge_m_sim",
            ),
            assets=AssetsConfig(asset_id="renderscale/fridge_7_m_sim"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same as pi05_droid_renderscale_fridge but with action_horizon=32.
        # Used to serve the run_fridge_h32_1710eps_20260430_222143 checkpoint.
        # Norm stats live inside the checkpoint, so no separate asset dir is required.
        name="pi05_droid_renderscale_fridge_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/fridge_open",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Fridge h32 trained on the fixed-prompt video-generated set staged under
        # /viscam/projects/egodex/renderscale/lerobot_data_generated/renderscale/fridge_fixed_prompt
        # (built by convert_molmodata_generated_to_lerobot.py from the matching
        # generated_*.mp4 + source h5 in molmodata/fridge). Set
        # HF_LEROBOT_HOME=/viscam/projects/egodex/renderscale/lerobot_data_generated
        # for compute_norm_stats.py / train.py.
        name="pi05_droid_renderscale_fridge_fixed_prompt_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/fridge_fixed_prompt",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Fridge-only h32, capped at the first 1000 episodes of
        # /viscam/projects/egodex/renderscale/lerobot_data/renderscale/fridge_open
        # (full repo has 2852 eps) via episode_filters. action_horizon=32, 40k steps.
        # Set HF_LEROBOT_HOME=/viscam/projects/egodex/renderscale/lerobot_data for both
        # compute_norm_stats.py and train.py.
        name="pi05_droid_renderscale_fridge_h32_1k",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/fridge_open",
            base_config=DataConfig(
                prompt_from_task=True,
                episode_filters={
                    "renderscale/fridge_open": list(range(1000)),
                },
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same as pi05_droid_renderscale_fridge but with action_horizon=50.
        # Used to serve the run_fridge_h50_1710eps_20260501_180144 checkpoint.
        # Norm stats live inside the checkpoint, so no separate asset dir is required.
        name="pi05_droid_renderscale_fridge_h50",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/fridge_open",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same as pi05_droid_renderscale_dishwasher but with action_horizon=50.
        # Used to serve the run_dishwasher_h50_2226eps_20260503_153723 checkpoint.
        # Norm stats live inside the checkpoint, so no separate asset dir is required.
        name="pi05_droid_renderscale_dishwasher_h50",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/dishwasher_open",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same as pi05_droid_renderscale_dishwasher but with action_horizon=32.
        # Used to serve the run_dishwasher_998eps_h32_20260501_122141 checkpoint.
        # Norm stats live inside the checkpoint, so no separate asset dir is required.
        name="pi05_droid_renderscale_dishwasher_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/dishwasher_open",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same recipe, trained on the dishwasher open molmospaces dataset
        # (998 trajectories assembled from three partial parallel-worker runs
        # with non-overlapping shard indices, so seeds don't collide).
        name="pi05_droid_renderscale_dishwasher",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/dishwasher_open",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same as pi05_droid_renderscale_cabinet but with action_horizon=32.
        # Used to serve the run_cabinet_h32_1337eps_40k_20260501_133805 checkpoint.
        name="pi05_droid_renderscale_cabinet_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/cabinet_open",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Same recipe, trained on the cabinet open molmospaces dataset
        # (1337 trajectories from the complete 15-shard parallel-worker run;
        # an earlier 8-shard run in the same dir was discarded since its
        # seeds 1001..1008 overlap with the _of_15 set).
        name="pi05_droid_renderscale_cabinet",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id="renderscale/cabinet_open",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Multi-task molmospaces drawer/fridge/cabinet/dishwasher trained
        # jointly. Uses the tuple-repo_id path through create_torch_dataset
        # (torch ConcatDataset over four LeRobotDatasets). Episodes are
        # sampled in proportion to their frame counts:
        #   drawer    886740 frames (3376 eps)
        #   fridge    834501 frames (2852 eps, 11 prompt variants)
        #   cabinet   332327 frames (1337 eps)
        #   dishwasher 660156 frames (2226 eps)
        # Norm stats are computed jointly under assets/<config>/renderscale/all_open/.
        # See pi05_droid_renderscale_drawer_fridge_h32_minimal below for the
        # drawer+fridge-only variant trained on the smaller "minimal" datasets.
        name="pi05_droid_renderscale_all_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=(
                "renderscale/drawer_open",
                "renderscale/fridge_open",
                "renderscale/cabinet_open",
                "renderscale/dishwasher_open",
            ),
            assets=AssetsConfig(asset_id="renderscale/all_open"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Two-task ConcatDataset on the "minimal" datasets at
        # /viscam/projects/egodex/renderscale/lerobot_data_minimal/. Source sizes:
        #   drawer_open    1100 eps / 298593 frames
        #   fridge_open    2424 eps / 708640 frames
        # episode_filters caps each repo at the first 1000 episodes for a balanced
        # ~1k+1k training set. action_horizon=32, 40k steps.
        # Set HF_LEROBOT_HOME=/viscam/projects/egodex/renderscale/lerobot_data_minimal
        # for both compute_norm_stats.py and train.py.
        name="pi05_droid_renderscale_drawer_fridge_h32_minimal",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=(
                "renderscale/drawer_open",
                "renderscale/fridge_open",
            ),
            assets=AssetsConfig(asset_id="renderscale/drawer_fridge_open_minimal"),
            base_config=DataConfig(
                prompt_from_task=True,
                episode_filters={
                    "renderscale/drawer_open": list(range(1000)),
                    "renderscale/fridge_open": list(range(1000)),
                },
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=16,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
    TrainConfig(
        # Combined 3drawers_v2 + dresser_sim multi-task (torch ConcatDataset).
        # 3drawers_v2: HF Ravenh97/lerobot_data:3drawers_v2/, local
        #   ~/.cache/huggingface/lerobot/3drawers_v2.
        # dresser_sim: HF Ravenh97/lerobot_data:dresser_sim/ (1207 eps /
        #   337,566 frames), local
        #   ~/.cache/huggingface/lerobot/Ravenh97/dresser_sim_lerobot.
        # Gripper *observation* (gripper_position) is pre-binarized on disk,
        # per-dataset thresholds; gripper *action* already binary in source.
        # Norm stats -> assets/<cfg>/3drawers_v2_plus_dresser_sim/norm_stats.json.
        name="pi05_droid_renderscale_3drawers_v2_plus_dresser_h32",
        project_name="renderscale-pi05",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotMolmospacesDroidDataConfig(
            repo_id=("3drawers_v2", "Ravenh97/dresser_sim_lerobot"),
            assets=AssetsConfig(asset_id="3drawers_v2_plus_dresser_sim"),
            base_config=DataConfig(prompt_from_task=True),
            binarize_gripper_threshold=0.5,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=64,  # divisible by 8x B200; launch overrides --batch_size after probe
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
