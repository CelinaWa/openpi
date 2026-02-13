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

# -----------------------------
# YOUR dataset-specific policy
# -----------------------------
from openpi.policies.drone_to_table_policy import DroneToTableInputs, DroneToTableOutputs


ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None

    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False

    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
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
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
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


# ============================================================
# STEP 2: Your LeRobot data processing config
# ============================================================
@dataclasses.dataclass(frozen=True)
class LeRobotDroneToTableDataConfig(DataConfigFactory):
    """
    Defines how to process drone_to_table data from LeRobot dataset for training.

    Assumes LeRobot per-frame keys:
      - image (room_view)
      - wrist_image (drone_front)
      - state (xyz, shape (3,))
      - actions (xyz, shape (3,))
      - task (instruction string)
    """

    # If True, convert actions to deltas for training and invert at inference.
    # Keep False if your actions are already deltas.
    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack LeRobot keys to the "common" keys expected by policy transforms.
        # (This follows LIBERO pattern.)
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",  # produced when prompt_from_task=True
                        # If you don't use prompt_from_task, use:
                        # "prompt": "task",
                    }
                )
            ]
        )

        # Use YOUR transforms instead of libero_policy.*
        data_transforms = _transforms.Group(
            inputs=[DroneToTableInputs(model_type=model_config.model_type)],
            outputs=[DroneToTableOutputs(action_dim=3)],
        )

        # Optional extra delta transform (xyz only => mask length 3)
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(3, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
        )


# ============================================================
# TrainConfig definition (same as example)
# ============================================================
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi"
    exp_name: str = tyro.MISSING

    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    pytorch_weight_path: str | None = None
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    data: DataConfigFactory = dataclasses.field(default_factory=lambda: FakeDataConfig())  # noqa: E731

    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"

    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000

    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000

    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True

    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# ============================================================
# _CONFIGS: add π0, π0-FAST, π0.5 configs for YOUR dataset
# ============================================================
@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


_CONFIGS = [
    # ------------------------------------------------------------
    # π0 on your dataset
    # ------------------------------------------------------------
    TrainConfig(
        name="pi0_drone_to_table",
        model=pi0_config.Pi0Config(
            action_horizon=10,
            # pi0 base typically uses action_dim=8 in the configs, but
            # OpenPi pads to model_config.action_dim anyway.
            # You can leave default action_dim from Pi0Config or set explicitly.
        ),
        data=LeRobotDroneToTableDataConfig(
            repo_id="Celina717/drone_to_table_lerobot_xyz_10hz",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        batch_size=64,
        num_workers=2,
    ),

    # ------------------------------------------------------------
    # π0-FAST on your dataset (FAST expects action_dim = YOUR true action dim)
    # ------------------------------------------------------------
    TrainConfig(
        name="pi0_fast_drone_to_table",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=3,        # xyz only
            action_horizon=10,
            max_token_len=180,
        ),
        data=LeRobotDroneToTableDataConfig(
            repo_id="Celina717/drone_to_table_lerobot_xyz_10hz",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        batch_size=64,
        num_workers=2,
    ),

    # ------------------------------------------------------------
    # π0.5 on your dataset (pi05 base uses action_dim=32)
    # ------------------------------------------------------------
    TrainConfig(
        name="pi05_drone_to_table",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            action_dim=32,   # IMPORTANT for pi05 base checkpoints
        ),
        data=LeRobotDroneToTableDataConfig(
            repo_id="Celina717/drone_to_table_lerobot_xyz_10hz",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=64,
        num_workers=2,
    ),

    # Optional: keep other provided configs (not required)
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
