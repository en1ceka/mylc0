"""Model / training / self-play configuration.

The model section mirrors ``lczero-training/proto/model_config.proto`` field for
field, so a config here can be read next to an upstream ``.textproto`` without
translation. The remaining sections cover the parts of
``proto/training_config.proto`` this project implements plus the self-play
knobs that live in lc0's ``selfplay/tournament.cc`` and ``search/classic/params.cc``.

Everything is loaded from YAML; see ``configs/``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class DefaultsConfig:
    compute_dtype: str = "f32"
    activation: str = "mish"
    ffn_activation: str = "mish"


@dataclass
class EmbeddingConfig:
    dense_size: int = 128
    embedding_size: int = 256
    dff: int = 384


@dataclass
class SmolgenConfig:
    hidden_channels: int = 32
    hidden_size: int = 256
    gen_size: int = 256
    activation: str = "swish"


@dataclass
class EncoderConfig:
    num_blocks: int = 6
    dff: int = 384
    d_model: int = 256
    heads: int = 8
    smolgen: Optional[SmolgenConfig] = field(default_factory=SmolgenConfig)


@dataclass
class PolicyHeadConfig:
    name: str = "vanilla"
    embedding_size: Optional[int] = 256
    d_model: int = 256


@dataclass
class ValueHeadConfig:
    name: str = "winner"
    num_channels: int = 32
    has_error_output: bool = False
    num_categorical_buckets: int = 0


@dataclass
class MovesLeftHeadConfig:
    name: str = "main"
    num_channels: int = 8


@dataclass
class ModelConfig:
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    shared_policy_embedding_size: Optional[int] = None
    policy_head: List[PolicyHeadConfig] = field(
        default_factory=lambda: [PolicyHeadConfig()])
    value_head: List[ValueHeadConfig] = field(
        default_factory=lambda: [ValueHeadConfig()])
    movesleft_head: List[MovesLeftHeadConfig] = field(
        default_factory=lambda: [MovesLeftHeadConfig()])
    # pblczero::NetworkFormat::InputFormat, see net/encoder.py.
    input_format: int = 5

    @property
    def primary_policy_head(self) -> str:
        return self.policy_head[0].name

    @property
    def primary_value_head(self) -> str:
        return self.value_head[0].name

    @property
    def primary_movesleft_head(self) -> Optional[str]:
        return self.movesleft_head[0].name if self.movesleft_head else None


# ---------------------------------------------------------------------------
# Losses / optimizer / schedule
# ---------------------------------------------------------------------------
# ValueType, from training_config.proto.
VALUE_TYPES = {"result": 0, "best": 1, "played": 2, "orig": 3, "root": 4, "st": 5}


@dataclass
class PolicyLossConfig:
    head_name: str = "vanilla"
    metric_name: str = ""
    weight: float = 1.0
    illegal_moves: str = "mask"          # "mask" | "train_to_zero"
    type: str = "cross_entropy"          # "cross_entropy" | "kl"
    temperature: float = 1.0


@dataclass
class ValueLossConfig:
    head_name: str = "winner"
    metric_name: str = ""
    weight: float = 1.0
    value_type: str = "result"


@dataclass
class MovesLeftLossConfig:
    head_name: str = "main"
    metric_name: str = ""
    weight: float = 1.0
    value_type: str = "result"


@dataclass
class RegularizationLossConfig:
    type: str = "l2"
    metric_name: str = "l2"
    weight: float = 0.0
    # Glob-ish rules, evaluated in order; first match wins.
    rules: List[Dict[str, Any]] = field(default_factory=list)
    otherwise_include: bool = True


@dataclass
class LossConfig:
    policy: List[PolicyLossConfig] = field(
        default_factory=lambda: [PolicyLossConfig()])
    value: List[ValueLossConfig] = field(
        default_factory=lambda: [ValueLossConfig()])
    movesleft: List[MovesLeftLossConfig] = field(
        default_factory=lambda: [MovesLeftLossConfig()])
    regularization: List[RegularizationLossConfig] = field(default_factory=list)


@dataclass
class NadamwConfig:
    beta_1: float = 0.9
    beta_2: float = 0.98
    epsilon: float = 1e-7
    weight_decay: float = 1e-4
    # Weight-decay selector, mirroring the example config's decay_selector.
    decay_rules: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"match": "*bias*", "include": False},
        {"match": "*.ln*", "include": False},
        {"match": "*norm*", "include": False},
        {"match": "*gate*", "include": False},
        {"match": "*policy_heads*", "include": True},
        {"match": "*value_heads*", "include": True},
        {"match": "*movesleft_heads*", "include": True},
    ])
    decay_otherwise_include: bool = False


@dataclass
class OptimizerConfig:
    type: str = "nadamw"
    nadamw: NadamwConfig = field(default_factory=NadamwConfig)


@dataclass
class LrScheduleConfig:
    starting_step: int = 0
    duration_steps: List[int] = field(default_factory=lambda: [1000, 0])
    lr: List[float] = field(default_factory=lambda: [0.0, 4e-4])
    transition: List[str] = field(default_factory=lambda: ["linear"])
    loop: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 128
    gradient_accumulation: int = 1
    steps_per_network: int = 250
    # Wait until this many *new* positions are available before each net.
    positions_per_network: int = 20000
    max_grad_norm: float = 10.0
    mixed_precision: bool = True
    # Recompute encoder activations in the backward pass to save VRAM.
    activation_checkpointing: bool = False
    checkpoint_path: str = "checkpoints"
    checkpoint_max_to_keep: int = 5
    checkpoint_every_steps: int = 250
    networks_path: str = "networks"
    tensorboard_path: str = "runs"
    # Data loader.
    chunk_pool_size: int = 4000
    position_sampling_rate: float = 0.10
    shuffle_buffer_size: int = 65536
    loader_workers: int = 2
    lr_schedule: List[LrScheduleConfig] = field(
        default_factory=lambda: [LrScheduleConfig()])
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    log_every_steps: int = 10


@dataclass
class SearchConfig:
    """Defaults from lc0 ``search/classic/params.cc``.

    Where ``selfplay/tournament.cc`` overrides a value for training games the
    self-play override is listed in ``SelfPlayConfig`` instead of here.
    """

    cpuct: float = 1.745
    cpuct_at_root: float = 1.745
    cpuct_base: float = 38739.0
    cpuct_base_at_root: float = 38739.0
    cpuct_factor: float = 3.894
    cpuct_factor_at_root: float = 3.894
    root_has_own_cpuct_params: bool = False
    fpu_strategy: str = "reduction"      # "reduction" | "absolute"
    fpu_value: float = 0.330
    fpu_strategy_at_root: str = "same"
    fpu_value_at_root: float = 1.0
    policy_softmax_temp: float = 1.359
    draw_score: float = 0.0
    two_fold_draws: bool = True
    minibatch_size: int = 32
    max_collision_visits: int = 80000
    max_collision_events: int = 917
    moves_left_max_effect: float = 0.0345
    moves_left_threshold: float = 0.8
    moves_left_slope: float = 0.0027
    moves_left_constant_factor: float = 0.0
    moves_left_scaled_factor: float = 1.6521
    moves_left_quadratic_factor: float = -0.6521
    history_fill: str = "fen_only"       # "no" | "fen_only" | "always"
    # Exploration (zero outside self-play, as in lc0).
    temperature: float = 0.0
    temp_decay_moves: int = 0
    temp_decay_delay_moves: int = 0
    temperature_cutoff_move: int = 0
    temperature_endgame: float = 0.0
    temperature_winpct_cutoff: float = 100.0
    temperature_visit_offset: float = 0.0
    noise_epsilon: float = 0.0
    noise_alpha: float = 0.3
    nncache_size: int = 200000


@dataclass
class SelfPlayConfig:
    """lc0 ``selfplay/tournament.cc`` training defaults."""

    visits: int = 800                # "go nodes" per move
    games_per_worker: int = 0        # 0 = unlimited
    workers: int = 1
    # Games in flight per worker, sharing one network batch. lc0's
    # selfplay/tournament.cc calls this "parallelism" and defaults it to 8.
    parallel_games: int = 8
    output_path: str = "data"
    max_game_ply: int = 450          # lc0 adjudicates at game ply 450
    resign_percentage: float = 0.0   # disabled by default, like lc0
    resign_wdl_style: bool = False
    resign_earliest_move: int = 0
    resign_playthrough: float = 0.0
    minimum_allowed_visits: int = 0
    reuse_tree: bool = False
    search: SearchConfig = field(default_factory=lambda: SearchConfig(
        # tournament.cc training overrides:
        cpuct=1.2,
        cpuct_at_root=1.2,
        cpuct_factor=0.0,
        cpuct_factor_at_root=0.0,
        policy_softmax_temp=1.0,
        minibatch_size=32,
        max_collision_visits=1,
        max_collision_events=1,
        temperature=1.0,
        noise_epsilon=0.25,
        fpu_value=0.0,
        history_fill="no",
        two_fold_draws=False,
    ))
    # Upper bound on one network call. This is an implementation limit only --
    # it does not touch the search, unlike SearchConfig.minibatch_size.
    batch_size: int = 256
    device: str = "cuda"
    fp16: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    name: str = "mylc0"


# ---------------------------------------------------------------------------
# YAML <-> dataclass
# ---------------------------------------------------------------------------
def _build(cls, data):
    if data is None:
        return cls()
    if not dataclasses.is_dataclass(cls):
        return data
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in fields:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        f = fields[key]
        kwargs[key] = _coerce(f.type, value)
    return cls(**kwargs)


_DATACLASS_BY_NAME = {}


def _coerce(type_hint, value):
    """Resolve nested dataclasses / lists of dataclasses from plain YAML."""
    name = type_hint if isinstance(type_hint, str) else getattr(type_hint, "__name__", "")
    # Optional[X] / List[X] arrive as strings because of `from __future__ import
    # annotations`; a tiny textual match is enough for this fixed schema.
    for cls_name, cls in _DATACLASS_BY_NAME.items():
        if name == cls_name or name == f"Optional[{cls_name}]":
            return _build(cls, value) if value is not None else None
        if name == f"List[{cls_name}]":
            return [_build(cls, v) for v in (value or [])]
    return value


for _cls in (DefaultsConfig, EmbeddingConfig, SmolgenConfig, EncoderConfig,
             PolicyHeadConfig, ValueHeadConfig, MovesLeftHeadConfig, ModelConfig,
             PolicyLossConfig, ValueLossConfig, MovesLeftLossConfig,
             RegularizationLossConfig, LossConfig, NadamwConfig, OptimizerConfig,
             LrScheduleConfig, TrainingConfig, SearchConfig, SelfPlayConfig,
             Config):
    _DATACLASS_BY_NAME[_cls.__name__] = _cls


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _build(Config, data)


def config_to_dict(cfg) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def model_config_from_dict(data: Dict[str, Any]) -> ModelConfig:
    return _build(ModelConfig, data)
