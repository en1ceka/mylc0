"""The Lc0 attention-body network, ported to PyTorch.

This is a direct transcription of the current official model in
``lczero-training/src/lczero_training/model/`` (JAX/Flax nnx):
``embedding.py``, ``encoder.py``, ``policy_head.py``, ``value_head.py``,
``movesleft_head.py``, ``shared.py`` and ``model.py``. It corresponds to
``NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT`` in ``net.proto``:

    input planes (112 x 8 x 8)
      -> per-square embedding with positional preprocessing + ma-gating + FFN
      -> N pre-computed-bias transformer encoder blocks (smolgen + DeepNorm)
      -> attention policy head(s)   -> 1858 logits
         WDL value head(s)          -> 3 logits (+ optional error / categorical)
         moves-left head(s)         -> 1 non-negative scalar

Details kept identical to upstream on purpose:

* DeepNorm scaling: ``alpha = (2N)^-1/4`` on both residual branches, and
  ``beta = (8N)^-1/4`` used as the variance-scaling factor for the FFN, value
  and output projections.
* LayerNorm epsilon 1e-3 everywhere (Flax default is 1e-6, Lc0 overrides it).
* Attention logits are divided by ``sqrt(head_depth)``; smolgen adds a
  per-head 64x64 bias *before* the softmax.
* The multiplicative gate is passed through ReLU before it multiplies.
* The policy head's promotion logits reuse the rank-7 x rank-8 block of the QK
  matrix and add a learned per-file offset; knight promotions share the plain
  from/to logit.
* Parameter initialisation reproduces Flax's ``lecun_normal`` (truncated
  normal, stddev = sqrt(1/fan_in) / 0.8796...) and its ``variance_scaling``
  with ``mode="fan_avg"`` for the DeepNorm-initialised layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from ..chessrules.policy_map import ATTENTION_POLICY_GATHER, POLICY_SIZE
from .config import ModelConfig

# Flax's truncated-normal correction: stddev of N(0,1) truncated to [-2, 2].
_TRUNC_STD = 0.87962566103423978


def get_activation(name: str):
    name = (name or "default").lower().replace("activation_", "")
    if name in ("default", "mish"):
        return F.mish
    if name == "relu":
        return F.relu
    if name == "none":
        return lambda x: x
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return torch.sigmoid
    if name == "selu":
        return F.selu
    if name == "swish":
        return F.silu
    if name == "relu_2":
        return lambda x: torch.square(F.relu(x))
    if name == "softmax":
        return lambda x: torch.softmax(x, dim=-1)
    raise ValueError(f"Unknown activation {name!r}")


def _variance_scaling_(tensor: torch.Tensor, scale: float, mode: str) -> None:
    """Flax ``variance_scaling(scale, mode, "truncated_normal")``."""
    fan_out, fan_in = tensor.shape[0], tensor.shape[1]
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(mode)
    stddev = math.sqrt(scale / denom) / _TRUNC_STD
    nn.init.trunc_normal_(tensor, mean=0.0, std=stddev,
                          a=-2.0 * stddev, b=2.0 * stddev)


def linear(in_features: int, out_features: int, bias: bool = True,
           init_scale: Optional[float] = None,
           init_mode: str = "fan_in") -> nn.Linear:
    """``nnx.Linear`` with Flax's default (lecun_normal) or DeepNorm init."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    if init_scale is None:
        _variance_scaling_(layer.weight, 1.0, "fan_in")   # lecun_normal
    else:
        _variance_scaling_(layer.weight, init_scale, init_mode)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


def layer_norm(dim: int) -> nn.LayerNorm:
    return nn.LayerNorm(dim, eps=1e-3)


class Ffn(nn.Module):
    """``shared.Ffn``: linear -> activation -> linear, DeepNorm-initialised."""

    def __init__(self, in_features: int, hidden_features: int,
                 hidden_activation: str, deepnorm_beta: float):
        super().__init__()
        self.linear1 = linear(in_features, hidden_features,
                              init_scale=deepnorm_beta, init_mode="fan_avg")
        self.linear2 = linear(hidden_features, in_features,
                              init_scale=deepnorm_beta, init_mode="fan_avg")
        self.activation = hidden_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(get_activation(self.activation)(self.linear1(x)))


class MaGating(nn.Module):
    """Per-square multiplicative + additive gating (``embedding.MaGating``)."""

    def __init__(self, squares: int, channels: int):
        super().__init__()
        self.mult_gate = nn.Parameter(torch.ones(squares, channels))
        self.add_gate = nn.Parameter(torch.zeros(squares, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu(self.mult_gate) + self.add_gate


class Embedding(nn.Module):
    """``embedding.Embedding``."""

    def __init__(self, input_channels: int, cfg, defaults,
                 deepnorm_alpha: float, deepnorm_beta: float):
        super().__init__()
        self.activation = defaults.activation
        self.dense_size = cfg.dense_size
        assert cfg.dense_size > 0 and cfg.embedding_size > 0
        # Positional preprocessing: the 12 piece planes of the *current*
        # position, seen as one flat 768-vector, are projected to a per-square
        # feature vector and concatenated to the raw input.
        self.preprocess = linear(64 * 12, 64 * cfg.dense_size)
        self.embedding = linear(input_channels + cfg.dense_size,
                                cfg.embedding_size)
        self.norm = layer_norm(cfg.embedding_size)
        self.ma_gating = MaGating(64, cfg.embedding_size)
        self.deepnorm_alpha = deepnorm_alpha
        self.ffn = Ffn(cfg.embedding_size, cfg.dff, defaults.ffn_activation,
                       deepnorm_beta)
        self.out_norm = layer_norm(cfg.embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        pos_info = self.preprocess(x[:, :, :12].reshape(b, 64 * 12))
        pos_info = pos_info.view(b, 64, self.dense_size)
        x = torch.cat([x, pos_info], dim=2)
        x = self.embedding(x)
        x = get_activation(self.activation)(x)
        x = self.norm(x)
        x = self.ma_gating(x)
        x = x + self.ffn(x) * self.deepnorm_alpha
        return self.out_norm(x)


class Smolgen(nn.Module):
    """``encoder.Smolgen``: generates a per-head 64x64 attention bias."""

    def __init__(self, in_features: int, cfg, defaults, heads: int,
                 weight_gen_dense: nn.Linear):
        super().__init__()
        self.heads = heads
        self.compress = linear(in_features, cfg.hidden_channels, bias=False)
        self.dense1 = linear(cfg.hidden_channels * 64, cfg.hidden_size)
        self.ln1 = layer_norm(cfg.hidden_size)
        self.dense2 = linear(cfg.hidden_size, cfg.gen_size * heads)
        self.ln2 = layer_norm(cfg.gen_size * heads)
        # Held in a plain list so the shared generator matrix is registered
        # once, on the tower, instead of once per encoder block.
        self._weight_gen = [weight_gen_dense]
        self.activation = cfg.activation or defaults.activation
        self.gen_size = cfg.gen_size

    @property
    def weight_gen_dense(self) -> nn.Linear:
        return self._weight_gen[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        act = get_activation(self.activation)
        compressed = self.compress(x).reshape(b, -1)
        hidden = self.ln1(act(self.dense1(compressed)))
        gen_from = self.ln2(act(self.dense2(hidden)))
        gen_from = gen_from.view(b * self.heads, self.gen_size)
        out = self.weight_gen_dense(gen_from)
        return out.view(b, self.heads, 64, 64)


class MultiHeadAttention(nn.Module):
    """``encoder.MultiHeadAttention``."""

    def __init__(self, in_features: int, cfg, defaults,
                 smol_gen_dense: Optional[nn.Linear], deepnorm_beta: float):
        super().__init__()
        depth = cfg.d_model
        assert depth % cfg.heads == 0, "d_model must be divisible by heads"
        self.depth = depth
        self.num_heads = cfg.heads
        self.head_depth = depth // cfg.heads
        self.q = linear(in_features, depth)
        self.k = linear(in_features, depth)
        self.v = linear(in_features, depth,
                        init_scale=deepnorm_beta, init_mode="fan_avg")
        self.output_dense = linear(depth, in_features,
                                   init_scale=deepnorm_beta, init_mode="fan_avg")
        self.smolgen = (Smolgen(in_features, cfg.smolgen, defaults, cfg.heads,
                                smol_gen_dense)
                        if smol_gen_dense is not None else None)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b = t.shape[0]
        return t.view(b, 64, self.num_heads, self.head_depth).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        logits = torch.matmul(q, k.transpose(-1, -2))
        logits = logits / math.sqrt(self.head_depth)
        if self.smolgen is not None:
            logits = logits + self.smolgen(x)
        weights = torch.softmax(logits, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(b, 64, self.depth)
        return self.output_dense(out)


class EncoderBlock(nn.Module):
    """``encoder.EncoderBlock`` (post-LN with DeepNorm residual scaling)."""

    def __init__(self, in_features: int, cfg, defaults,
                 smol_gen_dense: Optional[nn.Linear], deepnorm_beta: float):
        super().__init__()
        self.mha = MultiHeadAttention(in_features, cfg, defaults,
                                      smol_gen_dense, deepnorm_beta)
        self.alpha = math.pow(2.0 * cfg.num_blocks, -0.25)
        self.ln1 = layer_norm(in_features)
        self.ffn = Ffn(in_features, cfg.dff, defaults.ffn_activation,
                       deepnorm_beta)
        self.ln2 = layer_norm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(x) * self.alpha
        out1 = self.ln1(x)
        return self.ln2(out1 + self.ffn(out1) * self.alpha)


class EncoderTower(nn.Module):
    def __init__(self, in_features: int, cfg, defaults, deepnorm_beta: float):
        super().__init__()
        assert cfg.smolgen is not None, (
            "the current Lc0 attention body always uses smolgen")
        # One shared generator matrix for every block, as upstream.
        self.smolgen_weight_gen = linear(cfg.smolgen.gen_size, 64 * 64,
                                         bias=False)
        self.blocks = nn.ModuleList([
            EncoderBlock(in_features, cfg, defaults, self.smolgen_weight_gen,
                         deepnorm_beta)
            for _ in range(cfg.num_blocks)])
        # Recompute block activations during the backward pass instead of
        # keeping them: a pure VRAM/compute trade-off that leaves the maths
        # (and therefore the network) untouched.
        self.grad_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.grad_checkpointing and self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(block, x,
                                                      use_reentrant=False)
            else:
                x = block(x)
        return x


class PolicyHead(nn.Module):
    """``policy_head.PolicyHead`` (POLICY_ATTENTION)."""

    def __init__(self, in_features: int, cfg, defaults,
                 shared_embedding: Optional[nn.Linear] = None):
        super().__init__()
        assert (shared_embedding is not None) != (cfg.embedding_size is not None), (
            "a policy head uses either the shared embedding or its own")
        self.activation = defaults.activation
        if shared_embedding is not None:
            # Registered on the model, not here (see the note in model.py
            # upstream about the shared embedding living at parent level).
            self._shared_tokens = [shared_embedding]
            self.tokens = None
            embedding_size = shared_embedding.out_features
        else:
            self._shared_tokens = []
            self.tokens = linear(in_features, cfg.embedding_size)
            embedding_size = cfg.embedding_size
        self.q = linear(embedding_size, cfg.d_model)
        self.k = linear(embedding_size, cfg.d_model)
        self.dk = math.sqrt(cfg.d_model)
        self.promotion_dense = linear(cfg.d_model, 4, bias=False)
        self.register_buffer(
            "gather_idx",
            torch.from_numpy(np.asarray(ATTENTION_POLICY_GATHER)).long(),
            persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.tokens if self.tokens is not None else self._shared_tokens[0]
        x = get_activation(self.activation)(tokens(x))
        q = self.q(x)
        k = self.k(x)
        qk = torch.matmul(q, k.transpose(1, 2))            # (B, 64, 64)

        promotion_keys = k[:, -8:, :]                       # rank-8 squares
        offsets = self.promotion_dense(promotion_keys)      # (B, 8, 4)
        offsets = offsets.transpose(1, 2) * self.dk         # (B, 4, 8)
        # The knight offset is the baseline added to the other three.
        offsets = offsets[:, :3, :] + offsets[:, 3:4, :]    # (B, 3, 8)

        n_promo = qk[:, -16:-8, -8:]                        # (B, 8, 8)
        promo = torch.stack([n_promo + offsets[:, i:i + 1, :]
                             for i in range(3)], dim=-1)    # (B, 8, 8, 3)

        logits = torch.cat([qk.reshape(b, 64 * 64) / self.dk,
                            promo.reshape(b, 8 * 24) / self.dk], dim=1)
        return logits.index_select(1, self.gather_idx)


class ValueHead(nn.Module):
    """``value_head.ValueHead`` (VALUE_WDL, optional error / categorical)."""

    def __init__(self, in_features: int, cfg, defaults):
        super().__init__()
        self.activation = defaults.activation
        self.has_error_output = cfg.has_error_output
        self.num_categorical_buckets = cfg.num_categorical_buckets
        self.num_channels = cfg.num_channels
        self.embed = linear(in_features, cfg.num_channels)
        self.dense1 = linear(cfg.num_channels * 64, 128)
        self.wdl = linear(128, 3)
        if self.has_error_output:
            self.error = linear(128, 1)
        if self.num_categorical_buckets > 0:
            self.categorical = linear(128, self.num_categorical_buckets)

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        act = get_activation(self.activation)
        x = act(self.embed(x).reshape(b, 64 * self.num_channels))
        x = act(self.dense1(x))
        wdl = self.wdl(x)
        error = torch.sigmoid(self.error(x)) if self.has_error_output else None
        cat = self.categorical(x) if self.num_categorical_buckets > 0 else None
        return wdl, error, cat


class MovesLeftHead(nn.Module):
    """``movesleft_head.MovesLeftHead`` (MOVES_LEFT_V1)."""

    def __init__(self, in_features: int, cfg, defaults):
        super().__init__()
        self.activation = defaults.activation
        self.num_channels = cfg.num_channels
        self.embed = linear(in_features, cfg.num_channels)
        self.dense1 = linear(cfg.num_channels * 64, 128)
        self.out = linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        act = get_activation(self.activation)
        x = act(self.embed(x).reshape(b, 64 * self.num_channels))
        x = act(self.dense1(x))
        return F.relu(self.out(x))


@dataclass
class ModelPrediction:
    value: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor],
                           Optional[torch.Tensor]]]
    policy: Dict[str, torch.Tensor]
    movesleft: Dict[str, torch.Tensor]


class LczeroModel(nn.Module):
    """``model.LczeroModel``."""

    INPUT_CHANNELS = 112

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n = config.encoder.num_blocks
        assert n > 0
        deepnorm_beta = math.pow(8.0 * n, -0.25)
        deepnorm_alpha = math.pow(2.0 * n, -0.25)

        self.embedding = Embedding(self.INPUT_CHANNELS, config.embedding,
                                   config.defaults, deepnorm_alpha,
                                   deepnorm_beta)
        self.encoders = EncoderTower(config.embedding.embedding_size,
                                     config.encoder, config.defaults,
                                     deepnorm_beta)

        emb = config.embedding.embedding_size
        self.value_heads = nn.ModuleDict({
            h.name: ValueHead(emb, h, config.defaults)
            for h in config.value_head})
        if config.shared_policy_embedding_size:
            self.policy_embedding_shared = linear(
                emb, config.shared_policy_embedding_size)
        else:
            self.policy_embedding_shared = None
        self.policy_heads = nn.ModuleDict({
            h.name: PolicyHead(emb, h, config.defaults,
                               self.policy_embedding_shared)
            for h in config.policy_head})
        self.movesleft_heads = nn.ModuleDict({
            h.name: MovesLeftHead(emb, h, config.defaults)
            for h in config.movesleft_head})

    def forward(self, x: torch.Tensor) -> ModelPrediction:
        # (B, 112, 8, 8) -> (B, 64, 112): one token per square.
        if x.dim() == 4:
            x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.embedding(x)
        x = self.encoders(x)
        return ModelPrediction(
            value={k: h(x) for k, h in self.value_heads.items()},
            policy={k: h(x) for k, h in self.policy_heads.items()},
            movesleft={k: h(x) for k, h in self.movesleft_heads.items()})

    # -- convenience -------------------------------------------------------
    def num_parameters(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


def build_model(config: ModelConfig) -> LczeroModel:
    model = LczeroModel(config)
    assert POLICY_SIZE == 1858
    return model
