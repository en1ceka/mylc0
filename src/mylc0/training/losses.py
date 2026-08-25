"""Loss functions, ported from ``lczero-training/model/loss_function.py``.

The three losses this project trains with, exactly as upstream computes them:

**Policy** -- cross entropy (or KL) between the network's logits and the MCTS
visit distribution. Illegal moves are stored as ``-1`` in the training data;
with ``illegal_moves: mask`` their logits are set to ``-inf`` before the softmax
(the ``train_to_zero`` variant leaves them in). Targets are clamped at zero and
optionally temperature-scaled (``target ** (1/T)``, renormalised) -- that is how
the "soft" policy head is trained.

**Value** -- softmax cross entropy against the WDL target derived from ``(q, d)``:
``w = (1 + q - d) / 2``, ``l = (1 - q - d) / 2``. Which ``(q, d)`` pair is used
depends on ``value_type``: ``result`` is the game outcome, ``best``/``played``
are the search's Q, and so on.

**Moves left** -- Huber loss with ``delta = 10/20`` on targets and predictions
scaled by ``1/20``.

Weight decay is handled by the optimizer (NAdamW), not here; an explicit L2
term is available for parity with ``RegularizationLossConfig``.
"""

from __future__ import annotations

import fnmatch
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..net.config import (LossConfig, MovesLeftLossConfig, PolicyLossConfig,
                          RegularizationLossConfig, VALUE_TYPES,
                          ValueLossConfig)
from ..net.model import ModelPrediction

_NEG_INF = float("-inf")


def wdl_target_from_q_d(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    w = (1.0 + q - d) / 2.0
    l = (1.0 - q - d) / 2.0
    return torch.stack([w, d, l], dim=-1)


def q_from_wdl_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return probs[..., 0] - probs[..., 2]


class PolicyLoss:
    def __init__(self, cfg: PolicyLossConfig):
        self.cfg = cfg
        self.head_name = cfg.head_name
        self.metric_name = cfg.metric_name or cfg.head_name
        self.weight = cfg.weight
        self.mask_illegal = cfg.illegal_moves == "mask"
        self.kl = cfg.type == "kl"
        self.temperature = cfg.temperature if cfg.temperature > 0 else 1.0

    def __call__(self, predictions: ModelPrediction, values, probs) -> torch.Tensor:
        logits = predictions.policy[self.head_name].float()
        targets = probs
        if self.mask_illegal:
            logits = torch.where(targets >= 0, logits,
                                 torch.full_like(logits, _NEG_INF))
        targets = torch.relu(targets)
        if self.temperature != 1.0:
            targets = targets.pow(1.0 / self.temperature)
            total = targets.sum(dim=-1, keepdim=True)
            targets = targets / torch.where(total > 0, total,
                                            torch.ones_like(total))
        log_probs = torch.log_softmax(logits, dim=-1)
        # optax.safe_softmax_cross_entropy: rows that are entirely -inf would
        # produce NaNs; there is always at least one legal move, but guard the
        # product so that -inf * 0 does not appear.
        log_probs = torch.where(torch.isinf(log_probs),
                                torch.zeros_like(log_probs), log_probs)
        cross_entropy = -(targets * log_probs).sum(dim=-1)
        if self.kl:
            entropy = (targets * torch.log(targets.clamp_min(1e-30))).sum(dim=-1)
            return (cross_entropy + entropy).mean()
        return cross_entropy.mean()


class ValueLoss:
    def __init__(self, cfg: ValueLossConfig):
        self.head_name = cfg.head_name
        self.metric_name = cfg.metric_name or cfg.head_name
        self.weight = cfg.weight
        self.value_type = VALUE_TYPES[cfg.value_type]

    def __call__(self, predictions: ModelPrediction, values, probs) -> torch.Tensor:
        logits = predictions.value[self.head_name][0].float()
        q = values[:, self.value_type, 0]
        d = values[:, self.value_type, 1]
        target = wdl_target_from_q_d(q, d).detach()
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()


class MovesLeftLoss:
    def __init__(self, cfg: MovesLeftLossConfig):
        self.head_name = cfg.head_name
        self.metric_name = cfg.metric_name or cfg.head_name
        self.weight = cfg.weight
        self.value_type = VALUE_TYPES[cfg.value_type]
        self.scale = 20.0
        self.delta = 10.0 / 20.0

    def __call__(self, predictions: ModelPrediction, values, probs) -> torch.Tensor:
        pred = predictions.movesleft[self.head_name].float().view(-1) / self.scale
        target = values[:, self.value_type, 2] / self.scale
        return F.huber_loss(pred, target, delta=self.delta, reduction="mean")


class RegularizationLoss:
    def __init__(self, cfg: RegularizationLossConfig):
        self.metric_name = cfg.metric_name or "l2"
        self.weight = cfg.weight
        self.rules = cfg.rules
        self.otherwise_include = cfg.otherwise_include

    def selects(self, name: str) -> bool:
        for rule in self.rules:
            if fnmatch.fnmatch(name, rule["match"]):
                return bool(rule.get("include", True))
        return self.otherwise_include

    def __call__(self, model) -> torch.Tensor:
        total = None
        for name, param in model.named_parameters():
            if not self.selects(name):
                continue
            s = torch.sum(param.float() ** 2)
            total = s if total is None else total + s
        if total is None:
            return torch.zeros((), device=next(model.parameters()).device)
        return total


class LczeroLoss:
    """``LczeroLoss``: the weighted sum plus the per-head metrics."""

    def __init__(self, cfg: LossConfig):
        self.policy_losses = [PolicyLoss(c) for c in cfg.policy]
        self.value_losses = [ValueLoss(c) for c in cfg.value]
        self.movesleft_losses = [MovesLeftLoss(c) for c in cfg.movesleft]
        self.regularization_losses = [RegularizationLoss(c)
                                      for c in cfg.regularization]

    def __call__(self, model, predictions: ModelPrediction, probs, values
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        metrics: Dict[str, torch.Tensor] = {}
        total = None

        def accumulate(prefix: str, name: str, loss: torch.Tensor,
                       weight: float) -> None:
            nonlocal total
            metrics[f"{prefix}/{name}"] = loss.detach()
            weighted = loss * weight
            total = weighted if total is None else total + weighted

        for loss in self.policy_losses:
            accumulate("policy", loss.metric_name,
                       loss(predictions, values, probs), loss.weight)
        for loss in self.value_losses:
            accumulate("value", loss.metric_name,
                       loss(predictions, values, probs), loss.weight)
        for loss in self.movesleft_losses:
            accumulate("movesleft", loss.metric_name,
                       loss(predictions, values, probs), loss.weight)
        for loss in self.regularization_losses:
            if loss.weight == 0.0:
                continue
            accumulate("regularization", loss.metric_name, loss(model),
                       loss.weight)
        assert total is not None, "at least one loss must be configured"
        return total, metrics


def policy_accuracy(predictions: ModelPrediction, probs: torch.Tensor,
                    head: str) -> torch.Tensor:
    """Fraction of positions where the argmax matches the most-visited move."""
    logits = predictions.policy[head]
    logits = torch.where(probs >= 0, logits, torch.full_like(logits, _NEG_INF))
    return (logits.argmax(dim=-1) == probs.argmax(dim=-1)).float().mean()
