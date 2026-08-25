"""Optimizer and learning-rate schedule.

``NAdamW`` is a literal implementation of what ``lczero-training`` uses --
``optax.nadamw``, i.e. ``optax.adamw(..., nesterov=True)``:

    mu     = b1 * mu + (1 - b1) * g
    nu     = b2 * nu + (1 - b2) * g^2
    mu_hat = b1 * mu / (1 - b1^(t+1)) + (1 - b1) * g / (1 - b1^t)
    nu_hat = nu / (1 - b2^t)
    update = mu_hat / (sqrt(nu_hat) + eps) + weight_decay * param
    param -= lr * update

(PyTorch's own ``NAdam`` uses Dozat's momentum-decay schedule instead of this
Nesterov correction, which is why the optimizer is spelled out here.)

Weight decay is decoupled and applied only to the parameters selected by the
config's ``decay_rules`` -- upstream's ``decay_selector``, first matching rule
wins.

``make_lr_schedule`` reproduces ``training/lr_schedule.py``: a list of rules,
each a piecewise sequence of intervals with CONSTANT / LINEAR / COSINE
transitions, the last interval optionally open-ended, optionally looping.
"""

from __future__ import annotations

import fnmatch
import math
from typing import Callable, Dict, List, Sequence

import torch
from torch.optim import Optimizer

from ..net.config import LrScheduleConfig, NadamwConfig, OptimizerConfig


class NAdamW(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.98),
                 eps: float = 1e-7, weight_decay: float = 0.0,
                 eps_root: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        eps_root=eps_root)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            eps_root = group["eps_root"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["mu"] = torch.zeros_like(p)
                    state["nu"] = torch.zeros_like(p)
                mu, nu = state["mu"], state["nu"]
                state["step"] += 1
                t = state["step"]

                mu.mul_(b1).add_(grad, alpha=1 - b1)
                nu.mul_(b2).addcmul_(grad, grad, value=1 - b2)

                bias1_t = 1 - b1 ** t
                bias1_t1 = 1 - b1 ** (t + 1)
                bias2_t = 1 - b2 ** t

                mu_hat = mu.mul(b1 / bias1_t1).add_(grad, alpha=(1 - b1) / bias1_t)
                nu_hat = nu.div(bias2_t)
                update = mu_hat.div_(nu_hat.add_(eps_root).sqrt_().add_(eps))
                if wd:
                    update.add_(p, alpha=wd)
                p.add_(update, alpha=-lr)
        return loss


def _selects(name: str, rules: Sequence[Dict], otherwise: bool) -> bool:
    for rule in rules:
        if fnmatch.fnmatch(name, rule["match"]):
            return bool(rule.get("include", True))
    return otherwise


def build_optimizer(model, cfg: OptimizerConfig, lr: float) -> Optimizer:
    if cfg.type != "nadamw":
        raise ValueError(f"Unsupported optimizer type {cfg.type!r}")
    n: NadamwConfig = cfg.nadamw
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _selects(name, n.decay_rules, n.decay_otherwise_include):
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": n.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return NAdamW(groups, lr=lr, betas=(n.beta_1, n.beta_2), eps=n.epsilon)


def decayed_parameter_names(model, cfg: OptimizerConfig) -> List[str]:
    n = cfg.nadamw
    return [name for name, p in model.named_parameters()
            if p.requires_grad and _selects(name, n.decay_rules,
                                            n.decay_otherwise_include)]


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------
_TRANSITIONS = {"constant": 0, "linear": 1, "cosine": 2}


def _rule_fn(rule: LrScheduleConfig) -> Callable[[int], float]:
    durations = list(rule.duration_steps)
    lrs = list(rule.lr)
    if not durations or not lrs:
        value = lrs[-1] if lrs else 0.0
        return lambda step: value
    period = sum(durations)
    if period == 0:
        value = lrs[-1]
        return lambda step: value
    transitions = [
        _TRANSITIONS[(rule.transition[i] if i < len(rule.transition)
                      else "constant").lower()]
        for i in range(len(durations))]
    ends, acc = [], 0
    for d in durations:
        acc += d
        ends.append(acc)
    starts = [e - d for e, d in zip(ends, durations)]
    a_vals = [lrs[i] if i < len(lrs) else lrs[-1] for i in range(len(durations))]
    b_vals = [lrs[i + 1] if i + 1 < len(lrs) else a_vals[i]
              for i in range(len(durations))]
    last_lr = lrs[-1]
    start_step = rule.starting_step
    looping = rule.loop

    def fn(step: int) -> float:
        rel = step - start_step
        if looping:
            rel = rel % period
        for i, (s, e, dur) in enumerate(zip(starts, ends, durations)):
            if dur > 0 and s <= rel < e:
                t = min(1.0, max(0.0, (rel - s) / dur))
                a, b = a_vals[i], b_vals[i]
                if transitions[i] == 1:
                    return a + (b - a) * t
                if transitions[i] == 2:
                    return a + 0.5 * (1.0 - math.cos(math.pi * t)) * (b - a)
                return a
        return last_lr

    return fn


def make_lr_schedule(rules: Sequence[LrScheduleConfig]) -> Callable[[int], float]:
    """``training/lr_schedule.py: make_lr_schedule``."""
    if not rules:
        return lambda step: 0.0
    fns = [_rule_fn(r) for r in rules]
    starts = [r.starting_step for r in rules]
    first_lrs = [r.lr[0] if r.lr else 0.0 for r in rules]
    earliest = min(range(len(rules)), key=lambda i: starts[i])

    def schedule(step: int) -> float:
        if step < starts[earliest]:
            return first_lrs[earliest]
        active = max((i for i in range(len(rules)) if step >= starts[i]),
                     key=lambda i: starts[i])
        return float(fns[active](step))

    return schedule
