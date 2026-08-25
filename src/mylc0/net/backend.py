"""Neural-network evaluation backend.

Plays the role of lc0's ``Backend``: it turns positions into input planes,
batches them through the network, and returns for every position

* ``p`` -- priors for the legal moves, softmaxed over the *legal* logits only,
  with lc0's policy softmax temperature (``p = softmax(logit / T)``; note that
  ``trainingdata.cc`` undoes it with ``pow(p, T)``);
* ``q`` -- ``W - L`` from the WDL head, from the side-to-move's point of view;
* ``d`` -- the draw probability;
* ``m`` -- the moves-left head output.

It also owns the equivalent of lc0's NNCache. Cache keys include the last
``cache_history_length`` positions, because the input planes depend on history
(self-play uses 7, the engine default is 0).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..chessrules.position import PositionHistory
from .config import ModelConfig
from .encoder import (FILL_ALWAYS, FILL_FEN_ONLY, FILL_NO, TOTAL_PLANES,
                      encode_position)
from .model import LczeroModel

FILL_BY_NAME = {"no": FILL_NO, "fen_only": FILL_FEN_ONLY, "always": FILL_ALWAYS}


@dataclass
class EvalResult:
    """lc0's ``EvalResult``."""

    q: float
    d: float
    m: float
    p: np.ndarray  # priors aligned with the legal move list passed in


@dataclass
class BackendTiming:
    """Optional counters for profiling; ``None`` in normal runs."""

    batches: int = 0
    positions: int = 0
    min_batch: int = 10 ** 9
    max_batch: int = 0
    upload_s: float = 0.0     # staging copy + H2D
    gpu_s: float = 0.0        # pure forward, from CUDA events
    download_s: float = 0.0   # D2H (this is where the stream synchronises)
    post_s: float = 0.0       # per-position softmax over the legal moves
    total_s: float = 0.0
    histogram: dict = None

    def __post_init__(self):
        if self.histogram is None:
            self.histogram = {}

    def record_batch(self, n: int) -> None:
        self.batches += 1
        self.positions += n
        self.min_batch = min(self.min_batch, n)
        self.max_batch = max(self.max_batch, n)
        self.histogram[n] = self.histogram.get(n, 0) + 1

    @property
    def avg_batch(self) -> float:
        return self.positions / max(1, self.batches)


@dataclass
class EvalRequest:
    planes: np.ndarray          # (112, 8, 8) float32
    policy_indices: np.ndarray  # int32, one per legal move
    cache_key: Optional[tuple] = None


class NNCache:
    """Small LRU cache, the counterpart of lc0's NNCache."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: "OrderedDict[tuple, EvalResult]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key) -> Optional[EvalResult]:
        if self.capacity <= 0 or key is None:
            return None
        item = self._data.get(key)
        if item is None:
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return item

    def put(self, key, value: EvalResult) -> None:
        if self.capacity <= 0 or key is None:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


class Backend:
    def __init__(self,
                 model: LczeroModel,
                 model_config: ModelConfig,
                 device: str = "cuda",
                 fp16: bool = True,
                 max_batch_size: int = 32,
                 policy_softmax_temp: float = 1.359,
                 cache_size: int = 200000,
                 cache_history_length: int = 0,
                 history_fill: str = "fen_only",
                 policy_head: Optional[str] = None,
                 value_head: Optional[str] = None,
                 movesleft_head: Optional[str] = None):
        self.model = model
        self.config = model_config
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == "cuda"
        self.max_batch_size = max_batch_size
        self.policy_softmax_temp = policy_softmax_temp
        self.cache = NNCache(cache_size)
        self.cache_history_length = cache_history_length
        self.fill_empty_history = FILL_BY_NAME[history_fill]
        self.input_format = model_config.input_format
        self.policy_head = policy_head or model_config.primary_policy_head
        self.value_head = value_head or model_config.primary_value_head
        self.movesleft_head = movesleft_head or model_config.primary_movesleft_head
        self.evaluations = 0
        self.batches = 0
        self.timing: Optional[BackendTiming] = None

        self.model.to(self.device)
        self.model.eval()
        if self.fp16:
            self.model.half()
        self._scratch = np.zeros((max_batch_size, TOTAL_PLANES, 8, 8),
                                 dtype=np.float32)

    # -- position -> request ----------------------------------------------
    def cache_key(self, history: PositionHistory) -> tuple:
        snaps = history.snapshots
        depth = min(len(snaps), self.cache_history_length + 1)
        keys = tuple(s.key for s in snaps[len(snaps) - depth:])
        last = snaps[-1]
        return keys + (last.rule50, last.repetitions)

    def encode(self, history: PositionHistory) -> Tuple[np.ndarray, int]:
        return encode_position(history, self.input_format,
                               fill_empty_history=self.fill_empty_history)

    # -- evaluation --------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, requests: Sequence[EvalRequest]) -> List[EvalResult]:
        if not requests:
            return []
        results: List[Optional[EvalResult]] = [None] * len(requests)
        todo = []
        for i, req in enumerate(requests):
            cached = self.cache.get(req.cache_key)
            if cached is not None:
                results[i] = cached
            else:
                todo.append(i)
        for start in range(0, len(todo), self.max_batch_size):
            chunk = todo[start:start + self.max_batch_size]
            self._run_batch([requests[i] for i in chunk], chunk, results)
        return results  # type: ignore[return-value]

    def _run_batch(self, reqs, indices, results) -> None:
        timing = self.timing
        t_start = time.perf_counter() if timing is not None else 0.0
        n = len(reqs)
        batch = self._scratch[:n]
        for i, req in enumerate(reqs):
            batch[i] = req.planes
        tensor = torch.from_numpy(batch).to(self.device, non_blocking=True)
        if self.fp16:
            tensor = tensor.half()
        if timing is not None:
            t_upload = time.perf_counter()
            if self.device.type == "cuda":
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()

        out = self.model(tensor)
        policy = out.policy[self.policy_head].float()
        wdl = torch.softmax(out.value[self.value_head][0].float(), dim=-1)
        if self.movesleft_head is not None:
            mlh = out.movesleft[self.movesleft_head].float().view(-1)
        else:
            mlh = torch.zeros(n, device=policy.device)
        if timing is not None and self.device.type == "cuda":
            end_ev.record()
        # .cpu() is where the CPU waits for the queued GPU work.
        policy = policy.cpu().numpy()
        wdl = wdl.cpu().numpy()
        mlh = mlh.cpu().numpy()
        if timing is not None:
            t_download = time.perf_counter()

        self.evaluations += n
        self.batches += 1

        for i, (req, idx) in enumerate(zip(reqs, indices)):
            logits = policy[i][req.policy_indices]
            logits = logits / self.policy_softmax_temp
            logits -= logits.max()
            p = np.exp(logits)
            total = p.sum()
            if total > 0:
                p /= total
            else:  # degenerate; fall back to uniform
                p[:] = 1.0 / len(p)
            w, d, l = float(wdl[i][0]), float(wdl[i][1]), float(wdl[i][2])
            res = EvalResult(q=w - l, d=d, m=float(mlh[i]), p=p)
            results[idx] = res
            self.cache.put(req.cache_key, res)

        if timing is not None:
            now = time.perf_counter()
            timing.record_batch(n)
            timing.upload_s += t_upload - t_start
            timing.download_s += t_download - t_upload
            timing.post_s += now - t_download
            timing.total_s += now - t_start
            if self.device.type == "cuda":
                timing.gpu_s += start_ev.elapsed_time(end_ev) / 1000.0


def wdl_from_q_d(q: float, d: float) -> Tuple[float, float, float]:
    """(W, D, L) from Lc0's (Q, D) pair -- the inverse of ``q = w - l``."""
    w = (1.0 + q - d) / 2.0
    l = (1.0 - q - d) / 2.0
    return w, d, l
