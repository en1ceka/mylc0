"""One status line for a whole node, built from the workers' own telemetry.

Twenty-eight processes each logging what they are doing is unreadable; one
line every fifteen seconds is not. This aggregates what the workers already
publish and adds the machine-wide numbers only the parent can see.

**How the numbers get here.** Each worker keeps plain local counters and
writes a small JSON snapshot to its own file every couple of seconds, from
the tick it already runs between search batches. The parent reads those files.
There is no queue, no lock and no shared memory: a snapshot costs one ~1 kB
write per worker per second at most, which against 5000 positions/min of MCTS
is unmeasurable, and no worker can ever block on the parent or on another
worker. Adding an mp.Queue would introduce exactly the coupling this avoids --
a slow reader would apply backpressure to self-play itself.

**Two position counts, deliberately separate.**

``live_plies``
    Plies played, *including games still in progress*. This is the throughput
    number: it moves within seconds of start, so a misconfigured node is
    obvious immediately instead of after the first game ends minutes later.

``finalized_positions``
    Positions from finished games -- what a shard will actually contain. A
    game without a result has no value target and is never written, so this
    is the only count that describes the dataset.

Reporting the first as if it were the second would overstate the dataset by
everything currently in flight, which at 28x48 games is substantial.
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Dict, List

from ..perf import gpu_memory, sample_cpu, sample_gpu


def _fmt_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1e6:.1f}M"
    if value >= 10_000:
        return f"{value / 1e3:.0f}k"
    if value >= 1_000:
        return f"{value / 1e3:.1f}k"
    return f"{value:.0f}"


def _fmt_duration(seconds: float) -> str:
    if seconds != seconds or seconds < 0 or seconds > 86400 * 7:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class NodeStatus:
    """Aggregates worker snapshots into one line, and a final summary."""

    def __init__(self, perf_paths: List[str], target_positions: int = 0,
                 rate_window: float = 30.0):
        self.perf_paths = list(perf_paths)
        self.target_positions = max(0, int(target_positions))
        self.rate_window = max(1.0, rate_window)
        # Set on the first sample, from the same clock the samples use, so
        # elapsed cannot mix two time sources -- and so it measures the run
        # rather than however long the object happened to exist first.
        self.started = None
        # (timestamp, live_plies) for the instantaneous rate. A deque bounded
        # by time rather than by count so the window means the same thing
        # whatever the sampling interval is.
        self._history: deque = deque()
        self._gpu: List[Dict[str, float]] = []
        self._cpu: List[float] = []
        self._vram: List[float] = []
        self.last: Dict[str, float] = {}

    # -- collection --------------------------------------------------------
    def read_workers(self) -> List[Dict[str, float]]:
        """Whatever each worker last published. Missing files are simply not
        there yet -- a worker still loading its network has nothing to say."""
        out = []
        for path in self.perf_paths:
            try:
                with open(path, encoding="utf-8") as handle:
                    out.append(json.load(handle))
            except (OSError, ValueError):
                continue
        return out

    def sample(self, now: float = None) -> Dict[str, float]:
        """One aggregated reading. Safe to call before any worker exists.

        ``now`` is injectable so the rate arithmetic can be checked without
        sleeping through a heartbeat interval.
        """
        now = time.time() if now is None else now
        if self.started is None:
            self.started = now
        workers = self.read_workers()

        def total(key: str) -> float:
            return sum(float(w.get(key, 0) or 0) for w in workers)

        def mean(key: str) -> float:
            values = [float(w.get(key, 0) or 0) for w in workers]
            return sum(values) / len(values) if values else 0.0

        live_plies = total("live_plies") or total("plies")
        finalized = total("finalized_positions") or total("positions")

        self._history.append((now, live_plies))
        while (len(self._history) > 2
               and now - self._history[0][0] > self.rate_window):
            self._history.popleft()

        # Instantaneous rate over the window. Two samples separated by real
        # time are needed for a slope. Until then the rate is *unknown*, which
        # is not the same as zero: reporting 0 pos/min would be indis-
        # tinguishable from a node that has stopped dead, which is exactly the
        # condition this line exists to make obvious.
        rate_per_min = float("nan")
        if len(self._history) >= 2:
            t0, p0 = self._history[0]
            t1, p1 = self._history[-1]
            span = t1 - t0
            if span > 0.5:
                rate_per_min = max(0.0, (p1 - p0) / span * 60.0)

        elapsed = max(1e-9, now - self.started)
        average_per_min = live_plies / elapsed * 60.0 if elapsed > 1.0 else 0.0

        gpu = sample_gpu()
        if gpu:
            self._gpu.append(gpu)
        cpu_total, _cores = sample_cpu(per_core=False)
        if cpu_total is not None:
            self._cpu.append(cpu_total)
        memory = gpu_memory()
        if memory:
            self._vram.append(memory["used_mib"])

        gpu_busy = float(gpu["utilization.gpu"]) if gpu else float("nan")
        reading = {
            "workers_reporting": len(workers),
            "live_plies": live_plies,
            "finalized_positions": finalized,
            "games_done": total("games"),
            "games_in_flight": total("games_in_flight"),
            "positions_per_min": rate_per_min,
            "avg_positions_per_min": average_per_min,
            "nodes_per_s": total("nodes_per_s"),
            "evals_per_s": total("evals_per_s"),
            "avg_batch": mean("avg_batch"),
            "p50_batch": mean("p50_batch"),
            "p95_batch": mean("p95_batch"),
            "gpu_util": gpu_busy,
            "gpu_starvation_pct": (max(0.0, 100.0 - gpu_busy)
                                   if gpu_busy == gpu_busy else float("nan")),
            "cpu_wait_gpu_pct": mean("cpu_wait_gpu_pct"),
            "cpu_util": cpu_total if cpu_total is not None else float("nan"),
            "vram_used_mib": memory["used_mib"] if memory else float("nan"),
            "vram_total_mib": memory["total_mib"] if memory else float("nan"),
            "elapsed_s": now - self.started,
        }
        reading["progress"] = (live_plies / self.target_positions
                               if self.target_positions else 0.0)
        reading["eta_s"] = self.eta(live_plies, rate_per_min)
        self.last = reading
        return reading

    def eta(self, live_plies: float, rate_per_min: float) -> float:
        """Seconds until the shard target, or NaN when it cannot be known.

        Deliberately NaN rather than a guess before the rate has settled: a
        number that swings from 4 minutes to 4 hours between heartbeats is
        worse than admitting there is not enough information yet.
        """
        if (not self.target_positions or rate_per_min != rate_per_min
                or rate_per_min <= 0):
            return float("nan")
        remaining = self.target_positions - live_plies
        if remaining <= 0:
            return 0.0
        return remaining / rate_per_min * 60.0

    # -- output ------------------------------------------------------------
    def line(self, generation: int, network_sha: str, shard_index: int,
             backlog_gb: float = 0.0, uploaded: int = 0,
             failures: int = 0) -> str:
        r = self.last or self.sample()
        vram = ""
        if r["vram_used_mib"] == r["vram_used_mib"]:
            vram = (f" | VRAM {r['vram_used_mib'] / 1024:.1f}/"
                    f"{r['vram_total_mib'] / 1024:.1f}G")
        gpu = ("?" if r["gpu_util"] != r["gpu_util"]
               else f"{r['gpu_util']:.0f}%")
        cpu = ("?" if r["cpu_util"] != r["cpu_util"]
               else f"{r['cpu_util']:.0f}%")
        progress = (f" | shard {100 * r['progress']:.0f}%"
                    if self.target_positions else "")
        eta = (f" | ETA {_fmt_duration(r['eta_s'])}"
               if self.target_positions else "")
        rate = ("--" if r["positions_per_min"] != r["positions_per_min"]
                else f"{r['positions_per_min']:.0f}")
        return (
            f"gen {generation} ({network_sha[:8]}) | shard #{shard_index} | "
            f"{_fmt_count(r['live_plies'])} live pos "
            f"({_fmt_count(r['finalized_positions'])} final) | "
            f"{rate} pos/min | "
            f"{_fmt_count(r['nodes_per_s'])} nodes/s | "
            f"{_fmt_count(r['evals_per_s'])} evals/s | "
            f"GPU {gpu} | CPU {cpu}{vram} | "
            f"batch {r['avg_batch']:.0f} p50 {r['p50_batch']:.0f} "
            f"p95 {r['p95_batch']:.0f} | "
            f"{r['games_in_flight']:.0f} in flight | "
            f"{r['games_done']:.0f} done"
            f"{progress} | {_fmt_duration(r['elapsed_s'])} elapsed{eta} | "
            f"starv {r['gpu_starvation_pct']:.0f}% "
            f"waitGPU {r['cpu_wait_gpu_pct']:.0f}% | "
            f"backlog {backlog_gb:.1f}G | up {uploaded:.0f} | "
            f"fail {failures:.0f}")

    def summary(self) -> Dict[str, float]:
        """Final performance record for the shard manifest.

        Written into every shard so the nodes in a farm stay comparable after
        the fact: the same network and the same knobs on two rented machines
        should produce the same numbers, and when they do not, this says so
        without needing the node to still be running.
        """
        r = self.last or self.sample()
        elapsed = max(1e-9, r.get("elapsed_s", 0.0))
        return {
            "elapsed_s": round(elapsed, 1),
            "avg_positions_per_min": round(r["live_plies"] / elapsed * 60, 1),
            "finalized_positions": int(r["finalized_positions"]),
            "live_plies": int(r["live_plies"]),
            "nodes_per_s": round(r["nodes_per_s"], 1),
            "evals_per_s": round(r["evals_per_s"], 1),
            "avg_batch": round(r["avg_batch"], 1),
            "p50_batch": round(r["p50_batch"], 1),
            "p95_batch": round(r["p95_batch"], 1),
            "gpu_util_avg": round(_mean(
                [s["utilization.gpu"] for s in self._gpu]), 1),
            "gpu_starvation_pct": round(max(0.0, 100.0 - _mean(
                [s["utilization.gpu"] for s in self._gpu])), 1),
            "cpu_util_avg": round(_mean(self._cpu), 1),
            "cpu_wait_gpu_pct": round(r["cpu_wait_gpu_pct"], 1),
            "vram_peak_mib": round(max(self._vram), 1) if self._vram else None,
            "vram_total_mib": (round(r["vram_total_mib"], 1)
                               if r["vram_total_mib"] == r["vram_total_mib"]
                               else None),
        }


def _mean(values) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def confirmation_line(workers: int, runtime: Dict[str, object]) -> str:
    """The single line that replaces one startup log per worker."""
    return (f"workers confirmed: {workers} workers x "
            f"{runtime.get('parallel_games')} games, batch "
            f"{runtime.get('nn_batch')}, "
            f"{'fp16' if runtime.get('fp16') else 'fp32'}, visits "
            f"{runtime.get('visits')}, minibatch "
            f"{runtime.get('minibatch_size')}")


__all__ = ["NodeStatus", "confirmation_line"]
