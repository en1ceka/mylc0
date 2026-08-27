"""Self-play performance telemetry.

Answers one question: **is the GPU waiting for the CPU, or the CPU for the
GPU?** Everything else here exists to support that answer.

The two headline numbers:

``gpu_starvation_pct``
    Share of wall time the GPU had nothing to compute. Measured from CUDA
    events: the GPU is busy exactly while a forward pass is running, so
    ``1 - sum(forward time) / wall`` is the idle share. Summed across workers
    when they share one GPU.

``cpu_wait_gpu_pct``
    Share of wall time a worker's Python thread sat blocked in ``.cpu()``
    waiting for a queued forward to finish.

In the current design a worker alternates between the two (there is no
CPU/GPU overlap inside one process), so the pair tells you which side to add
capacity to: high starvation means add workers or games in flight, high
CPU-wait means the GPU is the limit.

Nothing here is on by default; ``PerfCounters`` is created only under
``--perf-debug`` and the collection cost is a handful of ``perf_counter``
calls per network batch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
def detect_gpu() -> Dict[str, object]:
    """Static GPU description (name, VRAM, driver) via torch and nvidia-smi."""
    info: Dict[str, object] = {"name": None, "vram_gb": None, "count": 0}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(name=props.name, count=torch.cuda.device_count(),
                        vram_gb=round(props.total_memory / 1e9, 1),
                        capability=f"{props.major}.{props.minor}",
                        cuda=torch.version.cuda,
                        torch=torch.__version__)
    except Exception:
        pass
    return info


def detect_cpu() -> Dict[str, object]:
    """CPU description and the core count actually usable by this process.

    Under a container the cgroup quota, not the host core count, is what
    matters; ``os.sched_getaffinity`` reflects it on Linux.
    """
    info: Dict[str, object] = {}
    if psutil:
        info["logical"] = psutil.cpu_count(logical=True)
        info["physical"] = psutil.cpu_count(logical=False)
    try:
        info["affinity"] = len(os.sched_getaffinity(0))       # Linux
    except AttributeError:
        info["affinity"] = info.get("logical")
    if psutil:
        try:
            info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        except Exception:
            pass
    model = None
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        else:
            model = os.environ.get("PROCESSOR_IDENTIFIER")
    except Exception:
        pass
    info["model"] = model
    return info


def usable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def limit_thread_pools(threads: int = 1) -> None:
    """Keep BLAS/OpenMP from spawning a pool inside every worker.

    Must run *before* numpy/torch are imported to take effect, which is why
    the launcher sets these in the environment the children inherit. Torch's
    own intra-op count can still be lowered afterwards.
    """
    value = str(max(1, threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, value)


def set_affinity(cpu_ids) -> bool:
    """Pin this process to ``cpu_ids``. False when unsupported."""
    ids = list(cpu_ids)
    if not ids:
        return False
    try:
        os.sched_setaffinity(0, ids)                          # Linux
        return True
    except AttributeError:
        pass
    if psutil:
        try:
            psutil.Process().cpu_affinity(ids)                # Windows
            return True
        except Exception:
            return False
    return False


def affinity_slice(worker_id: int, workers: int) -> List[int]:
    """Contiguous block of CPUs for one worker, sized by what is available."""
    try:
        available = sorted(os.sched_getaffinity(0))
    except AttributeError:
        available = list(range(os.cpu_count() or 1))
    if workers <= 0 or not available:
        return available
    per = max(1, len(available) // workers)
    start = (worker_id * per) % len(available)
    return available[start:start + per] or available


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
_GPU_FIELDS = ("utilization.gpu", "utilization.memory", "memory.used",
               "memory.total", "power.draw", "power.limit",
               "clocks.current.sm")


def sample_gpu() -> Optional[Dict[str, float]]:
    """One nvidia-smi sample, or None when the tool is unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(_GPU_FIELDS),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        values = out.stdout.strip().splitlines()[0].split(",")
        parsed = {}
        for key, raw in zip(_GPU_FIELDS, values):
            raw = raw.strip()
            parsed[key] = float(raw) if raw not in ("", "[N/A]") else float("nan")
        return parsed
    except Exception:
        return None


def gpu_memory():
    """Total / used / free VRAM in MiB, or None without nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        total, used, free = [float(x) for x in
                             out.stdout.strip().splitlines()[0].split(",")]
        return {"total_mib": total, "used_mib": used, "free_mib": free}
    except Exception:
        return None


def gpu_processes():
    """Per-process VRAM: list of {"pid", "used_mib"}, newest driver first.

    ``used_mib`` is None where the driver refuses to break the number down --
    notably every consumer GPU on Windows, which runs in WDDM mode and hands
    memory management to the OS, so nvidia-smi reports ``[N/A]`` per process.
    The process list itself is still accurate there, which is what the CUDA
    context leak check actually needs.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    procs = []
    for line in out.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        try:
            used = float(parts[1])
        except ValueError:
            used = None          # "[N/A]" under WDDM
        procs.append({"pid": int(parts[0]), "used_mib": used})
    return procs


def wait_for_vram(baseline_mib: float, tolerance_mib: float = 256.0,
                  timeout: float = 30.0, poll: float = 0.5):
    """Block until VRAM falls back to ``baseline_mib`` (+ tolerance).

    Returns ``(ok, used_mib, waited_s)``. A process that died still holds its
    CUDA context until the driver reaps it, which takes a moment; starting the
    next trial before that finishes would charge the leftovers to it and can
    push a borderline configuration into a spurious OOM.
    """
    deadline = time.time() + timeout
    used = float("nan")
    start = time.time()
    while True:
        mem = gpu_memory()
        if mem is None:
            return True, float("nan"), 0.0
        used = mem["used_mib"]
        if used <= baseline_mib + tolerance_mib:
            return True, used, time.time() - start
        if time.time() >= deadline:
            return False, used, time.time() - start
        time.sleep(poll)


def sample_cpu(per_core: bool = True):
    if not psutil:
        return None, None
    total = psutil.cpu_percent(None)
    cores = psutil.cpu_percent(None, percpu=True) if per_core else None
    return total, cores


def runnable_threads(pids=None) -> int:
    """Threads of the given processes (default: every python process).

    Pass the worker PIDs when benchmarking -- counting every python process on
    the box would include editors, TensorBoard and the benchmark driver, and
    turn the oversubscription check into noise.
    """
    if not psutil:
        return 0
    total = 0
    if pids is not None:
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                total += proc.num_threads()
                for child in proc.children(recursive=True):
                    total += child.num_threads()
            except Exception:
                continue
        return total
    for proc in psutil.process_iter(["name", "num_threads"]):
        try:
            if "python" in (proc.info.get("name") or "").lower():
                total += proc.info.get("num_threads") or 0
        except Exception:
            continue
    return total


# ---------------------------------------------------------------------------
# Per-worker counters
# ---------------------------------------------------------------------------
@dataclass
class PerfCounters:
    """Wall-clock split for one self-play worker."""

    started: float = field(default_factory=time.perf_counter)
    # Phases, all measured on the worker's single Python thread.
    gather_s: float = 0.0        # tree descent (PUCT + make/unmake)
    encode_s: float = 0.0        # 112 planes + policy indices
    terminal_s: float = 0.0      # legal move generation for game-end checks
    apply_s: float = 0.0         # expansion + backup
    upload_s: float = 0.0        # staging + H2D
    gpu_wait_s: float = 0.0      # blocked in .cpu() waiting for the forward
    post_s: float = 0.0          # softmax over the legal moves
    gpu_busy_s: float = 0.0      # forward time from CUDA events
    # Volumes.
    batches: int = 0
    evals: int = 0
    cache_hits: int = 0
    nodes: int = 0
    plies: int = 0
    games: int = 0
    positions: int = 0
    batch_sizes: List[int] = field(default_factory=list)
    # Everything below is measured since the last rebaseline(), so a trial can
    # drop the ramp-up (games starting one by one, cold CUDA, empty cache).
    _base: Dict[str, float] = field(default_factory=dict)
    _base_batches: int = 0

    games_in_flight: int = 0
    phase: str = "running"

    def rebaseline(self) -> None:
        """Start a fresh measurement window at the current totals."""
        self._base = {
            "gather_s": self.gather_s, "encode_s": self.encode_s,
            "terminal_s": self.terminal_s, "apply_s": self.apply_s,
            "upload_s": self.upload_s, "gpu_wait_s": self.gpu_wait_s,
            "post_s": self.post_s, "gpu_busy_s": self.gpu_busy_s,
            "batches": self.batches, "evals": self.evals,
            "nodes": self.nodes, "plies": self.plies,
            "games": self.games, "positions": self.positions,
            "cache_hits": self.cache_hits,
        }
        self._base_batches = len(self.batch_sizes)
        self.started = time.perf_counter()

    def _since(self, key: str) -> float:
        return getattr(self, key) - self._base.get(key, 0.0)

    def wall(self) -> float:
        return max(1e-9, time.perf_counter() - self.started)

    def absorb_backend(self, timing) -> None:
        """Copy the backend's counters (they are cumulative)."""
        self.upload_s = timing.upload_s
        self.gpu_wait_s = timing.download_s
        self.post_s = timing.post_s
        self.gpu_busy_s = timing.gpu_s
        self.batches = timing.batches
        self.evals = timing.positions

    def absorb_search(self, timing) -> None:
        self.gather_s = timing.select_s - timing.encode_s - timing.terminal_s
        self.encode_s = timing.encode_s
        self.terminal_s = timing.terminal_s
        self.apply_s = timing.backup_s

    def percentiles(self):
        window = self.batch_sizes[self._base_batches:]
        if not window:
            return (0, 0, 0, 0)
        ordered = sorted(window)
        def pct(p):
            return ordered[min(len(ordered) - 1, int(p * len(ordered)))]
        return (sum(ordered) / len(ordered), pct(0.5), pct(0.95), ordered[-1])

    def snapshot(self) -> Dict[str, float]:
        wall = self.wall()
        avg, p50, p95, mx = self.percentiles()
        cpu_side = sum(self._since(k) for k in
                       ("gather_s", "encode_s", "terminal_s", "apply_s",
                        "post_s", "upload_s"))
        return {
            "wall_s": wall,
            "nodes_per_s": self._since("nodes") / wall,
            "evals_per_s": self._since("evals") / wall,
            "batches_per_s": self._since("batches") / wall,
            "avg_batch": avg, "p50_batch": p50, "p95_batch": p95,
            "max_batch": mx,
            "gather_pct": 100 * self._since("gather_s") / wall,
            "encode_pct": 100 * self._since("encode_s") / wall,
            "terminal_pct": 100 * self._since("terminal_s") / wall,
            "apply_pct": 100 * self._since("apply_s") / wall,
            "post_pct": 100 * self._since("post_s") / wall,
            "upload_pct": 100 * self._since("upload_s") / wall,
            "cpu_wait_gpu_pct": 100 * self._since("gpu_wait_s") / wall,
            # Only meaningful for a single process: with several processes
            # sharing the GPU each one's CUDA events also cover the time its
            # kernels waited behind another process, so they overlap.
            "gpu_forward_pct": 100 * self._since("gpu_busy_s") / wall,
            "cpu_busy_pct": 100 * cpu_side / wall,
            "games": self._since("games"), "positions": self._since("positions"),
            "plies": self._since("plies"), "evals": self._since("evals"),
            "cache_hits": self._since("cache_hits"),
            # Explicit names for two very different counts. ``live_plies``
            # includes games still being played, so throughput is measurable
            # seconds after start instead of only once games finish;
            # ``finalized_positions`` counts what finished games actually
            # wrote, and is what ends up in a shard. Reporting the first as
            # the second would overstate the dataset by everything in flight.
            "live_plies": self._since("plies"),
            "finalized_positions": self._since("positions"),
            "games_in_flight": self.games_in_flight,
            "phase": self.phase,
        }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_perf_block(global_stats: Dict[str, float],
                      workers: List[Dict[str, float]],
                      gpu: Optional[Dict[str, float]],
                      cpu_total: Optional[float],
                      cpu_cores: Optional[List[float]],
                      threads: int = 0) -> str:
    lines = ["", "PERF " + "-" * 60]
    g = global_stats
    lines.append(f"  positions/min      {g.get('positions_per_min', 0):>10.0f}")
    lines.append(f"  global nodes/s     {g.get('nodes_per_s', 0):>10.0f}")
    lines.append(f"  global evals/s     {g.get('evals_per_s', 0):>10.0f}")
    lines.append(f"  games done / flight{g.get('games', 0):>7.0f} /"
                 f" {g.get('in_flight', 0):.0f}")
    lines.append("")
    if cpu_total is not None:
        busy = sum(1 for c in (cpu_cores or []) if c > 50)
        lines.append(f"  CPU total          {cpu_total:>9.0f}%   "
                     f"cores >50%: {busy}/{len(cpu_cores or [])}")
        if cpu_cores:
            row = "  ".join(f"{c:>3.0f}" for c in cpu_cores)
            lines.append(f"  per core           {row}")
    if threads:
        lines.append(f"  python threads     {threads:>10}")
    if gpu:
        lines.append(f"  GPU util           {gpu.get('utilization.gpu', 0):>9.0f}%"
                     f"   mem-bus {gpu.get('utilization.memory', 0):.0f}%")
        lines.append(f"  VRAM               "
                     f"{gpu.get('memory.used', 0) / 1024:>7.1f} /"
                     f" {gpu.get('memory.total', 0) / 1024:.1f} GB")
        lines.append(f"  power / clock      "
                     f"{gpu.get('power.draw', 0):>7.0f} /"
                     f" {gpu.get('power.limit', 0):.0f} W"
                     f"   {gpu.get('clocks.current.sm', 0):.0f} MHz")
    lines.append("")
    lines.append(f"  NN avg / p50 / p95 / max batch   "
                 f"{g.get('avg_batch', 0):.0f} / {g.get('p50_batch', 0):.0f} /"
                 f" {g.get('p95_batch', 0):.0f} / {g.get('max_batch', 0):.0f}")
    lines.append(f"  NN batches/s       {g.get('batches_per_s', 0):>10.1f}")
    lines.append("")
    lines.append("  TIMING (share of one worker's wall clock, averaged)")
    for label, key in (("MCTS gather (PUCT)", "gather_pct"),
                       ("terminal / legal moves", "terminal_pct"),
                       ("input encoding", "encode_pct"),
                       ("apply + backup", "apply_pct"),
                       ("policy softmax", "post_pct"),
                       ("H2D upload", "upload_pct"),
                       ("blocked on GPU", "cpu_wait_gpu_pct")):
        value = g.get(key, 0.0)
        lines.append(f"    {label:<24s} {value:>5.1f}%  "
                     + "#" * int(round(value / 3)))
    lines.append("")
    lines.append(f"  GPU busy (CUDA events)          "
                 f"{g.get('gpu_busy_pct', 0):>5.1f}%")
    lines.append(f"  GPU starvation (idle, no batch) "
                 f"{g.get('gpu_starvation_pct', 0):>5.1f}%")
    lines.append(f"  CPU waiting for GPU             "
                 f"{g.get('cpu_wait_gpu_pct', 0):>5.1f}%")
    if len(workers) > 1:
        lines.append("")
        lines.append("  PER WORKER")
        for i, w in enumerate(workers):
            lines.append(
                f"    worker-{i}  nodes/s {w.get('nodes_per_s', 0):>6.0f}"
                f"  evals/s {w.get('evals_per_s', 0):>6.0f}"
                f"  batch {w.get('avg_batch', 0):>5.0f}"
                f"  cpu {w.get('cpu_busy_pct', 0):>4.0f}%"
                f"  waitGPU {w.get('cpu_wait_gpu_pct', 0):>4.0f}%"
                f"  games {w.get('games', 0):>3.0f}")
    alerts = diagnose(global_stats, gpu, cpu_total, threads)
    if alerts:
        lines.append("")
        for alert in alerts:
            lines.append("  " + alert.replace("\n", "\n  "))
    lines.append("-" * 65)
    return "\n".join(lines)


def diagnose(g: Dict[str, float], gpu: Optional[Dict[str, float]],
             cpu_total: Optional[float], threads: int = 0,
             cpus: Optional[int] = None) -> List[str]:
    """Turn the numbers into a named bottleneck."""
    out = []
    cpus = cpus or usable_cpus()
    gpu_util = gpu.get("utilization.gpu") if gpu else None
    starvation = g.get("gpu_starvation_pct", 0.0)
    wait = g.get("cpu_wait_gpu_pct", 0.0)
    avg_batch = g.get("avg_batch", 0.0)
    max_batch = g.get("max_batch", 0.0)

    if gpu_util is not None and gpu_util < 70 and starvation > 25:
        out.append(
            f"WARNING: GPU UNDERUTILIZED\n"
            f"  GPU util = {gpu_util:.0f}%, starvation = {starvation:.0f}%\n"
            f"  Likely bottleneck: the CPU cannot produce requests fast "
            f"enough.\n"
            f"  Try: more workers, or more parallel_games per worker.")
    if wait > 30 and (gpu_util is None or gpu_util > 85):
        out.append(
            f"WARNING: CPU WAITING FOR GPU\n"
            f"  CPU wait = {wait:.0f}%, GPU util = "
            f"{gpu_util if gpu_util is not None else float('nan'):.0f}%\n"
            f"  Likely bottleneck: network inference throughput.\n"
            f"  Try: a larger NN batch, or fewer workers (they are queueing).")
    if max_batch and avg_batch < 0.4 * max_batch and starvation > 15:
        out.append(
            f"WARNING: SMALL NN BATCHES\n"
            f"  avg batch = {avg_batch:.0f}, max = {max_batch:.0f}\n"
            f"  Likely cause: not enough games in flight to fill a batch.\n"
            f"  Try: raise parallel_games.")
    if threads and threads > 2.5 * cpus:
        out.append(
            f"WARNING: CPU OVERSUBSCRIBED\n"
            f"  python threads = {threads}, usable CPUs = {cpus}\n"
            f"  Try: OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 and "
            f"torch.set_num_threads(1) in every worker.")
    if cpu_total is not None and cpu_total > 92 and starvation > 20:
        out.append(
            f"WARNING: CPU SATURATED\n"
            f"  CPU = {cpu_total:.0f}% and the GPU is still idle "
            f"{starvation:.0f}% of the time.\n"
            f"  More workers will not help; the Python search is the limit.")
    return out


def write_json(path: str, payload) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        for _ in range(6):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.02)
    except OSError:
        pass
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
