"""Measure where self-play time actually goes.

    python scripts/profile_selfplay.py --config configs/small.yaml --seconds 90
    python scripts/profile_selfplay.py --config configs/small.yaml --workers 3
    python scripts/profile_selfplay.py --config configs/small.yaml --cprofile

Nothing here changes the algorithm: the search parameters, the visit count and
the network are exactly the ones in the config. The timers are the optional
counters on ``Backend``/``Search`` (``BackendTiming`` / ``SearchTiming``),
which are ``None`` outside this script.

With ``--workers N`` it instead launches the normal worker processes and only
measures end-to-end throughput plus GPU/CPU load, which is what you want for a
scaling test.
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import threading
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

from mylc0.net.backend import BackendTiming
from mylc0.net.config import load_config
from mylc0.net.model import build_model
from mylc0.search.search import SearchTiming
from mylc0.selfplay.batched import BatchedSelfPlay
from mylc0.selfplay.worker import (make_backend,
                                   read_live_stats, run_worker)

try:
    import psutil
except ImportError:
    psutil = None


class Sampler(threading.Thread):
    """Background sampling of GPU utilisation, VRAM and CPU load."""

    def __init__(self, interval: float = 0.25, stats_paths=None):
        super().__init__(daemon=True)
        self.interval = interval
        # Optional: follow the workers' live counters so throughput can be read
        # off the slope while the run is going, instead of from the final file
        # (which is written after the games in flight have been abandoned).
        self.stats_paths = stats_paths
        self.progress = []   # (t, plies played so far including in flight)
        self.gpu = []
        self.vram = []
        self.cpu_total = []
        self.cpu_proc = []
        self._stop = threading.Event()
        self._proc = psutil.Process() if psutil else None
        if self._proc:
            self._proc.cpu_percent(None)
            psutil.cpu_percent(None)

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                util, mem = out.stdout.strip().splitlines()[0].split(",")
                self.gpu.append(int(util))
                self.vram.append(int(mem))
            except Exception:
                pass
            if self._proc:
                self.cpu_total.append(psutil.cpu_percent(None))
                self.cpu_proc.append(self._proc.cpu_percent(None))
            if self.stats_paths:
                live = read_live_stats(self.stats_paths)
                self.progress.append(
                    (time.perf_counter(),
                     live.get("plies", 0) + live.get("current_plies", 0)))
            self._stop.wait(self.interval)

    def throughput(self):
        """Plies per second from the steady middle of the run."""
        pts = [p for p in self.progress if p[1] > 0]
        if len(pts) < 4:
            return None
        lo = pts[len(pts) // 5]
        hi = pts[-1]
        span = hi[0] - lo[0]
        if span <= 0:
            return None
        return (hi[1] - lo[1]) / span, hi[1]

    def stop(self):
        self._stop.set()
        self.join(timeout=3)

    def summary(self):
        def stat(values):
            if not values:
                return None
            return (float(np.mean(values)), int(np.max(values)))
        return {"gpu": stat(self.gpu), "vram": stat(self.vram),
                "cpu_total": stat(self.cpu_total),
                "cpu_proc": stat(self.cpu_proc)}


def profile_single(args, config):
    """One worker in-process, with the fine-grained timers switched on."""
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    cfg = config.selfplay
    device = args.device or cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.network and os.path.isfile(args.network):
        backend = make_backend(args.network, cfg, device, cfg.fp16)
        source = os.path.basename(args.network)
    else:
        from mylc0.net.backend import Backend
        model = build_model(config.model)
        backend = Backend(model, config.model, device=device, fp16=cfg.fp16,
                          max_batch_size=cfg.batch_size,
                          policy_softmax_temp=cfg.search.policy_softmax_temp,
                          cache_size=cfg.search.nncache_size,
                          cache_history_length=7,
                          history_fill=cfg.search.history_fill)
        source = "freshly initialised network"

    backend.timing = BackendTiming()
    search_timing = SearchTiming()

    # Hand the shared SearchTiming to every Search the game creates.
    import mylc0.search.search as search_module
    original_init = search_module.Search.__init__

    def patched_init(self, *a, **kw):
        original_init(self, *a, **kw)
        self.timing = search_timing
    search_module.Search.__init__ = patched_init

    print(f"network: {source}")
    print(f"device: {device}  fp16: {cfg.fp16}  visits/move: {cfg.visits}  "
          f"minibatch: {cfg.search.minibatch_size}  "
          f"max_collision_events: {cfg.search.max_collision_events}")
    print(f"games in flight: {cfg.parallel_games}  "
          f"max NN batch: {cfg.batch_size}")
    print(f"profiling for {args.seconds:.0f}s ...\n")

    sampler = Sampler()
    sampler.start()
    t0 = time.perf_counter()
    profiler = None
    if args.cprofile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    driver = BatchedSelfPlay(backend, cfg, max(1, cfg.parallel_games),
                             seed=args.seed)
    deadline = t0 + args.seconds
    totals = {"games": 0, "positions": 0, "nodes": 0, "seconds": 0.0,
              "plies": 0}

    def on_game(game):
        totals["games"] += 1
        totals["positions"] += len(game.data)
        totals["nodes"] += game.stats.nodes
        totals["seconds"] += game.stats.seconds
        totals["plies"] += game.stats.plies

    driver.run(on_game=on_game,
               should_stop=lambda: time.perf_counter() >= deadline,
               hard_stop=lambda: time.perf_counter() >= deadline)
    games = totals["games"]
    positions = totals["positions"]
    finished_game_seconds = totals["seconds"]
    # Count the games still in flight too: their moves were really played, they
    # just will not be written as training data.
    in_flight = [r.game for r in driver.runners if r.game is not None]
    plies = totals["plies"] + sum(g.stats.plies for g in in_flight)
    nodes = totals["nodes"] + sum(g.stats.nodes for g in in_flight)

    if profiler:
        profiler.disable()
    elapsed = time.perf_counter() - t0
    sampler.stop()
    search_module.Search.__init__ = original_init

    report(args, config, backend, search_timing, sampler, elapsed, games,
           plies, positions, nodes, finished_game_seconds, driver)
    if profiler:
        print("\n--- cProfile, top 18 by cumulative time -------------------")
        import io
        import pstats
        buf = io.StringIO()
        pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(18)
        text = buf.getvalue()
        start = text.find("ncalls")
        print(text[start:start + 2600])


def report(args, config, backend, st, sampler, elapsed, games, plies,
           positions, nodes, game_seconds, driver=None):
    bt = backend.timing
    cfg = config.selfplay
    moves = max(1, plies)
    evals = bt.positions
    cache_hits = backend.cache.hits
    s = sampler.summary()

    def line(label, value):
        print(f"  {label:<34s} {value}")

    print("=" * 66)
    print("MEASURED PERFORMANCE (single worker, in-process)")
    print("=" * 66)
    line("wall clock", f"{elapsed:.1f} s")
    line("games finished", games)
    line("plies (moves) played", plies)
    line("positions (training frames)", positions)
    print()
    line("positions/s", f"{plies / elapsed:.2f}")
    line("positions/min", f"{plies / elapsed * 60:.0f}")
    if games:
        line("games/hour (finished games only)",
             f"{games / max(game_seconds, 1e-9) * 3600:.1f}")
    print()
    line("MCTS nodes/s (visits)", f"{nodes / elapsed:.0f}")
    line("MCTS nodes/move", f"{nodes / moves:.0f}")
    line("configured visits/move", cfg.visits)
    print()
    line("NN evaluations/s", f"{evals / elapsed:.0f}")
    line("NN evaluations/move", f"{evals / moves:.1f}")
    line("NN cache hits (total)", cache_hits)
    line("cache hit rate",
         f"{cache_hits / max(1, cache_hits + evals) * 100:.1f} %")
    print()
    line("NN batches/s", f"{bt.batches / elapsed:.1f}")
    line("NN batches/move", f"{bt.batches / moves:.1f}")
    line("average NN batch size", f"{bt.avg_batch:.1f}")
    if driver is not None:
        line("avg requests per driver step", f"{driver.avg_batch:.1f}")
    line("min / max NN batch size", f"{bt.min_batch} / {bt.max_batch}")
    line("configured minibatch size (search)", cfg.search.minibatch_size)
    line("games in flight", cfg.parallel_games)
    top = sorted(bt.histogram.items(), key=lambda kv: -kv[1])[:6]
    line("most common batch sizes",
         ", ".join(f"{k}x{v}" for k, v in top))
    print()

    print("TIME SPLIT (single-threaded, so these add up to the wall clock)")
    select_pure = st.select_s - st.encode_s - st.terminal_s
    rows = [
        ("MCTS tree descent (PUCT, make/unmake)", select_pure),
        ("terminal detection (legal moves)", st.terminal_s),
        ("position encoding (112 planes + idx)", st.encode_s),
        ("backup (value propagation)", st.backup_s),
        ("inside Backend.evaluate (total)", bt.total_s),
        ("  - staging + H2D copy", bt.upload_s),
        ("  - forward + D2H (CPU blocked here)", bt.download_s),
        ("  - GPU forward only (CUDA events)", bt.gpu_s),
        ("  - policy softmax over legal moves", bt.post_s),
    ]
    for label, value in rows:
        share = value / elapsed * 100
        bar = "#" * int(round(share / 4))
        print(f"  {label:<40s} {value:7.2f} s  {share:5.1f} %  {bar}")
    cpu_side = select_pure + st.terminal_s + st.encode_s + st.backup_s + bt.post_s
    accounted = select_pure + st.terminal_s + st.encode_s + st.backup_s + bt.total_s
    print(f"  {'-> CPU-bound work':<40s} {cpu_side:7.2f} s  "
          f"{cpu_side / elapsed * 100:5.1f} %")
    print(f"  {'-> blocked on the GPU':<40s} {bt.download_s:7.2f} s  "
          f"{bt.download_s / elapsed * 100:5.1f} %")
    print(f"  {'accounted for':<40s} {accounted:7.2f} s  "
          f"{accounted / elapsed * 100:5.1f} %")
    print()

    print("HARDWARE")
    if s["gpu"]:
        line("GPU utilisation (avg / max)",
             f"{s['gpu'][0]:.0f} % / {s['gpu'][1]} %")
        line("VRAM used (avg / max)",
             f"{s['vram'][0]:.0f} / {s['vram'][1]} MiB")
    if s["cpu_total"]:
        cores = psutil.cpu_count(logical=True)
        line("CPU total (avg of all cores)", f"{s['cpu_total'][0]:.0f} %")
        line("this process CPU",
             f"{s['cpu_proc'][0]:.0f} % of one core "
             f"({s['cpu_proc'][0] / cores:.0f} % of {cores} logical cores)")
    print()

    theoretical = evals / max(bt.gpu_s, 1e-9)
    line("GPU-only throughput at this batch size",
         f"{theoretical:.0f} positions/s")
    line("actual NN throughput", f"{evals / elapsed:.0f} positions/s")
    line("=> fraction of the GPU's capability",
         f"{evals / elapsed / max(theoretical, 1e-9) * 100:.0f} %")


def profile_workers(args, config):
    """Throughput and hardware load with the normal worker processes."""
    stats_dir = os.path.join(args.stats_dir, "profile")
    os.makedirs(stats_dir, exist_ok=True)
    paths = [os.path.join(stats_dir, f"w{i:02d}.json")
             for i in range(args.workers)]
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
    common = dict(config_path=args.config, network_path=args.network,
                  output_dir=args.output, num_games=0, device=args.device,
                  max_seconds=args.seconds, log_every=0)
    sampler = Sampler(stats_paths=paths)
    sampler.start()
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    procs = []
    for i in range(args.workers):
        p = ctx.Process(target=run_worker,
                        kwargs=dict(worker_id=i, seed=args.seed + i * 7919,
                                    stats_path=paths[i], **common))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - t0
    sampler.stop()
    total = read_live_stats(paths)
    s = sampler.summary()
    rate = sampler.throughput()
    print("=" * 66)
    print(f"SCALING: {args.workers} worker process(es), {elapsed:.0f} s")
    print("=" * 66)
    print(f"  games finished     {total.get('games', 0):.0f}")
    print(f"  positions written  {total.get('positions', 0):.0f}")
    if rate:
        print(f"  plies played       {rate[1]:.0f} (incl. games in flight)")
        print(f"  positions/min      {rate[0] * 60:.0f}  "
              f"(slope during the run)")
    else:
        print("  positions/min      ? (run too short to measure a slope)")
    print(f"  MCTS nodes/s       {total.get('nodes', 0) / elapsed:.0f} "
          f"(finished games only)")
    print(f"  NN evals/s         {total.get('nn_evals', 0) / elapsed:.0f} "
          f"(finished games only)")
    if s["gpu"]:
        print(f"  GPU util avg/max   {s['gpu'][0]:.0f} % / {s['gpu'][1]} %")
        print(f"  VRAM max           {s['vram'][1]} MiB")
    if s["cpu_total"]:
        print(f"  CPU total avg      {s['cpu_total'][0]:.0f} %")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--network", default="networks/latest.mylc0")
    parser.add_argument("--output", default="data_profile")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats-dir", default="stats")
    parser.add_argument("--cprofile", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.workers > 1:
        profile_workers(args, config)
    else:
        profile_single(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
