"""Run self-play workers.

    python scripts/selfplay.py --config configs/small.yaml \
        --network networks/latest.mylc0 --output data --workers 2 --games 20

With ``--workers 1`` the games run in this process; with more, each worker is
its own process (they only share the network file and the output directory,
so this scales to as many processes -- or machines -- as you like).
"""

import argparse
import logging
import multiprocessing as mp
import os
import threading
import time

import _bootstrap  # noqa: F401

from mylc0.net.config import load_config
from mylc0.progress import Progress, attach_logging, format_duration
from mylc0.selfplay.worker import (aggregate_stats, monitor_selfplay,
                                   run_worker)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--network", default="networks/latest.mylc0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--games", type=int, default=0,
                        help="games per worker (0 = unlimited)")
    parser.add_argument("--target-positions", type=int, default=0,
                        help="stop once this many positions have been produced")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats-dir", default="stats")
    parser.add_argument("--progress", default="auto",
                        choices=["auto", "on", "off"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    progress = Progress(enabled={"on": True, "off": False}.get(args.progress))
    attach_logging(progress)
    config = load_config(args.config)
    workers = args.workers or config.selfplay.workers
    output = args.output or config.selfplay.output_path
    os.makedirs(args.stats_dir, exist_ok=True)
    per_worker_positions = (args.target_positions // workers
                            if args.target_positions else 0)

    common = dict(config_path=args.config, network_path=args.network,
                  output_dir=output, num_games=args.games,
                  device=args.device, max_seconds=args.max_seconds,
                  target_positions=per_worker_positions)

    t0 = time.perf_counter()
    stats_paths = [os.path.join(args.stats_dir, f"selfplay_{i:02d}.json")
                   for i in range(workers)]
    for path in stats_paths:
        if os.path.exists(path):
            os.remove(path)

    quiet_games = 0 if progress.enabled else 1
    procs = []
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=monitor_selfplay,
        args=(stats_paths, args.target_positions, progress, stop_monitor,
              workers, procs), daemon=True)
    monitor.start()
    try:
        if workers == 1:
            run_worker(worker_id=0, seed=args.seed, stats_path=stats_paths[0],
                       log_every=quiet_games, **common)
        else:
            ctx = mp.get_context("spawn")
            for i in range(workers):
                p = ctx.Process(target=run_worker,
                                kwargs=dict(worker_id=i,
                                            seed=args.seed + i * 7919,
                                            stats_path=stats_paths[i],
                                            log_every=quiet_games, **common))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            failed = [p for p in procs if p.exitcode]
            if failed:
                print("ERROR: self-play worker(s) failed with exit code(s) "
                      + ", ".join(str(p.exitcode) for p in failed))
    finally:
        stop_monitor.set()
        monitor.join(timeout=2.0)
        progress.close()

    total = aggregate_stats(stats_paths)
    elapsed = time.perf_counter() - t0
    print(f"\nself-play finished in {format_duration(elapsed)}")
    for key in ("games", "positions", "white_wins", "draws", "black_wins",
                "adjudicated", "avg_game_length", "nodes", "nn_evals"):
        if key in total:
            print(f"  {key:18s} {total[key]:,.2f}" if isinstance(total[key], float)
                  else f"  {key:18s} {total[key]:,}")
    if elapsed > 0 and total.get("games"):
        print(f"  wall-clock games/h {total['games'] / elapsed * 3600:,.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
