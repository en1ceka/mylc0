"""The reinforcement-learning loop: self-play -> training -> next generation.

    python scripts/loop.py --config configs/small.yaml

Each iteration:

    1. self-play workers play games with the newest exported network and write
       V6 chunks under ``data/``
    2. the trainer consumes a shuffled window over the most recent chunks for
       ``steps_per_network`` optimizer steps
    3. a checkpoint is written and ``networks/gen_NNNNNN.mylc0`` is exported
    4. repeat, now playing with the new network

The loop is restartable: it picks up from the newest checkpoint and keeps
every exported generation, so any of them can be run later as a UCI engine.
"""

import argparse
import logging
import multiprocessing as mp
import os
import threading
import time

import _bootstrap  # noqa: F401

from mylc0.net.config import load_config
from mylc0.net.netfile import generation_filename
from mylc0.progress import Progress, attach_logging, format_duration
from mylc0.selfplay.worker import (aggregate_stats, monitor_selfplay,
                                   run_worker)
from mylc0.training.dataset import TrainingDataLoader
from mylc0.training.trainer import Trainer

log = logging.getLogger("mylc0.loop")


def run_selfplay_phase(config_path: str, network: str, output: str,
                       workers: int, target_positions: int, stats_dir: str,
                       seed: int, device=None, max_seconds: float = 0.0,
                       progress=None):
    os.makedirs(stats_dir, exist_ok=True)
    stats_paths = [os.path.join(stats_dir, f"selfplay_{i:02d}.json")
                   for i in range(workers)]
    for path in stats_paths:
        if os.path.exists(path):
            os.remove(path)
    per_worker = max(1, target_positions // workers) if target_positions else 0
    common = dict(config_path=config_path, network_path=network,
                  output_dir=output, num_games=0, device=device,
                  target_positions=per_worker, max_seconds=max_seconds)

    # With a live bar the per-game lines would only add noise; without one
    # (redirected output) they stay, so a log file still tells the story.
    quiet_games = 0 if (progress is not None and progress.enabled) else 1
    stop_monitor = threading.Event()
    monitor = None
    procs = []
    if progress is not None:
        monitor = threading.Thread(
            target=monitor_selfplay,
            args=(stats_paths, target_positions, progress, stop_monitor,
                  workers, procs), daemon=True)
        monitor.start()
    try:
        if workers == 1:
            # A single worker runs in this process; the monitor thread still
            # reads its stats file, so progress is reported either way.
            run_worker(worker_id=0, seed=seed, stats_path=stats_paths[0],
                       log_every=quiet_games, **common)
        else:
            ctx = mp.get_context("spawn")
            for i in range(workers):
                p = ctx.Process(target=run_worker,
                                kwargs=dict(worker_id=i, seed=seed + i * 7919,
                                            stats_path=stats_paths[i],
                                            log_every=quiet_games, **common))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            failed = [p for p in procs if p.exitcode]
            if failed:
                raise RuntimeError(
                    "self-play worker(s) failed with exit code(s) "
                    + ", ".join(str(p.exitcode) for p in failed)
                    + " -- their tracebacks are above")
    finally:
        stop_monitor.set()
        if monitor is not None:
            monitor.join(timeout=2.0)
    return aggregate_stats(stats_paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=0,
                        help="0 = run until interrupted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selfplay-device", default=None)
    parser.add_argument("--stats-dir", default="stats")
    parser.add_argument("--positions-per-network", type=int, default=None)
    parser.add_argument("--selfplay-max-seconds", type=float, default=0.0)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--progress", default="auto",
                        choices=["auto", "on", "off"],
                        help="live progress line (auto = only on a terminal)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [loop] %(message)s")
    progress = Progress(enabled={"on": True, "off": False}.get(args.progress))
    attach_logging(progress)
    config = load_config(args.config)
    data_dir = args.data or config.selfplay.output_path
    workers = args.workers or config.selfplay.workers
    target_positions = (args.positions_per_network
                        or config.training.positions_per_network)

    trainer = Trainer(config, device=args.device)
    trainer.report_device()
    if not args.fresh:
        trainer.load_checkpoint()

    network = generation_filename(trainer.networks_dir, trainer.generation)
    if not os.path.isfile(network):
        log.info("no network for generation %d; exporting one from the "
                 "current weights", trainer.generation)
        network = trainer.export_network()

    loader = TrainingDataLoader(
        [data_dir], batch_size=config.training.batch_size,
        chunk_pool_size=config.training.chunk_pool_size,
        position_sampling_rate=config.training.position_sampling_rate,
        shuffle_buffer_size=config.training.shuffle_buffer_size,
        workers=config.training.loader_workers, seed=trainer.step + 1)
    loader.start()

    iteration = 0
    try:
        while args.iterations <= 0 or iteration < args.iterations:
            iteration += 1
            log.info("=== iteration %d | generation %d | step %d ===",
                     iteration, trainer.generation, trainer.step)
            log.info("phase 1/2 self-play: %s with %d worker(s), network %s",
                     f"{target_positions} positions" if target_positions
                     else "unlimited", workers, os.path.basename(network))

            t0 = time.perf_counter()
            sp = run_selfplay_phase(
                args.config, network, data_dir, workers, target_positions,
                args.stats_dir, seed=trainer.step * 1009 + iteration,
                device=args.selfplay_device,
                max_seconds=args.selfplay_max_seconds,
                progress=progress)
            sp_time = time.perf_counter() - t0
            progress.close()
            log.info("self-play: %d games, %d positions, avg length %.1f, "
                     "W%d D%d B%d in %.1fs",
                     sp.get("games", 0), sp.get("positions", 0),
                     sp.get("avg_game_length", 0.0), sp.get("white_wins", 0),
                     sp.get("draws", 0), sp.get("black_wins", 0), sp_time)
            trainer.log_scalars({
                "selfplay/games": sp.get("games", 0),
                "selfplay/positions": sp.get("positions", 0),
                "selfplay/avg_game_length": sp.get("avg_game_length", 0.0),
                "selfplay/white_wins": sp.get("white_wins", 0),
                "selfplay/black_wins": sp.get("black_wins", 0),
                "selfplay/draws": sp.get("draws", 0),
                "selfplay/adjudicated": sp.get("adjudicated", 0),
                "selfplay/nn_evals": sp.get("nn_evals", 0),
                "selfplay/nodes_per_second": sp.get("nodes_per_second", 0.0),
                "selfplay/nodes_per_move": sp.get("nodes_per_move", 0.0),
                "selfplay/positions_per_second": sp.get("positions_per_second", 0.0),
                "selfplay/games_per_hour": sp.get("games_per_hour", 0.0),
                "selfplay/wall_seconds": sp_time,
            })

            added = loader.maybe_rescan(force=True)
            log.info("data: +%d chunks, pool now %d", added, len(loader.pool))
            if len(loader.pool) == 0:
                log.warning("no chunks available; skipping training")
                continue

            log.info("phase 2/2 training: %d steps, batch %d x %d, "
                     "chunk pool %d",
                     config.training.steps_per_network,
                     config.training.batch_size,
                     max(1, config.training.gradient_accumulation),
                     len(loader.pool))
            t1 = time.perf_counter()
            trainer.train_generation(loader, progress=progress)
            progress.close()
            log.info("training: generation %d ready in %s",
                     trainer.generation,
                     format_duration(time.perf_counter() - t1))
            network = generation_filename(trainer.networks_dir,
                                          trainer.generation)
    except KeyboardInterrupt:
        log.info("interrupted; saving checkpoint")
        trainer.save_checkpoint()
    finally:
        progress.close()
        loader.stop()
        trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
