"""The local half of the loop: sync from R2, train when there is enough, publish.

    python scripts/loop_r2.py

One process, running for as long as you leave it. Every pass it pulls whatever
the self-play nodes have finished, and once enough genuinely new positions have
arrived it trains one generation and publishes it. The nodes pick the new
network up by themselves at their next shard boundary, so nothing here ever
waits for them and nothing there ever waits for this.

The gate is the point. ``train.py`` on its own does not wait for data -- it
runs its steps on whatever is in the directory, stale or not -- so a bare
"sync; train; publish" loop happily trains the same positions over and over
when the nodes fall behind. This waits for ``--min-new-positions`` first.

Each generation gets a fresh data loader over the current replay window,
because the window moves: a loader built once at startup keeps reading the
same three directories while newer ones pile up beside it.

Nothing about the search, the network or the training targets changes here.
This only decides *when* to run the existing trainer.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time

import _bootstrap  # noqa: F401

from mylc0.cloud.index import ShardIndex
from mylc0.cloud.models import fetch_latest, publish_model
from mylc0.cloud.replay import generation_dirs, policy_from_config
from mylc0.cloud.storage import StorageError, describe_env, store_from_env
from mylc0.net.config import load_config
from mylc0.net.netfile import generation_filename
from mylc0.progress import Progress, attach_logging, format_duration
from mylc0.training.dataset import TrainingDataLoader
from mylc0.training.trainer import Trainer

log = logging.getLogger("loop")

_STOP = {"requested": False}


def _install_signal_handlers() -> None:
    """Ctrl-C finishes the phase it is in rather than abandoning it.

    Killing the process mid-training loses the generation's work; killing it
    between publish and the next sync loses nothing. So the first signal asks
    the loop to stop at the next boundary, and only the second is immediate.
    """
    def handler(signum, _frame):
        if _STOP["requested"]:
            log.warning("second signal (%s); exiting now", signum)
            sys.exit(130)
        _STOP["requested"] = True
        log.warning("signal %s received: finishing this phase, then stopping. "
                    "Signal again to abort immediately.", signum)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


class Watermark:
    """How many positions had been downloaded when we last trained.

    Counting *total ever downloaded* rather than "positions in the window"
    makes this monotonic, so it stays correct when the replay window moves on
    and old generations drop out of it. Persisted so a restart neither
    retrains immediately on data it already used nor waits for a threshold it
    has already passed.
    """

    def __init__(self, path: str):
        self.path = path
        self.value = 0
        try:
            with open(path, encoding="utf-8") as handle:
                self.value = int(json.load(handle).get("positions", 0))
        except (OSError, ValueError, TypeError):
            self.value = 0

    def set(self, value: int) -> None:
        self.value = int(value)
        tmp = self.path + f".tmp-{os.getpid()}"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)),
                        exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump({"positions": self.value, "at": time.time()}, handle)
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("could not persist the watermark: %s", exc)


def sync(store, args, index, sync_module) -> dict:
    """One pull from R2. Never fatal: the nodes keep the data either way."""
    import argparse as _argparse
    sync_args = _argparse.Namespace(
        data=args.data, replay_generations=args.replay_generations,
        all_generations=False, max_shards=args.max_shards_per_sync,
        retry_attempts=args.retry_attempts, retry_backoff=args.retry_backoff)
    try:
        return sync_module.sync_once(store, sync_args, index)
    except StorageError as exc:
        log.warning("sync failed (%s); the shards stay in R2 and will be "
                    "picked up next pass", exc)
        return {"new": 0, "bytes": 0, "generations": [], "failed": 0}


def train_one_generation(config, args, trainer, data_paths, progress):
    """Fresh loader over the current window, then one generation."""
    loader = TrainingDataLoader(
        data_paths, batch_size=config.training.batch_size,
        chunk_pool_size=config.training.chunk_pool_size,
        position_sampling_rate=config.training.position_sampling_rate,
        shuffle_buffer_size=config.training.shuffle_buffer_size,
        workers=config.training.loader_workers,
        seed=trainer.step + 1)
    loader.maybe_rescan(force=True)
    if len(loader.pool) == 0:
        loader.stop()
        return None, 0
    chunks = len(loader.pool)
    loader.start()
    try:
        started = time.perf_counter()
        trainer.train_generation(loader, steps=args.steps, progress=progress)
        progress.close()
        return time.perf_counter() - started, chunks
    finally:
        loader.stop()


def publish(store, trainer, args) -> bool:
    """Upload the generation just exported. latest.json moves last."""
    path = generation_filename(trainer.networks_dir, trainer.generation)
    if not os.path.isfile(path):
        log.error("generation %d was not exported to %s; not publishing",
                  trainer.generation, path)
        return False
    try:
        pointer = publish_model(
            store, path, int(trainer.generation),
            metadata={"step": trainer.step, "generation": trainer.generation},
            attempts=args.retry_attempts, base_delay=args.retry_backoff)
    except StorageError as exc:
        log.error("publish failed: %s", exc)
        log.error("latest.json was NOT moved, so the nodes keep the previous "
                  "generation. The network is on disk at %s and will be "
                  "published on the next pass.", path)
        return False
    log.info("published generation %d (%s...); nodes will switch after their "
             "current shard", pointer.generation, pointer.sha256[:12])
    return True


def status(index, window, watermark, threshold) -> str:
    stats = index.stats_by_generation()
    total = index.total_positions()
    new = total - watermark.value
    parts = [f"gen {g}: {stats[g]['positions']}" for g in window
             if g in stats]
    return (f"window [{', '.join(parts) or 'empty'}] | "
            f"new since last training {new}/{threshold} "
            f"({100.0 * new / threshold if threshold else 100:.0f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--data", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay-generations", type=int, default=3)
    parser.add_argument("--min-new-positions", type=int, default=500_000,
                        help="new positions required before a generation is "
                             "trained. Pass 0 to fall back to "
                             "training.positions_per_network from the config.")
    parser.add_argument("--steps", type=int, default=None,
                        help="override training.steps_per_network")
    parser.add_argument("--generations", type=int, default=0,
                        help="stop after this many (0 = run forever)")
    parser.add_argument("--sync-interval", type=float, default=120.0)
    parser.add_argument("--status-interval", type=float, default=60.0)
    parser.add_argument("--max-shards-per-sync", type=int, default=0)
    parser.add_argument("--retry-attempts", type=int, default=6)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--no-publish", action="store_true",
                        help="train but keep the networks local")
    parser.add_argument("--train-now", action="store_true",
                        help="ignore the gate for the first generation")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore existing checkpoints")
    parser.add_argument("--progress", default="auto",
                        choices=["auto", "on", "off"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [loop] %(message)s")
    progress = Progress(enabled={"on": True, "off": False}.get(args.progress))
    attach_logging(progress)
    _install_signal_handlers()

    config = load_config(args.config)
    # A shard is hundreds of thousands of positions, so a gate of a few tens
    # of thousands would fire on almost every sync and train generation after
    # generation on a replay window that had barely moved. Half a million new
    # positions is roughly one full shard from a 4090-class node.
    threshold = (args.min_new_positions
                 or config.training.positions_per_network)
    policy = policy_from_config(args.replay_generations)
    os.makedirs(args.data, exist_ok=True)

    print("local training loop")
    print(f"  config           {args.config}")
    print(f"  data             {os.path.abspath(args.data)}")
    print(f"  replay window    {policy.describe()}")
    print(f"  train when       {threshold:,} new positions have arrived"
          .replace(",", " "))
    print(f"  steps/generation {args.steps or config.training.steps_per_network}"
          f" at batch {config.training.batch_size}")
    print(f"  publish          {'no (local only)' if args.no_publish else 'yes'}")
    print("R2 configuration")
    print(describe_env())

    try:
        store = store_from_env()
    except StorageError as exc:
        print(f"\n{exc}")
        return 2

    # sync_selfplay owns the download logic; importing it keeps one
    # implementation rather than a second copy that can drift.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sync_selfplay", os.path.join(os.path.dirname(__file__),
                                      "sync_selfplay.py"))
    sync_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync_module)

    index = ShardIndex(os.path.join(args.data, "shard_index.db"))
    watermark = Watermark(os.path.join(args.data, "trained_watermark.json"))
    trainer = Trainer(config, device=args.device)
    trainer.report_device()
    if not args.fresh:
        trainer.load_checkpoint()

    pointer = fetch_latest(store)
    print(f"\nlocal generation {trainer.generation} (step {trainer.step})")
    print(f"published        {pointer.generation if pointer else 'nothing yet'}")
    print(f"downloaded       {index.total_positions()} positions, "
          f"watermark {watermark.value}\n")

    produced = 0
    force_next = args.train_now
    last_status = 0.0
    try:
        while not _STOP["requested"]:
            if args.generations and produced >= args.generations:
                log.info("reached --generations %d", args.generations)
                break

            result = sync(store, args, index, sync_module)
            if result["new"]:
                log.info("synced %d shard(s), %.2f GB", result["new"],
                         result["bytes"] / 1e9)

            paths, window = generation_dirs([args.data], policy)
            total = index.total_positions()
            new = total - watermark.value

            now = time.time()
            if now - last_status >= args.status_interval:
                last_status = now
                log.info("%s", status(index, window, watermark, threshold))

            if not paths:
                log.info("no generation directories yet; waiting for the "
                         "nodes to upload their first shard")
                _sleep_until(args.sync_interval)
                continue

            if new < threshold and not force_next:
                _sleep_until(args.sync_interval)
                continue

            if force_next:
                log.info("--train-now: training without waiting for the gate")
            force_next = False

            log.info("=== training generation %d -> %d | window %s | "
                     "%d new positions ===",
                     trainer.generation, trainer.generation + 1, window, new)
            elapsed, chunks = train_one_generation(
                config, args, trainer, paths, progress)
            progress.close()
            if elapsed is None:
                log.warning("the replay window holds no readable chunks; "
                            "waiting")
                _sleep_until(args.sync_interval)
                continue

            produced += 1
            # The watermark moves to the count at training time, not after,
            # so shards that arrived while training still count as new.
            watermark.set(total)
            log.info("generation %d ready in %s (%d chunks, step %d)",
                     trainer.generation, format_duration(elapsed), chunks,
                     trainer.step)
            trainer.log_scalars({
                "r2/new_positions": new,
                "r2/window_positions": sum(
                    index.stats_by_generation().get(g, {}).get("positions", 0)
                    for g in window),
                "r2/downloaded_positions": total,
                "r2/shards": sum(
                    r["shards"] for r in index.stats_by_generation().values()),
            })

            if not args.no_publish:
                publish(store, trainer, args)
    except KeyboardInterrupt:
        log.warning("interrupted; saving a checkpoint")
        trainer.save_checkpoint()
    finally:
        progress.close()
        trainer.close()
        index.close()

    log.info("stopping after %d generation(s)", produced)
    return 0


def _sleep_until(seconds: float) -> None:
    """Sleep, but stay responsive to a stop request."""
    deadline = time.time() + max(1.0, seconds)
    while time.time() < deadline and not _STOP["requested"]:
        time.sleep(min(1.0, max(0.0, deadline - time.time())))


if __name__ == "__main__":
    raise SystemExit(main())
