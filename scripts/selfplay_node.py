"""A self-play node: generate continuously, ship mini-shards to R2.

    python scripts/selfplay_node.py --config configs/small.yaml \
        --workers 28 --parallel-games 48 --nn-batch 512 \
        --upload-shard-positions 20000

Meant to run unattended for days on a rented box. It owns no state that
matters: the bucket holds the models and the finished shards, the local disk
holds only work in progress. Kill the machine, rent another, clone the repo,
export four secrets -- the new node reads ``models/latest.json`` and carries on.

**Self-play never waits for storage.** Workers are long-lived and keep every
game slot busy. When a game ends its chunk is renamed into a spool directory,
which makes it durable and visible atomically, and the worker starts the next
game immediately -- it does not wait for the other games, for a shard to close,
or for an upload. Three background threads do the rest:

    workers --> spool --> collector --> outbox --> uploader --> R2
                       (groups games)           (compresses, sends)

Compression and upload live in those threads, never in a worker, so nothing
touching the network or the CPU-heavy zstd path can land between two search
batches.

The search itself is untouched: this drives the existing worker with the
existing config. Visits, minibatch semantics, the network and the encoding
are whatever ``--config`` says.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import signal
import sys
import threading
import time

import _bootstrap  # noqa: F401

from mylc0.cloud.collector import ShardCollector
from mylc0.cloud.layout import load_or_create_node_id
from mylc0.cloud.models import ensure_model, fetch_latest
from mylc0.cloud.storage import StorageError, describe_env, store_from_env
from mylc0.cloud.uploader import Backpressure, Outbox, ShardUploader
from mylc0.net.config import load_config
from mylc0.perf import limit_thread_pools
from mylc0.selfplay.status import NodeStatus, confirmation_line
from mylc0.selfplay.worker import run_worker

log = logging.getLogger("selfplay-node")

_STOP = {"requested": False, "hard": False}


def _install_signal_handlers() -> None:
    """The first signal stops admitting games; the second gives up.

    A node is usually stopped by the provider reclaiming the box. Everything
    already finished is durable in the spool, so the graceful path is about
    the games still in flight and the partial shard -- not about avoiding data
    loss, which the spool already handles.
    """
    def handler(signum, _frame):
        if _STOP["requested"]:
            _STOP["hard"] = True
            log.warning("second signal (%s); exiting now. Finished games are "
                        "durable and will be picked up on the next start.",
                        signum)
            sys.exit(130)
        _STOP["requested"] = True
        log.warning("signal %s: no new games; finishing those in flight, "
                    "flushing the shard and draining uploads. Signal again "
                    "to exit immediately.", signum)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


class ConfigMismatch(RuntimeError):
    """A worker started with knobs the node did not ask for."""


def verify_runtime_config(paths, expected, timeout=300.0, alive=None):
    """Read back what each worker actually started with, and insist it match.

    The knobs travel through a YAML file, a process boundary and a driver
    constructor. A mismatch anywhere in that chain still produces correct
    games -- just far fewer of them -- so it is invisible in the output and
    shows up only as a throughput number nobody is watching.
    """
    deadline = time.time() + timeout
    seen = {}
    while time.time() < deadline:
        for index, path in enumerate(paths):
            if index in seen:
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    seen[index] = json.load(handle)
            except (OSError, ValueError):
                continue
        if len(seen) == len(paths):
            break
        if alive is not None and not alive():
            break
        time.sleep(0.25)

    if not seen:
        raise ConfigMismatch(
            f"no worker reported its runtime configuration within "
            f"{timeout:.0f}s; self-play did not start")

    problems = []
    for index, actual in sorted(seen.items()):
        for key, want in expected.items():
            got = actual.get(key)
            if got != want:
                problems.append(f"worker {index}: {key} is {got!r}, "
                                f"expected {want!r}")
    if problems:
        raise ConfigMismatch(
            "workers did not start with the requested configuration:\n  "
            + "\n  ".join(problems[:10])
            + ("\n  ..." if len(problems) > 10 else ""))
    return seen


def overshoot_floor(workers, parallel_games, max_game_ply):
    """Positions one full round of games across every worker would produce.

    Only advisory now: a mini-shard closes on finished games as they arrive,
    so nothing has to wait for a full round. It still tells you how much data
    is in flight at any moment, and therefore how much a hard kill would cost.
    """
    return workers * parallel_games * max(1, max_game_ply // 2)


def _dump_config(source: str, target: str, parallel_games: int,
                 nn_batch: int) -> None:
    """Copy the config, overriding only the two knobs, only under selfplay:.

    ``batch_size`` appears twice in the reference configs -- once under
    ``training:`` and once under ``selfplay:`` -- at the same indentation, so
    matching on the key alone silently rewrites the trainer's batch size too.
    """
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    out = []
    section = None
    for line in text.splitlines():
        if line[:1] not in (" ", "\t", "#", ""):
            section = line.split(":", 1)[0].strip()
        stripped = line.strip()
        if section == "selfplay":
            if parallel_games and stripped.startswith("parallel_games:"):
                line = f"  parallel_games: {parallel_games}"
            elif nn_batch and stripped.startswith("batch_size:"):
                line = f"  batch_size: {nn_batch}"
        out.append(line)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out) + "\n")


class WorkerPool:
    """Long-lived self-play processes, replaceable a few at a time.

    One driver holds one network: ``backend.evaluate`` runs a single model
    over a single tensor, so games from two generations cannot share a batch
    without rewriting both the backend and the batching driver. Rather than
    force that, a worker is *retired* in order to change networks -- it stops
    admitting games, finishes the ones it holds, and is restarted on the new
    weights.

    Retiring a few at a time is what stops a generation change from becoming a
    farm-wide barrier. The workers not being retired never pause, the games
    still finishing on the old network stay attributed to it, and their chunks
    keep flowing into the spool throughout.
    """

    def __init__(self, args, config_path, control_dir, spool_dir):
        self.args = args
        self.config_path = config_path
        self.control_dir = control_dir
        self.spool_dir = spool_dir
        os.makedirs(control_dir, exist_ok=True)
        self.procs = {}
        self.network_path = None
        self.generation = None
        self.ctx = mp.get_context("spawn")
        self._epoch = 0

    def stop_file(self, index):
        return os.path.join(self.control_dir, f"stop_{index}")

    def runtime_path(self, index):
        return os.path.join(self.control_dir, f"runtime_{index}.json")

    def perf_path(self, index):
        return os.path.join(self.control_dir, f"perf_{index}.json")

    def perf_paths(self):
        return [self.perf_path(i) for i in range(self.args.workers)]

    def runtime_paths(self):
        return [self.runtime_path(i) for i in range(self.args.workers)]

    def _spawn(self, index):
        for path in (self.stop_file(index), self.runtime_path(index)):
            try:
                os.remove(path)
            except OSError:
                pass
        self._epoch += 1
        proc = self.ctx.Process(target=run_worker, kwargs=dict(
            config_path=self.config_path, network_path=self.network_path,
            output_dir=self.spool_dir, chunk_dir=self.spool_dir,
            worker_id=index,
            seed=(int(time.time() * 1000) + index * 7919
                  + self._epoch * 104729) & 0x7FFFFFFF,
            device=self.args.device, num_games=0, target_positions=0,
            max_seconds=0.0, log_every=0, stats_path=None,
            runtime_config_path=self.runtime_path(index),
            perf_debug=True, perf_path=self.perf_path(index),
            log_level=self.args.worker_log_level,
            # There is no position quota at all here: the node runs until it
            # is told to stop, and a mini-shard closes around finished games
            # rather than around a worker's lifetime.
            scale_parallel_to_target=False,
            freeze_perf_on_drain=False,
            stop_file=self.stop_file(index),
            watchdog_seconds=self.args.watchdog_seconds,
            heartbeat_seconds=0, torch_threads=self.args.torch_threads,
            affinity=self.args.affinity, workers_total=self.args.workers))
        proc.start()
        self.procs[index] = proc
        return proc

    def start_all(self, network_path, generation):
        self.network_path = network_path
        self.generation = generation
        for index in range(self.args.workers):
            self._spawn(index)

    def retire(self, index):
        """Ask one worker to finish its games and exit."""
        try:
            with open(self.stop_file(index), "w", encoding="utf-8") as handle:
                handle.write("retire")
        except OSError as exc:
            log.warning("could not signal worker %d: %s", index, exc)

    def alive(self):
        return [i for i, p in self.procs.items() if p.is_alive()]

    def restart_dead(self):
        """A worker that died on its own is restarted; a farm must self-heal."""
        restarted = []
        for index, proc in list(self.procs.items()):
            if proc.is_alive():
                continue
            if os.path.exists(self.stop_file(index)):
                continue          # retired on purpose, handled elsewhere
            log.warning("worker %d exited with %s; restarting", index,
                        proc.exitcode)
            self._spawn(index)
            restarted.append(index)
        return restarted

    def rolling_switch(self, network_path, generation, concurrency,
                       should_stop):
        """Move every worker onto a new network, a few at a time."""
        previous = self.generation
        self.network_path = network_path
        self.generation = generation
        indices = list(range(self.args.workers))
        done = 0
        for start in range(0, len(indices), max(1, concurrency)):
            if should_stop():
                break
            group = indices[start:start + max(1, concurrency)]
            for index in group:
                self.retire(index)
            for index in group:
                while self.procs[index].is_alive() and not should_stop():
                    time.sleep(0.5)
                self.procs[index].join(timeout=10.0)
                self._spawn(index)
                done += 1
            log.info("rolling switch %s -> %d: %d/%d worker(s) moved",
                     previous, generation, done, len(indices))
        return done

    def stop_all(self, timeout=3600.0):
        for index in list(self.procs):
            self.retire(index)
        deadline = time.time() + timeout
        while self.alive() and time.time() < deadline and not _STOP["hard"]:
            time.sleep(1.0)
        for proc in self.procs.values():
            if proc.is_alive():
                proc.terminate()
        for proc in self.procs.values():
            proc.join(timeout=10.0)


def _heartbeat(status, stop, args, context):
    """One aggregated line per interval, in its own thread."""
    status.sample()
    while not stop.wait(args.status_seconds):
        try:
            status.sample()
            log.info("%s", status.continuous_line(**context()))
        except Exception as exc:                       # noqa: BLE001
            log.debug("status line failed: %s", exc)   # never fatal


def _uploader_loop(uploader, backpressure, stop, interval):
    """Upload in the background. Nothing here can block a worker."""
    while not stop.wait(interval):
        try:
            uploader.drain(should_continue=lambda: not _STOP["hard"])
            backpressure.check()
        except Exception as exc:                       # noqa: BLE001
            log.warning("upload pass failed: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--node-id", default=None,
                        help="default: hostname plus a random suffix, kept in "
                             "the cache dir across restarts")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-games", type=int, default=0)
    parser.add_argument("--nn-batch", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--affinity", action="store_true")
    parser.add_argument("--watchdog-seconds", type=float, default=300.0)

    parser.add_argument("--upload-shard-positions", type=int, default=20000,
                        help="close a mini-shard once this many finished "
                             "positions have collected. Whole games only, so "
                             "a shard overshoots rather than splitting one.")
    parser.add_argument("--collect-seconds", type=float, default=5.0,
                        help="how often the collector looks at the spool")
    parser.add_argument("--upload-seconds", type=float, default=10.0)
    parser.add_argument("--max-runtime-hours", type=float, default=0.0,
                        help="stop gracefully after this long (0 = forever)")

    parser.add_argument("--cache-dir", default="node_cache",
                        help="models, node id, spool and the upload outbox")
    parser.add_argument("--keep-local-shards", action="store_true")
    parser.add_argument("--max-backlog-gb", type=float, default=20.0,
                        help="pause generation when unsent shards exceed this")
    parser.add_argument("--retry-attempts", type=int, default=6)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--poll-latest-seconds", type=float, default=120.0,
                        help="how often to check models/latest.json")
    parser.add_argument("--switch-concurrency", type=int, default=4,
                        help="workers retired at once when the network "
                             "changes. Lower means a smaller dip in "
                             "throughput and a longer switch.")
    parser.add_argument("--status-seconds", type=float, default=15.0)
    parser.add_argument("--worker-log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="per-worker log level. The default hides one "
                             "startup banner per worker; errors and OOM are "
                             "logged above it and stay visible.")
    parser.add_argument("--verbose-workers", action="store_true",
                        help="shorthand for --worker-log-level INFO")
    parser.add_argument("--dry-run", action="store_true",
                        help="check config and R2 access, then exit")
    args = parser.parse_args()

    if args.verbose_workers:
        args.worker_log_level = "INFO"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [node] %(levelname)s %(message)s")
    limit_thread_pools(max(1, args.torch_threads))
    _install_signal_handlers()

    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    node_id = load_or_create_node_id(cache_dir, args.node_id)
    models_dir = os.path.join(cache_dir, "models")
    spool_dir = os.path.join(cache_dir, "spool")
    staging_dir = os.path.join(cache_dir, "staging")
    control_dir = os.path.join(cache_dir, "control")
    outbox = Outbox(os.path.join(cache_dir, "outbox"))
    for path in (spool_dir, staging_dir, control_dir):
        os.makedirs(path, exist_ok=True)

    load_config(args.config)          # fail fast on a broken config
    run_config = os.path.join(cache_dir, "run_config.yaml")
    _dump_config(args.config, run_config, args.parallel_games, args.nn_batch)
    effective = load_config(run_config)
    cfg = effective.selfplay

    print(f"node {node_id}")
    print(f"  config           {args.config} -> {run_config}")
    print(f"  visits           {cfg.visits} "
          f"(fixed; minibatch {cfg.search.minibatch_size})")
    print(f"  workers          {args.workers} x {cfg.parallel_games} games, "
          f"nn_batch {cfg.batch_size}")
    print(f"  mini-shard       {args.upload_shard_positions} finished "
          f"positions (whole games; overshoot expected)")
    print(f"  in flight        up to "
          f"{overshoot_floor(args.workers, cfg.parallel_games, cfg.max_game_ply)}"
          f" positions, durable only once each game ends")
    print(f"  cache            {cache_dir}")
    print(f"  backlog limit    {args.max_backlog_gb:.1f} GB, keep local: "
          f"{args.keep_local_shards}")
    print("R2 configuration")
    print(describe_env())

    try:
        store = store_from_env()
    except StorageError as exc:
        print(f"\n{exc}")
        return 2

    try:
        pointer = fetch_latest(store, attempts=args.retry_attempts,
                               base_delay=args.retry_backoff)
    except StorageError as exc:
        print(f"\n{exc}")
        return 3
    if pointer is None:
        print("\nmodels/latest.json does not exist. Publish a network first:\n"
              "  python scripts/publish_model.py --network "
              "networks/latest.mylc0")
        return 3
    network_path = ensure_model(store, pointer, models_dir,
                                attempts=args.retry_attempts,
                                base_delay=args.retry_backoff)
    print(f"\nplaying generation {pointer.generation} "
          f"(sha256 {pointer.sha256[:16]}...)\n")
    if args.dry_run:
        print("dry run: R2 reachable, model verified, nothing generated")
        return 0

    known_sha = {pointer.generation: pointer.sha256}
    collector = ShardCollector(
        spool_dir=spool_dir, staging_dir=staging_dir, outbox=outbox,
        node_id=node_id, target_positions=args.upload_shard_positions,
        network_sha_for=lambda g: known_sha.get(g, ""),
        visits=cfg.visits,
        extra={"workers": args.workers,
               "parallel_games": cfg.parallel_games,
               "nn_batch": cfg.batch_size})
    uploader = ShardUploader(store, outbox, attempts=args.retry_attempts,
                             base_delay=args.retry_backoff,
                             keep_local=args.keep_local_shards)
    backpressure = Backpressure(outbox, args.max_backlog_gb)

    # Whatever a previous life left behind goes out before new work is made.
    collector.recover_staging()
    leftover = len(outbox.pending())
    spooled = collector.pending_positions()
    if leftover or spooled:
        log.info("resuming: %d shard(s) in the outbox, %d finished position(s)"
                 " in the spool", leftover, spooled)
        uploader.drain()
        collector.build_ready()

    pool = WorkerPool(args, run_config, control_dir, spool_dir)
    pool.start_all(network_path, pointer.generation)

    expected = {"parallel_games": cfg.parallel_games,
                "nn_batch": cfg.batch_size, "visits": cfg.visits,
                "minibatch_size": cfg.search.minibatch_size,
                "fp16": bool(cfg.fp16)}
    try:
        observed = verify_runtime_config(
            pool.runtime_paths(), expected, alive=lambda: bool(pool.alive()))
    except ConfigMismatch as exc:
        log.error("%s", exc)
        log.error("Refusing to generate slow data; this is a configuration "
                  "bug, not a transient failure.")
        pool.stop_all(timeout=30.0)
        return 5
    log.info("%s", confirmation_line(args.workers, observed[min(observed)]))

    status = NodeStatus(pool.perf_paths(), target_positions=0,
                        rate_window=max(30.0, args.status_seconds * 2))
    switching = {"text": ""}

    def context():
        stats = uploader.stats
        return {"generation": pool.generation or 0,
                "network_sha": known_sha.get(pool.generation, ""),
                "shard_fill": collector.pending_positions(),
                "shard_target": args.upload_shard_positions,
                "outbox_shards": len(outbox.pending()),
                "outbox_gb": outbox.backlog_bytes() / 1e9,
                "uploaded": stats.uploaded,
                "uploaded_positions": collector.stats.positions_packed,
                "failures": stats.failed,
                "switching": switching["text"]}

    stop_threads = threading.Event()
    threads = [
        threading.Thread(target=collector.run, daemon=True,
                         args=(stop_threads, args.collect_seconds)),
        threading.Thread(target=_uploader_loop, daemon=True,
                         args=(uploader, backpressure, stop_threads,
                               args.upload_seconds)),
        threading.Thread(target=_heartbeat, daemon=True,
                         args=(status, stop_threads, args, context)),
    ]
    for thread in threads:
        thread.start()

    started = time.time()
    last_poll = time.time()
    try:
        while not _STOP["requested"]:
            time.sleep(1.0)
            if (args.max_runtime_hours
                    and time.time() - started > args.max_runtime_hours * 3600):
                log.info("reached --max-runtime-hours %.1f",
                         args.max_runtime_hours)
                break

            pool.restart_dead()

            if not backpressure.check():
                # Only now does storage get to slow self-play down, and only
                # because the disk is finite. Retiring the workers keeps every
                # finished game: the spool is already durable.
                log.warning("backlog %.2f GB over the limit; pausing "
                            "generation until uploads catch up",
                            backpressure.backlog_gb)
                for index in range(args.workers):
                    pool.retire(index)
                while not backpressure.check() and not _STOP["requested"]:
                    uploader.drain()
                    time.sleep(5.0)
                if _STOP["requested"]:
                    break
                log.info("backlog cleared; resuming generation")
                pool.start_all(pool.network_path, pool.generation)

            if time.time() - last_poll < args.poll_latest_seconds:
                continue
            last_poll = time.time()
            try:
                latest = fetch_latest(store, attempts=2,
                                      base_delay=args.retry_backoff)
            except StorageError as exc:
                log.warning("could not read latest.json: %s", exc)
                continue
            if latest is None or latest.generation == pool.generation:
                continue

            log.info("new generation published: %s -> %d; rolling the workers "
                     "over %d at a time", pool.generation, latest.generation,
                     args.switch_concurrency)
            try:
                path = ensure_model(store, latest, models_dir,
                                    attempts=args.retry_attempts,
                                    base_delay=args.retry_backoff)
            except StorageError as exc:
                log.warning("could not fetch generation %d: %s; staying on %s",
                            latest.generation, exc, pool.generation)
                continue
            known_sha[latest.generation] = latest.sha256
            switching["text"] = f"switching -> gen {latest.generation}"
            pool.rolling_switch(path, latest.generation,
                                args.switch_concurrency,
                                lambda: _STOP["requested"])
            switching["text"] = ""
            log.info("all workers on generation %d", latest.generation)
    finally:
        log.info("stopping: finishing games in flight, then flushing")
        pool.stop_all()
        stop_threads.set()
        for thread in threads:
            thread.join(timeout=5.0)

        # Everything left in the spool becomes a shard even if it is short of
        # the target: a node stopping with 14k positions should ship them.
        collector.recover_staging()
        built = collector.build_ready(force=True)
        if built:
            log.info("flushed %d partial shard(s) on shutdown", built)
        uploader.drain()
        remaining = outbox.pending()
        if remaining:
            log.warning("%d shard(s) could not be uploaded and remain in %s; "
                        "they go out when this node is started again",
                        len(remaining), outbox.directory)
        stats = uploader.stats
        log.info("node stopped: %d shard(s) built, %d uploaded, %d already "
                 "present, %d failed, %d positions packed",
                 collector.stats.shards_built, stats.uploaded, stats.skipped,
                 stats.failed, collector.stats.positions_packed)
        shutil.rmtree(control_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
