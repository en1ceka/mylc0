"""A self-play node: generate games, ship shards to R2, follow the newest net.

    python scripts/selfplay_node.py --config configs/small.yaml \
        --workers 28 --parallel-games 48 --nn-batch 512 \
        --shard-positions 50000

Meant to run unattended for days on a rented box. It owns no state that
matters: the bucket holds the models and the finished shards, the local disk
holds only work in progress. Kill the machine, rent another, clone the repo,
export four secrets -- the new node reads ``models/latest.json`` and carries on.

One round of the loop is one shard:

    check backlog -> generate into a staging dir -> pack -> queue -> upload
    -> re-read latest.json -> maybe switch model -> repeat

The model is only ever swapped between rounds. Workers are separate processes
started per round with a fixed weights file, so "never switch mid-shard" is
not a rule the code has to remember -- there is no path by which it could.

The search itself is untouched: this drives the existing worker with the
existing config. Visits, minibatch semantics, the network and the encoding are
whatever ``--config`` says.
"""

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time

import _bootstrap  # noqa: F401

from mylc0.cloud.layout import (make_shard_id, load_or_create_node_id,
                                shard_key, today)
from mylc0.cloud.models import LatestPointer, ensure_model, fetch_latest
from mylc0.cloud.shards import build_manifest, collect_chunks, pack_shard
from mylc0.cloud.storage import (StorageError, describe_env,
                                 store_from_env)
from mylc0.cloud.uploader import Backpressure, Outbox, ShardUploader
from mylc0.net.config import load_config
from mylc0.perf import limit_thread_pools
from mylc0.selfplay.worker import run_worker

log = logging.getLogger("selfplay-node")

_STOP = {"requested": False}


def _install_signal_handlers() -> None:
    """Ctrl-C and SIGTERM finish the current shard, they do not discard it.

    A node is usually stopped by the provider reclaiming the box. Losing ten
    minutes of finished games because the process died between generating and
    packing would be the most annoying possible failure, so the loop is asked
    to stop at the next safe point instead.
    """
    def handler(signum, _frame):
        if _STOP["requested"]:
            log.warning("second signal (%s); exiting now", signum)
            sys.exit(130)
        _STOP["requested"] = True
        log.warning("signal %s received: finishing the current shard, then "
                    "stopping. Signal again to abort immediately.", signum)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


def _dump_config(source: str, target: str, parallel_games: int,
                 nn_batch: int) -> None:
    """Copy the config, overriding only the two knobs, only under selfplay:.

    ``batch_size`` appears twice in the reference configs -- once under
    ``training:`` and once under ``selfplay:`` -- at the same indentation, so
    matching on the key alone silently rewrites the trainer's batch size too.
    Tracking which top-level section a line belongs to is what keeps the
    override where it was aimed.
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

class ConfigMismatch(RuntimeError):
    """A worker started with knobs the node did not ask for."""


def verify_runtime_config(paths, expected, timeout=180.0, alive=None):
    """Read back what each worker actually started with, and insist it match.

    The knobs travel through a YAML file, a process boundary and a driver
    constructor. A mismatch anywhere in that chain still produces correct
    games -- just far fewer of them -- so it is invisible in the output and
    shows up only as a throughput number nobody is watching. Checking costs
    one small file per worker and turns a silent 6x slowdown into a startup
    error.
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
            "no worker reported its runtime configuration within "
            f"{timeout:.0f}s; self-play did not start")

    problems = []
    for index, actual in sorted(seen.items()):
        for key, want in expected.items():
            got = actual.get(key)
            if got != want:
                problems.append(f"worker {index}: {key} is {got!r}, "
                                f"expected {want!r}")
    missing = [i for i in range(len(paths)) if i not in seen]
    if missing and not problems:
        log.warning("worker(s) %s did not report a runtime config", missing)
    if problems:
        raise ConfigMismatch(
            "workers did not start with the requested configuration:\n  "
            + "\n  ".join(problems[:10])
            + ("\n  ..." if len(problems) > 10 else ""))
    return seen


def overshoot_floor(workers, parallel_games, max_game_ply):
    """Positions below which a shard target cannot be honoured at all.

    Every game in flight plays to its result, so the smallest shard a node can
    produce is roughly one full round of games across every worker. Asking for
    less does not make a smaller shard -- it makes the same shard while the
    target fires immediately.
    """
    return workers * parallel_games * max(1, max_game_ply // 2)


def generate_shard(args, config_path: str, network_path: str,
                   staging_dir: str, target_positions: int, expected: dict):
    """Run one round of self-play into ``staging_dir``. Returns totals.

    Raises ConfigMismatch if the workers did not start with ``expected``.
    """
    shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir, exist_ok=True)
    stats_dir = os.path.join(staging_dir, "_stats")
    os.makedirs(stats_dir, exist_ok=True)

    workers = max(1, args.workers)
    per_worker = max(1, math.ceil(target_positions / workers))
    stats_paths = [os.path.join(stats_dir, f"stats_{i}.json")
                   for i in range(workers)]
    runtime_paths = [os.path.join(stats_dir, f"runtime_{i}.json")
                     for i in range(workers)]

    ctx = mp.get_context("spawn")
    procs = []
    started = time.perf_counter()
    for i in range(workers):
        proc = ctx.Process(target=run_worker, kwargs=dict(
            config_path=config_path, network_path=network_path,
            output_dir=staging_dir, worker_id=i,
            seed=(int(time.time() * 1000) + i * 7919) & 0x7FFFFFFF,
            device=args.device, num_games=0,
            target_positions=per_worker, log_every=0,
            stats_path=stats_paths[i],
            runtime_config_path=runtime_paths[i],
            # A shard boundary is not a quota: whatever the games in flight
            # produce past it goes into the shard. Scaling parallel_games
            # down to fit the target -- which is right for loop.py's exact
            # per-generation quota -- would collapse 48 games to 1 here.
            scale_parallel_to_target=False,
            watchdog_seconds=args.watchdog_seconds,
            heartbeat_seconds=0, torch_threads=args.torch_threads,
            affinity=args.affinity, workers_total=workers))
        proc.start()
        procs.append(proc)

    try:
        observed = verify_runtime_config(
            runtime_paths, expected,
            alive=lambda: any(p.is_alive() for p in procs))
    except ConfigMismatch:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(5.0)
        raise
    first = observed[min(observed)]
    log.info("workers confirmed: %d games in flight, NN batch %d, fp16=%s, "
             "visits=%d", first["parallel_games"], first["nn_batch"],
             first["fp16"], first["visits"])

    for proc in procs:
        proc.join()
    elapsed = time.perf_counter() - started

    totals = {"games": 0, "positions": 0, "plies": 0, "seconds": elapsed,
              "failed": [p.exitcode for p in procs if p.exitcode]}
    for path in stats_paths:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        totals["games"] += int(payload.get("games", 0))
        totals["positions"] += int(payload.get("positions", 0))
        totals["plies"] += int(payload.get("plies", 0))
    shutil.rmtree(stats_dir, ignore_errors=True)
    return totals


def resolve_model(store, cache_dir: str, args, current):
    """The pointer the node should be playing with, and its local file."""
    pointer = fetch_latest(store, attempts=args.retry_attempts,
                           base_delay=args.retry_backoff)
    if pointer is None:
        if current is not None:
            log.warning("latest.json disappeared; staying on generation %d",
                        current[0].generation)
            return current
        raise StorageError(
            "models/latest.json does not exist. Publish a network first:\n"
            "  python scripts/publish_model.py --network networks/latest.mylc0")
    if current is not None and current[0].generation == pointer.generation \
            and current[0].sha256 == pointer.sha256:
        return current
    path = ensure_model(store, pointer, cache_dir,
                        attempts=args.retry_attempts,
                        base_delay=args.retry_backoff)
    return pointer, path


def status_line(node_id, pointer: LatestPointer, shard_index, totals,
                backpressure, uploader) -> str:
    rate = (totals["positions"] / totals["seconds"] * 60
            if totals["seconds"] else 0.0)
    stats = uploader.stats
    return (f"[{node_id}] gen {pointer.generation} "
            f"(net {pointer.sha256[:8]}) | shard #{shard_index} "
            f"{totals['games']} games / {totals['positions']} pos "
            f"| {rate:.0f} pos/min "
            f"| backlog {backpressure.backlog_gb:.2f} GB "
            f"| sent {stats.uploaded} shards {stats.bytes_sent / 1e6:.0f} MB "
            f"at {stats.mb_per_s:.1f} MB/s "
            f"| failed {stats.failed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--node-id", default=None,
                        help="default: hostname plus a random suffix, kept in "
                             "the cache dir across restarts")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-games", type=int, default=0,
                        help="override selfplay.parallel_games")
    parser.add_argument("--nn-batch", type=int, default=0,
                        help="override selfplay.batch_size")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--affinity", action="store_true")
    parser.add_argument("--watchdog-seconds", type=float, default=120.0)

    parser.add_argument("--shard-positions", type=int, default=50000,
                        help="close the shard after roughly this many "
                             "positions")
    parser.add_argument("--shard-games", type=int, default=0,
                        help="alternative target, in finished games")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="stop after this many shards (0 = forever)")

    parser.add_argument("--cache-dir", default="node_cache",
                        help="models, node id and the upload outbox")
    parser.add_argument("--keep-local-shards", action="store_true",
                        help="do not delete a shard after it is uploaded")
    parser.add_argument("--max-backlog-gb", type=float, default=20.0,
                        help="pause generation when unsent shards exceed this")
    parser.add_argument("--retry-attempts", type=int, default=6)
    parser.add_argument("--retry-backoff", type=float, default=2.0,
                        help="first backoff in seconds; doubles per attempt")
    parser.add_argument("--poll-latest-seconds", type=float, default=0.0,
                        help="extra latest.json poll while draining a backlog "
                             "(0 = only between shards)")
    parser.add_argument("--status-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="check config and R2 access, then exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [node] %(levelname)s %(message)s")
    limit_thread_pools(max(1, args.torch_threads))
    _install_signal_handlers()

    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    node_id = load_or_create_node_id(cache_dir, args.node_id)
    models_dir = os.path.join(cache_dir, "models")
    outbox = Outbox(os.path.join(cache_dir, "outbox"))
    staging = os.path.join(cache_dir, "staging")

    load_config(args.config)   # fail fast on a broken config
    run_config = os.path.join(cache_dir, "run_config.yaml")
    _dump_config(args.config, run_config, args.parallel_games, args.nn_batch)
    effective = load_config(run_config)

    print(f"node {node_id}")
    print(f"  config           {args.config} -> {run_config}")
    print(f"  visits           {effective.selfplay.visits} "
          f"(fixed; minibatch {effective.selfplay.search.minibatch_size})")
    print(f"  workers          {args.workers}, "
          f"parallel_games {effective.selfplay.parallel_games}, "
          f"nn_batch {effective.selfplay.batch_size}")
    target = args.shard_positions
    print(f"  shard target     {target} positions"
          if not args.shard_games else
          f"  shard target     {args.shard_games} games")
    print(f"  cache            {cache_dir}")
    floor = overshoot_floor(args.workers, effective.selfplay.parallel_games,
                            effective.selfplay.max_game_ply)
    if not args.shard_games and args.shard_positions < floor:
        print(f"\n  NOTE: --shard-positions {args.shard_positions} is below "
              f"{floor}, which is one full\n"
              f"        round of {args.workers} workers x "
              f"{effective.selfplay.parallel_games} games x ~"
              f"{effective.selfplay.max_game_ply // 2} plies. Every game in "
              f"flight plays to its\n"
              f"        result, so the shard will be roughly that size "
              f"regardless. For a short\n"
              f"        test use fewer workers and games, not a smaller "
              f"target.")
    print(f"  backlog limit    {args.max_backlog_gb:.1f} GB, "
          f"keep local: {args.keep_local_shards}")
    print("R2 configuration")
    print(describe_env())

    try:
        store = store_from_env()
    except StorageError as exc:
        print(f"\n{exc}")
        return 2

    try:
        current = resolve_model(store, models_dir, args, None)
    except StorageError as exc:
        print(f"\n{exc}")
        return 3
    pointer, network_path = current
    print(f"\nplaying generation {pointer.generation} "
          f"(sha256 {pointer.sha256[:16]}...)\n")

    if args.dry_run:
        print("dry run: R2 reachable, model verified, nothing generated")
        return 0

    expected_runtime = {
        "parallel_games": effective.selfplay.parallel_games,
        "nn_batch": effective.selfplay.batch_size,
        "visits": effective.selfplay.visits,
        "minibatch_size": effective.selfplay.search.minibatch_size,
        "fp16": bool(effective.selfplay.fp16),
    }

    uploader = ShardUploader(store, outbox, attempts=args.retry_attempts,
                             base_delay=args.retry_backoff,
                             keep_local=args.keep_local_shards)
    backpressure = Backpressure(outbox, args.max_backlog_gb)
    sequence = 0
    last_status = 0.0

    # Anything left from a previous life goes out before new work is made.
    leftovers = outbox.pending()
    if leftovers:
        log.info("resuming: %d shard(s) still in the outbox", len(leftovers))
        uploader.drain()

    while not _STOP["requested"]:
        if args.max_shards and sequence >= args.max_shards:
            log.info("reached --max-shards %d", args.max_shards)
            break

        if not backpressure.check():
            uploader.drain(should_continue=lambda: not _STOP["requested"])
            if not backpressure.check():
                log.warning("backlog still %.2f GB; waiting %.0fs",
                            backpressure.backlog_gb, args.retry_backoff * 10)
                time.sleep(args.retry_backoff * 10)
                continue

        sequence += 1
        shard_id = make_shard_id(node_id, sequence)
        generation = pointer.generation
        log.info("shard %s: generating on generation %d", shard_id, generation)

        target_positions = args.shard_positions
        if args.shard_games:
            # Approximate: a game of this network averages about half the ply
            # cap. The shard target is a batching decision, not a training one,
            # so landing near it is enough.
            target_positions = args.shard_games * max(
                1, effective.selfplay.max_game_ply // 2)

        try:
            totals = generate_shard(args, run_config, network_path, staging,
                                    target_positions, expected_runtime)
        except ConfigMismatch as exc:
            log.error("%s", exc)
            log.error("Refusing to generate slow data. This is a "
                      "configuration bug, not a transient failure.")
            return 5
        if totals["failed"]:
            log.error("worker(s) exited with %s", totals["failed"])

        chunks = collect_chunks(staging)
        if not chunks:
            log.error("shard %s produced no chunks; not packing", shard_id)
            if totals["failed"]:
                log.error("self-play is failing; stopping rather than "
                          "spinning")
                return 4
            continue

        data_key = shard_key(generation, node_id, today(), shard_id)
        shard_path = outbox.path_for(shard_id)
        packed = pack_shard(chunks, shard_path)
        manifest = build_manifest(
            shard_id=shard_id, generation=generation,
            network_sha256=pointer.sha256, node_id=node_id, packed=packed,
            data_key=data_key, games=totals["games"],
            positions=totals["positions"],
            visits=effective.selfplay.visits,
            extra={"plies": totals["plies"],
                   "seconds": round(totals["seconds"], 1),
                   "model_key": pointer.key})
        # The manifest lands next to the data before any upload is attempted,
        # so a node killed here still has a complete, self-describing shard.
        with open(outbox.manifest_path_for(shard_id), "wb") as handle:
            handle.write(manifest.to_json())
        shutil.rmtree(staging, ignore_errors=True)

        log.info("shard %s packed: %d chunks, %.1f MB, sha256 %s...",
                 shard_id, packed["chunks"], packed["size"] / 1e6,
                 str(packed["sha256"])[:12])

        uploader.drain(should_continue=lambda: True)

        now = time.time()
        if now - last_status >= args.status_seconds:
            last_status = now
            log.info("%s", status_line(node_id, pointer, sequence, totals,
                                       backpressure, uploader))

        # Only now, between shards, may the network change.
        try:
            candidate = resolve_model(store, models_dir, args,
                                      (pointer, network_path))
        except StorageError as exc:
            log.warning("could not re-check latest.json: %s; staying on "
                        "generation %d", exc, pointer.generation)
            continue
        if candidate[0].generation != pointer.generation:
            log.info("new generation published: %d -> %d; switching for the "
                     "next shard", pointer.generation,
                     candidate[0].generation)
        pointer, network_path = candidate

    log.info("draining the outbox before exit")
    uploader.drain()
    remaining = outbox.pending()
    if remaining:
        log.warning("%d shard(s) could not be uploaded and remain in %s; "
                    "they will be sent when this node is started again",
                    len(remaining), outbox.directory)
    stats = uploader.stats
    log.info("node stopping: %d shard(s) uploaded, %d already present, "
             "%d failed, %.1f MB sent",
             stats.uploaded, stats.skipped, stats.failed,
             stats.bytes_sent / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
