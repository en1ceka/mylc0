"""Pull finished self-play shards from R2 into the local training directory.

    python scripts/sync_selfplay.py --data data --replay-generations 3
    python scripts/sync_selfplay.py --watch --interval 120

Lists the manifests in the bucket, downloads whatever this machine does not
already have, verifies it, and unpacks it into
``data/gen_NNNNNN/<shard_id>/``. That is
exactly the directory layout the existing loader walks, so the trainer needs
no knowledge of R2 at all.

A shard is downloaded once. The local index records it only after the bytes
have been verified and extracted, so an interrupted sync retries the shard it
was in the middle of and nothing else.
"""

import argparse
import logging
import os
import shutil
import time

import _bootstrap  # noqa: F401

from mylc0.cloud.index import ShardIndex
from mylc0.cloud.layout import (MANIFESTS_PREFIX, generation_dir,
                                generation_of_manifest_key, manifest_prefix)
from mylc0.cloud.models import fetch_latest
from mylc0.cloud.replay import policy_from_config
from mylc0.cloud.shards import parse_manifest, unpack_shard
from mylc0.cloud.storage import (NotFound, StorageError, describe_env,
                                 store_from_env, with_retries)

log = logging.getLogger("sync")


def list_manifests(store, generations=None):
    """Every completed shard in the bucket, newest generation last.

    Only manifests are listed. A shard whose data is still uploading has no
    manifest yet, so it simply is not in this list -- that is the whole
    mechanism that stops a trainer from seeing a partial shard.
    """
    prefixes = ([manifest_prefix(g) for g in generations] if generations
                else [f"{MANIFESTS_PREFIX}/"])
    out = []
    for prefix in prefixes:
        for info in store.list(prefix):
            if info.key.endswith(".json"):
                out.append(info)
    out.sort(key=lambda i: i.key)
    return out


def known_generations(store):
    seen = set()
    for info in store.list(f"{MANIFESTS_PREFIX}/"):
        gen = generation_of_manifest_key(info.key)
        if gen is not None:
            seen.add(gen)
    return sorted(seen)


def download_shard(store, manifest, data_root, tmp_dir, attempts,
                   backoff) -> int:
    """Fetch, verify and unpack one shard. Returns the chunk count."""
    # One directory per shard, not one per generation: chunk file names are
    # only unique within the node that made them (generation, worker, ms,
    # index), so two nodes can collide and one game would silently overwrite
    # another. The loader walks the tree, so nesting costs nothing.
    dest = os.path.join(data_root, generation_dir(manifest.generation),
                        manifest.shard_id)
    os.makedirs(tmp_dir, exist_ok=True)
    local = os.path.join(tmp_dir, manifest.shard_id + ".tar.zst")

    def fetch():
        store.get_file(manifest.data_key, local)
        # unpack_shard checks the hash before extracting anything, so a
        # truncated transfer raises here and the retry gets a clean slate.
        return unpack_shard(local, dest, expected_sha256=manifest.sha256)

    try:
        return with_retries(fetch, attempts=attempts, base_delay=backoff,
                            what=f"download shard {manifest.shard_id}")
    finally:
        try:
            os.remove(local)
        except OSError:
            pass


def sync_once(store, args, index: ShardIndex) -> dict:
    policy = policy_from_config(args.replay_generations)
    available = known_generations(store)
    if not available:
        log.info("no shards in the bucket yet")
        return {"new": 0, "bytes": 0, "generations": []}

    wanted, _weights = policy.select(available)
    if args.all_generations:
        wanted = available
    log.info("bucket has generations %s; syncing %s (%s)",
             available, wanted, policy.describe())

    manifests = list_manifests(store, wanted)
    known = index.known_ids()
    fetched, total_bytes, failed = 0, 0, 0
    tmp_dir = os.path.join(args.data, "_incoming")

    for info in manifests:
        try:
            manifest = parse_manifest(store.get_bytes(info.key))
        except (StorageError, NotFound) as exc:
            log.warning("skipping unreadable manifest %s: %s", info.key, exc)
            continue
        if manifest.shard_id in known:
            continue
        if args.max_shards and fetched >= args.max_shards:
            log.info("stopping at --max-shards %d", args.max_shards)
            break
        try:
            chunks = download_shard(store, manifest, args.data, tmp_dir,
                                    args.retry_attempts, args.retry_backoff)
        except StorageError as exc:
            failed += 1
            log.warning("shard %s not downloaded: %s", manifest.shard_id, exc)
            continue
        index.record(manifest, os.path.join(
            args.data, generation_dir(manifest.generation),
            manifest.shard_id), time.time())
        fetched += 1
        total_bytes += manifest.size
        log.info("shard %s: gen %d, %d games, %d positions, %d chunks, "
                 "%.1f MB", manifest.shard_id, manifest.generation,
                 manifest.games, manifest.positions, chunks,
                 manifest.size / 1e6)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    if failed:
        log.warning("%d shard(s) failed and will be retried next sync", failed)
    return {"new": fetched, "bytes": total_bytes, "generations": wanted,
            "failed": failed}


def report(index: ShardIndex, wanted) -> None:
    stats = index.stats_by_generation()
    if not stats:
        print("  nothing downloaded yet")
        return
    print(f"  {'gen':>6} {'shards':>7} {'games':>9} {'positions':>11} "
          f"{'size':>9}   window")
    for gen in sorted(stats):
        row = stats[gen]
        mark = "  <- in replay window" if gen in wanted else ""
        print(f"  {gen:>6} {row['shards']:>7} {row['games']:>9} "
              f"{row['positions']:>11} {row['size'] / 1e9:>8.2f}G{mark}")
    in_window = sum(stats[g]["positions"] for g in wanted if g in stats)
    print(f"  {'total':>6} {sum(r['shards'] for r in stats.values()):>7} "
          f"{sum(r['games'] for r in stats.values()):>9} "
          f"{sum(r['positions'] for r in stats.values()):>11}")
    print(f"\n  positions available to the trainer right now: {in_window}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data",
                        help="root for gen_NNNNNN/ chunk directories")
    parser.add_argument("--index", default=None,
                        help="SQLite index (default: <data>/shard_index.db)")
    parser.add_argument("--replay-generations", type=int, default=3)
    parser.add_argument("--all-generations", action="store_true",
                        help="download everything, not just the window")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="stop after this many new shards per pass")
    parser.add_argument("--retry-attempts", type=int, default=6)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--watch", action="store_true",
                        help="keep syncing on an interval")
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--status-only", action="store_true",
                        help="report what is local, download nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [sync] %(message)s")
    os.makedirs(args.data, exist_ok=True)
    index_path = args.index or os.path.join(args.data, "shard_index.db")

    with ShardIndex(index_path) as index:
        if args.status_only:
            policy = policy_from_config(args.replay_generations)
            wanted, _ = policy.select(index.generations())
            print(f"local index {index_path}")
            report(index, wanted)
            return 0

        print("R2 configuration")
        print(describe_env())
        try:
            store = store_from_env()
        except StorageError as exc:
            print(f"\n{exc}")
            return 2

        pointer = fetch_latest(store)
        if pointer is not None:
            log.info("latest published model: generation %d (%s...)",
                     pointer.generation, pointer.sha256[:12])

        while True:
            started = time.perf_counter()
            try:
                result = sync_once(store, args, index)
            except StorageError as exc:
                log.error("sync failed: %s", exc)
                result = {"new": 0, "bytes": 0, "generations": []}
            log.info("%d new shard(s), %.2f GB, in %.1fs",
                     result["new"], result["bytes"] / 1e9,
                     time.perf_counter() - started)
            print()
            report(index, result["generations"])
            if not args.watch:
                return 0
            print()
            time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
