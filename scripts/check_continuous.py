"""Checks for continuous self-play: spool, mini-shards, recovery, shutdown.

    python scripts/check_continuous.py

The producer boundary is a directory of atomically renamed chunk files, which
is exactly what makes this testable without a GPU: a "finished game" is a file
appearing in the spool, and every property worth asserting -- shards never
split a game, generations never mix, a crash loses nothing, a partial upload
stays invisible -- is a property of what the collector and uploader do with
those files.

One check does run the real self-play driver, with a stub network, to prove
that a worker starts its next game the moment one ends.
"""

import argparse
import gzip
import os
import shutil
import tempfile
import threading
import time
import traceback

import _bootstrap  # noqa: F401

from mylc0.cloud.collector import (CHUNK_RE, ShardCollector,
                                   parse_chunk_name, scan_spool)
from mylc0.cloud.index import ShardIndex
from mylc0.cloud.layout import manifest_key
from mylc0.cloud.shards import parse_manifest, unpack_shard
from mylc0.cloud.storage import MemoryStore
from mylc0.cloud.uploader import Backpressure, Outbox, ShardUploader

RESULTS = []


def check(name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                detail = fn(*args, **kwargs)
                RESULTS.append((name, True, detail or ""))
                print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
                return True
            except Exception as exc:
                RESULTS.append((name, False, repr(exc)))
                print(f"  FAIL  {name}: {exc}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator


_SEQ = {"n": 0}


def finish_game(spool, generation=36, worker=0, positions=110):
    """Simulate a worker finishing a game: an atomically named chunk file."""
    _SEQ["n"] += 1
    os.makedirs(spool, exist_ok=True)
    name = (f"g{generation:06d}_w{worker:02d}_"
            f"{1787663329920 + _SEQ['n']:013d}_{_SEQ['n']:06d}"
            f"_n{positions:05d}.gz")
    path = os.path.join(spool, name)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as handle:
        handle.write(bytes((_SEQ["n"] + i) % 251 for i in range(positions * 8)))
    os.replace(tmp, path)
    return path


def make_collector(tmp, name, target=20000, shas=None):
    root = os.path.join(tmp, name)
    spool = os.path.join(root, "spool")
    staging = os.path.join(root, "staging")
    outbox = Outbox(os.path.join(root, "outbox"))
    shas = shas or {}
    collector = ShardCollector(
        spool_dir=spool, staging_dir=staging, outbox=outbox,
        node_id=f"node-{name}", target_positions=target,
        network_sha_for=lambda g: shas.get(g, "f" * 64), visits=800)
    return collector, spool, outbox


# ---------------------------------------------------------------------------
# producer -> spool
# ---------------------------------------------------------------------------
@check("a finished game is durable immediately, without any other game ending")
def check_game_is_durable(tmp):
    collector, spool, _outbox = make_collector(tmp, "durable")
    finish_game(spool, positions=97)
    chunks = scan_spool(spool)
    assert len(chunks) == 1, chunks
    assert chunks[0].positions == 97, chunks[0]
    assert chunks[0].generation == 36
    # 1343 other games are still running; this one is already safe on disk.
    assert os.path.getsize(chunks[0].path) > 0
    return "one game, 97 positions, visible on its own"


@check("a partially written game is invisible to the collector")
def check_partial_game_invisible(tmp):
    collector, spool, _outbox = make_collector(tmp, "partial")
    os.makedirs(spool, exist_ok=True)
    # What a worker mid-write looks like: the .tmp before the rename.
    half = os.path.join(spool, "g000036_w00_1787663329999_000099_n00110.gz.tmp")
    with open(half, "wb") as handle:
        handle.write(b"\x1f\x8b truncated")
    assert scan_spool(spool) == [], "a half-written game was picked up"
    assert not CHUNK_RE.match(os.path.basename(half))
    finish_game(spool)
    assert len(scan_spool(spool)) == 1
    return ".tmp ignored, the renamed file is taken"


@check("a worker starts the next game the moment one finishes")
def check_worker_keeps_going(tmp):
    """The real driver, a stub network: game slots must refill immediately."""
    import torch
    from mylc0.net.backend import Backend
    from mylc0.net.config import load_config
    from mylc0.net.model import build_model
    from mylc0.selfplay.batched import BatchedSelfPlay

    class Stub:
        def __init__(self, real):
            self._real = real
            self.input_format = real.input_format
            self.movesleft_head = real.movesleft_head
            self.cache = real.cache
            self.evaluations = 0
            self.batches = 0
            self.timing = None

        def encode(self, history):
            return self._real.encode(history)

        def cache_key(self, history):
            return self._real.cache_key(history)

        def evaluate(self, requests):
            import hashlib

            import numpy as np
            from mylc0.net.backend import EvalResult
            self.evaluations += len(requests)
            self.batches += 1 if requests else 0
            out = []
            for req in requests:
                digest = hashlib.blake2b(req.planes.tobytes(),
                                         digest_size=8).digest()
                rng = np.random.default_rng(int.from_bytes(digest, "little"))
                p = rng.random(len(req.policy_indices)).astype(np.float32)
                p /= p.sum()
                q = float(rng.random() * 2 - 1)
                out.append(EvalResult(q=q, d=float(rng.random() * (1 - abs(q))),
                                      m=float(rng.random() * 50), p=p))
            return out

    config = load_config("configs/tiny.yaml")
    cfg = config.selfplay
    cfg.visits = 8              # fixture: keeps the check to a few seconds
    cfg.max_game_ply = 24
    cfg.parallel_games = 4
    torch.manual_seed(3)
    real = Backend(build_model(config.model), config.model, device="cpu",
                   fp16=False, max_batch_size=64)
    driver = BatchedSelfPlay(Stub(real), cfg, 4, seed=9)

    finished = []
    slots = []

    def on_game(game):
        finished.append(game)

    def on_tick():
        slots.append(driver.active_games())

    # Stop only after several games have completed, so refilling is exercised.
    driver.run(on_game=on_game, should_stop=lambda: len(finished) >= 6,
               on_tick=on_tick)

    assert len(finished) >= 6, len(finished)
    # While games were still being admitted the slots stayed full: a finished
    # game was replaced rather than leaving the worker idle.
    early = slots[:len(slots) // 2]
    assert max(early) == 4, max(early)
    assert sum(1 for s in early if s == 4) > len(early) * 0.5, \
        f"slots sat below capacity: {early[:20]}"
    return f"{len(finished)} games, slots stayed at {max(early)}/4"


@check("the position count comes only from the trailing _nNNNNN")
def check_parser_fields():
    """The failure this guards against reads as a plausible small number.

    Every field in a chunk name is a run of digits, so a parser that takes the
    wrong one reports, say, the worker id as a position count -- 7 instead of
    112 -- which looks like a slow node rather than like a bug.
    """
    cases = [
        # name, generation, worker, timestamp, index, positions
        ("g000036_w07_1787663329920_000000_n00112.gz",
         36, 7, 1787663329920, 0, 112),
        ("g000036_w09_1787663331004_000001_n00150.gz",
         36, 9, 1787663331004, 1, 150),
        # Multi-digit worker: 28 workers means ids up to 27.
        ("g000036_w27_1787663340117_000042_n00098.gz",
         36, 27, 1787663340117, 42, 98),
        ("g000036_w127_1787663340117_123456_n01234.gz",
         36, 127, 1787663340117, 123456, 1234),
        # A genuinely tiny game, so a small count is not assumed to be a bug.
        ("g000037_w00_1787663999999_000000_n00007.gz",
         37, 0, 1787663999999, 0, 7),
        # Position count larger than any other field.
        ("g000999_w01_0000000000001_000002_n99999.gz",
         999, 1, 1, 2, 99999),
    ]
    for name, gen, worker, stamp, index, positions in cases:
        fields = parse_chunk_name(name)
        assert fields is not None, f"rejected a valid name: {name}"
        assert fields["positions"] == positions, (name, fields)
        assert fields["worker"] == worker, (name, fields)
        assert fields["generation"] == gen, (name, fields)
        assert fields["timestamp"] == stamp, (name, fields)
        assert fields["index"] == index, (name, fields)
        # The specific confusion: the worker id must never be the count.
        if worker != positions:
            assert fields["positions"] != worker, (name, fields)

    # w07 with n00112 is the exact case from the report.
    assert parse_chunk_name(
        "g000036_w07_1787663329920_000000_n00112.gz")["positions"] == 112
    return f"{len(cases)} names; w07 -> 112, w09 -> 150, w27 -> 98"


@check("a name that is not a chunk is rejected, never guessed")
def check_parser_rejects():
    bad = [
        "g000036_w07_1787663329920_000000_n00112.gz.tmp",   # mid-write
        "g00036_w07_1787663329920_000000_n00112.gz",        # short generation
        "g000036_w_1787663329920_000000_n00112.gz",         # no worker id
        "g000036_w07_1787663329920_n00112.gz",              # no index
        "g000036_w07_1787663329920_000000_n.gz",            # empty count
        "g000036_w07_1787663329920_000000_x00112.gz",       # wrong marker
        "gABCDEF_w07_1787663329920_000000_n00112.gz",       # not digits
        "shard_000001.tar.zst",
        "junk.gz",
        "",
    ]
    for name in bad:
        assert parse_chunk_name(name) is None, f"accepted {name!r}"
        assert CHUNK_RE.match(name) is None, f"regex accepted {name!r}"

    # A chunk written before the suffix existed reports zero rather than
    # borrowing a number from another field.
    old = parse_chunk_name("g000036_w07_1787663329920_000000.gz")
    assert old is not None and old["positions"] == 0, old
    assert old["worker"] == 7, old
    return f"{len(bad)} malformed names rejected; a legacy name reports 0"


@check("shard sizing and the heartbeat read the same parser")
def check_one_parser(tmp):
    collector, spool, outbox = make_collector(tmp, "oneparser", target=300)
    # Worker ids that would be obvious if they leaked into a count.
    for worker, positions in ((7, 112), (9, 150), (27, 98)):
        finish_game(spool, worker=worker, positions=positions)

    # What the heartbeat prints as "shard X/20k".
    assert collector.pending_positions() == 360, collector.pending_positions()
    assert collector.pending()[36]["chunks"] == 3

    # What the shard is actually sized and manifested by.
    assert collector.build_ready() == 1
    manifest = outbox.pending()[0]
    assert manifest.positions == 360, manifest.positions
    assert manifest.games == 3, manifest.games
    # 7 + 9 + 27 = 43, which is what a worker-id bug would have produced.
    assert manifest.positions != 43
    return "spool fill and manifest both 360, not 43"


# ---------------------------------------------------------------------------
# mini-shards
# ---------------------------------------------------------------------------
@check("a mini-shard closes on the target without a global drain")
def check_minishard_closes(tmp):
    collector, spool, outbox = make_collector(tmp, "close", target=20000)
    for _ in range(9):
        finish_game(spool, positions=2000)      # 18000: not yet
    assert collector.build_ready() == 0, "closed a shard early"
    assert len(outbox.pending()) == 0

    finish_game(spool, positions=2000)          # 20000: now
    assert collector.build_ready() == 1
    pending = outbox.pending()
    assert len(pending) == 1, pending
    assert pending[0].positions == 20000, pending[0].positions
    assert pending[0].games == 10
    # The other games are still running; nothing was drained to get here.
    assert scan_spool(spool) == [], scan_spool(spool)
    return "20000 positions, 10 games, no drain"


@check("a shard never splits a game and may overshoot the target")
def check_never_splits(tmp):
    collector, spool, outbox = make_collector(tmp, "split", target=20000)
    for _ in range(4):
        finish_game(spool, positions=4975)      # 19900
    finish_game(spool, positions=150)           # 20050
    assert collector.build_ready() == 1
    manifest = outbox.pending()[0]
    assert manifest.positions == 20050, manifest.positions
    assert manifest.games == 5, manifest.games
    assert manifest.positions > 20000, "the shard was trimmed to the target"

    # A single game larger than the whole target still goes in whole.
    finish_game(spool, positions=45000)
    assert collector.build_ready() == 1
    big = [m for m in outbox.pending() if m.positions == 45000]
    assert len(big) == 1 and big[0].games == 1, outbox.pending()
    return "20050 for a 20000 target; a 45000-position game kept whole"


@check("shards never mix generations")
def check_no_generation_mixing(tmp):
    collector, spool, outbox = make_collector(tmp, "gens", target=1000)
    # A rolling switch in progress: old and new workers spool side by side.
    for _ in range(6):
        finish_game(spool, generation=36, positions=200)
    for _ in range(6):
        finish_game(spool, generation=37, positions=200)

    assert collector.build_ready() == 2, "expected one shard per generation"
    shards = outbox.pending()
    assert len(shards) == 2, shards
    by_gen = {m.generation: m for m in shards}
    assert set(by_gen) == {36, 37}, by_gen
    for generation, manifest in by_gen.items():
        # Five games of 200 reach the target; the sixth stays spooled for the
        # next shard rather than being trimmed into this one.
        assert manifest.positions == 1000, manifest.positions
        assert manifest.games == 5, manifest.games
        assert f"/gen_{generation:06d}/" in manifest.data_key
    left = collector.pending()
    assert left[36]["positions"] == 200 and left[37]["positions"] == 200, left
    return "gen 36 and gen 37 shipped separately, remainder kept per generation"


@check("a shard's chunks all belong to its generation")
def check_shard_contents(tmp):
    collector, spool, outbox = make_collector(tmp, "contents", target=500)
    for _ in range(3):
        finish_game(spool, generation=36, positions=200)
    for _ in range(3):
        finish_game(spool, generation=37, positions=200)
    collector.build_ready()

    for manifest in outbox.pending():
        dest = os.path.join(tmp, "contents", "out", manifest.shard_id)
        unpack_shard(outbox.path_for(manifest.shard_id), dest,
                     expected_sha256=manifest.sha256)
        for name in os.listdir(dest):
            assert name.startswith(f"g{manifest.generation:06d}_"), \
                f"{name} is not from generation {manifest.generation}"
    return "unpacked shards contain only their own generation"


# ---------------------------------------------------------------------------
# uploads alongside generation
# ---------------------------------------------------------------------------
@check("uploads run while games keep finishing")
def check_upload_concurrent(tmp):
    collector, spool, outbox = make_collector(tmp, "concurrent", target=500)
    store = MemoryStore()
    uploader = ShardUploader(store, outbox, attempts=3, base_delay=0.0)

    stop = threading.Event()
    produced = {"n": 0}

    def produce():
        while not stop.is_set():
            finish_game(spool, positions=100)
            produced["n"] += 1
            time.sleep(0.002)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        collector.build_ready()
        uploader.drain()
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=2.0)
    collector.build_ready()
    uploader.drain()

    assert produced["n"] > 20, produced
    assert uploader.stats.uploaded > 0, uploader.stats
    manifests = list(store.list("manifests/"))
    assert manifests, "nothing reached the bucket while games were running"
    return (f"{produced['n']} games produced, "
            f"{uploader.stats.uploaded} shards uploaded concurrently")


@check("a stalled uploader does not stop games until the backlog limit")
def check_slow_upload_does_not_block(tmp):
    collector, spool, outbox = make_collector(tmp, "slow", target=300)
    store = MemoryStore()
    store.fail_next = 10 ** 6            # R2 is simply gone
    uploader = ShardUploader(store, outbox, attempts=1, base_delay=0.0)
    pressure = Backpressure(outbox, max_gb=1.0)      # far from the limit

    for _ in range(30):
        finish_game(spool, positions=100)
        assert pressure.check(), "generation was paused while under the limit"
    built = collector.build_ready()
    uploader.drain()

    assert built >= 10, built
    assert uploader.stats.uploaded == 0, "the outage was not simulated"
    assert len(outbox.pending()) == built, "shards were lost in the outage"
    assert pressure.check(), "still under the limit, must not pause"

    # Only crossing the configured limit is allowed to apply backpressure.
    tight = Backpressure(outbox, max_gb=1e-9)
    assert not tight.check(), "the limit never engaged"
    return f"{built} shards queued during a total outage, generation ran on"


# ---------------------------------------------------------------------------
# crash recovery
# ---------------------------------------------------------------------------
@check("a restart recovers spooled games and queued shards")
def check_restart_recovery(tmp):
    collector, spool, outbox = make_collector(tmp, "restart", target=500)
    store = MemoryStore()

    for _ in range(5):
        finish_game(spool, positions=100)
    collector.build_ready()                       # one shard in the outbox
    for _ in range(3):
        finish_game(spool, positions=100)         # 300 still spooled

    queued = len(outbox.pending())
    spooled = collector.pending_positions()
    assert queued == 1 and spooled == 300, (queued, spooled)

    # The process dies here. A new one is constructed over the same directory.
    reborn, spool2, outbox2 = make_collector(tmp, "restart", target=500)
    assert len(outbox2.pending()) == queued, "queued shard was lost"
    assert reborn.pending_positions() == spooled, "spooled games were lost"

    uploader = ShardUploader(store, outbox2, attempts=2, base_delay=0.0)
    uploader.drain()
    assert uploader.stats.uploaded == 1
    reborn.build_ready(force=True)
    uploader.drain()
    assert uploader.stats.uploaded == 2, uploader.stats
    total = sum(parse_manifest(store.get_bytes(i.key)).positions
                for i in store.list("manifests/"))
    assert total == 800, total
    return "1 queued shard and 300 spooled positions both survived"


@check("an interrupted pack returns its games to the spool")
def check_interrupted_pack(tmp):
    collector, spool, _outbox = make_collector(tmp, "interrupted", target=500)
    # A crash between moving chunks into staging and writing the manifest
    # leaves finished games owned by nobody.
    orphan = os.path.join(collector.staging_dir, "half-built-shard")
    os.makedirs(orphan, exist_ok=True)
    for _ in range(3):
        path = finish_game(spool, positions=100)
        os.replace(path, os.path.join(orphan, os.path.basename(path)))
    assert collector.pending_positions() == 0, "premise wrong"

    assert collector.recover_staging() == 3
    assert collector.pending_positions() == 300, collector.pending_positions()
    assert not os.path.isdir(orphan), "staging was not cleaned up"
    return "3 games recovered from a half-built shard"


@check("shutdown flushes a partial shard rather than dropping it")
def check_shutdown_flush(tmp):
    collector, spool, outbox = make_collector(tmp, "shutdown", target=20000)
    for _ in range(7):
        finish_game(spool, positions=2000)       # 14000, well short
    assert collector.build_ready() == 0, "closed early"

    # SIGTERM: take what there is.
    assert collector.build_ready(force=True) == 1
    manifest = outbox.pending()[0]
    assert manifest.positions == 14000, manifest.positions
    assert collector.pending_positions() == 0
    return "14000 of a 20000 target shipped on shutdown"


@check("re-uploading a recovered shard is idempotent")
def check_recovery_idempotent(tmp):
    collector, spool, outbox = make_collector(tmp, "idem", target=300)
    store = MemoryStore()
    for _ in range(3):
        finish_game(spool, positions=100)
    collector.build_ready()
    manifest = outbox.pending()[0]

    uploader = ShardUploader(store, outbox, attempts=2, base_delay=0.0,
                             keep_local=True)
    assert uploader.upload_one(manifest)
    puts = store.calls["put"]
    # A restart re-offers it from the outbox; the payload must not be resent.
    assert uploader.upload_one(manifest)
    assert uploader.stats.skipped == 1, uploader.stats
    assert store.calls["put"] - puts == 1, "resent the payload"
    return "payload sent once across a restart"


@check("a shard whose manifest never landed is invisible to the trainer")
def check_partial_upload_invisible(tmp):
    collector, spool, outbox = make_collector(tmp, "invisible", target=300)
    store = MemoryStore()
    for _ in range(3):
        finish_game(spool, positions=100)
    collector.build_ready()
    manifest = outbox.pending()[0]

    store.put_file(manifest.data_key, outbox.path_for(manifest.shard_id),
                   metadata={"sha256": manifest.sha256})
    assert not list(store.list("manifests/")), "a partial shard was visible"

    ShardUploader(store, outbox, attempts=2,
                  base_delay=0.0).upload_one(manifest)
    assert len(list(store.list("manifests/"))) == 1
    assert store.head(manifest_key(36, manifest.shard_id)) is not None
    return "invisible until the manifest lands"


# ---------------------------------------------------------------------------
# many nodes, and the trainer's view
# ---------------------------------------------------------------------------
@check("mini-shards from several nodes never collide")
def check_multi_node(tmp):
    store = MemoryStore()
    ids = set()
    for n in range(4):
        collector, spool, outbox = make_collector(tmp, f"multi{n}", target=300)
        uploader = ShardUploader(store, outbox, attempts=2, base_delay=0.0)
        for _ in range(9):
            finish_game(spool, positions=100)
        collector.build_ready()
        for manifest in outbox.pending():
            assert uploader.upload_one(manifest)
            ids.add(manifest.shard_id)
    assert len(ids) == 12, len(ids)
    published = list(store.list("manifests/gen_000036/"))
    assert len(published) == 12, len(published)
    return "4 nodes x 3 mini-shards, 12 distinct ids"


@check("sync pulls mini-shards continuously and the trainer sums the nodes")
def check_sync_and_trainer_view(tmp):
    import argparse as _argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sync_selfplay", os.path.join(os.path.dirname(__file__),
                                      "sync_selfplay.py"))
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)

    store = MemoryStore()
    for n in range(3):
        collector, spool, outbox = make_collector(tmp, f"farm{n}", target=300)
        uploader = ShardUploader(store, outbox, attempts=2, base_delay=0.0)
        for _ in range(6):
            finish_game(spool, positions=100)
        collector.build_ready()
        uploader.drain()

    data = os.path.join(tmp, "farm-data")
    args = _argparse.Namespace(data=data, replay_generations=3,
                               all_generations=False, max_shards=0,
                               retry_attempts=3, retry_backoff=0.0)
    index = ShardIndex(os.path.join(data, "index.db"))
    first = sync.sync_once(store, args, index)
    assert first["new"] == 6, first          # 3 nodes x 2 mini-shards
    # loop_r2 gates on this number, summed across every node.
    assert index.total_positions() == 1800, index.total_positions()

    # More arrive while the trainer is running; the next pass takes only those.
    collector, spool, outbox = make_collector(tmp, "farm0", target=300)
    uploader = ShardUploader(store, outbox, attempts=2, base_delay=0.0)
    for _ in range(3):
        finish_game(spool, positions=100)
    collector.build_ready()
    uploader.drain()
    second = sync.sync_once(store, args, index)
    assert second["new"] == 1, second
    assert index.total_positions() == 2100, index.total_positions()
    index.close()
    return "6 shards from 3 nodes = 1800 positions, then +300"


@check("the collector reports what is waiting, for the heartbeat")
def check_pending_report(tmp):
    collector, spool, _outbox = make_collector(tmp, "pending", target=20000)
    assert collector.pending_positions() == 0
    for _ in range(4):
        finish_game(spool, generation=36, positions=1000)
    for _ in range(2):
        finish_game(spool, generation=37, positions=1000)
    pending = collector.pending()
    assert pending[36]["positions"] == 4000, pending
    assert pending[37]["chunks"] == 2, pending
    assert collector.pending_positions() == 6000
    return "4000 in gen 36, 2000 in gen 37"


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    tmp = tempfile.mkdtemp(prefix="mylc0-continuous-")
    try:
        print("== producer -> spool ==")
        check_game_is_durable(tmp)
        check_partial_game_invisible(tmp)
        check_worker_keeps_going(tmp)

        print("\n== chunk name parser ==")
        check_parser_fields()
        check_parser_rejects()
        check_one_parser(tmp)

        print("\n== mini-shards ==")
        check_minishard_closes(tmp)
        check_never_splits(tmp)
        check_no_generation_mixing(tmp)
        check_shard_contents(tmp)
        check_pending_report(tmp)

        print("\n== uploads alongside generation ==")
        check_upload_concurrent(tmp)
        check_slow_upload_does_not_block(tmp)

        print("\n== crash and shutdown ==")
        check_restart_recovery(tmp)
        check_interrupted_pack(tmp)
        check_shutdown_flush(tmp)
        check_recovery_idempotent(tmp)
        check_partial_upload_invisible(tmp)

        print("\n== farm and trainer ==")
        check_multi_node(tmp)
        check_sync_and_trainer_view(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [name for name, ok, _d in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
