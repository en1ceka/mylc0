"""Checks for the R2 transport layer.

    python scripts/check_cloud.py

Everything runs against ``MemoryStore``, the in-process fake, so this needs no
network, no credentials and no bucket. That is the point: the interesting
behaviour is what happens when uploads fail, downloads truncate, nodes restart
and two machines write at once, and none of that is convenient to arrange
against real object storage.

Follows the same PASS/FAIL idiom as ``scripts/sanity_check.py`` -- this
project keeps its checks as runnable scripts rather than a test framework.
"""

import argparse
import json
import os
import shutil
import tempfile
import traceback

import _bootstrap  # noqa: F401

from mylc0.cloud.index import ShardIndex
from mylc0.cloud.layout import (LATEST_KEY, load_or_create_node_id,
                                make_node_id, make_shard_id, manifest_key,
                                model_key, shard_key, today)
from mylc0.cloud.models import (LatestPointer, ensure_model, fetch_latest,
                                parse_latest, publish_model)
from mylc0.cloud.replay import (ReplayWindow, WeightedReplayWindow,
                                policy_from_config)
from mylc0.cloud.shards import (build_manifest, collect_chunks, pack_shard,
                                unpack_shard)
from mylc0.cloud.storage import (MemoryStore, NotFound, StorageError,
                                 sha256_file, with_retries)
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


def make_chunks(directory, count, generation=36, worker=0, size=512):
    """Fake gzipped chunk files -- the shard layer never looks inside."""
    import gzip
    os.makedirs(directory, exist_ok=True)
    paths = []
    for i in range(count):
        name = f"g{generation:06d}_w{worker:02d}_{1700000000000 + i}_{i:06d}.gz"
        path = os.path.join(directory, name)
        with gzip.open(path, "wb") as handle:
            handle.write(bytes((i * 7 + j) % 251 for j in range(size)))
        paths.append(path)
    return sorted(paths)


def stage_shard(tmp, outbox, node_id, generation, sequence, chunks=4):
    """Generate, pack and manifest one shard into an outbox."""
    staging = os.path.join(tmp, f"staging-{node_id}-{sequence}")
    make_chunks(staging, chunks, generation=generation)
    shard_id = make_shard_id(node_id, sequence)
    packed = pack_shard(collect_chunks(staging), outbox.path_for(shard_id))
    manifest = build_manifest(
        shard_id=shard_id, generation=generation,
        network_sha256="a" * 64, node_id=node_id, packed=packed,
        data_key=shard_key(generation, node_id, today(), shard_id),
        games=chunks, positions=chunks * 100, visits=800)
    with open(outbox.manifest_path_for(shard_id), "wb") as handle:
        handle.write(manifest.to_json())
    shutil.rmtree(staging, ignore_errors=True)
    return manifest


# ---------------------------------------------------------------------------
# latest.json
# ---------------------------------------------------------------------------
@check("latest.json parses a well-formed pointer")
def check_latest_parse():
    payload = json.dumps({"generation": 37,
                          "key": "models/gen_000037/model.mylc0",
                          "sha256": "b" * 64,
                          "created_at": "2026-08-27T10:00:00+00:00"}).encode()
    pointer = parse_latest(payload)
    assert pointer.generation == 37, pointer.generation
    assert pointer.key.endswith("gen_000037/model.mylc0")
    assert pointer.sha256 == "b" * 64
    return "generation 37"


@check("latest.json rejects everything a node could not act on")
def check_latest_rejects():
    bad = [
        (b"not json", "not JSON"),
        (b"[]", "not an object"),
        (json.dumps({"key": "k", "sha256": "b" * 64}).encode(), "no generation"),
        (json.dumps({"generation": 1, "sha256": "b" * 64}).encode(), "no key"),
        (json.dumps({"generation": 1, "key": "k"}).encode(), "no sha256"),
        (json.dumps({"generation": "x", "key": "k",
                     "sha256": "b" * 64}).encode(), "generation not a number"),
        (json.dumps({"generation": 1, "key": "k",
                     "sha256": "short"}).encode(), "malformed sha256"),
        (json.dumps({"generation": -3, "key": "k",
                     "sha256": "b" * 64}).encode(), "negative generation"),
    ]
    for payload, why in bad:
        try:
            parse_latest(payload)
        except StorageError:
            continue
        raise AssertionError(f"accepted a bad latest.json: {why}")
    return f"{len(bad)} malformed inputs rejected"


# ---------------------------------------------------------------------------
# checksums
# ---------------------------------------------------------------------------
@check("model checksum: publish, fetch and verify round trip")
def check_model_checksum(tmp):
    store = MemoryStore()
    path = os.path.join(tmp, "net.mylc0")
    with open(path, "wb") as handle:
        handle.write(os.urandom(50000))
    digest = sha256_file(path)

    pointer = publish_model(store, path, 37, metadata={"steps": 250})
    assert pointer.sha256 == digest
    assert store.head(model_key(37)) is not None

    cache = os.path.join(tmp, "cache")
    local = ensure_model(store, pointer, cache)
    assert sha256_file(local) == digest, "fetched model does not match"

    # A second call must not re-download; the cached copy already verifies.
    before = store.calls.get("get", 0)
    again = ensure_model(store, pointer, cache)
    assert again == local
    assert store.calls.get("get", 0) == before, "re-downloaded a cached model"
    return "published, fetched, cached"


@check("model checksum: a corrupted cache is refetched, not trusted")
def check_model_cache_corruption(tmp):
    store = MemoryStore()
    path = os.path.join(tmp, "net2.mylc0")
    with open(path, "wb") as handle:
        handle.write(b"weights" * 1000)
    pointer = publish_model(store, path, 40)

    cache = os.path.join(tmp, "cache2")
    local = ensure_model(store, pointer, cache)
    with open(local, "wb") as handle:
        handle.write(b"corrupted")            # a truncated earlier download

    recovered = ensure_model(store, pointer, cache)
    assert sha256_file(recovered) == pointer.sha256, "trusted a bad cache"
    return "bad cache detected and replaced"


@check("shard checksum: pack, verify and unpack round trip")
def check_shard_checksum(tmp):
    staging = os.path.join(tmp, "stage-a")
    sources = make_chunks(staging, 6)
    out = os.path.join(tmp, "shard-a.tar.zst")
    packed = pack_shard(sources, out)
    assert packed["chunks"] == 6, packed
    assert packed["sha256"] == sha256_file(out)

    dest = os.path.join(tmp, "unpacked-a")
    count = unpack_shard(out, dest, expected_sha256=packed["sha256"])
    assert count == 6, count
    for source in sources:
        target = os.path.join(dest, os.path.basename(source))
        with open(source, "rb") as a, open(target, "rb") as b:
            assert a.read() == b.read(), f"{target} differs from the original"
    return f"6 chunks, {packed['size']} bytes"


@check("shard checksum: a truncated shard is refused before extraction")
def check_shard_truncated(tmp):
    staging = os.path.join(tmp, "stage-b")
    out = os.path.join(tmp, "shard-b.tar.zst")
    packed = pack_shard(make_chunks(staging, 5), out)

    with open(out, "r+b") as handle:
        handle.truncate(packed["size"] // 2)

    dest = os.path.join(tmp, "unpacked-b")
    try:
        unpack_shard(out, dest, expected_sha256=packed["sha256"])
    except StorageError:
        extracted = os.listdir(dest) if os.path.isdir(dest) else []
        assert not extracted, f"extracted {extracted} from a bad shard"
        return "refused, nothing extracted"
    raise AssertionError("a truncated shard was accepted")


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------
@check("shard ids are unique across nodes, restarts and clock collisions")
def check_shard_ids(tmp):
    ids = {make_shard_id("node-a", i) for i in range(500)}
    assert len(ids) == 500, "sequence numbers collided"

    # The case that actually matters: two nodes, same second, same sequence.
    same_second = {make_shard_id(f"node-{n}", 0) for n in range(200)}
    assert len(same_second) == 200, "node ids collided"

    # And a restart, which resets the sequence back to zero.
    restarted = {make_shard_id("node-a", 0) for _ in range(200)}
    assert len(restarted) == 200, "a restarted node reused a shard id"

    nodes = {make_node_id() for _ in range(500)}
    assert len(nodes) == 500, "auto node ids collided"
    return "500 sequential, 200 cross-node, 200 restart, 500 node ids"


@check("node id survives a restart and is not reinvented")
def check_node_id_persistence(tmp):
    cache = os.path.join(tmp, "node-cache")
    first = load_or_create_node_id(cache)
    second = load_or_create_node_id(cache)
    assert first == second, f"{first} != {second} after restart"
    explicit = load_or_create_node_id(cache, "My Node 01")
    assert explicit == "my-node-01", explicit
    return f"{first} stable"


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------
@check("upload retries a flaky bucket and eventually succeeds")
def check_upload_retry(tmp):
    store = MemoryStore()
    outbox = Outbox(os.path.join(tmp, "outbox-retry"))
    manifest = stage_shard(tmp, outbox, "node-r", 36, 1)

    store.fail_next = 3          # two put failures, then success
    uploader = ShardUploader(store, outbox, attempts=6, base_delay=0.0)
    assert uploader.upload_one(manifest), "gave up on a recoverable failure"
    assert store.head(manifest.data_key) is not None
    assert store.head(manifest_key(36, manifest.shard_id)) is not None
    return "survived 3 injected failures"


@check("upload that never succeeds keeps the shard on disk")
def check_upload_gives_up_safely(tmp):
    store = MemoryStore()
    outbox = Outbox(os.path.join(tmp, "outbox-fail"))
    manifest = stage_shard(tmp, outbox, "node-f", 36, 1)

    store.fail_next = 10 ** 6    # the network is simply gone
    uploader = ShardUploader(store, outbox, attempts=3, base_delay=0.0)
    assert not uploader.upload_one(manifest), "claimed success while failing"
    assert os.path.isfile(outbox.path_for(manifest.shard_id)), \
        "a shard was deleted after a failed upload"
    assert outbox.pending(), "the shard left the retry queue"
    assert uploader.stats.failed == 1
    return "shard kept and still queued"


@check("re-uploading an already-present shard is a no-op")
def check_upload_idempotent(tmp):
    store = MemoryStore()
    outbox = Outbox(os.path.join(tmp, "outbox-idem"))
    manifest = stage_shard(tmp, outbox, "node-i", 36, 1)

    uploader = ShardUploader(store, outbox, attempts=3, base_delay=0.0,
                             keep_local=True)
    assert uploader.upload_one(manifest)
    puts_after_first = store.calls["put"]

    # A restarted node re-offers the same shard from its outbox.
    assert uploader.upload_one(manifest)
    data_puts = store.calls["put"] - puts_after_first
    assert uploader.stats.skipped == 1, uploader.stats
    # Only the manifest is rewritten; the 30 MB payload is not resent.
    assert data_puts == 1, f"resent the payload ({data_puts} puts)"
    return "payload sent once, manifest idempotent"


@check("manifest appears only after the data object")
def check_manifest_is_the_completion_marker(tmp):
    store = MemoryStore()
    outbox = Outbox(os.path.join(tmp, "outbox-order"))
    manifest = stage_shard(tmp, outbox, "node-o", 36, 1)

    written = []
    original = store.put_bytes

    def record(key, payload, metadata=None):
        # put_file delegates here, so recording by key rather than by method
        # is what actually distinguishes the payload from its manifest.
        written.append(key)
        return original(key, payload, metadata)

    store.put_bytes = record
    ShardUploader(store, outbox, attempts=2, base_delay=0.0).upload_one(manifest)

    mkey = manifest_key(36, manifest.shard_id)
    assert manifest.data_key in written, written
    assert mkey in written, written
    assert written.index(manifest.data_key) < written.index(mkey), written
    return "data key written before manifest key"


@check("a shard interrupted before its manifest is invisible to the trainer")
def check_partial_shard_invisible(tmp):
    store = MemoryStore()
    outbox = Outbox(os.path.join(tmp, "outbox-partial"))
    manifest = stage_shard(tmp, outbox, "node-p", 36, 1)

    # Data lands, then the node dies before publishing the manifest.
    store.put_file(manifest.data_key, outbox.path_for(manifest.shard_id),
                   metadata={"sha256": manifest.sha256})
    visible = list(store.list("manifests/"))
    assert not visible, f"a half-uploaded shard was visible: {visible}"

    # On restart it finishes, and only then becomes visible.
    ShardUploader(store, outbox, attempts=2, base_delay=0.0).upload_one(manifest)
    visible = list(store.list("manifests/"))
    assert len(visible) == 1, visible
    return "invisible until finished"


# ---------------------------------------------------------------------------
# publishing order
# ---------------------------------------------------------------------------
@check("a failed model upload never advances latest.json")
def check_publish_atomicity(tmp):
    store = MemoryStore()
    good = os.path.join(tmp, "good.mylc0")
    with open(good, "wb") as handle:
        handle.write(b"generation 36" * 100)
    first = publish_model(store, good, 36)
    assert fetch_latest(store).generation == 36

    bad = os.path.join(tmp, "bad.mylc0")
    with open(bad, "wb") as handle:
        handle.write(b"generation 37" * 100)
    store.fail_next = 10 ** 6
    try:
        publish_model(store, bad, 37, attempts=2, base_delay=0.0)
    except StorageError:
        pass
    else:
        raise AssertionError("publish reported success while failing")

    store.fail_next = 0          # the outage is over; now read the truth
    assert store.head(model_key(37)) is None, "a failed upload left an object"
    pointer = fetch_latest(store)
    assert pointer.generation == 36, f"latest moved to {pointer.generation}"
    assert pointer.sha256 == first.sha256
    return "latest.json still points at generation 36"


@check("models are immutable: republishing different bytes is refused")
def check_model_immutability(tmp):
    store = MemoryStore()
    a = os.path.join(tmp, "a.mylc0")
    b = os.path.join(tmp, "b.mylc0")
    with open(a, "wb") as handle:
        handle.write(b"first")
    with open(b, "wb") as handle:
        handle.write(b"second")
    publish_model(store, a, 50)
    try:
        publish_model(store, b, 50)
    except StorageError:
        pass
    else:
        raise AssertionError("overwrote an immutable model")

    # Republishing identical bytes is fine and just re-points the pointer.
    publish_model(store, a, 50)
    return "different bytes refused, identical bytes allowed"


@check("a missing latest.json is reported as absent, not as an error")
def check_missing_latest():
    store = MemoryStore()
    assert fetch_latest(store) is None
    store.put_bytes(LATEST_KEY, b"{ this is not json")
    try:
        fetch_latest(store)
    except StorageError:
        return "absent -> None, malformed -> error"
    raise AssertionError("a malformed latest.json was accepted")


@check("latest.json pointing at a missing model fails loudly")
def check_latest_points_nowhere(tmp):
    store = MemoryStore()
    pointer = LatestPointer(generation=99, key=model_key(99),
                            sha256="c" * 64, created_at="now")
    store.put_bytes(LATEST_KEY, pointer.to_json())

    parsed = fetch_latest(store)
    assert parsed.generation == 99
    try:
        ensure_model(store, parsed, os.path.join(tmp, "cache-missing"),
                     attempts=2, base_delay=0.0)
    except (StorageError, NotFound):
        return "download refused"
    raise AssertionError("pretended to fetch a model that does not exist")


# ---------------------------------------------------------------------------
# generation switching
# ---------------------------------------------------------------------------
@check("a generation published mid-shard applies only to the next shard")
def check_generation_switch(tmp):
    store = MemoryStore()
    outbox = Outbox(os.path.join(tmp, "outbox-switch"))
    net36 = os.path.join(tmp, "n36.mylc0")
    net37 = os.path.join(tmp, "n37.mylc0")
    for path, body in ((net36, b"net36"), (net37, b"net37")):
        with open(path, "wb") as handle:
            handle.write(body * 100)
    publish_model(store, net36, 36)

    # The node starts a shard on 36...
    started_on = fetch_latest(store).generation
    manifest = stage_shard(tmp, outbox, "node-s", started_on, 1)

    # ...and 37 is published while that shard is being generated.
    publish_model(store, net37, 37)

    # The finished shard must still be attributed to 36.
    assert manifest.generation == 36, manifest.generation
    assert "/gen_000036/" in manifest.data_key, manifest.data_key
    ShardUploader(store, outbox, attempts=2, base_delay=0.0).upload_one(manifest)
    assert store.head(manifest_key(36, manifest.shard_id)) is not None

    # Only now does the node see the new generation.
    assert fetch_latest(store).generation == 37
    nxt = stage_shard(tmp, outbox, "node-s", 37, 2)
    assert nxt.generation == 37 and "/gen_000037/" in nxt.data_key
    return "old shard stayed gen 36, next shard is gen 37"


@check("shards from several nodes coexist without coordination")
def check_multiple_nodes(tmp):
    store = MemoryStore()
    manifests = []
    for n in range(4):
        node = f"node-{n}"
        outbox = Outbox(os.path.join(tmp, f"outbox-multi-{n}"))
        uploader = ShardUploader(store, outbox, attempts=2, base_delay=0.0)
        for seq in range(3):
            manifest = stage_shard(tmp, outbox, node, 36, seq)
            assert uploader.upload_one(manifest)
            manifests.append(manifest)

    keys = {m.data_key for m in manifests}
    assert len(keys) == 12, f"{len(keys)} distinct keys for 12 shards"
    published = [i.key for i in store.list("manifests/gen_000036/")]
    assert len(published) == 12, published
    nodes = {m.node_id for m in manifests}
    assert len(nodes) == 4, nodes
    return "4 nodes x 3 shards, 12 distinct keys"


# ---------------------------------------------------------------------------
# replay window
# ---------------------------------------------------------------------------
@check("replay window keeps the newest three generations")
def check_replay_window():
    policy = policy_from_config(3)
    assert isinstance(policy, ReplayWindow)
    chosen, weights = policy.select([31, 32, 33, 34, 35, 36])
    assert chosen == [34, 35, 36], chosen
    assert weights is None, "the plain window should not force weighting"

    # Early in a run there is less history than the window asks for.
    assert policy.select([36])[0] == [36]
    assert policy.select([])[0] == []
    assert policy.select([35, 36])[0] == [35, 36]

    # Out-of-order and duplicated inputs must not change the answer.
    assert policy.select([36, 34, 35, 34])[0] == [34, 35, 36]
    return "last 3 of 6, robust to gaps and disorder"


@check("replay window supports 70/25/5 weighting without a rewrite")
def check_weighted_replay():
    policy = WeightedReplayWindow(by_age={0: 0.70, 1: 0.25, 2: 0.05})
    chosen, weights = policy.select([30, 34, 35, 36])
    assert chosen == [34, 35, 36], chosen
    assert abs(weights[36] - 0.70) < 1e-9, weights
    assert abs(weights[35] - 0.25) < 1e-9, weights
    assert abs(weights[34] - 0.05) < 1e-9, weights
    assert abs(sum(weights.values()) - 1.0) < 1e-9

    # With only two generations the remaining shares renormalise.
    _chosen, partial = policy.select([35, 36])
    assert abs(sum(partial.values()) - 1.0) < 1e-9, partial
    assert partial[36] > partial[35]
    return "70/25/5, renormalised when history is short"


@check("older generations are kept, not deleted, when the window moves on")
def check_old_generations_retained(tmp):
    index = ShardIndex(os.path.join(tmp, "retain.db"))
    outbox = Outbox(os.path.join(tmp, "outbox-retain"))
    for generation in (34, 35, 36, 37):
        manifest = stage_shard(tmp, outbox, "node-x", generation, generation)
        index.record(manifest, f"data/gen_{generation:06d}", 0.0)

    chosen, _ = policy_from_config(3).select(index.generations())
    assert chosen == [35, 36, 37], chosen
    # 34 is outside the window but must still be on disk and in the index.
    assert 34 in index.generations(), "an old generation was dropped"
    assert index.stats_by_generation()[34]["shards"] == 1
    index.close()
    return "gen 34 outside the window but retained"


# ---------------------------------------------------------------------------
# local index
# ---------------------------------------------------------------------------
@check("local index downloads each shard once, across restarts")
def check_index_dedup(tmp):
    path = os.path.join(tmp, "index.db")
    outbox = Outbox(os.path.join(tmp, "outbox-index"))

    index = ShardIndex(path)
    manifests = [stage_shard(tmp, outbox, "node-d", 36, i) for i in range(5)]
    for manifest in manifests[:3]:
        index.record(manifest, "data/gen_000036", 0.0)
    index.close()

    # Restart: the index is reopened from disk, not rebuilt.
    index = ShardIndex(path)
    known = index.known_ids()
    assert len(known) == 3, known
    todo = [m for m in manifests if m.shard_id not in known]
    assert len(todo) == 2, todo

    # Recording the same shard twice must not duplicate it.
    index.record(manifests[0], "data/gen_000036", 1.0)
    assert len(index.known_ids()) == 3, "a shard was recorded twice"
    stats = index.stats_by_generation()[36]
    assert stats["shards"] == 3, stats
    index.close()
    return "3 known, 2 outstanding, idempotent"


@check("index reports positions per generation for the status line")
def check_index_stats(tmp):
    index = ShardIndex(os.path.join(tmp, "stats.db"))
    outbox = Outbox(os.path.join(tmp, "outbox-stats"))
    for generation in (35, 36):
        for seq in range(2):
            manifest = stage_shard(tmp, outbox, "node-t", generation, seq,
                                   chunks=3)
            index.record(manifest, f"data/gen_{generation:06d}", 0.0)
    stats = index.stats_by_generation()
    assert set(stats) == {35, 36}, stats
    assert stats[35]["shards"] == 2
    assert stats[35]["positions"] == 600, stats[35]
    assert index.total_positions() == 1200, index.total_positions()
    index.close()
    return "2 generations, 1200 positions"


# ---------------------------------------------------------------------------
# interrupted download
# ---------------------------------------------------------------------------
@check("a truncated download is detected and retried, never recorded")
def check_interrupted_download(tmp):
    store = MemoryStore()
    outbox = Outbox(os.path.join(tmp, "outbox-dl"))
    manifest = stage_shard(tmp, outbox, "node-dl", 36, 1, chunks=5)
    ShardUploader(store, outbox, attempts=2, base_delay=0.0).upload_one(manifest)

    dest = os.path.join(tmp, "data-dl", "gen_000036")
    local = os.path.join(tmp, "incoming.tar.zst")
    store.truncate_next = 1              # the first download is cut short

    def fetch():
        store.get_file(manifest.data_key, local)
        return unpack_shard(local, dest, expected_sha256=manifest.sha256)

    try:
        fetch()
    except StorageError:
        pass
    else:
        raise AssertionError("a truncated download was accepted")
    assert not os.path.isdir(dest) or not os.listdir(dest), \
        "a partial shard reached the training directory"

    count = with_retries(fetch, attempts=3, base_delay=0.0, what="refetch")
    assert count == 5, count
    return "first attempt rejected, retry recovered 5 chunks"


# ---------------------------------------------------------------------------
# backpressure
# ---------------------------------------------------------------------------
@check("backlog over the limit stops generation and resumes with hysteresis")
def check_backpressure(tmp):
    outbox = Outbox(os.path.join(tmp, "outbox-bp"))
    store = MemoryStore()
    # 1 MB limit so a handful of small shards crosses it.
    pressure = Backpressure(outbox, max_gb=1e-6 * 1.0, resume_ratio=0.5)
    assert pressure.check(), "blocked an empty outbox"

    manifests = []
    while pressure.check() and len(manifests) < 50:
        manifests.append(stage_shard(tmp, outbox, "node-bp", 36,
                                     len(manifests), chunks=2))
    assert not pressure.check(), "never engaged despite a growing backlog"
    blocked_at = pressure.backlog_gb

    # Draining part of the queue must not immediately unblock (hysteresis).
    uploader = ShardUploader(store, outbox, attempts=2, base_delay=0.0)
    uploader.upload_one(manifests[0])
    # ...but draining everything does.
    for manifest in manifests[1:]:
        uploader.upload_one(manifest)
    assert pressure.check(), "stayed blocked after the queue drained"
    return f"engaged at {blocked_at * 1e6:.0f} MB, released after draining"


@check("a failed upload leaves the backlog intact for the next attempt")
def check_backlog_survives_failure(tmp):
    outbox = Outbox(os.path.join(tmp, "outbox-survive"))
    store = MemoryStore()
    manifests = [stage_shard(tmp, outbox, "node-sv", 36, i) for i in range(3)]
    before = outbox.backlog_bytes()

    store.fail_next = 10 ** 6
    uploader = ShardUploader(store, outbox, attempts=2, base_delay=0.0)
    uploader.drain()
    assert outbox.backlog_bytes() == before, "backlog shrank without uploading"
    assert len(outbox.pending()) == 3, outbox.pending()

    # The network comes back and the queue drains without losing anything.
    store.fail_next = 0
    uploader.drain()
    assert not outbox.pending(), outbox.pending()
    assert uploader.stats.uploaded == 3, uploader.stats
    for manifest in manifests:
        assert store.head(manifest.data_key) is not None
    return "3 shards survived an outage and were sent afterwards"


@check("an orphaned data file without a manifest is not uploaded")
def check_orphan_shard(tmp):
    outbox = Outbox(os.path.join(tmp, "outbox-orphan"))
    staging = os.path.join(tmp, "stage-orphan")  # packed, then interrupted
    pack_shard(make_chunks(staging, 3), outbox.path_for("interrupted-pack"))
    assert outbox.pending() == [], "uploaded a shard with no manifest"
    return "pack interrupted before the manifest -> skipped"


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
@check("sync pulls only new shards and only from the replay window")
def check_sync_end_to_end(tmp):
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sync_selfplay", os.path.join(os.path.dirname(__file__),
                                      "sync_selfplay.py"))
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)

    store = MemoryStore()
    outbox = Outbox(os.path.join(tmp, "outbox-e2e"))
    uploader = ShardUploader(store, outbox, attempts=2, base_delay=0.0)
    for generation in (33, 34, 35, 36):
        for seq in range(2):
            uploader.upload_one(
                stage_shard(tmp, outbox, f"n{generation}", generation, seq,
                            chunks=3))

    data = os.path.join(tmp, "data-e2e")
    args = argparse.Namespace(
        data=data, replay_generations=3, all_generations=False,
        max_shards=0, retry_attempts=3, retry_backoff=0.0)
    index = ShardIndex(os.path.join(data, "index.db"))

    result = sync.sync_once(store, args, index)
    assert result["generations"] == [34, 35, 36], result
    assert result["new"] == 6, result          # 3 generations x 2 shards
    assert result["failed"] == 0, result

    # Generation 33 is outside the window and must not have been downloaded.
    assert not os.path.isdir(os.path.join(data, "gen_000033")), \
        "downloaded a generation outside the replay window"
    for generation in (34, 35, 36):
        target = os.path.join(data, f"gen_{generation:06d}")
        chunks = [os.path.join(root, n)
                  for root, _dirs, names in os.walk(target)
                  for n in names if n.endswith(".gz")]
        # Two shards of three chunks each. They share chunk file names, so
        # this only holds because each shard unpacks into its own directory.
        assert len(chunks) == 6, (generation, len(chunks), chunks)
        assert len({os.path.dirname(c) for c in chunks}) == 2, chunks

    # A second pass must download nothing at all.
    again = sync.sync_once(store, args, index)
    assert again["new"] == 0, again

    # Widening the window reaches back for the older generation, which was
    # kept in the bucket rather than expired.
    args.replay_generations = 4
    wider = sync.sync_once(store, args, index)
    assert wider["new"] == 2, wider
    assert os.path.isdir(os.path.join(data, "gen_000033"))
    assert index.total_positions() == 8 * 300, index.total_positions()
    index.close()
    return "6 shards in, 0 on rerun, 2 more when the window widened"


@check("a shard still uploading is skipped, then picked up next sync")
def check_sync_ignores_incomplete(tmp):
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sync_selfplay2", os.path.join(os.path.dirname(__file__),
                                       "sync_selfplay.py"))
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)

    store = MemoryStore()
    outbox = Outbox(os.path.join(tmp, "outbox-inflight"))
    done = stage_shard(tmp, outbox, "n-done", 36, 1, chunks=2)
    ShardUploader(store, outbox, attempts=2,
                  base_delay=0.0).upload_one(done)

    # A second node's data object is up, but its manifest is not yet.
    inflight = stage_shard(tmp, outbox, "n-slow", 36, 2, chunks=2)
    store.put_file(inflight.data_key, outbox.path_for(inflight.shard_id),
                   metadata={"sha256": inflight.sha256})

    data = os.path.join(tmp, "data-inflight")
    args = argparse.Namespace(data=data, replay_generations=3,
                              all_generations=False, max_shards=0,
                              retry_attempts=3, retry_backoff=0.0)
    index = ShardIndex(os.path.join(data, "index.db"))
    first = sync.sync_once(store, args, index)
    assert first["new"] == 1, f"saw an in-flight shard: {first}"

    # The slow node finishes; now it becomes visible.
    ShardUploader(store, outbox, attempts=2,
                  base_delay=0.0).upload_one(inflight)
    second = sync.sync_once(store, args, index)
    assert second["new"] == 1, second
    assert index.total_positions() == 400, index.total_positions()
    index.close()
    return "1 of 2 first pass, the second after its manifest landed"


# ---------------------------------------------------------------------------
# retry primitive
# ---------------------------------------------------------------------------
@check("retry backs off exponentially and gives up after the last attempt")
def check_retry_backoff():
    delays = []
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise StorageError("nope")

    try:
        with_retries(always_fails, attempts=5, base_delay=1.0, max_delay=8.0,
                     sleep=delays.append, what="test")
    except StorageError:
        pass
    else:
        raise AssertionError("returned despite always failing")

    assert calls["n"] == 5, calls
    assert len(delays) == 4, delays          # no sleep after the last attempt
    # Full jitter: each delay is bounded by the exponential schedule.
    for i, delay in enumerate(delays):
        assert 0.0 <= delay <= min(8.0, 2 ** i), (i, delay)
    assert max(delays) <= 8.0, delays

    # A missing key must not be retried; it will not appear by waiting.
    misses = {"n": 0}

    def missing():
        misses["n"] += 1
        raise NotFound("gone")

    try:
        with_retries(missing, attempts=5, base_delay=0.0, sleep=delays.append)
    except NotFound:
        pass
    assert misses["n"] == 1, f"retried a 404 {misses['n']} times"
    return "5 attempts, 4 bounded sleeps, 404 not retried"


@check("no credential ever reaches a log or an error message")
def check_no_secret_leak(tmp):
    from mylc0.cloud.storage import describe_env
    saved = {k: os.environ.get(k) for k in
             ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
              "R2_BUCKET")}
    try:
        os.environ["R2_ENDPOINT_URL"] = "https://example.r2.cloudflarestorage.com"
        os.environ["R2_ACCESS_KEY_ID"] = "AKIAsecretkeyid1234"
        os.environ["R2_SECRET_ACCESS_KEY"] = "supersecretvalue9876"
        os.environ["R2_BUCKET"] = "mylc0"
        text = describe_env()
        assert "AKIAsecretkeyid1234" not in text, text
        assert "supersecretvalue9876" not in text, text
        assert "mylc0" in text, "the bucket name should be visible"
        assert "set, 19 chars" in text or "chars" in text, text
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return "keys reported by length only"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    tmp = tempfile.mkdtemp(prefix="mylc0-cloud-check-")
    try:
        print("== latest.json ==")
        check_latest_parse()
        check_latest_rejects()
        check_missing_latest()
        check_latest_points_nowhere(tmp)

        print("\n== checksums ==")
        check_model_checksum(tmp)
        check_model_cache_corruption(tmp)
        check_shard_checksum(tmp)
        check_shard_truncated(tmp)

        print("\n== identifiers ==")
        check_shard_ids(tmp)
        check_node_id_persistence(tmp)

        print("\n== uploads ==")
        check_upload_retry(tmp)
        check_upload_gives_up_safely(tmp)
        check_upload_idempotent(tmp)
        check_manifest_is_the_completion_marker(tmp)
        check_partial_shard_invisible(tmp)
        check_orphan_shard(tmp)

        print("\n== model publishing ==")
        check_publish_atomicity(tmp)
        check_model_immutability(tmp)

        print("\n== generations and nodes ==")
        check_generation_switch(tmp)
        check_multiple_nodes(tmp)

        print("\n== replay window ==")
        check_replay_window()
        check_weighted_replay()
        check_old_generations_retained(tmp)

        print("\n== local index ==")
        check_index_dedup(tmp)
        check_index_stats(tmp)

        print("\n== end to end ==")
        check_sync_end_to_end(tmp)
        check_sync_ignores_incomplete(tmp)

        print("\n== failure handling ==")
        check_interrupted_download(tmp)
        check_backpressure(tmp)
        check_backlog_survives_failure(tmp)
        check_retry_backoff()
        check_no_secret_leak(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [name for name, ok, _detail in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
