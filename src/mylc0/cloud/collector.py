"""Turn a stream of finished games into mini-shards, off the hot path.

The producer boundary already existed and was not being used. A worker writes
each finished game to a ``.tmp`` file and renames it into place, so the moment
a game ends its training chunk is durable and atomically visible -- and the
worker immediately starts the next game. Nothing about that waits for a shard.

What forced enormous shards was the supervisor: it gave every worker a
position target, waited for all of them to exit, and only then packed. With 28
workers holding 48 games each, "wait for everyone" means draining 1344 games,
so the smallest shard a node could produce was around 300k positions and the
first byte reached R2 an hour after start.

This module watches the spool instead. It groups completed chunk files into
mini-shards and hands them to the uploader, while self-play keeps running:

    worker ends a game -> chunk renamed into the spool   (durable, atomic)
    collector groups chunks by generation                (no decompression)
    >= target positions -> pack .tar.zst + manifest      (into the outbox)
    uploader sends it                                    (separate thread)

Three properties the grouping has to keep:

*A game is never split.* The unit is a whole chunk file, so a shard boundary
can only ever fall between games. Overshooting the target is fine and
expected; a shard of 20,050 for a target of 20,000 is correct.

*Generations never mix.* Each chunk names the generation that produced it, so
chunks are bucketed by generation and a shard is always the output of exactly
one network -- which is what the manifest claims and what the replay window
relies on.

*Nothing is lost or double counted.* Chunks are moved into a staging
directory before packing, so a crash mid-pack leaves them recoverable and a
restart cannot pack the same game twice.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .layout import make_shard_id, shard_key, today
from .shards import build_manifest, pack_shard
from .storage import StorageError

log = logging.getLogger("mylc0.cloud.collect")

# g000036_w00_1787663329920_000000_n00112.gz
CHUNK_RE = re.compile(r"^g(\d{6})_w(\d+)_(\d+)_(\d+)(?:_n(\d+))?\.gz$")


@dataclass
class Chunk:
    path: str
    generation: int
    positions: int
    mtime: float


@dataclass
class CollectorStats:
    shards_built: int = 0
    chunks_packed: int = 0
    positions_packed: int = 0
    games_seen: int = 0
    by_generation: Dict[int, int] = field(default_factory=dict)


def scan_spool(spool_dir: str) -> List[Chunk]:
    """Completed chunks, oldest first.

    Only fully renamed files appear: a chunk being written is a ``.tmp``,
    which this pattern does not match, so a partially written game can never
    be picked up.
    """
    out = []
    try:
        names = os.listdir(spool_dir)
    except OSError:
        return out
    for name in names:
        match = CHUNK_RE.match(name)
        if not match:
            continue
        path = os.path.join(spool_dir, name)
        try:
            mtime = os.path.getmtime(path)
            size = os.path.getsize(path)
        except OSError:
            continue
        if size <= 0:
            continue
        positions = int(match.group(5) or 0)
        out.append(Chunk(path=path, generation=int(match.group(1)),
                         positions=positions, mtime=mtime))
    out.sort(key=lambda c: (c.mtime, c.path))
    return out


def group_by_generation(chunks: List[Chunk]) -> Dict[int, List[Chunk]]:
    groups: Dict[int, List[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.generation, []).append(chunk)
    return groups


def select_for_shard(chunks: List[Chunk], target_positions: int,
                     force: bool = False) -> List[Chunk]:
    """The chunks that make up the next shard, or [] if it is not full yet.

    Accumulates whole chunks until the target is met or passed. ``force``
    takes whatever is there, which is what a shutdown does with a partial
    shard rather than discarding it.
    """
    if not chunks:
        return []
    if force:
        return list(chunks)
    total = 0
    for index, chunk in enumerate(chunks):
        total += chunk.positions
        if total >= target_positions:
            return chunks[:index + 1]
    return []


class ShardCollector:
    """Builds mini-shards from the spool. Runs in its own thread."""

    def __init__(self, spool_dir: str, staging_dir: str, outbox,
                 node_id: str, target_positions: int,
                 network_sha_for: Callable[[int], str],
                 visits: int = 0, extra: Optional[Dict[str, object]] = None):
        self.spool_dir = spool_dir
        self.staging_dir = staging_dir
        self.outbox = outbox
        self.node_id = node_id
        self.target_positions = max(1, int(target_positions))
        self.network_sha_for = network_sha_for
        self.visits = visits
        self.extra = dict(extra or {})
        self.stats = CollectorStats()
        self.sequence = 0
        os.makedirs(spool_dir, exist_ok=True)
        os.makedirs(staging_dir, exist_ok=True)

    # -- state the heartbeat asks about -----------------------------------
    def pending(self) -> Dict[int, Dict[str, int]]:
        """Positions waiting in the spool, per generation."""
        out = {}
        for generation, chunks in group_by_generation(
                scan_spool(self.spool_dir)).items():
            out[generation] = {
                "chunks": len(chunks),
                "positions": sum(c.positions for c in chunks)}
        return out

    def pending_positions(self) -> int:
        return sum(g["positions"] for g in self.pending().values())

    # -- the work ----------------------------------------------------------
    def build_ready(self, force: bool = False) -> int:
        """Pack every shard that is ready. Returns how many were built.

        ``force`` also packs partial shards, which is what shutdown wants: a
        node stopping with 14k positions in the spool should ship them, not
        leave them for a restart that may never happen.
        """
        built = 0
        for generation, chunks in sorted(
                group_by_generation(scan_spool(self.spool_dir)).items()):
            while True:
                selected = select_for_shard(chunks, self.target_positions,
                                            force=force)
                if not selected:
                    break
                if self._build_one(generation, selected):
                    built += 1
                chunks = chunks[len(selected):]
                if force:
                    break        # force takes everything in one shard
        return built

    def _build_one(self, generation: int, chunks: List[Chunk]) -> bool:
        self.sequence += 1
        shard_id = make_shard_id(self.node_id, self.sequence)
        staging = os.path.join(self.staging_dir, shard_id)
        os.makedirs(staging, exist_ok=True)

        # Move out of the spool first. Once a chunk is in the staging
        # directory the collector owns it, so a crash cannot leave it to be
        # packed a second time by the next scan.
        moved = []
        for chunk in chunks:
            target = os.path.join(staging, os.path.basename(chunk.path))
            try:
                os.replace(chunk.path, target)
                moved.append((target, chunk.positions))
            except OSError as exc:
                # A worker may still be renaming, or the file is gone. Leave
                # it for the next pass rather than failing the whole shard.
                log.debug("skipping %s: %s", chunk.path, exc)
        if not moved:
            shutil.rmtree(staging, ignore_errors=True)
            return False

        positions = sum(p for _path, p in moved)
        paths = [path for path, _p in moved]
        try:
            packed = pack_shard(paths, self.outbox.path_for(shard_id))
        except StorageError as exc:
            log.error("could not pack shard %s: %s; returning its chunks to "
                      "the spool", shard_id, exc)
            for path, _p in moved:
                try:
                    os.replace(path, os.path.join(self.spool_dir,
                                                  os.path.basename(path)))
                except OSError:
                    pass
            shutil.rmtree(staging, ignore_errors=True)
            return False

        manifest = build_manifest(
            shard_id=shard_id, generation=generation,
            network_sha256=self.network_sha_for(generation),
            node_id=self.node_id, packed=packed,
            data_key=shard_key(generation, self.node_id, today(), shard_id),
            games=len(moved), positions=positions, visits=self.visits,
            extra=dict(self.extra))
        # The manifest lands beside the data before any upload is attempted,
        # so a node killed here still has a complete, self-describing shard.
        with open(self.outbox.manifest_path_for(shard_id), "wb") as handle:
            handle.write(manifest.to_json())
        shutil.rmtree(staging, ignore_errors=True)

        self.stats.shards_built += 1
        self.stats.chunks_packed += len(moved)
        self.stats.positions_packed += positions
        self.stats.by_generation[generation] = (
            self.stats.by_generation.get(generation, 0) + positions)
        log.info("shard %s: gen %d, %d games, %d positions, %.1f MB",
                 shard_id, generation, len(moved), positions,
                 packed["size"] / 1e6)
        return True

    def recover_staging(self) -> int:
        """Return chunks from an interrupted pack to the spool.

        A node killed between moving chunks and writing the manifest leaves
        them in staging, owned by nobody. They are finished games with
        results, so they go back to the spool to be packed again.
        """
        recovered = 0
        try:
            names = os.listdir(self.staging_dir)
        except OSError:
            return 0
        for name in names:
            path = os.path.join(self.staging_dir, name)
            if not os.path.isdir(path):
                continue
            for chunk in os.listdir(path):
                if not CHUNK_RE.match(chunk):
                    continue
                try:
                    os.replace(os.path.join(path, chunk),
                               os.path.join(self.spool_dir, chunk))
                    recovered += 1
                except OSError:
                    pass
            shutil.rmtree(path, ignore_errors=True)
        if recovered:
            log.info("recovered %d finished game(s) from an interrupted pack",
                     recovered)
        return recovered

    def run(self, stop, interval: float = 5.0) -> None:
        """Loop until ``stop`` is set. Compression happens here, not in a
        worker: packing is CPU work that would otherwise sit between two
        search batches."""
        while not stop.wait(interval):
            try:
                self.build_ready()
            except Exception as exc:                       # noqa: BLE001
                log.warning("collector pass failed: %s", exc)
                time.sleep(interval)


__all__ = ["ShardCollector", "CollectorStats", "Chunk", "scan_spool",
           "group_by_generation", "select_for_shard", "CHUNK_RE"]
