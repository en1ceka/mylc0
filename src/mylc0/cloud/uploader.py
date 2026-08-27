"""The upload queue: never lose a shard, never fill the disk.

A self-play node has one job the network can take away from it. The rule here
is that a shard which was generated correctly is never lost because R2 was
unreachable: it stays on local disk, in a queue that survives restarts,
until it is confirmed uploaded.

The queue is the filesystem. A pending shard is a ``.tar.zst`` next to its
``.json`` manifest in the outbox directory; "uploaded" means the manifest key
exists in the bucket. There is no separate database to fall out of sync with
what is actually on disk, and a node that is killed mid-upload rediscovers its
backlog by listing a directory.

Backpressure is the other half. Disk is finite, and a node that keeps
generating into a stalled queue eventually fills it and loses everything.
Past ``max_backlog_gb`` the node stops generating and does nothing but drain.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .layout import SHARD_SUFFIX, manifest_key
from .shards import ShardManifest, parse_manifest
from .storage import ObjectStore, StorageError, with_retries

log = logging.getLogger("mylc0.cloud.upload")


@dataclass
class UploadStats:
    uploaded: int = 0
    skipped: int = 0            # already in the bucket, byte-identical
    failed: int = 0
    bytes_sent: int = 0
    seconds: float = 0.0
    retries: int = 0

    @property
    def mb_per_s(self) -> float:
        return (self.bytes_sent / 1e6 / self.seconds) if self.seconds else 0.0


class Outbox:
    """Shards on disk waiting to reach the bucket."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def path_for(self, shard_id: str) -> str:
        return os.path.join(self.directory, shard_id + SHARD_SUFFIX)

    def manifest_path_for(self, shard_id: str) -> str:
        return os.path.join(self.directory, shard_id + ".json")

    def pending(self) -> List[ShardManifest]:
        """Every shard that has a manifest and its data file, oldest first.

        A ``.tar.zst`` with no manifest is a pack that was interrupted; it is
        skipped rather than uploaded, because nothing knows what is in it.
        """
        out = []
        try:
            names = sorted(os.listdir(self.directory))
        except OSError:
            return out
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.directory, name)
            try:
                with open(path, "rb") as handle:
                    manifest = parse_manifest(handle.read())
            except (OSError, StorageError) as exc:
                log.warning("ignoring unreadable manifest %s: %s", name, exc)
                continue
            if not os.path.isfile(self.path_for(manifest.shard_id)):
                log.warning("manifest %s has no data file; skipping",
                            manifest.shard_id)
                continue
            out.append(manifest)
        out.sort(key=lambda m: m.created_at)
        return out

    def backlog_bytes(self) -> int:
        total = 0
        try:
            for name in os.listdir(self.directory):
                if name.endswith(SHARD_SUFFIX):
                    try:
                        total += os.path.getsize(
                            os.path.join(self.directory, name))
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def discard(self, shard_id: str) -> None:
        for path in (self.path_for(shard_id),
                     self.manifest_path_for(shard_id)):
            try:
                os.remove(path)
            except OSError:
                pass


class ShardUploader:
    """Sends what the outbox holds, and says what happened."""

    def __init__(self, store: ObjectStore, outbox: Outbox,
                 attempts: int = 5, base_delay: float = 2.0,
                 max_delay: float = 120.0, keep_local: bool = False):
        self.store = store
        self.outbox = outbox
        self.attempts = attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.keep_local = keep_local
        self.stats = UploadStats()

    def upload_one(self, manifest: ShardManifest) -> bool:
        """Data first, manifest last. Returns True when the shard is safe.

        The manifest is what makes a shard visible to the trainer, so it is
        written only after the data object has been confirmed present. A shard
        interrupted between the two uploads is simply invisible and gets
        finished on the next pass.
        """
        data_path = self.outbox.path_for(manifest.shard_id)
        key = manifest.data_key
        mkey = manifest_key(manifest.generation, manifest.shard_id)
        started = time.perf_counter()

        try:
            if self.store.exists_with_sha(key, manifest.sha256):
                # A restart re-offers shards that already made it. Confirming
                # by hash is one HEAD; re-sending would be tens of megabytes.
                log.info("shard %s already in the bucket; not resending",
                         manifest.shard_id)
                self.stats.skipped += 1
            else:
                before = self.stats.retries
                with_retries(
                    lambda: self.store.put_file(
                        key, data_path,
                        metadata={"sha256": manifest.sha256,
                                  "generation": str(manifest.generation),
                                  "node_id": manifest.node_id}),
                    attempts=self.attempts, base_delay=self.base_delay,
                    max_delay=self.max_delay,
                    what=f"upload shard {manifest.shard_id}")
                self.stats.uploaded += 1
                self.stats.bytes_sent += manifest.size
                self.stats.retries = before

            check = with_retries(lambda: self.store.head(key),
                                 attempts=self.attempts,
                                 base_delay=self.base_delay,
                                 max_delay=self.max_delay,
                                 what=f"verify shard {manifest.shard_id}")
            if check is None:
                raise StorageError(f"{key} is not readable after upload")

            with_retries(
                lambda: self.store.put_bytes(mkey, manifest.to_json()),
                attempts=self.attempts, base_delay=self.base_delay,
                max_delay=self.max_delay,
                what=f"publish manifest {manifest.shard_id}")
        except StorageError as exc:
            self.stats.failed += 1
            log.warning("shard %s stays in the outbox: %s",
                        manifest.shard_id, exc)
            return False
        finally:
            self.stats.seconds += time.perf_counter() - started

        if not self.keep_local:
            self.outbox.discard(manifest.shard_id)
        return True

    def drain(self, should_continue: Optional[Callable[[], bool]] = None,
              limit: int = 0) -> UploadStats:
        """Try every pending shard once. Returns cumulative stats."""
        pending = self.outbox.pending()
        if limit:
            pending = pending[:limit]
        for manifest in pending:
            if should_continue is not None and not should_continue():
                break
            self.upload_one(manifest)
        return self.stats


class Backpressure:
    """Stop generating before the disk is the thing that fails.

    Hysteresis on purpose: generation stops at ``max_gb`` and does not resume
    until the backlog is back under ``resume_ratio`` of it, so a node sitting
    exactly at the limit does not start and stop a shard at a time.
    """

    def __init__(self, outbox: Outbox, max_gb: float,
                 resume_ratio: float = 0.7):
        self.outbox = outbox
        self.max_bytes = max_gb * 1e9
        self.resume_bytes = self.max_bytes * resume_ratio
        self.blocked = False

    def check(self) -> bool:
        """True when generation may continue."""
        if self.max_bytes <= 0:
            return True
        backlog = self.outbox.backlog_bytes()
        if self.blocked:
            if backlog <= self.resume_bytes:
                log.info("backlog down to %.2f GB; resuming generation",
                         backlog / 1e9)
                self.blocked = False
        elif backlog >= self.max_bytes:
            log.warning("backlog %.2f GB reached the %.2f GB limit; pausing "
                        "generation until uploads catch up",
                        backlog / 1e9, self.max_bytes / 1e9)
            self.blocked = True
        return not self.blocked

    @property
    def backlog_gb(self) -> float:
        return self.outbox.backlog_bytes() / 1e9


__all__ = ["Outbox", "ShardUploader", "UploadStats", "Backpressure"]
