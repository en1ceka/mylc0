"""Publishing and fetching networks, with ``latest.json`` as the last write.

The ordering is the whole point:

    put models/gen_000037/model.mylc0     <- immutable, written once
    put models/gen_000037/metadata.json
    head models/gen_000037/model.mylc0    <- prove it is really there
    put models/latest.json                <- only now does anyone see it

If any earlier step fails, ``latest.json`` still names the previous
generation, which is still downloadable. A pointer to a model that does not
exist would take down every self-play node at once, so it is written only
after the thing it points at has been read back.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

from .layout import (LATEST_KEY, model_key, model_metadata_key, utc_now)
from .storage import (NotFound, ObjectStore, StorageError, sha256_file,
                      with_retries)

log = logging.getLogger("mylc0.cloud.models")

LATEST_SCHEMA = 1


@dataclass
class LatestPointer:
    generation: int
    key: str
    sha256: str
    created_at: str = ""
    schema: int = LATEST_SCHEMA

    def to_json(self) -> bytes:
        return json.dumps({
            "schema": self.schema,
            "generation": self.generation,
            "key": self.key,
            "sha256": self.sha256,
            "created_at": self.created_at or utc_now(),
        }, indent=1).encode("utf-8")


def parse_latest(payload: bytes) -> LatestPointer:
    """Read ``latest.json``, rejecting anything a node could not act on.

    A node does exactly one thing with this file -- download the key and check
    the hash -- so the parse fails loudly if either is missing rather than
    letting the node run against a model it cannot verify.
    """
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"latest.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise StorageError("latest.json must be a JSON object")

    missing = [f for f in ("generation", "key", "sha256") if not data.get(f)]
    if missing:
        raise StorageError("latest.json is missing " + ", ".join(missing))
    try:
        generation = int(data["generation"])
    except (TypeError, ValueError) as exc:
        raise StorageError(
            f"latest.json has a non-numeric generation: "
            f"{data['generation']!r}") from exc
    if generation < 0:
        raise StorageError(f"latest.json has a negative generation: "
                           f"{generation}")
    digest = str(data["sha256"]).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise StorageError("latest.json has a malformed sha256")
    return LatestPointer(generation=generation, key=str(data["key"]),
                         sha256=digest,
                         created_at=str(data.get("created_at", "")),
                         schema=int(data.get("schema", LATEST_SCHEMA)))


def fetch_latest(store: ObjectStore, attempts: int = 5,
                 base_delay: float = 1.0) -> Optional[LatestPointer]:
    """The current pointer, or None when nothing has been published yet.

    A missing ``latest.json`` is a legitimate state -- an empty bucket at the
    very start of a run -- so it is not an error. Anything else is.
    """
    try:
        payload = with_retries(lambda: store.get_bytes(LATEST_KEY),
                               attempts=attempts, base_delay=base_delay,
                               what="fetch latest.json")
    except NotFound:
        return None
    return parse_latest(payload)


def publish_model(store: ObjectStore, local_path: str, generation: int,
                  metadata: Optional[Dict[str, object]] = None,
                  attempts: int = 5, base_delay: float = 1.0,
                  overwrite: bool = False) -> LatestPointer:
    """Upload one generation and make it current.

    Refuses to replace an existing generation: models are immutable, and a
    silently rewritten ``gen_000037`` would mean two different networks share
    a name in every shard manifest that referenced it.
    """
    if not os.path.isfile(local_path):
        raise StorageError(f"no such model file: {local_path}")

    key = model_key(generation)
    digest = sha256_file(local_path)
    size = os.path.getsize(local_path)

    existing = with_retries(lambda: store.head(key), attempts=attempts,
                            base_delay=base_delay, what=f"head {key}")
    if existing is not None and not overwrite:
        if existing.sha256 and existing.sha256 == digest:
            log.info("generation %d already published with the same bytes; "
                     "re-pointing latest at it", generation)
        else:
            raise StorageError(
                f"{key} already exists with different contents. Models are "
                f"immutable -- publish the next generation instead, or pass "
                f"overwrite=True if you are certain.")
    else:
        log.info("uploading %s (%.1f MB, sha256 %s...)", key,
                 size / 1e6, digest[:12])
        with_retries(
            lambda: store.put_file(key, local_path,
                                   metadata={"sha256": digest,
                                             "generation": str(generation)}),
            attempts=attempts, base_delay=base_delay, what=f"upload {key}")

    payload = dict(metadata or {})
    payload.update({"generation": generation, "sha256": digest,
                    "size": size, "created_at": utc_now(),
                    "key": key})
    with_retries(
        lambda: store.put_bytes(model_metadata_key(generation),
                                json.dumps(payload, indent=1,
                                           default=str).encode("utf-8")),
        attempts=attempts, base_delay=base_delay, what="upload metadata.json")

    # Read it back before advertising it. An upload that returned success but
    # left nothing behind would otherwise become every node's problem.
    check = with_retries(lambda: store.head(key), attempts=attempts,
                         base_delay=base_delay, what=f"verify {key}")
    if check is None:
        raise StorageError(
            f"{key} is not readable after upload; latest.json was NOT updated")
    if check.sha256 and check.sha256 != digest:
        raise StorageError(
            f"{key} reads back with a different sha256; latest.json was NOT "
            f"updated")

    pointer = LatestPointer(generation=generation, key=key, sha256=digest,
                            created_at=utc_now())
    with_retries(lambda: store.put_bytes(LATEST_KEY, pointer.to_json()),
                 attempts=attempts, base_delay=base_delay,
                 what="update latest.json")
    log.info("published generation %d and updated latest.json", generation)
    return pointer


def ensure_model(store: ObjectStore, pointer: LatestPointer, cache_dir: str,
                 attempts: int = 5, base_delay: float = 1.0) -> str:
    """Return a local path holding exactly the bytes ``pointer`` names.

    Cached by generation. A cached file whose hash does not match is deleted
    and fetched again rather than trusted: a truncated download from a
    previous run is exactly the kind of thing that would otherwise be used as
    a network and produce silently corrupt games.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"gen_{pointer.generation:06d}.mylc0")

    if os.path.isfile(path):
        actual = sha256_file(path)
        if actual == pointer.sha256:
            return path
        log.warning("cached %s has sha256 %s..., expected %s...; refetching",
                    os.path.basename(path), actual[:12], pointer.sha256[:12])
        os.remove(path)

    def download():
        store.get_file(pointer.key, path)
        actual = sha256_file(path)
        if actual != pointer.sha256:
            os.remove(path)
            raise StorageError(
                f"{pointer.key} downloaded with sha256 {actual[:12]}..., "
                f"expected {pointer.sha256[:12]}...")
        return path

    log.info("fetching %s -> %s", pointer.key, path)
    return with_retries(download, attempts=attempts, base_delay=base_delay,
                        what=f"download {pointer.key}")


__all__ = ["LatestPointer", "parse_latest", "fetch_latest", "publish_model",
           "ensure_model", "LATEST_SCHEMA"]
