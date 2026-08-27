"""Key layout of the bucket, and the identifiers that keep nodes apart.

    models/gen_000037/model.mylc0      immutable weights
    models/gen_000037/metadata.json    what produced them
    models/latest.json                 pointer to one immutable model

    selfplay/gen_000036/<node>/<date>/<shard_id>.tar.zst    the data
    manifests/gen_000036/<shard_id>.json                    "it is complete"

Two rules the rest of the code depends on:

*Models are immutable.* ``gen_000037/model.mylc0`` is written once and never
replaced. ``latest.json`` is the only mutable object, and it names a specific
generation, so a node that reads it either gets the previous model or the new
one -- never a half-written file under a stable name.

*A shard is complete only when its manifest exists.* The data object is
uploaded first and the manifest last, so a trainer listing ``manifests/``
cannot see a shard whose bytes are still in flight. The manifest carries the
data key, so the two are found together.

Manifests live in their own prefix rather than beside the data because that is
what the trainer lists on every sync: a flat prefix with one small object per
shard, instead of walking every node and date directory.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import re
import secrets
from typing import Optional

MODELS_PREFIX = "models"
SELFPLAY_PREFIX = "selfplay"
MANIFESTS_PREFIX = "manifests"
LATEST_KEY = f"{MODELS_PREFIX}/latest.json"

SHARD_SUFFIX = ".tar.zst"
SHARD_FORMAT_VERSION = 1

_SAFE = re.compile(r"[^a-z0-9-]+")


def generation_dir(generation: int) -> str:
    return f"gen_{generation:06d}"


def model_key(generation: int) -> str:
    return f"{MODELS_PREFIX}/{generation_dir(generation)}/model.mylc0"


def model_metadata_key(generation: int) -> str:
    return f"{MODELS_PREFIX}/{generation_dir(generation)}/metadata.json"


def shard_key(generation: int, node_id: str, date: str, shard_id: str) -> str:
    return (f"{SELFPLAY_PREFIX}/{generation_dir(generation)}/{node_id}/"
            f"{date}/{shard_id}{SHARD_SUFFIX}")


def manifest_key(generation: int, shard_id: str) -> str:
    return f"{MANIFESTS_PREFIX}/{generation_dir(generation)}/{shard_id}.json"


def manifest_prefix(generation: int) -> str:
    return f"{MANIFESTS_PREFIX}/{generation_dir(generation)}/"


def generation_of_manifest_key(key: str) -> Optional[int]:
    match = re.search(r"/gen_(\d{6})/", key)
    return int(match.group(1)) if match else None


def today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def slug(text: str, limit: int = 24) -> str:
    out = _SAFE.sub("-", text.strip().lower()).strip("-")
    return out[:limit] or "node"


def make_node_id(explicit: Optional[str] = None) -> str:
    """A name that will not collide with another machine's.

    Hostname alone is not enough: Vast.ai containers are routinely called
    ``C.12345`` or just ``localhost``, and two rented boxes can easily share
    one. The random suffix is what actually guarantees uniqueness; the
    hostname is there so a human can tell the nodes apart in the logs.
    """
    if explicit:
        return slug(explicit, limit=48)
    host = slug(platform.node() or "node")
    return f"{host}-{secrets.token_hex(4)}"


def load_or_create_node_id(cache_dir: str,
                           explicit: Optional[str] = None) -> str:
    """Keep the id across restarts so a node's shards stay attributable.

    A restarted node must not pick a new identity: its unsent shards are
    already named after the old one, and the retry queue would otherwise
    upload them under a name nothing else refers to.
    """
    if explicit:
        return slug(explicit, limit=48)
    path = os.path.join(cache_dir, "node_id")
    try:
        with open(path, encoding="utf-8") as handle:
            existing = handle.read().strip()
        if existing:
            return existing
    except OSError:
        pass
    node_id = make_node_id()
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + f".tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(node_id)
    os.replace(tmp, path)
    return node_id


def make_shard_id(node_id: str, sequence: int) -> str:
    """Unique across nodes, restarts and clock resets.

    ``<node>-<utc timestamp>-<sequence>-<random>``. The timestamp sorts
    usefully, the sequence disambiguates shards produced inside one second,
    and the random tail covers the case that matters most: a node that
    restarts, resets its sequence to zero and lands in the same second.
    """
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{node_id}-{stamp}-{sequence:06d}-{secrets.token_hex(3)}"


__all__ = [
    "MODELS_PREFIX", "SELFPLAY_PREFIX", "MANIFESTS_PREFIX", "LATEST_KEY",
    "SHARD_SUFFIX", "SHARD_FORMAT_VERSION", "generation_dir", "model_key",
    "model_metadata_key", "shard_key", "manifest_key", "manifest_prefix",
    "generation_of_manifest_key", "today", "utc_now", "slug", "make_node_id",
    "load_or_create_node_id", "make_shard_id",
]
