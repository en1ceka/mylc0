"""Shards: many games in one object, plus the manifest that completes them.

A self-play worker writes one gzipped V6 chunk per game, 40-80 kB each. At
5000 positions/min a node produces a few hundred of those an hour, and
uploading each as its own object would mean a request per game and a listing
the trainer cannot walk. A shard is simply a tar of those chunk files,
compressed with zstd.

The chunks go in **untouched**. The training records inside are the same
8356-byte V6 frames Lc0 writes, in the same gzip containers, under the same
file names -- so unpacking a shard yields exactly the directory the existing
loader already reads, and no training semantics depend on this module.

Why tar+zstd rather than one big blob: the chunk files are already gzipped, so
the zstd layer is not there to compress the game data. It is there to collapse
tar's 512-byte block padding (several hundred small files per shard) and to
give the whole shard one checksum and one streaming decoder. Level 3 is fast
enough to keep up with generation and costs nothing measurable.
"""

from __future__ import annotations


import json
import logging
import os
import tarfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .layout import SHARD_FORMAT_VERSION, SHARD_SUFFIX, utc_now
from .storage import StorageError, sha256_bytes, sha256_file

log = logging.getLogger("mylc0.cloud.shards")

ZSTD_LEVEL = 3


def _zstd():
    try:
        import zstandard
    except ImportError as exc:      # pragma: no cover - environment issue
        raise StorageError(
            "zstandard is required for shard packing: "
            "pip install zstandard") from exc
    return zstandard


@dataclass
class ShardManifest:
    """Everything needed to attribute a shard without opening it."""

    shard_id: str
    generation: int
    network_sha256: str
    node_id: str
    created_at: str
    games: int
    positions: int
    visits: int
    data_key: str
    sha256: str
    size: int
    chunks: int
    format_version: int = SHARD_FORMAT_VERSION
    extra: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> bytes:
        payload = {
            "format_version": self.format_version,
            "shard_id": self.shard_id,
            "generation": self.generation,
            "network_sha256": self.network_sha256,
            "node_id": self.node_id,
            "created_at": self.created_at,
            "games": self.games,
            "positions": self.positions,
            "visits": self.visits,
            "data_key": self.data_key,
            "sha256": self.sha256,
            "size": self.size,
            "chunks": self.chunks,
        }
        payload.update(self.extra)
        return json.dumps(payload, indent=1, default=str).encode("utf-8")


def parse_manifest(payload: bytes) -> ShardManifest:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"shard manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise StorageError("shard manifest must be a JSON object")

    required = ("shard_id", "generation", "data_key", "sha256")
    missing = [f for f in required if data.get(f) in (None, "")]
    if missing:
        raise StorageError("shard manifest is missing " + ", ".join(missing))

    version = int(data.get("format_version", 0))
    if version > SHARD_FORMAT_VERSION:
        raise StorageError(
            f"shard {data['shard_id']} is format version {version}, but this "
            f"build only understands up to {SHARD_FORMAT_VERSION}. Update "
            f"before syncing it.")

    known = {f.name for f in ShardManifest.__dataclass_fields__.values()}
    extra = {k: v for k, v in data.items() if k not in known}
    return ShardManifest(
        shard_id=str(data["shard_id"]),
        generation=int(data["generation"]),
        network_sha256=str(data.get("network_sha256", "")),
        node_id=str(data.get("node_id", "")),
        created_at=str(data.get("created_at", "")),
        games=int(data.get("games", 0)),
        positions=int(data.get("positions", 0)),
        visits=int(data.get("visits", 0)),
        data_key=str(data["data_key"]),
        sha256=str(data["sha256"]).lower(),
        size=int(data.get("size", 0)),
        chunks=int(data.get("chunks", 0)),
        format_version=version or SHARD_FORMAT_VERSION,
        extra=extra)


def collect_chunks(directory: str) -> List[str]:
    """Every ``*.gz`` chunk under a staging directory, in a stable order."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in sorted(filenames):
            if name.endswith(".gz"):
                out.append(os.path.join(dirpath, name))
    out.sort()
    return out


def pack_shard(chunk_paths: List[str], out_path: str,
               level: int = ZSTD_LEVEL) -> Dict[str, object]:
    """Tar the chunks and zstd the tar. Returns size, sha256 and count.

    Written to a ``.part`` file and renamed, so a crash mid-pack cannot leave
    something the upload queue would mistake for a finished shard.
    """
    if not chunk_paths:
        raise StorageError("refusing to pack an empty shard")
    zstandard = _zstd()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".part"

    compressor = zstandard.ZstdCompressor(level=level)
    with open(tmp, "wb") as raw:
        with compressor.stream_writer(raw) as encoder:
            # Stream straight through: a shard is tens of MB and there is no
            # reason for any of it to be resident twice.
            with tarfile.open(fileobj=encoder, mode="w|") as tar:
                for path in chunk_paths:
                    tar.add(path, arcname=os.path.basename(path))
    os.replace(tmp, out_path)

    return {"size": os.path.getsize(out_path),
            "sha256": sha256_file(out_path),
            "chunks": len(chunk_paths)}


def unpack_shard(shard_path: str, dest_dir: str,
                 expected_sha256: Optional[str] = None) -> int:
    """Extract a shard into ``dest_dir``. Returns the number of chunks.

    The hash is checked before a single byte is extracted: a truncated
    download that unpacked cleanly up to the cut would otherwise put a
    partial, plausible-looking set of games into the training data.
    """
    if expected_sha256:
        actual = sha256_file(shard_path)
        if actual != expected_sha256:
            raise StorageError(
                f"{os.path.basename(shard_path)} has sha256 {actual[:12]}..., "
                f"expected {expected_sha256[:12]}...")

    zstandard = _zstd()
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    decompressor = zstandard.ZstdDecompressor()
    with open(shard_path, "rb") as raw:
        with decompressor.stream_reader(raw) as decoder:
            with tarfile.open(fileobj=decoder, mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = os.path.basename(member.name)
                    if not name.endswith(".gz") or name != member.name:
                        # A shard only ever contains flat chunk file names.
                        # Anything else is either corruption or a crafted
                        # archive trying to write outside dest_dir.
                        raise StorageError(
                            f"unexpected member {member.name!r} in "
                            f"{os.path.basename(shard_path)}")
                    source = tar.extractfile(member)
                    if source is None:
                        continue
                    target = os.path.join(dest_dir, name)
                    tmp = target + f".part-{os.getpid()}"
                    with open(tmp, "wb") as handle:
                        while True:
                            block = source.read(1 << 20)
                            if not block:
                                break
                            handle.write(block)
                    os.replace(tmp, target)
                    count += 1
    return count


def verify_shard(shard_path: str, expected_sha256: str) -> bool:
    try:
        return sha256_file(shard_path) == expected_sha256
    except OSError:
        return False


def shard_filename(shard_id: str) -> str:
    return f"{shard_id}{SHARD_SUFFIX}"


def build_manifest(shard_id: str, generation: int, network_sha256: str,
                   node_id: str, packed: Dict[str, object], data_key: str,
                   games: int, positions: int, visits: int,
                   extra: Optional[Dict[str, object]] = None) -> ShardManifest:
    return ShardManifest(
        shard_id=shard_id, generation=generation,
        network_sha256=network_sha256, node_id=node_id,
        created_at=utc_now(), games=games, positions=positions,
        visits=visits, data_key=data_key,
        sha256=str(packed["sha256"]), size=int(packed["size"]),
        chunks=int(packed["chunks"]), extra=dict(extra or {}))


def manifest_bytes_sha(manifest: ShardManifest) -> str:
    return sha256_bytes(manifest.to_json())


__all__ = ["ShardManifest", "parse_manifest", "pack_shard", "unpack_shard",
           "verify_shard", "collect_chunks", "shard_filename",
           "build_manifest", "manifest_bytes_sha", "ZSTD_LEVEL"]
