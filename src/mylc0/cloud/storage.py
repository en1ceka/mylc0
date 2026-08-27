"""Object storage, as much of it as this project needs.

Two implementations behind one interface:

``S3Store``
    Cloudflare R2 over the S3 API, via boto3. R2 is S3-compatible, so nothing
    here is Cloudflare-specific beyond the endpoint URL and ``region_name`` of
    ``auto``.

``MemoryStore``
    An in-process fake, used by ``scripts/check_cloud.py`` to exercise the
    retry, idempotency and manifest logic without a network.

The interface is deliberately tiny -- put, get, head, list, delete -- because
everything above it (shards, manifests, model publishing) is built from those
five calls and stays testable against the fake.

Nothing here ever logs a credential. Errors carry the bucket and key, never
the signing material.
"""

from __future__ import annotations

import hashlib

import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

log = logging.getLogger("mylc0.cloud")

# 5 MiB is the smallest part size the S3 multipart API accepts.
MULTIPART_THRESHOLD = 16 * 1024 * 1024
MULTIPART_CHUNKSIZE = 8 * 1024 * 1024


class StorageError(RuntimeError):
    """Anything the object store refused to do."""


class NotFound(StorageError):
    """The key does not exist."""


@dataclass
class ObjectInfo:
    key: str
    size: int
    etag: str = ""
    sha256: str = ""
    last_modified: float = 0.0


def sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ObjectStore:
    """The five operations everything else is built from."""

    def put_file(self, key: str, path: str,
                 metadata: Optional[Dict[str, str]] = None) -> ObjectInfo:
        raise NotImplementedError

    def put_bytes(self, key: str, payload: bytes,
                  metadata: Optional[Dict[str, str]] = None) -> ObjectInfo:
        raise NotImplementedError

    def get_file(self, key: str, path: str) -> ObjectInfo:
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def head(self, key: str) -> Optional[ObjectInfo]:
        raise NotImplementedError

    def list(self, prefix: str) -> Iterator[ObjectInfo]:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------
    def exists_with_sha(self, key: str, sha256: str) -> bool:
        """True when the key is already there and holds exactly these bytes.

        This is what makes uploads idempotent: a node that restarts mid-run
        re-offers shards it may already have sent, and re-sending 30 MB to
        find out is wasteful when a HEAD answers the question.
        """
        info = self.head(key)
        if info is None:
            return False
        if info.sha256 and sha256:
            return info.sha256 == sha256
        # Without our own checksum in the metadata we cannot prove equality,
        # so treat a same-size object as present. Shard keys carry a random
        # component, so a same-size collision would have to be the same upload.
        return True


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------
def with_retries(operation, attempts: int = 5, base_delay: float = 1.0,
                 max_delay: float = 60.0, what: str = "operation",
                 sleep=time.sleep, rng: Optional[random.Random] = None):
    """Run ``operation``; on failure back off exponentially and try again.

    Full jitter (``uniform(0, delay)``) rather than a fixed delay: when a node
    farm all loses the network at once, a fixed schedule makes them retry in
    lockstep and hammer R2 the moment it comes back.
    """
    rng = rng or random
    last: Optional[BaseException] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return operation()
        except NotFound:
            raise                      # a missing key will not appear by waiting
        except Exception as exc:       # noqa: BLE001 - deliberately broad
            last = exc
            if attempt >= attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = rng.uniform(0.0, delay)
            log.warning("%s failed (attempt %d/%d): %s; retrying in %.1fs",
                        what, attempt, attempts, exc, delay)
            sleep(delay)
    raise StorageError(f"{what} failed after {attempts} attempt(s): {last}")


# ---------------------------------------------------------------------------
# Cloudflare R2
# ---------------------------------------------------------------------------
class S3Store(ObjectStore):
    """Cloudflare R2 (or any S3-compatible endpoint)."""

    def __init__(self, bucket: str, endpoint_url: str, access_key: str,
                 secret_key: str, region: str = "auto",
                 connect_timeout: float = 10.0, read_timeout: float = 60.0,
                 max_attempts: int = 3):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:      # pragma: no cover - environment issue
            raise StorageError(
                "boto3 is required for R2 access: pip install boto3") from exc

        if not bucket:
            raise StorageError("no bucket configured (set R2_BUCKET)")
        if not endpoint_url:
            raise StorageError("no endpoint configured (set R2_ENDPOINT_URL)")

        self.bucket = bucket
        # botocore's own retries handle a single flaky call; the coarse retry
        # with backoff around whole operations lives in with_retries().
        self._client = boto3.client(
            "s3", endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(retries={"max_attempts": max_attempts,
                                   "mode": "standard"},
                          connect_timeout=connect_timeout,
                          read_timeout=read_timeout,
                          # R2 does not support the newer default checksums.
                          request_checksum_calculation="when_required",
                          response_checksum_validation="when_required"))
        from boto3.s3.transfer import TransferConfig
        self._transfer = TransferConfig(
            multipart_threshold=MULTIPART_THRESHOLD,
            multipart_chunksize=MULTIPART_CHUNKSIZE,
            max_concurrency=4, use_threads=True)

    # -- writes -----------------------------------------------------------
    def put_file(self, key, path, metadata=None) -> ObjectInfo:
        meta = dict(metadata or {})
        digest = meta.get("sha256") or sha256_file(path)
        meta["sha256"] = digest
        self._client.upload_file(
            path, self.bucket, key, Config=self._transfer,
            ExtraArgs={"Metadata": meta})
        return ObjectInfo(key=key, size=os.path.getsize(path), sha256=digest)

    def put_bytes(self, key, payload, metadata=None) -> ObjectInfo:
        meta = dict(metadata or {})
        digest = meta.get("sha256") or sha256_bytes(payload)
        meta["sha256"] = digest
        self._client.put_object(Bucket=self.bucket, Key=key, Body=payload,
                                Metadata=meta)
        return ObjectInfo(key=key, size=len(payload), sha256=digest)

    # -- reads ------------------------------------------------------------
    def get_file(self, key, path) -> ObjectInfo:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Download to a sibling temp file and rename: an interrupted transfer
        # must never leave something that looks like a complete shard.
        tmp = f"{path}.part-{os.getpid()}"
        try:
            self._client.download_file(self.bucket, key, tmp,
                                       Config=self._transfer)
            os.replace(tmp, path)
        except Exception as exc:
            try:
                os.remove(tmp)
            except OSError:
                pass
            if _is_missing(exc):
                raise NotFound(f"{self.bucket}/{key} does not exist") from exc
            raise StorageError(f"download of {self.bucket}/{key} failed: "
                               f"{exc}") from exc
        return ObjectInfo(key=key, size=os.path.getsize(path))

    def get_bytes(self, key) -> bytes:
        try:
            out = self._client.get_object(Bucket=self.bucket, Key=key)
            return out["Body"].read()
        except Exception as exc:
            if _is_missing(exc):
                raise NotFound(f"{self.bucket}/{key} does not exist") from exc
            raise StorageError(f"read of {self.bucket}/{key} failed: "
                               f"{exc}") from exc

    def head(self, key) -> Optional[ObjectInfo]:
        try:
            out = self._client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _is_missing(exc):
                return None
            raise StorageError(f"head of {self.bucket}/{key} failed: "
                               f"{exc}") from exc
        meta = out.get("Metadata") or {}
        modified = out.get("LastModified")
        return ObjectInfo(key=key, size=int(out.get("ContentLength", 0)),
                          etag=str(out.get("ETag", "")).strip('"'),
                          sha256=meta.get("sha256", ""),
                          last_modified=modified.timestamp() if modified else 0.0)

    def list(self, prefix) -> Iterator[ObjectInfo]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                modified = item.get("LastModified")
                yield ObjectInfo(
                    key=item["Key"], size=int(item.get("Size", 0)),
                    etag=str(item.get("ETag", "")).strip('"'),
                    last_modified=modified.timestamp() if modified else 0.0)

    def delete(self, key) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)


def _is_missing(exc: BaseException) -> bool:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
    return str(code) in ("404", "NoSuchKey", "NotFound") or \
        exc.__class__.__name__ == "NoSuchKey"


# ---------------------------------------------------------------------------
# fake, for the checks
# ---------------------------------------------------------------------------
class MemoryStore(ObjectStore):
    """In-process object store with injectable faults.

    ``fail_next`` makes the next N mutating calls raise, which is how the
    retry and resume paths get exercised without unplugging anything.
    """

    def __init__(self):
        self._data: Dict[str, bytes] = {}
        self._meta: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()
        self.fail_next = 0
        self.calls: Dict[str, int] = {}
        self.truncate_next = 0

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        if self.fail_next > 0:
            self.fail_next -= 1
            raise StorageError(f"injected failure in {name}")

    def put_file(self, key, path, metadata=None) -> ObjectInfo:
        with open(path, "rb") as handle:
            return self.put_bytes(key, handle.read(), metadata)

    def put_bytes(self, key, payload, metadata=None) -> ObjectInfo:
        with self._lock:
            self._count("put")
            meta = dict(metadata or {})
            meta.setdefault("sha256", sha256_bytes(payload))
            self._data[key] = bytes(payload)
            self._meta[key] = meta
            return ObjectInfo(key=key, size=len(payload),
                              sha256=meta["sha256"], last_modified=time.time())

    def get_bytes(self, key) -> bytes:
        with self._lock:
            self._count("get")
            if key not in self._data:
                raise NotFound(key)
            payload = self._data[key]
        if self.truncate_next > 0:
            self.truncate_next -= 1
            return payload[:max(1, len(payload) // 2)]   # corrupt download
        return payload

    def get_file(self, key, path) -> ObjectInfo:
        payload = self.get_bytes(key)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.part-{os.getpid()}"
        with open(tmp, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, path)
        return ObjectInfo(key=key, size=len(payload))

    def head(self, key) -> Optional[ObjectInfo]:
        with self._lock:
            if key not in self._data:
                return None
            return ObjectInfo(key=key, size=len(self._data[key]),
                              sha256=self._meta[key].get("sha256", ""))

    def list(self, prefix) -> Iterator[ObjectInfo]:
        with self._lock:
            keys = sorted(k for k in self._data if k.startswith(prefix))
            out = [ObjectInfo(key=k, size=len(self._data[k]),
                              sha256=self._meta[k].get("sha256", ""))
                   for k in keys]
        return iter(out)

    def delete(self, key) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._meta.pop(key, None)


# ---------------------------------------------------------------------------
# construction from the environment
# ---------------------------------------------------------------------------
ENV_VARS = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET", "R2_REGION")


def _project_root() -> str:
    # src/mylc0/cloud/storage.py -> the repo root
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir))


def load_env_file(path: Optional[str] = None) -> int:
    """Read ``.env`` into the environment. Returns how many names it set.

    Sourcing a file is shell-specific -- ``set -a && . ./.env`` in bash does
    nothing in PowerShell -- and this project runs the trainer on Windows and
    the nodes on Linux. Reading the file here means one instruction works
    everywhere.

    Variables already present in the environment always win, so an explicit
    ``export`` on a node still overrides the file, and nothing is printed:
    the file holds secrets.
    """
    candidates = [path] if path else [
        os.environ.get("MYLC0_ENV_FILE"),
        os.path.join(os.getcwd(), ".env"),
        os.path.join(_project_root(), ".env"),
    ]
    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        applied = 0
        try:
            with open(candidate, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:]
            name, _sep, value = line.partition("=")
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and value and not os.environ.get(name):
                os.environ[name] = value
                applied += 1
        return applied
    return 0


def store_from_env(**overrides) -> ObjectStore:
    """Build an S3Store from R2_* environment variables.

    Credentials only ever come from the environment; nothing reads them from
    a config file that could end up committed.
    """
    def pick(name, default=""):
        return overrides.get(name.lower()) or os.environ.get(name, default)

    if any(not pick(n) for n in ENV_VARS[:4]):
        load_env_file()

    missing = [n for n in ENV_VARS[:4] if not pick(n)]
    if missing:
        raise StorageError(
            "missing R2 configuration: " + ", ".join(missing)
            + "\n\nEasiest fix: put them in a .env file in the project root "
              "(see .env.example);\nit is gitignored and every script reads "
              "it automatically.\n\nOr set them in the shell:\n"
            + "  PowerShell:  "
            + "; ".join(f'$env:{n}="..."' for n in missing)
            + "\n  bash:        "
            + " ".join(f"export {n}=..." for n in missing))
    return S3Store(bucket=pick("R2_BUCKET"),
                   endpoint_url=pick("R2_ENDPOINT_URL"),
                   access_key=pick("R2_ACCESS_KEY_ID"),
                   secret_key=pick("R2_SECRET_ACCESS_KEY"),
                   region=pick("R2_REGION", "auto"))


def describe_env() -> str:
    """What is configured, with the secrets reduced to a length."""
    load_env_file()
    lines = []
    for name in ENV_VARS:
        value = os.environ.get(name)
        if not value:
            lines.append(f"  {name:<22} (unset)")
        elif "SECRET" in name or "ACCESS_KEY" in name:
            lines.append(f"  {name:<22} set, {len(value)} chars")
        else:
            lines.append(f"  {name:<22} {value}")
    return "\n".join(lines)


__all__ = ["ObjectStore", "S3Store", "MemoryStore", "ObjectInfo",
           "StorageError", "NotFound", "with_retries", "store_from_env",
           "load_env_file",
           "describe_env", "sha256_file", "sha256_bytes", "ENV_VARS"]
