"""What this machine has already downloaded.

SQLite rather than a JSONL append log: the sync needs "do I have shard X"
thousands of times per run, and it must survive being killed halfway through
a download without leaving a half-written line that breaks the next parse.
A single-writer file with a primary key gives both for free, and the file is
small enough that ``sqlite3`` in the standard library is the whole dependency.

A shard is recorded only after its bytes are verified and extracted. The row
is therefore a claim about the local filesystem, and anything that fails
before that point simply gets retried on the next sync.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS shards (
    shard_id      TEXT PRIMARY KEY,
    generation    INTEGER NOT NULL,
    node_id       TEXT,
    data_key      TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    size          INTEGER NOT NULL,
    games         INTEGER NOT NULL DEFAULT 0,
    positions     INTEGER NOT NULL DEFAULT 0,
    chunks        INTEGER NOT NULL DEFAULT 0,
    local_dir     TEXT NOT NULL,
    created_at    TEXT,
    downloaded_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS shards_by_generation ON shards (generation);
"""


@dataclass
class ShardRow:
    shard_id: str
    generation: int
    node_id: str
    data_key: str
    sha256: str
    size: int
    games: int
    positions: int
    chunks: int
    local_dir: str
    created_at: str
    downloaded_at: float


class ShardIndex:
    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        # The sync is one writer plus a trainer that only reads; WAL lets the
        # trainer keep reading while a download commits.
        with contextlib.suppress(sqlite3.DatabaseError):
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with contextlib.suppress(sqlite3.DatabaseError):
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    # -- queries -----------------------------------------------------------
    def has(self, shard_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM shards WHERE shard_id = ?", (shard_id,)).fetchone()
        return row is not None

    def known_ids(self) -> set:
        return {r["shard_id"] for r in
                self._conn.execute("SELECT shard_id FROM shards")}

    def generations(self) -> List[int]:
        return [r[0] for r in self._conn.execute(
            "SELECT DISTINCT generation FROM shards ORDER BY generation")]

    def stats_by_generation(self) -> Dict[int, Dict[str, int]]:
        out = {}
        for row in self._conn.execute(
                "SELECT generation, COUNT(*) AS shards, SUM(games) AS games, "
                "SUM(positions) AS positions, SUM(size) AS size "
                "FROM shards GROUP BY generation ORDER BY generation"):
            out[int(row["generation"])] = {
                "shards": int(row["shards"] or 0),
                "games": int(row["games"] or 0),
                "positions": int(row["positions"] or 0),
                "size": int(row["size"] or 0)}
        return out

    def rows(self, generation: Optional[int] = None) -> Iterator[ShardRow]:
        if generation is None:
            cursor = self._conn.execute("SELECT * FROM shards")
        else:
            cursor = self._conn.execute(
                "SELECT * FROM shards WHERE generation = ?", (generation,))
        for row in cursor:
            yield ShardRow(**{k: row[k] for k in row.keys()})

    def total_positions(self) -> int:
        row = self._conn.execute(
            "SELECT SUM(positions) FROM shards").fetchone()
        return int(row[0] or 0)

    # -- writes ------------------------------------------------------------
    def record(self, manifest, local_dir: str, downloaded_at: float) -> None:
        """Mark a shard as present locally. Idempotent by shard_id."""
        self._conn.execute(
            "INSERT OR REPLACE INTO shards (shard_id, generation, node_id, "
            "data_key, sha256, size, games, positions, chunks, local_dir, "
            "created_at, downloaded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (manifest.shard_id, manifest.generation, manifest.node_id,
             manifest.data_key, manifest.sha256, manifest.size,
             manifest.games, manifest.positions, manifest.chunks,
             local_dir, manifest.created_at, downloaded_at))
        self._conn.commit()

    def forget(self, shard_id: str) -> None:
        self._conn.execute("DELETE FROM shards WHERE shard_id = ?",
                           (shard_id,))
        self._conn.commit()


__all__ = ["ShardIndex", "ShardRow"]
