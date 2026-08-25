"""Training data pipeline.

Mirrors the stages of lczero-training's C++ data loader
(``csrc/loader/stages/``), in the same order and with the same meaning:

===========================  ==================================================
``file_path_provider``       find chunk files on disk, newest first
``chunk_source_loader``      read a gzipped chunk into V6 frames
``shuffling_chunk_pool``     keep a large pool of chunks and draw from it at
                             random, evicting the oldest as new data arrives
``chunk_unpacker``           keep each position with probability
                             ``position_sampling_rate`` (decorrelates frames
                             coming from the same game)
``shuffling_frame_sampler``  reservoir shuffle over individual frames
``tensor_generator``         build ``(planes, probabilities, values)`` batches
===========================  ==================================================

The ``chunk_rescorer`` stage is deliberately absent: it rewrites targets using
Syzygy tablebases, which this project must not use.

The ``values`` tensor is ``(batch, 6, 3)`` and is indexed exactly like
``tensor_generator.cc``: ``[result, best, played, orig, root, st] x [q, d, m]``.
"""

from __future__ import annotations

import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..selfplay.trainingdata import read_chunk, unpack_planes

VALUE_RESULT, VALUE_BEST, VALUE_PLAYED, VALUE_ORIG, VALUE_ROOT, VALUE_ST = range(6)


def find_chunks(paths: Sequence[str]) -> List[str]:
    """All ``*.gz`` chunk files under the given roots, oldest first."""
    out = []
    for root in paths:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.endswith(".gz"):
                    full = os.path.join(dirpath, name)
                    try:
                        out.append((os.path.getmtime(full), full))
                    except OSError:
                        pass
    out.sort()
    return [p for _, p in out]


def frames_to_tensors(frames: Sequence[np.ndarray]):
    """``TensorGenerator::ConvertFramesToTensors``."""
    n = len(frames)
    planes = np.empty((n, 112, 8, 8), dtype=np.float32)
    probs = np.empty((n, 1858), dtype=np.float32)
    values = np.empty((n, 6, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        planes[i] = unpack_planes(frame)
        probs[i] = frame["probabilities"]
        values[i, VALUE_RESULT] = (frame["result_q"], frame["result_d"],
                                   frame["plies_left"])
        values[i, VALUE_BEST] = (frame["best_q"], frame["best_d"], frame["best_m"])
        values[i, VALUE_PLAYED] = (frame["played_q"], frame["played_d"],
                                   frame["played_m"])
        values[i, VALUE_ORIG] = (frame["orig_q"], frame["orig_d"], frame["orig_m"])
        values[i, VALUE_ROOT] = (frame["root_q"], frame["root_d"], frame["root_m"])
        values[i, VALUE_ST] = (frame["q_st"], 0.0, np.nan)
    return planes, probs, values


@dataclass
class LoaderStats:
    chunks_read: int = 0
    frames_read: int = 0
    frames_sampled: int = 0
    batches: int = 0


class ChunkPool:
    """``shuffling_chunk_pool``: a sliding window over the newest chunks."""

    def __init__(self, data_paths: Sequence[str], pool_size: int):
        self.data_paths = list(data_paths)
        self.pool_size = pool_size
        self.chunks: List[str] = []
        self._seen = set()
        self.rescan()

    def rescan(self) -> int:
        found = find_chunks(self.data_paths)
        added = 0
        for path in found:
            if path not in self._seen:
                self._seen.add(path)
                self.chunks.append(path)
                added += 1
        if len(self.chunks) > self.pool_size:
            drop = len(self.chunks) - self.pool_size
            for path in self.chunks[:drop]:
                self._seen.discard(path)
            self.chunks = self.chunks[drop:]
        return added

    def __len__(self) -> int:
        return len(self.chunks)

    def sample(self, rng: random.Random) -> Optional[str]:
        if not self.chunks:
            return None
        return rng.choice(self.chunks)


class TrainingDataLoader:
    """Background loader producing shuffled batches of training tensors."""

    def __init__(self, data_paths: Sequence[str], batch_size: int,
                 chunk_pool_size: int = 4000,
                 position_sampling_rate: float = 0.10,
                 shuffle_buffer_size: int = 65536,
                 workers: int = 2, seed: int = 0,
                 rescan_interval_s: float = 30.0,
                 queue_size: int = 8):
        self.pool = ChunkPool(data_paths, chunk_pool_size)
        self.batch_size = batch_size
        self.position_sampling_rate = position_sampling_rate
        self.shuffle_buffer_size = max(shuffle_buffer_size, 4 * batch_size)
        self.workers = max(1, workers)
        self.seed = seed
        self.rescan_interval_s = rescan_interval_s
        self.stats = LoaderStats()
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_rescan = 0.0

    def start(self) -> None:
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, args=(self.seed + i,),
                                 daemon=True, name=f"dataloader-{i}")
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        # Drain so the workers can exit.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def maybe_rescan(self, force: bool = False) -> int:
        now = time.time()
        if not force and now - self._last_rescan < self.rescan_interval_s:
            return 0
        with self._lock:
            self._last_rescan = now
            return self.pool.rescan()

    def next_batch(self, timeout: Optional[float] = None):
        return self._queue.get(timeout=timeout)

    # -- worker ------------------------------------------------------------
    def _worker(self, seed: int) -> None:
        """One loader thread: chunk -> sampled frames -> shuffle -> batch."""
        rng = random.Random(seed)
        buffer: List[np.ndarray] = []
        batch: List[np.ndarray] = []
        chunks_read = 0
        rate = self.position_sampling_rate

        while not self._stop.is_set():
            self.maybe_rescan()
            with self._lock:
                path = self.pool.sample(rng)
                pool_size = len(self.pool)
            if path is None:
                time.sleep(1.0)
                continue
            try:
                frames = read_chunk(path)
            except (OSError, EOFError, ValueError):
                # A chunk that is still being written; try another one.
                time.sleep(0.05)
                continue
            chunks_read += 1
            self.stats.chunks_read += 1
            self.stats.frames_read += len(frames)

            # Emit only once the shuffle buffer is warm, but do not stall
            # forever on a small pool: after a couple of passes over every
            # available chunk, start emitting with whatever we have.
            warm = (len(buffer) >= self.shuffle_buffer_size
                    or (chunks_read > 2 * pool_size + 4
                        and len(buffer) >= 4 * self.batch_size))

            for frame in frames:
                if rate < 1.0 and rng.random() >= rate:
                    continue
                # Copy the record out of the chunk: a numpy view would keep the
                # whole decompressed game alive for as long as it sits in the
                # shuffle buffer.
                frame = frame.copy()
                self.stats.frames_sampled += 1
                if not warm and len(buffer) < self.shuffle_buffer_size:
                    buffer.append(frame)
                    continue
                if buffer:
                    j = rng.randrange(len(buffer))
                    frame, buffer[j] = buffer[j], frame
                batch.append(frame)
                if len(batch) >= self.batch_size:
                    self._push(frames_to_tensors(batch))
                    batch = []
                    if self._stop.is_set():
                        return

    def _push(self, tensors) -> None:
        self.stats.batches += 1
        while not self._stop.is_set():
            try:
                self._queue.put(tensors, timeout=0.5)
                return
            except queue.Full:
                continue


def iterate_batches(loader: TrainingDataLoader) -> Iterator[Tuple[np.ndarray, ...]]:
    while True:
        yield loader.next_batch()
