"""V6 training data -- byte-compatible with Lc0.

The record is ``V6TrainingData`` from ``lc0/src/trainingdata/trainingdata_v6.h``
(8356 bytes, packed, little endian) and the fields are filled exactly like
``V6TrainingDataArray::Add`` / ``::Write`` do. A chunk is the gzipped
concatenation of the frames of one game, which is what
``TrainingDataWriter::WriteChunk`` produces and what lczero-training's
``chunk_source_loader`` reads.

Because the format is the upstream one, the data written here can be fed to
lczero-training as-is (and, conversely, this project could train on real Lc0
data -- which of course would defeat the point of learning from zero).

Two details worth remembering when reading the code below:

* The 104 board planes are stored bit-reversed within each byte
  (``ReverseBitsInBytes``). ``np.packbits(..., bitorder="big")`` produces
  exactly those bytes, and ``np.unpackbits(..., bitorder="big")`` inverts it,
  so square ``i`` of a plane is bit ``i`` of the original mask.
* ``plies_left`` is not known when a frame is created: ``Write`` fills it in
  backwards from the end of the game, starting at ``best_m`` of the last
  position, as upstream does.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

V6_VERSION = 6
POLICY_SIZE = 1858

V6_DTYPE = np.dtype([
    ("version", "<u4"),
    ("input_format", "<u4"),
    ("probabilities", "<f4", (POLICY_SIZE,)),
    ("planes", "<u8", (104,)),
    ("castling_us_ooo", "u1"),
    ("castling_us_oo", "u1"),
    ("castling_them_ooo", "u1"),
    ("castling_them_oo", "u1"),
    ("side_to_move_or_enpassant", "u1"),
    ("rule50_count", "u1"),
    ("invariance_info", "u1"),
    ("dummy", "u1"),
    ("root_q", "<f4"),
    ("best_q", "<f4"),
    ("root_d", "<f4"),
    ("best_d", "<f4"),
    ("root_m", "<f4"),
    ("best_m", "<f4"),
    ("plies_left", "<f4"),
    ("result_q", "<f4"),
    ("result_d", "<f4"),
    ("played_q", "<f4"),
    ("played_d", "<f4"),
    ("played_m", "<f4"),
    ("orig_q", "<f4"),
    ("orig_d", "<f4"),
    ("orig_m", "<f4"),
    ("visits", "<u4"),
    ("played_idx", "<u2"),
    ("best_idx", "<u2"),
    ("policy_kld", "<f4"),
    ("q_st", "<f4"),
])
assert V6_DTYPE.itemsize == 8356, V6_DTYPE.itemsize

# Invariance-info bits, see trainingdata_v6.h.
INV_TRANSFORM_MASK = 0b0000_0111
INV_BEST_IS_PROVEN = 1 << 3
INV_MAX_GAME_LENGTH = 1 << 4
INV_ADJUDICATED = 1 << 5
INV_DELETED = 1 << 6
INV_BLACK_TO_MOVE = 1 << 7

WHITE_WON, BLACK_WON, DRAW, UNDECIDED = 1, -1, 0, 2


@dataclass
class Eval:
    """lc0's ``classic::Eval``."""

    wl: float
    d: float
    ml: float


def drift_correct(q: float, d: float):
    """``trainingdata.cc: DriftCorrect``."""
    if q > 1.0:
        q = 1.0
    if q < -1.0:
        q = -1.0
    if d > 1.0:
        d = 1.0
    if d < 0.0:
        d = 0.0
    w = (1.0 - d + q) / 2.0
    l = w - q
    if w < 0.0 or l < 0.0:
        d += 2.0 * min(w, l)
        if d < 0.0:
            d = 0.0
    return q, d


def pack_planes(planes: np.ndarray) -> np.ndarray:
    """(112, 8, 8) float planes -> the 104 uint64 masks stored in a frame."""
    bits = planes.reshape(112, 64)[:104].astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="big")   # (104, 8) bytes
    return np.ascontiguousarray(packed).view("<u8").reshape(104)


def unpack_planes(frame) -> np.ndarray:
    """Rebuild the full (112, 8, 8) input tensor from a V6 frame.

    This is the counterpart of lczero-training's ``tensor_generator.cc``
    ``ProcessPlanes`` plus ``chunkparser.py``'s ``middle_planes`` handling, for
    the input formats this project supports.
    """
    out = np.zeros((112, 64), dtype=np.float32)
    board_bytes = np.ascontiguousarray(frame["planes"]).view(np.uint8).reshape(104, 8)
    out[:104] = np.unpackbits(board_bytes, axis=1, bitorder="big").astype(np.float32)

    fmt = int(frame["input_format"])
    if fmt == 1:  # INPUT_CLASSICAL_112_PLANE
        out[104, :] = 1.0 if frame["castling_us_ooo"] else 0.0
        out[105, :] = 1.0 if frame["castling_us_oo"] else 0.0
        out[106, :] = 1.0 if frame["castling_them_ooo"] else 0.0
        out[107, :] = 1.0 if frame["castling_them_oo"] else 0.0
        out[108, :] = 1.0 if frame["side_to_move_or_enpassant"] else 0.0
        out[109, :] = frame["rule50_count"] / 99.0
    else:
        # Castling rights as rook positions: ours on rank 1, theirs on rank 8.
        out[104, 0:8] = _expand_byte(int(frame["castling_us_ooo"]))
        out[104, 56:64] = _expand_byte(int(frame["castling_them_ooo"]))
        out[105, 0:8] = _expand_byte(int(frame["castling_us_oo"]))
        out[105, 56:64] = _expand_byte(int(frame["castling_them_oo"]))
        if fmt >= 3:  # canonical formats: plane 108 holds the en-passant file
            out[108, 56:64] = _expand_byte(int(frame["side_to_move_or_enpassant"]))
        else:
            out[108, :] = 1.0 if frame["side_to_move_or_enpassant"] else 0.0
        out[109, :] = (frame["rule50_count"] / 100.0 if fmt >= 4
                       else frame["rule50_count"] / 99.0)
    if fmt in (132, 133) and (int(frame["invariance_info"]) & INV_BLACK_TO_MOVE):
        out[110, :] = 1.0
    out[111, :] = 1.0
    return out.reshape(112, 8, 8)


def _expand_byte(value: int) -> np.ndarray:
    """Byte -> 8 floats, LSB first (file a .. file h)."""
    return np.array([(value >> i) & 1 for i in range(8)], dtype=np.float32)


class TrainingDataArray:
    """``V6TrainingDataArray`` -- the frames of one game."""

    def __init__(self, input_format: int):
        self.input_format = input_format
        # Each entry is a 1-element structured array (a view-friendly frame).
        self.frames: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.frames)

    def add(self, *, planes: np.ndarray, transform: int, snapshot,
            visit_counts: Sequence[int], total_children_visits: int,
            legal_policy_indices: Sequence[int], nneval_p: Optional[np.ndarray],
            policy_softmax_temp: float, root_q: float, root_d: float,
            root_m: float, root_visits: int, best_eval: Eval, played_eval: Eval,
            orig_eval: Optional[Eval], best_idx: int, played_idx: int,
            best_is_proven: bool) -> None:
        record = np.zeros(1, dtype=V6_DTYPE)
        frame = record[0]
        frame["version"] = V6_VERSION
        frame["input_format"] = self.input_format
        frame["planes"] = pack_planes(planes)

        probs = np.full(POLICY_SIZE, -1.0, dtype=np.float32)
        kld_sum = 0.0
        total_p = 0.0
        for i, policy_idx in enumerate(legal_policy_indices):
            fracv = (visit_counts[i] / total_children_visits
                     if total_children_visits > 0 else 1.0)
            probs[policy_idx] = fracv
            if nneval_p is not None:
                # Undo the policy softmax temperature applied during search.
                p = float(nneval_p[i]) ** policy_softmax_temp
                if fracv > 0 and p > 0:
                    kld_sum += fracv * np.log(fracv / p)
                total_p += p
        if nneval_p is not None:
            # Add a small epsilon for backward compatibility with the earlier
            # value of 0, exactly as trainingdata.cc does.
            adjusted = kld_sum + float(np.log(total_p)) if total_p > 0 else 0.0
            kld_sum = max(adjusted, 0.0) + float(np.finfo(np.float32).tiny)
        frame["probabilities"] = probs
        frame["policy_kld"] = kld_sum

        # Castling: for non-FRC nets lc0 just sends 1; the 960 formats send the
        # rook file as a bit mask.
        if self.input_format >= 2:
            us_ooo = 1 << snapshot.our_queenside_rook if snapshot.our_queenside_rook >= 0 else 0
            us_oo = 1 << snapshot.our_kingside_rook if snapshot.our_kingside_rook >= 0 else 0
            them_ooo = 1 << snapshot.their_queenside_rook if snapshot.their_queenside_rook >= 0 else 0
            them_oo = 1 << snapshot.their_kingside_rook if snapshot.their_kingside_rook >= 0 else 0
        else:
            us_ooo = 1 if snapshot.our_queenside_rook >= 0 else 0
            us_oo = 1 if snapshot.our_kingside_rook >= 0 else 0
            them_ooo = 1 if snapshot.their_queenside_rook >= 0 else 0
            them_oo = 1 if snapshot.their_kingside_rook >= 0 else 0
        frame["castling_us_ooo"] = us_ooo
        frame["castling_us_oo"] = us_oo
        frame["castling_them_ooo"] = them_ooo
        frame["castling_them_oo"] = them_oo

        invariance = 0
        if self.input_format >= 3:  # canonical
            ep_byte = (snapshot.ep >> 56) & 0xFF
            if transform & 1:  # FlipTransform mirrors the file order
                ep_byte = int(f"{ep_byte:08b}"[::-1], 2)
            frame["side_to_move_or_enpassant"] = ep_byte
            invariance = transform | (INV_BLACK_TO_MOVE if snapshot.flipped else 0)
        else:
            frame["side_to_move_or_enpassant"] = 1 if snapshot.flipped else 0
        if best_is_proven:
            invariance |= INV_BEST_IS_PROVEN
        frame["invariance_info"] = invariance
        frame["dummy"] = 0
        frame["rule50_count"] = snapshot.rule50

        # Overwritten by write() once the game result is known.
        frame["result_q"] = 0.0
        frame["result_d"] = 1.0

        best_q, best_d = drift_correct(best_eval.wl, best_eval.d)
        rq, rd = drift_correct(root_q, root_d)
        pq, pd = drift_correct(played_eval.wl, played_eval.d)
        frame["root_q"], frame["root_d"] = rq, rd
        frame["best_q"], frame["best_d"] = best_q, best_d
        frame["played_q"], frame["played_d"] = pq, pd
        frame["root_m"] = root_m
        frame["best_m"] = best_eval.ml
        frame["played_m"] = played_eval.ml
        if orig_eval is None:
            frame["orig_q"] = np.nan
            frame["orig_d"] = np.nan
            frame["orig_m"] = np.nan
        else:
            frame["orig_q"] = orig_eval.wl
            frame["orig_d"] = orig_eval.d
            frame["orig_m"] = orig_eval.ml
        frame["visits"] = root_visits
        frame["best_idx"] = best_idx
        frame["played_idx"] = played_idx
        frame["q_st"] = 0.0
        frame["plies_left"] = 0.0
        self.frames.append(record)

    def write(self, path: str, result: int, adjudicated: bool) -> int:
        """``V6TrainingDataArray::Write``; returns the number of frames."""
        if not self.frames:
            return 0
        frames = np.concatenate(self.frames)
        m_estimate = float(frames[-1]["best_m"]) + len(frames) - 1
        for frame in frames:
            black_to_move = bool(frame["side_to_move_or_enpassant"])
            if self.input_format >= 3:
                black_to_move = bool(int(frame["invariance_info"]) & INV_BLACK_TO_MOVE)
            if result == WHITE_WON:
                frame["result_q"] = -1.0 if black_to_move else 1.0
                frame["result_d"] = 0.0
            elif result == BLACK_WON:
                frame["result_q"] = 1.0 if black_to_move else -1.0
                frame["result_d"] = 0.0
            else:
                frame["result_q"] = 0.0
                frame["result_d"] = 1.0
            if adjudicated:
                frame["invariance_info"] = int(frame["invariance_info"]) | INV_ADJUDICATED
                if result == UNDECIDED:
                    frame["invariance_info"] = (int(frame["invariance_info"])
                                                | INV_MAX_GAME_LENGTH)
            frame["plies_left"] = m_estimate
            m_estimate -= 1.0

        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            f.write(frames.tobytes())
        os.replace(tmp, path)
        return len(frames)


def read_chunk(path: str) -> np.ndarray:
    """Read a gzipped chunk into an array of V6 frames."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if len(raw) % V6_DTYPE.itemsize != 0:
        raise ValueError(f"{path}: size {len(raw)} is not a multiple of "
                         f"{V6_DTYPE.itemsize}")
    return np.frombuffer(raw, dtype=V6_DTYPE)
