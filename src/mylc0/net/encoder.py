"""Lc0 input-plane encoder (112 planes), transcribed from lc0/src/neural/encoder.cc.

Layout of the 112 planes (all from the side-to-move's point of view):

===========  ===============================================================
0 .. 103     8 history positions x 13 planes:
             our P, N, B, R, Q, K, their P, N, B, R, Q, K, "repetitions >= 1"
104          castling: rooks with a-side (queenside) rights  [format >= 2]
             or "we can castle queenside" as a full plane    [format 1]
105          castling: rooks with h-side (kingside) rights   [format >= 2]
             or "we can castle kingside" as a full plane     [format 1]
106          zero [format >= 2] / "they can castle queenside" [format 1]
107          zero [format >= 2] / "they can castle kingside"  [format 1]
108          en-passant file marker on rank 8 [canonical formats]
             or "black to move" as a full plane [formats 1, 2]
109          rule-50 counter, scaled
110          zeros (used to be the move counter; ones for the black side of an
             armageddon game in the armageddon formats)
111          all ones -- lets the network find the edges of the board
===========  ===============================================================

Two things about the rule-50 plane are worth spelling out, because upstream is
not self-consistent:

* the hectoplies formats (4, 5, 132, 133) divide the ply count by 100 in both
  lc0's encoder and lczero-training's data loader -- consistent;
* format 1/2/3 fill the plane with the *raw* ply count in lc0's encoder while
  lczero-training divides by 99 (``chunkparser.py``: ``rule50_divisor``, and
  the same 99 in the new C++ ``tensor_generator.cc``).

Since self-play and training share this module there is no mismatch here; the
default format is 5 (hectoplies), where upstream agrees with itself anyway.
See ARCHITECTURE.md, "Input representation".
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..chessrules.policy_map import (
    FLIP_TRANSFORM,
    MIRROR_TRANSFORM,
    NO_TRANSFORM,
    TRANSPOSE_TRANSFORM,
    _reverse_bits_in_bytes,
    _reverse_bytes_in_bytes,
    _transpose_bits_in_bytes,
    transform_bitboard,
)
from ..chessrules.position import PositionHistory, Snapshot

# pblczero::NetworkFormat::InputFormat
INPUT_CLASSICAL_112_PLANE = 1
INPUT_112_WITH_CASTLING_PLANE = 2
INPUT_112_WITH_CANONICALIZATION = 3
INPUT_112_WITH_CANONICALIZATION_HECTOPLIES = 4
INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON = 132
INPUT_112_WITH_CANONICALIZATION_V2 = 5
INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON = 133

INPUT_FORMAT_NAMES = {
    1: "INPUT_CLASSICAL_112_PLANE",
    2: "INPUT_112_WITH_CASTLING_PLANE",
    3: "INPUT_112_WITH_CANONICALIZATION",
    4: "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES",
    5: "INPUT_112_WITH_CANONICALIZATION_V2",
    132: "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON",
    133: "INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON",
}

MOVE_HISTORY = 8
PLANES_PER_BOARD = 13
AUX_PLANE_BASE = PLANES_PER_BOARD * MOVE_HISTORY  # 104
TOTAL_PLANES = AUX_PLANE_BASE + 8  # 112

# FillEmptyHistory
FILL_NO = 0
FILL_FEN_ONLY = 1
FILL_ALWAYS = 2

_MASK64 = 0xFFFFFFFFFFFFFFFF
_STARTPOS_KEYS = None


def is_canonical_format(fmt: int) -> bool:
    return fmt >= INPUT_112_WITH_CANONICALIZATION


def is_canonical_armageddon_format(fmt: int) -> bool:
    return fmt in (INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON,
                   INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON)


def is_hectoplies_format(fmt: int) -> bool:
    return fmt >= INPUT_112_WITH_CANONICALIZATION_HECTOPLIES


def is_960_castling_format(fmt: int) -> bool:
    return fmt >= INPUT_112_WITH_CASTLING_PLANE


def rule50_divisor(fmt: int) -> float:
    """Scaling of the rule-50 plane, as applied by lczero-training."""
    return 100.0 if is_hectoplies_format(fmt) else 99.0


# --------------------------------------------------------------------------
# Canonicalization (encoder.cc: ChooseTransform / CompareTransposing)
# --------------------------------------------------------------------------
def _compare_transposing(bb: int, initial_transform: int) -> int:
    value = bb
    if initial_transform & FLIP_TRANSFORM:
        value = _reverse_bits_in_bytes(value)
    if initial_transform & MIRROR_TRANSFORM:
        value = _reverse_bytes_in_bytes(value)
    alternative = _transpose_bits_in_bytes(value)
    if value < alternative:
        return -1
    if value > alternative:
        return 1
    return 0


def choose_transform(snap: Snapshot) -> int:
    """encoder.cc ``ChooseTransform``: put our king in the bottom-right octant."""
    # Any castling right makes every transform invalid: even under FRC rules
    # a-side and h-side castling are not symmetrical.
    if not snap.no_legal_castle():
        return NO_TRANSFORM

    our_king = snap.kings & snap.ours
    transform = NO_TRANSFORM
    if our_king & 0x0F0F0F0F0F0F0F0F:
        transform |= FLIP_TRANSFORM
        our_king = _reverse_bits_in_bytes(our_king)
    # With pawns on the board only the horizontal flip is valid.
    if snap.pawns:
        return transform
    if our_king & 0xFFFFFFFF00000000:
        transform |= MIRROR_TRANSFORM
        our_king = _reverse_bytes_in_bytes(our_king)
    # Our king is now always in the bottom right quadrant. Transpose for a king
    # in the top right triangle, or -- on the diagonal -- for whichever side
    # gives the smaller integer value, tested piece type by piece type.
    if our_king & 0xE0C08000:
        transform |= TRANSPOSE_TRANSFORM
    elif our_king & 0x10204080:
        for bb in (snap.ours | snap.theirs, snap.ours, snap.kings, snap.queens,
                   snap.rooks, snap.knights, snap.bishops):
            outcome = _compare_transposing(bb, transform)
            if outcome == -1:
                return transform
            if outcome == 1:
                return transform | TRANSPOSE_TRANSFORM
        # Everything is symmetrical, so transposing is a no-op.
    return transform


def transform_for_position(history: PositionHistory, input_format: int) -> int:
    if not is_canonical_format(input_format):
        return NO_TRANSFORM
    return choose_transform(history.last())


# --------------------------------------------------------------------------
# Board mirroring while walking back through history
# --------------------------------------------------------------------------
class _Board:
    """The subset of ChessBoard the encoder touches, mirrorable in place."""

    __slots__ = ("ours", "theirs", "pawns", "knights", "bishops", "rooks",
                 "queens", "kings", "ep", "castling_key")

    def __init__(self, snap: Snapshot):
        self.reset(snap)

    def reset(self, snap: Snapshot) -> None:
        self.ours = snap.ours
        self.theirs = snap.theirs
        self.pawns = snap.pawns
        self.knights = snap.knights
        self.bishops = snap.bishops
        self.rooks = snap.rooks
        self.queens = snap.queens
        self.kings = snap.kings
        self.ep = snap.ep
        self.castling_key = snap.castling_key()

    def mirror(self, snap: Snapshot) -> None:
        """``ChessBoard::Mirror``: flip ranks, swap the two sides."""
        f = _reverse_bytes_in_bytes
        self.ours, self.theirs = f(snap.theirs), f(snap.ours)
        self.pawns = f(snap.pawns)
        self.knights = f(snap.knights)
        self.bishops = f(snap.bishops)
        self.rooks = f(snap.rooks)
        self.queens = f(snap.queens)
        self.kings = f(snap.kings)
        self.ep = f(snap.ep)
        # Castling rights swap sides together with the board.
        self.castling_key = ((snap.their_kingside_rook + 1)
                             | ((snap.their_queenside_rook + 1) << 4)
                             | ((snap.our_kingside_rook + 1) << 8)
                             | ((snap.our_queenside_rook + 1) << 12))


def _unpack(masks: List[int]) -> np.ndarray:
    """Expand 64-bit masks into a (n, 64) float32 array, bit i -> square i."""
    if not masks:
        return np.zeros((0, 64), dtype=np.float32)
    buf = b"".join((m & _MASK64).to_bytes(8, "little") for m in masks)
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    return bits.reshape(len(masks), 64).astype(np.float32)


def encode_position(history: PositionHistory,
                    input_format: int = INPUT_112_WITH_CANONICALIZATION_V2,
                    history_planes: int = MOVE_HISTORY,
                    fill_empty_history: int = FILL_NO,
                    out: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """``EncodePositionForNN``. Returns (planes[112, 8, 8] float32, transform)."""
    if out is None:
        planes = np.zeros((TOTAL_PLANES, 64), dtype=np.float32)
    else:
        planes = out.reshape(TOTAL_PLANES, 64)
        planes.fill(0.0)

    snaps = history.snapshots
    last = snaps[-1]
    transform = NO_TRANSFORM
    canonical = is_canonical_format(input_format)
    # The canonical formats stop walking back early: it avoids applying the
    # transform across incompatible transitions, and history before those
    # points is not relevant to the result anyway.
    stop_early = canonical
    if canonical:
        transform = choose_transform(last)

    aux_masks = [0] * 5  # planes 104..108 are bitboards; 109..111 are scalars

    if input_format == INPUT_CLASSICAL_112_PLANE:
        if last.our_queenside_rook >= 0:
            aux_masks[0] = _MASK64
        if last.our_kingside_rook >= 0:
            aux_masks[1] = _MASK64
        if last.their_queenside_rook >= 0:
            aux_masks[2] = _MASK64
        if last.their_kingside_rook >= 0:
            aux_masks[3] = _MASK64
    elif input_format in (INPUT_112_WITH_CASTLING_PLANE,
                          INPUT_112_WITH_CANONICALIZATION,
                          INPUT_112_WITH_CANONICALIZATION_HECTOPLIES,
                          INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON,
                          INPUT_112_WITH_CANONICALIZATION_V2,
                          INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON):
        # Plane 104: rooks (ours on rank 1, theirs on rank 8) with a-side
        # rights; plane 105: the same for h-side rights.
        aux_masks[0] = last.castling_plane_queenside()
        aux_masks[1] = last.castling_plane_kingside()
    else:
        raise ValueError(f"Unsupported input plane encoding {input_format}")

    if canonical:
        aux_masks[4] = last.ep
    elif last.flipped:
        aux_masks[4] = _MASK64

    if is_hectoplies_format(input_format):
        planes[AUX_PLANE_BASE + 5, :] = last.rule50 / 100.0
    else:
        planes[AUX_PLANE_BASE + 5, :] = last.rule50 / 99.0
    # Plane 110 used to be the move counter; it is zero except for the black
    # side of an armageddon game in the armageddon formats.
    if is_canonical_armageddon_format(input_format) and last.flipped:
        planes[AUX_PLANE_BASE + 6, :] = 1.0
    # Plane 111 is all ones so the network can find the edges of the board.
    planes[AUX_PLANE_BASE + 7, :] = 1.0

    castlings = last.castling_key() if stop_early else 0
    skip_non_repeats = input_format in (
        INPUT_112_WITH_CANONICALIZATION_V2,
        INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON)

    board = _Board(last)
    flip = False
    history_idx = len(snaps) - 1
    history_masks: List[int] = []
    history_slots: List[int] = []

    i = 0
    while i < min(history_planes, MOVE_HISTORY):
        snap = snaps[history_idx] if history_idx >= 0 else snaps[0]
        if flip:
            board.mirror(snap)
        else:
            board.reset(snap)

        # Castling changes cannot be repeated, so we can stop early.
        if stop_early and board.castling_key != castlings:
            break
        # En passant cannot be repeated either, but the current position must
        # always be sent.
        if (stop_early and history_idx != len(snaps) - 1 and board.ep):
            break
        # If en passant is possible we know the previous move, so one extra
        # position can be reconstructed even without real history.
        if fill_empty_history == FILL_NO and (
                history_idx < -1 or (history_idx == -1 and not board.ep)):
            break
        if (history_idx < 0 and fill_empty_history == FILL_FEN_ONLY
                and _is_startpos(snaps[0])):
            break

        repetitions = snap.repetitions
        # Canonical v2 only writes an entry if it is a repeat, unless it is the
        # most recent position.
        if skip_non_repeats and repetitions == 0 and i > 0:
            if history_idx > 0:
                flip = not flip
            # rule50 == 0 means the previous ply was the start of the game, a
            # capture or a pawn push: no further repeats are worth considering.
            if snap.rule50 == 0:
                break
            history_idx -= 1
            continue

        base = i * PLANES_PER_BOARD
        masks = [
            board.ours & board.pawns,
            board.ours & board.knights,
            board.ours & board.bishops,
            board.ours & board.rooks,
            board.ours & board.queens,
            board.ours & board.kings,
            board.theirs & board.pawns,
            board.theirs & board.knights,
            board.theirs & board.bishops,
            board.theirs & board.rooks,
            board.theirs & board.queens,
            board.theirs & board.kings,
            _MASK64 if repetitions >= 1 else 0,
        ]
        # If the en-passant flag is set on a reconstructed position, undo the
        # last pawn move: take the pawn off the square it moved to and put it
        # back on the square it came from.
        if history_idx < 0 and board.ep:
            idx = (board.ep & -board.ep).bit_length() - 1
            if idx < 8:  # "us" board
                masks[0] += (0x0000000000000100 - 0x0000000001000000) << idx
            else:
                masks[6] += ((0x0001000000000000 - 0x0000000100000000)
                             << (idx - 56))
        for k, m in enumerate(masks):
            history_masks.append(m & _MASK64)
            history_slots.append(base + k)

        if history_idx > 0:
            flip = not flip
        if stop_early and snap.rule50 == 0:
            break
        history_idx -= 1
        i += 1

    for k, mask in enumerate(aux_masks):
        if mask:
            history_masks.append(mask)
            history_slots.append(AUX_PLANE_BASE + k)

    if transform != NO_TRANSFORM:
        # Lc0 transforms the raw masks of every board-shaped plane (0..108) and
        # leaves the all-zero / all-one ones alone.
        history_masks = [m if (m == 0 or m == _MASK64)
                         else transform_bitboard(m, transform)
                         for m in history_masks]

    if history_masks:
        planes[history_slots, :] = _unpack(history_masks)

    return planes.reshape(TOTAL_PLANES, 8, 8), transform


def _is_startpos(snap: Snapshot) -> bool:
    """``position.GetBoard() == ChessBoard::kStartposBoard``."""
    global _STARTPOS_KEYS
    if _STARTPOS_KEYS is None:
        import chess
        from ..chessrules.position import make_snapshot
        s = make_snapshot(chess.Board(), 0)
        _STARTPOS_KEYS = (s.ours, s.theirs, s.pawns, s.knights, s.bishops,
                          s.rooks, s.queens, s.kings)
    return (snap.ours, snap.theirs, snap.pawns, snap.knights, snap.bishops,
            snap.rooks, snap.queens, snap.kings) == _STARTPOS_KEYS
