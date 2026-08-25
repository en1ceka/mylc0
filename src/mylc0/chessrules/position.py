"""Position and position history in Lc0's representation.

Lc0 does not store a board "as seen from white". Its ``ChessBoard`` is always
stored from the point of view of the side to move: when black is to move the
board is mirrored vertically (rank 1 <-> rank 8) and the colours are swapped,
so the side to move always owns ``ours()`` and always moves "up" the board.
Every plane of the network input, every move index and every value/WDL target
in this project follows that convention.

The rules themselves come from python-chess (move legality, FEN parsing,
Chess960 castling). Nothing here encodes chess *knowledge*: no piece values, no
piece-square tables, no opening theory -- only the rules, plus the bookkeeping
Lc0 needs (repetitions, the rule-50 counter, the en-passant marker and the
castling rook files).

Notable Lc0 conventions reproduced here:

* The en-passant marker is a phantom bit on **rank 8** at the file of the pawn
  that just double-pushed (``board.cc``: the flag is set on rank 1 before the
  board is mirrored for the opponent, and ``BitBoard en_passant()`` returns
  ``pawns_ - kPawnMask``). It is only set when an enemy pawn actually attacks
  the en-passant square, matching ``kPawnAttacks[ep_sq].intersects(...)``.
* Castling rights are kept as *rook files*, which is what input formats >= 2
  put on the board (Chess960-compatible).
* ``compute_game_result`` follows ``PositionHistory::ComputeGameResult``:
  no legal moves, no mating material, rule50 >= 100, or repetitions >= 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import chess

WHITE_WON = 1
BLACK_WON = -1
DRAW = 0
UNDECIDED = 2

_RANK_1 = 0x00000000000000FF
_RANK_8 = 0xFF00000000000000


def flip_vertical(bb: int) -> int:
    """Mirror a bitboard across the horizontal axis (rank 1 <-> rank 8)."""
    return int.from_bytes((bb & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"), "big")


@dataclass(frozen=True)
class Snapshot:
    """One position, already rotated into the side-to-move's frame."""

    ours: int
    theirs: int
    pawns: int
    knights: int
    bishops: int
    rooks: int
    queens: int
    kings: int
    # En-passant phantom bit (rank 8), Lc0 style. 0 when there is none.
    ep: int
    # Castling rook files from our/their point of view, -1 when unavailable.
    our_kingside_rook: int
    our_queenside_rook: int
    their_kingside_rook: int
    their_queenside_rook: int
    rule50: int
    repetitions: int
    # Plies back to the previous occurrence of this position (0 if none).
    # Lc0's Position::GetPliesSincePrevRepetition, used by two-fold draws.
    plies_since_prev_repetition: int
    # True when black is to move, i.e. the board above has been mirrored.
    flipped: bool
    ply: int
    key: int

    def no_legal_castle(self) -> bool:
        return (self.our_kingside_rook < 0 and self.our_queenside_rook < 0
                and self.their_kingside_rook < 0
                and self.their_queenside_rook < 0)

    def castling_key(self) -> int:
        """Packed castling rights, used to detect changes across history."""
        return ((self.our_kingside_rook + 1)
                | ((self.our_queenside_rook + 1) << 4)
                | ((self.their_kingside_rook + 1) << 8)
                | ((self.their_queenside_rook + 1) << 12))

    def castling_plane_queenside(self) -> int:
        """Bitboard of rooks with a-side castling rights (input format >= 2)."""
        bb = 0
        if self.our_queenside_rook >= 0:
            bb |= 1 << self.our_queenside_rook
        if self.their_queenside_rook >= 0:
            bb |= 1 << (56 + self.their_queenside_rook)
        return bb

    def castling_plane_kingside(self) -> int:
        bb = 0
        if self.our_kingside_rook >= 0:
            bb |= 1 << self.our_kingside_rook
        if self.their_kingside_rook >= 0:
            bb |= 1 << (56 + self.their_kingside_rook)
        return bb


def _rook_files(board: chess.Board, color: bool):
    """(kingside_file, queenside_file) of the rooks that may still castle."""
    king_sq = board.king(color)
    if king_sq is None:
        return -1, -1
    king_file = chess.square_file(king_sq)
    backrank = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
    rights = board.clean_castling_rights() & backrank
    kingside, queenside = -1, -1
    for sq in chess.scan_forward(rights):
        f = chess.square_file(sq)
        if f > king_file:
            kingside = f
        elif f < king_file:
            queenside = f
    return kingside, queenside


def make_snapshot(board: chess.Board, repetitions: int = 0,
                  plies_since_prev_repetition: int = 0,
                  key: Optional[int] = None) -> Snapshot:
    """Convert a python-chess board into Lc0's side-to-move-relative frame."""
    us = board.turn
    them = not us
    flipped = us == chess.BLACK

    ours = board.occupied_co[us]
    theirs = board.occupied_co[them]
    pawns = board.pawns
    knights = board.knights
    bishops = board.bishops
    rooks = board.rooks
    queens = board.queens
    kings = board.kings

    ep = 0
    if board.ep_square is not None and board.has_pseudo_legal_en_passant():
        # Phantom pawn on the 8th rank of the mirrored board, i.e. on the rank
        # closest to the opponent, at the file of the pawn that double-pushed.
        ep_file = chess.square_file(board.ep_square)
        ep = 1 << (56 + ep_file)

    if flipped:
        # ``ours``/``theirs`` were already picked by side to move, so this is
        # only the geometric half of ``ChessBoard::Mirror``.
        ours, theirs = flip_vertical(ours), flip_vertical(theirs)
        pawns = flip_vertical(pawns)
        knights = flip_vertical(knights)
        bishops = flip_vertical(bishops)
        rooks = flip_vertical(rooks)
        queens = flip_vertical(queens)
        kings = flip_vertical(kings)

    our_ks, our_qs = _rook_files(board, us)
    their_ks, their_qs = _rook_files(board, them)

    return Snapshot(
        ours=ours,
        theirs=theirs,
        pawns=pawns,
        knights=knights,
        bishops=bishops,
        rooks=rooks,
        queens=queens,
        kings=kings,
        ep=ep,
        our_kingside_rook=our_ks,
        our_queenside_rook=our_qs,
        their_kingside_rook=their_ks,
        their_queenside_rook=their_qs,
        rule50=board.halfmove_clock,
        repetitions=repetitions,
        plies_since_prev_repetition=plies_since_prev_repetition,
        flipped=flipped,
        ply=2 * (board.fullmove_number - 1) + (1 if flipped else 0),
        key=board._transposition_key() if key is None else key,
    )


def move_to_our_uci(move: chess.Move, flipped: bool) -> str:
    """UCI string of ``move`` in Lc0's "our" (side-to-move) orientation."""
    from_sq = move.from_square
    to_sq = move.to_square
    if flipped:
        from_sq ^= 56
        to_sq ^= 56
    s = chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq]
    if move.promotion:
        s += chess.piece_symbol(move.promotion)
    return s


def our_uci_to_move(uci: str, flipped: bool) -> chess.Move:
    """Inverse of :func:`move_to_our_uci`."""
    mv = chess.Move.from_uci(uci)
    if flipped:
        mv = chess.Move(mv.from_square ^ 56, mv.to_square ^ 56, mv.promotion)
    return mv


class PositionHistory:
    """A game (or search line) together with the positions that led to it.

    Mirrors ``lczero::PositionHistory``: ``append``/``pop`` walk the line while
    the snapshot list keeps everything the input encoder needs about earlier
    positions.
    """

    def __init__(self, board: Optional[chess.Board] = None):
        self.board = board.copy(stack=False) if board is not None else chess.Board()
        self.snapshots: List[Snapshot] = []
        self._key_stack: List[int] = []
        self._move_stack: List[chess.Move] = []
        self._reset_snapshots()

    # -- construction ------------------------------------------------------
    def _reset_snapshots(self) -> None:
        self.snapshots = [make_snapshot(self.board, 0, 0)]
        self._key_stack = [self.board._transposition_key()]

    @classmethod
    def from_fen(cls, fen: str) -> "PositionHistory":
        return cls(chess.Board(fen))

    def clone(self) -> "PositionHistory":
        other = PositionHistory.__new__(PositionHistory)
        other.board = self.board.copy(stack=False)
        other.snapshots = list(self.snapshots)
        other._key_stack = list(self._key_stack)
        other._move_stack = list(self._move_stack)
        return other

    # -- basic accessors ---------------------------------------------------
    def last(self) -> Snapshot:
        return self.snapshots[-1]

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def is_black_to_move(self) -> bool:
        return self.board.turn == chess.BLACK

    def legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def fen(self) -> str:
        return self.board.fen()

    # -- moves -------------------------------------------------------------
    def _count_repetitions(self, key):
        """(occurrences, plies back to the most recent one).

        Only the current rule-50 window can contain repetitions, so the scan
        stops at the last irreversible move -- the same reasoning Lc0 uses.
        """
        count = 0
        cycle = 0
        limit = self.board.halfmove_clock
        distance = 0
        # self._key_stack[-1] is the position before the move just played.
        for i in range(len(self._key_stack) - 1, -1, -1):
            if limit <= 0:
                break
            distance += 1
            if self._key_stack[i] == key:
                count += 1
                if cycle == 0:
                    cycle = distance
            limit -= 1
        return count, cycle

    def append(self, move: chess.Move) -> None:
        self.board.push(move)
        self._move_stack.append(move)
        key = self.board._transposition_key()
        reps, cycle = self._count_repetitions(key)
        self._key_stack.append(key)
        self.snapshots.append(make_snapshot(self.board, reps, cycle, key))

    def pop(self) -> chess.Move:
        move = self._move_stack.pop()
        self.board.pop()
        self._key_stack.pop()
        self.snapshots.pop()
        return move

    def moves(self) -> List[chess.Move]:
        return list(self._move_stack)

    # -- game end ----------------------------------------------------------
    def compute_game_result(self) -> int:
        """``PositionHistory::ComputeGameResult`` (result from white's view)."""
        if not any(self.board.generate_legal_moves()):
            if self.board.is_check():
                return BLACK_WON if self.board.turn == chess.WHITE else WHITE_WON
            return DRAW
        if not self._has_mating_material():
            return DRAW
        if self.board.halfmove_clock >= 100:
            return DRAW
        if self.snapshots[-1].repetitions >= 2:
            return DRAW
        return UNDECIDED

    def _has_mating_material(self) -> bool:
        """``ChessBoard::HasMatingMaterial``, transcribed.

        Lc0's ``rooks_``/``bishops_`` bitboards both contain the queens, so the
        first test covers rooks, queens and pawns. Note that Lc0 keeps KNN v K
        as "has mating material" (any knight returns true) and only calls
        bishop endings dead when every bishop sits on one square colour.
        """
        b = self.board
        if b.rooks or b.queens or b.pawns:
            return True
        if chess.popcount(b.occupied) < 4:
            # K v K, K+B v K, K+N v K.
            return False
        if b.knights:
            return True
        # Only kings and bishops remain.
        light = bool(b.bishops & chess.BB_LIGHT_SQUARES)
        dark = bool(b.bishops & chess.BB_DARK_SQUARES)
        return light and dark
