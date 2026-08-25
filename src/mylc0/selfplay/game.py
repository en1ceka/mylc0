"""One self-play game -- lc0's ``selfplay/game.cc`` (``SelfPlayGame::Play``).

Both sides are the same network and the same search parameters, as in a real
Lc0 training run. Per move:

    search (``visits`` nodes, Dirichlet noise at the root)
      -> best move / best eval for the training record and resign logic
      -> move actually played, sampled from the visit counts with temperature
      -> one V6 training frame
      -> play the move

When the game ends, the frames are written as one gzipped chunk with the game
result filled in (``V6TrainingDataArray::Write``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..chessrules.policy_map import policy_indices
from ..chessrules.position import (BLACK_WON, DRAW, UNDECIDED, WHITE_WON,
                                   PositionHistory)
from ..net.backend import Backend
from ..net.config import SelfPlayConfig
from ..search.node import NodeTree
from ..search.search import GoParams, Search
from .trainingdata import Eval, TrainingDataArray


@dataclass
class GameStats:
    result: int = UNDECIDED
    adjudicated: bool = False
    plies: int = 0
    frames: int = 0
    nodes: int = 0
    nn_evals: int = 0
    cache_hits: int = 0
    seconds: float = 0.0
    moves: List[str] = field(default_factory=list)
    chunk_path: Optional[str] = None


class SelfPlayGame:
    def __init__(self, backend: Backend, config: SelfPlayConfig,
                 rng: Optional[np.random.Generator] = None,
                 start_fen: Optional[str] = None):
        self.backend = backend
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.history = (PositionHistory.from_fen(start_fen) if start_fen
                        else PositionHistory())
        self.start_fen = self.history.fen()
        self.tree = NodeTree(self.history)
        self.data = TrainingDataArray(backend.input_format)
        self.stats = GameStats()
        self._t0 = None
        self._cache_hits0 = 0
        self._evals0 = 0

    # -- the game as two steps, so a driver can interleave several games ---
    def start_move(self) -> Optional[Search]:
        """Begin the search for the next move, or end the game.

        Returns the fresh :class:`Search`, or ``None`` when the game is over
        (the result is then already recorded in ``stats``).
        """
        if self._t0 is None:
            self._t0 = time.perf_counter()
            self._cache_hits0 = self.backend.cache.hits
            self._evals0 = self.backend.evaluations

        result = self.history.compute_game_result()
        if result != UNDECIDED:
            self.stats.result = result
            return None
        if self.history.last().ply >= self.config.max_game_ply:
            self.stats.adjudicated = True
            self.stats.result = UNDECIDED
            return None

        if not self.config.reuse_tree:
            self.tree.trim()
        return Search(self.tree, self.backend, self.config.search,
                      rng=self.rng, sticky_endgames=False)

    def finish_move(self, search: Search, enable_resign: bool = True) -> bool:
        """Record the training frame and play the move. False = game over."""
        if not self.tree.root.is_expanded:
            # The root turned out to be terminal (mate or stalemate).
            self.stats.result = self.history.compute_game_result()
            return False
        self.stats.nodes += search.stats.nodes

        best_eval, best_is_terminal, best_idx = search.get_best_eval()
        if best_idx < 0:
            self.stats.result = self.history.compute_game_result()
            return False

        if enable_resign and self._check_resign(best_eval):
            self.stats.adjudicated = True
            return False

        played_idx = self._pick_played_move(search)
        played_eval = search.get_edge_eval(played_idx)
        self._record_frame(search, best_idx, played_idx, best_eval,
                           played_eval, best_is_terminal)

        move = self.tree.root.moves[played_idx]
        self.stats.moves.append(move.uci())
        self.tree.make_move(move, reuse=self.config.reuse_tree)
        self.stats.plies += 1
        return True

    def finalize(self) -> GameStats:
        """Close the books once the game has ended."""
        self.stats.seconds = time.perf_counter() - (self._t0 or time.perf_counter())
        self.stats.nn_evals = self.backend.evaluations - self._evals0
        self.stats.cache_hits = self.backend.cache.hits - self._cache_hits0
        self.stats.frames = len(self.data)
        return self.stats

    def play(self, enable_resign: bool = True, on_move=None) -> GameStats:
        """Play the whole game in this thread (one game at a time)."""
        while True:
            search = self.start_move()
            if search is None:
                break
            search.run(GoParams(nodes=self.config.visits))
            if not self.finish_move(search, enable_resign):
                break
            if on_move is not None:
                on_move(self.stats.plies)
        return self.finalize()

    # -- move choice -------------------------------------------------------
    def _pick_played_move(self, search: Search) -> int:
        """Temperature sampling plus lc0's ``MinimumAllowedVisits`` retry."""
        game_ply = self.history.last().ply
        idx = search.pick_move(game_ply)
        minimum = self.config.minimum_allowed_visits
        if minimum <= 0:
            return idx
        for _ in range(8):
            counts = [c.n if c is not None else 0 for c in search.root.children]
            max_n = max(counts) if counts else 0
            if counts[idx] == max_n or counts[idx] >= minimum:
                return idx
            idx = search.pick_move(game_ply)
        return search.get_best_child_no_temperature(search.root, 0)

    def _check_resign(self, best_eval) -> bool:
        cfg = self.config
        if cfg.resign_percentage <= 0.0:
            return False
        move_number = len(self.history.snapshots) // 2 + 1
        if move_number < cfg.resign_earliest_move:
            return False
        wl, d, _ = best_eval
        blacks_move = self.history.is_black_to_move
        w = (wl + 1.0 - d) / 2.0
        l = w - wl
        resignpct = cfg.resign_percentage / 100.0
        if cfg.resign_wdl_style:
            threshold = 1.0 - resignpct
            if w > threshold:
                self.stats.result = BLACK_WON if blacks_move else WHITE_WON
                return True
            if l > threshold:
                self.stats.result = WHITE_WON if blacks_move else BLACK_WON
                return True
            if d > threshold:
                self.stats.result = DRAW
                return True
            return False
        eval01 = (wl + 1) / 2
        if eval01 < resignpct:
            self.stats.result = WHITE_WON if blacks_move else BLACK_WON
            return True
        return False

    # -- training data -----------------------------------------------------
    def _record_frame(self, search: Search, best_idx: int, played_idx: int,
                      best_eval, played_eval, best_is_terminal: bool) -> None:
        root = search.root
        transform = search.transform_at_root
        planes = search.root_planes
        if planes is None:
            planes, transform = self.backend.encode(self.history)
        snapshot = self.history.last()
        flipped = snapshot.flipped

        visit_counts = [c.n if c is not None else 0 for c in root.children]
        total_children_visits = root.children_visits()
        indices = policy_indices(root.moves, flipped, transform)

        nneval = search.get_root_nn_eval()
        orig = (Eval(nneval.q, nneval.d, nneval.m) if nneval is not None else None)
        nneval_p = nneval.p if nneval is not None else None

        best_is_proven = search.best_is_proven(best_eval[0], best_is_terminal)

        self.data.add(
            planes=planes,
            transform=transform,
            snapshot=snapshot,
            visit_counts=visit_counts,
            total_children_visits=total_children_visits,
            legal_policy_indices=indices,
            nneval_p=nneval_p,
            policy_softmax_temp=self.config.search.policy_softmax_temp,
            root_q=root.wl,
            root_d=root.d,
            root_m=root.m,
            root_visits=root.n,
            best_eval=Eval(*best_eval),
            played_eval=Eval(*played_eval),
            orig_eval=orig,
            best_idx=int(indices[best_idx]),
            played_idx=int(indices[played_idx]),
            best_is_proven=best_is_proven,
        )

    def write(self, path: str) -> int:
        n = self.data.write(path, self.stats.result, self.stats.adjudicated)
        self.stats.chunk_path = path
        return n
