"""Several self-play games sharing one network batch.

This is Lc0's self-play arrangement. ``selfplay/tournament.cc`` defaults to

    options->Add<IntOption>(kParallelGamesId, 1, 256) = 8;   // parallelism
    defaults->Set<...>(SharedBackendParams::kBackendId, "multiplexing");

i.e. eight games in flight, and a *multiplexing* backend that merges the
evaluation requests coming from all of them into a single GPU batch::

    game 1 --.
    game 2 --+
    game 3 --+--> one batch --> network --> results split back per game
    ...    --'

Why it matters here: the network is latency-bound at small batch sizes. On an
RTX 3060 Ti a batch of 32 takes 8.8 ms and a batch of 128 takes 18.3 ms, so
four times the work costs twice the time. One game alone can only ever offer
``MinibatchSize`` (32) leaves at a time.

**The mathematics of each game is untouched.** Every game keeps its own tree,
its own history, its own Dirichlet noise and its own RNG; a search's
gather/apply sequence is identical to what it would be running alone. The only
thing that is shared is the network evaluation -- which is a pure function of
the position.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..net.backend import Backend, EvalRequest, EvalResult
from ..net.config import SelfPlayConfig
from ..search.search import Search
from .game import SelfPlayGame

log = logging.getLogger("mylc0.selfplay")

# Runner states.
_NEED_GAME = 0
_NEED_ROOT = 1
_SEARCHING = 2
_STATE_NAMES = {_NEED_GAME: "NEED_GAME", _NEED_ROOT: "NEED_ROOT",
                _SEARCHING: "SEARCHING"}
_PENDING_NAMES = {0: "none", 1: "root", 2: "gathered"}


@dataclass
class _Pending:
    """What a runner is waiting for in the current batch."""

    kind: int = 0          # 0 = nothing, 1 = root expansion, 2 = gathered leaves
    gathered: object = None
    count: int = 0


class GameRunner:
    """One self-play game, advanced in steps instead of in a blocking loop."""

    def __init__(self, backend: Backend, config: SelfPlayConfig,
                 rng: np.random.Generator, index: int = 0):
        self.draining = False
        self.backend = backend
        self.config = config
        self.rng = rng
        self.index = index
        self.game: Optional[SelfPlayGame] = None
        self.search: Optional[Search] = None
        self.state = _NEED_GAME
        self.pending = _Pending()
        self.finished: List[SelfPlayGame] = []
        self.plies_in_progress = 0

    # -- driving -----------------------------------------------------------
    def collect(self) -> List[EvalRequest]:
        """Advance until this game needs the network, and return its requests.

        Returns an empty list when the step needed no evaluation at all (a game
        ended, a search finished, a root was already terminal); the driver just
        calls again on the next turn.
        """
        # Guard for the bug this method used to have: a reserved gather that
        # never reached apply() leaks its in-flight visits and freezes the
        # search forever.
        assert self.pending.kind == 0, (
            f"runner {self.index}: gather from the previous step was never "
            f"applied (kind={self.pending.kind}, count={self.pending.count})")

        if self.state == _NEED_GAME:
            if self.draining:
                return []
            self.game = SelfPlayGame(self.backend, self.config, rng=self.rng)
            self.state = _NEED_ROOT
            return []

        if self.state == _NEED_ROOT:
            self.search = self.game.start_move()
            if self.search is None:
                self._end_game()
                return []
            request = self.search.prepare_root()
            if request is None:
                # Terminal root: nothing to evaluate, the move step will end
                # the game.
                self.state = _SEARCHING
                return []
            self.pending = _Pending(kind=1, count=1)
            self.state = _SEARCHING
            return [request]

        # _SEARCHING
        search = self.search
        if self._search_complete(search):
            self._finish_move()
            return []
        budget = max(1, self.config.visits - search.root.n)
        limit = min(self.config.search.minibatch_size, budget)
        gathered = search.gather(limit)
        requests = search.requests_of(gathered)
        self.pending = _Pending(kind=2, gathered=gathered, count=len(requests))
        return requests

    def apply(self, results: Sequence[EvalResult]) -> None:
        pending = self.pending
        if pending.kind == 1:
            self.search.apply_root(results[0])
        elif pending.kind == 2:
            if not self.search.apply(pending.gathered, results):
                # Nothing could be expanded (every descent collided); the same
                # condition ends the single-game loop in Search.run.
                self._finish_move()
        self.pending = _Pending()

    # -- internals ---------------------------------------------------------
    def _search_complete(self, search: Search) -> bool:
        """The stop conditions of ``Search.run`` for a ``go nodes`` limit."""
        root = search.root
        if root.is_terminal:
            return True
        if root.n >= self.config.visits:
            return True
        if root.is_expanded and len(root.moves) == 1 and root.n > 1:
            return True
        return False

    def _finish_move(self) -> None:
        alive = self.game.finish_move(self.search)
        self.search = None
        self.plies_in_progress = self.game.stats.plies
        self.state = _NEED_ROOT if alive else _NEED_GAME
        if not alive:
            self._end_game()

    def _end_game(self) -> None:
        game = self.game
        if game is not None:
            game.finalize()
            self.finished.append(game)
        self.game = None
        self.search = None
        self.plies_in_progress = 0
        self.state = _NEED_GAME

    @property
    def is_busy(self) -> bool:
        return self.game is not None

    def describe(self) -> str:
        """One line of state, for the watchdog dump."""
        parts = [f"game[{self.index}] {_STATE_NAMES[self.state]:<9s}"]
        if self.game is None:
            parts.append("no game")
        else:
            parts.append(f"ply={self.game.stats.plies:3d}")
            parts.append(f"frames={len(self.game.data):3d}")
        if self.search is None:
            parts.append("search=none")
        else:
            root = self.search.root
            parts.append(f"root.n={root.n:4d}/{self.config.visits}")
            parts.append(f"root.in_flight={root.n_in_flight:3d}")
            parts.append(f"reserved={self.search.reserved:3d}")
            parts.append("terminal" if root.is_terminal else
                         ("expanded" if root.is_expanded else "unexpanded"))
        parts.append(f"pending={_PENDING_NAMES.get(self.pending.kind, '?')}"
                     f"({self.pending.count})")
        return "  ".join(parts)


@dataclass
class BatchedStats:
    games: int = 0
    positions: int = 0
    plies: int = 0
    batches: int = 0
    batch_positions: int = 0
    requests_per_batch_min: int = 10 ** 9
    requests_per_batch_max: int = 0
    last_batch_size: int = 0
    last_batch_at: float = 0.0


class BatchedSelfPlay:
    """Drives ``parallel_games`` runners against one shared backend."""

    def __init__(self, backend: Backend, config: SelfPlayConfig,
                 num_games: int, seed: int = 0):
        self.backend = backend
        self.config = config
        # Each game gets its own generator, so a game's moves depend only on
        # its own seed and not on how the games happened to interleave.
        seeds = np.random.SeedSequence(seed).spawn(num_games)
        self.runners = [GameRunner(backend, config,
                                   np.random.default_rng(s), index=i)
                        for i, s in enumerate(seeds)]
        self.stats = BatchedStats()
        # When draining, no new game is started; the ones in flight are played
        # out, because a game without a result cannot be written as training
        # data.
        self.draining = False

    def step(self) -> int:
        """One collect/evaluate/apply cycle across all games.

        Every runner that reserved leaves must be applied afterwards, *even if
        it produced no evaluation requests*. A gather whose leaves are all
        terminal positions is exactly that case: there is nothing for the
        network to do, but the in-flight visits on those paths still have to be
        released, and their (known) values still have to be backed up.
        """
        requests: List[EvalRequest] = []
        owners: List[GameRunner] = []
        for runner in self.runners:
            reqs = runner.collect()
            if runner.pending.kind:
                owners.append(runner)
                if reqs:
                    requests.extend(reqs)
        if not owners:
            return 0
        # A partial batch is evaluated immediately: nothing ever waits for the
        # batch to fill up.
        results = self.backend.evaluate(requests) if requests else []
        offset = 0
        for runner in owners:
            count = runner.pending.count
            runner.apply(results[offset:offset + count])
            offset += count
        if requests:
            self.stats.batches += 1
            self.stats.batch_positions += len(requests)
            self.stats.requests_per_batch_min = min(
                self.stats.requests_per_batch_min, len(requests))
            self.stats.requests_per_batch_max = max(
                self.stats.requests_per_batch_max, len(requests))
            self.stats.last_batch_size = len(requests)
            self.stats.last_batch_at = time.monotonic()
        return len(requests)

    def start_draining(self) -> None:
        """Stop starting new games; let the ones in flight finish."""
        self.draining = True
        for runner in self.runners:
            runner.draining = True

    def drain_finished(self) -> List[SelfPlayGame]:
        out = []
        for runner in self.runners:
            if runner.finished:
                out.extend(runner.finished)
                runner.finished = []
        return out

    def describe(self) -> List[str]:
        """Full state of every game in flight, for the watchdog dump."""
        return [r.describe() for r in self.runners]

    def plies_in_flight(self) -> int:
        return sum(r.plies_in_progress for r in self.runners)

    def nodes_in_flight(self) -> int:
        return sum(r.game.stats.nodes for r in self.runners
                   if r.game is not None)

    def active_games(self) -> int:
        return sum(1 for r in self.runners if r.is_busy)

    @property
    def avg_batch(self) -> float:
        return self.stats.batch_positions / max(1, self.stats.batches)

    def run(self, on_game: Callable[[SelfPlayGame], None],
            should_stop: Callable[[], bool],
            on_tick: Optional[Callable[[], None]] = None,
            hard_stop: Optional[Callable[[], bool]] = None) -> None:
        """Play until ``should_stop``, then let the games in flight finish.

        ``hard_stop`` (a deadline, say) abandons the games still running --
        those are simply dropped, since a game without a result has no value
        target and must not be written.
        """
        while True:
            if not self.draining and should_stop():
                self.start_draining()
            if self.draining and self.active_games() == 0:
                break
            if hard_stop is not None and hard_stop():
                break
            self.step()
            for game in self.drain_finished():
                on_game(game)
            if on_tick is not None:
                on_tick()
