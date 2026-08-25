"""PUCT search, transcribed from lc0's ``search/classic/search.cc``.

The pieces reproduced here, with their upstream names:

* ``ComputeCpuct``      -- ``cpuct + cpuct_factor * log((N + base) / base)``
* ``GetFpu``            -- "reduction" (``Q_parent - fpu * sqrt(visited_P)``)
  and "absolute" strategies
* ``PickNodesToExtend`` -- the PUCT formula
  ``score = P * cpuct * sqrt(max(N_children, 1)) / (1 + N_started) + U``
  where ``U`` is the child's ``Q`` seen from the parent, or the FPU value for
  unvisited children, plus the moves-left utility
* ``MEvaluator``        -- how the moves-left head influences selection
* ``ExtendNode``        -- terminal detection (mate/stalemate, no mating
  material, rule 50, threefold, optional twofold)
* ``DoBackupUpdateSingleNode`` / ``MaybeSetBounds`` -- backup with running
  averages, sign flip per ply, and proven-terminal ("sticky endgame")
  propagation
* ``ApplyDirichletNoise``, ``EnsureBestMoveKnown``,
  ``GetBestChildrenNoTemperature``, ``GetBestRootChildWithTemperature``
* ``LegacyTimeManager`` from ``stoppers/legacy.cc`` (lc0's default)

Search here is single-threaded but batched: leaves are collected into a
minibatch using ``n_in_flight`` exactly like lc0 does (in-flight visits enter
the PUCT denominator, they do *not* change Q), then evaluated in one network
call and backed up. That is the same mathematics as lc0 running with one search
thread and ``MinibatchSize`` leaves per iteration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..chessrules.position import PositionHistory
from ..chessrules.policy_map import policy_indices
from ..net.backend import Backend, EvalRequest, EvalResult
from ..net.config import SearchConfig
from .node import (BOUND_DRAW, BOUND_LOSS, BOUND_WIN, TERMINAL_TWOFOLD,
                   Node, NodeTree)


@dataclass
class GoParams:
    """The ``go`` command's arguments."""

    wtime: Optional[int] = None
    btime: Optional[int] = None
    winc: Optional[int] = None
    binc: Optional[int] = None
    movestogo: Optional[int] = None
    movetime: Optional[int] = None
    nodes: Optional[int] = None
    depth: Optional[int] = None
    infinite: bool = False
    ponder: bool = False


def compute_estimated_moves_to_go(ply: int, midpoint: float = 51.5,
                                  steepness: float = 7.0) -> float:
    """``stoppers/legacy.cc: ComputeEstimatedMovesToGo``."""
    move = ply / 2.0
    return (midpoint * math.pow(1 + 2 * math.pow(move / midpoint, steepness),
                                1 / steepness) - move)


class LegacyTimeManager:
    """lc0's default time manager (``stoppers/legacy.cc``)."""

    def __init__(self, move_overhead_ms: int = 200, slowmover: float = 1.0,
                 midpoint: float = 51.5, steepness: float = 7.0,
                 immediate_use: float = 1.0, first_move_bonus: float = 1.8,
                 book_ply_bonus: float = 0.25):
        self.move_overhead = move_overhead_ms
        self.slowmover = slowmover
        self.midpoint = midpoint
        self.steepness = steepness
        self.spend_saved_time = immediate_use
        self.first_move_bonus = first_move_bonus
        self.book_ply_bonus = book_ply_bonus
        self.first_move_of_game = True
        self.time_spared_ms = 0.0

    def reset(self) -> None:
        self.first_move_of_game = True
        self.time_spared_ms = 0.0

    def deadline_ms(self, params: GoParams, game_ply: int,
                    is_black: bool) -> Optional[float]:
        if params.infinite or params.ponder:
            return None
        remaining = params.btime if is_black else params.wtime
        if remaining is None:
            return None
        inc = params.binc if is_black else params.winc
        increment = max(0, inc or 0)

        movestogo = compute_estimated_moves_to_go(game_ply, self.midpoint,
                                                  self.steepness)
        if params.movestogo and 0 < params.movestogo < movestogo:
            movestogo = params.movestogo

        total_moves_time = max(
            0.0, remaining + increment * (movestogo - 1) - self.move_overhead)

        time_to_squander = 0.0
        if self.time_spared_ms > 0:
            total_moves_time = max(0.0, total_moves_time - self.time_spared_ms)
            time_to_squander = self.time_spared_ms * self.spend_saved_time
            self.time_spared_ms -= time_to_squander

        this_move_time = total_moves_time / movestogo

        if self.first_move_of_game:
            this_move_time *= (1.0 + self.first_move_bonus
                               + self.book_ply_bonus * min(12, game_ply))
            self.first_move_of_game = False

        if self.slowmover < 1.0 or this_move_time * self.slowmover > 200:
            self.time_spared_ms -= this_move_time * (self.slowmover - 1)
            this_move_time *= self.slowmover

        this_move_time += time_to_squander
        return min(this_move_time, remaining - self.move_overhead)

    def on_search_done(self, budget_ms: Optional[float],
                       elapsed_ms: float) -> None:
        if budget_ms is not None:
            self.time_spared_ms += budget_ms - elapsed_ms


@dataclass
class SearchTiming:
    """Optional wall-clock split of the search loop, for profiling."""

    select_s: float = 0.0   # tree descent: PUCT + make/unmake moves
    encode_s: float = 0.0   # legal moves -> 112 planes + policy indices
    eval_s: float = 0.0     # everything inside Backend.evaluate
    backup_s: float = 0.0   # value propagation up the tree
    terminal_s: float = 0.0  # game-end detection at the leaves
    total_s: float = 0.0


@dataclass
class SearchStats:
    nodes: int = 0        # leaves evaluated by the network in this search
    playouts: int = 0
    batches: int = 0
    max_depth: int = 0
    cum_depth: int = 0
    collisions: int = 0


@dataclass
class _PendingLeaf:
    path: List[Node]
    node: Node
    legal_moves: list
    request: Optional[EvalRequest]
    eval: Optional[EvalResult] = None
    # Only filled for the root, whose planes and transform the training data
    # records alongside the visit counts.
    planes: Optional[np.ndarray] = None
    transform: int = 0


@dataclass
class _Gathered:
    """Leaves reserved by one ``gather`` call, waiting for their evaluation."""

    pending: List[_PendingLeaf]
    collisions: List[_PendingLeaf]
    started_at: float = 0.0


class Search:
    """One search from one root position (lc0's ``classic::Search``)."""

    def __init__(self, tree: NodeTree, backend: Backend, params: SearchConfig,
                 rng: Optional[np.random.Generator] = None,
                 sticky_endgames: bool = False,
                 info_callback: Optional[Callable[[dict], None]] = None):
        self.tree = tree
        self.root = tree.root
        self.history: PositionHistory = tree.history
        self.backend = backend
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sticky_endgames = sticky_endgames
        self.info_callback = info_callback
        self.stats = SearchStats()
        self._root_noise_applied = False
        self.transform_at_root = 0
        # The raw network evaluation of the root, kept for the "orig_*" fields
        # of the training data (lc0 reads it back out of the NN cache).
        self.root_nn_eval: Optional[EvalResult] = None
        self.root_planes: Optional[np.ndarray] = None
        self.timing: Optional[SearchTiming] = None
        self._root_legal_moves = None
        # Leaves reserved by gather() and not yet released by apply(). Must be
        # zero between iterations; anything else means leaked in-flight visits.
        self.reserved = 0

    # -- helpers -----------------------------------------------------------
    def _draw_score(self, depth: int) -> float:
        """Draw score in the own frame of a node at ``depth`` plies from root."""
        ds = self.params.draw_score
        return ds if (depth % 2 == 0) else -ds

    def _cpuct(self, n: int, is_root: bool) -> float:
        if is_root and self.params.root_has_own_cpuct_params:
            init = self.params.cpuct_at_root
            k = self.params.cpuct_factor_at_root
            base = self.params.cpuct_base_at_root
        else:
            init = self.params.cpuct
            k = self.params.cpuct_factor
            base = self.params.cpuct_base
        if k:
            return init + k * math.log((n + base) / base)
        return init

    def _fpu(self, node: Node, is_root: bool, draw_score: float,
             visited_pol: float) -> float:
        if is_root and self.params.fpu_strategy_at_root != "same":
            absolute = self.params.fpu_strategy_at_root == "absolute"
            value = self.params.fpu_value_at_root
        else:
            absolute = self.params.fpu_strategy == "absolute"
            value = self.params.fpu_value
        if absolute:
            return value
        return node.q(draw_score) - value * math.sqrt(visited_pol)

    # -- selection ---------------------------------------------------------
    def _pick_child(self, node: Node, depth: int) -> int:
        is_root = node is self.root
        draw_score = self._draw_score(depth)
        child_draw_score = self._draw_score(depth + 1)
        pol = node.policy
        children = node.children

        visited_pol = 0.0
        for i, child in enumerate(children):
            if child is not None and child.n > 0:
                visited_pol += float(pol[i])
        fpu = self._fpu(node, is_root, draw_score, visited_pol)

        cpuct = self._cpuct(node.n, is_root)
        puct_mult = cpuct * math.sqrt(max(node.children_visits(), 1))

        # MEvaluator: only active once the parent evaluation is decisive.
        m_enabled = (self.backend.movesleft_head is not None
                     and self.params.moves_left_max_effect > 0.0
                     and abs(node.q(0.0)) > self.params.moves_left_threshold)
        parent_m = node.m

        best_score = -math.inf
        best_idx = 0
        for i, child in enumerate(children):
            if child is not None and child.n > 0:
                util = -child.q(child_draw_score)
                if m_enabled:
                    util += self._m_utility(child.m, parent_m, util)
            else:
                util = fpu  # GetDefaultMUtility() is 0
            nstarted = 0 if child is None else child.n + child.n_in_flight
            score = float(pol[i]) * puct_mult / (1 + nstarted) + util
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _m_utility(self, child_m: float, parent_m: float, q: float) -> float:
        """``MEvaluator::GetMUtility``: prefer shorter wins, longer losses."""
        p = self.params
        m = p.moves_left_slope * (child_m - parent_m)
        m = max(-p.moves_left_max_effect, min(p.moves_left_max_effect, m))
        m *= 1.0 if -q > 0 else (-1.0 if -q < 0 else 0.0)
        thr = p.moves_left_threshold
        if 0.0 < thr < 1.0:
            q = max(0.0, abs(q) - thr) / (1.0 - thr)
        m *= (p.moves_left_constant_factor + p.moves_left_scaled_factor * abs(q)
              + p.moves_left_quadratic_factor * q * q)
        return m

    # -- node extension ----------------------------------------------------
    def _terminal_check(self, node: Node, depth: int, legal_moves) -> bool:
        """``SearchWorker::ExtendNode`` terminal part; True if terminal."""
        last = self.history.last()
        if not legal_moves:
            if self.history.board.is_check():
                node.make_terminal(BOUND_LOSS)   # we are mated
            else:
                node.make_terminal(BOUND_DRAW)   # stalemate
            return True
        # Draws by rule are only short-circuited away from the root: at the
        # root, thinking about them is the point.
        if node is not self.root:
            if not self.history._has_mating_material():
                node.make_terminal(BOUND_DRAW)
                return True
            if last.rule50 >= 100:
                node.make_terminal(BOUND_DRAW)
                return True
            if last.repetitions >= 2:
                node.make_terminal(BOUND_DRAW)
                return True
            if (last.repetitions == 1 and depth >= 4
                    and self.params.two_fold_draws
                    and depth >= last.plies_since_prev_repetition):
                node.make_terminal(BOUND_DRAW,
                                   float(last.plies_since_prev_repetition),
                                   TERMINAL_TWOFOLD)
                return True
        return False

    def _make_request(self, legal_moves=None) -> Tuple[EvalRequest, list, int,
                                                       np.ndarray]:
        """Encode the current position and map its legal moves to policy slots."""
        t0 = time.perf_counter() if self.timing is not None else 0.0
        if legal_moves is None:
            legal_moves = self.history.legal_moves()
        planes, transform = self.backend.encode(self.history)
        indices = policy_indices(legal_moves, self.history.is_black_to_move,
                                 transform)
        if self.timing is not None:
            self.timing.encode_s += time.perf_counter() - t0
        request = EvalRequest(planes=planes, policy_indices=indices,
                              cache_key=self.backend.cache_key(self.history))
        return request, legal_moves, transform, planes

    # -- one descent -------------------------------------------------------
    def _descend(self) -> Tuple[Optional[_PendingLeaf], str]:
        node = self.root
        node.n_in_flight += 1
        path = [node]
        depth = 0
        while True:
            if node.is_terminal:
                self._pop_to_root(depth)
                return _PendingLeaf(path, node, [], None), "terminal"
            if not node.is_expanded:
                if node.n_in_flight > 1:
                    # Already queued in this batch -- a collision.
                    self._pop_to_root(depth)
                    return _PendingLeaf(path, node, [], None), "collision"
                t_term = time.perf_counter() if self.timing is not None else 0.0
                legal_moves = self.history.legal_moves()
                terminal = self._terminal_check(node, depth, legal_moves)
                if self.timing is not None:
                    self.timing.terminal_s += time.perf_counter() - t_term
                if terminal:
                    self._pop_to_root(depth)
                    return _PendingLeaf(path, node, [], None), "terminal"
                request, legal_moves, transform, planes = self._make_request(
                    legal_moves)
                self._pop_to_root(depth)
                if depth > self.stats.max_depth:
                    self.stats.max_depth = depth
                self.stats.cum_depth += depth
                leaf = _PendingLeaf(path, node, legal_moves, request)
                if node is self.root:
                    leaf.planes = planes
                    leaf.transform = transform
                return leaf, "leaf"

            idx = self._pick_child(node, depth)
            child = node.children[idx]
            if child is None:
                child = Node(node, idx)
                node.children[idx] = child
            self.history.append(node.moves[idx])
            node = child
            node.n_in_flight += 1
            path.append(node)
            depth += 1

    def _pop_to_root(self, depth: int) -> None:
        for _ in range(depth):
            self.history.pop()

    def _unwind(self, path: List[Node]) -> None:
        for node in path:
            node.cancel_in_flight(1)

    # -- backup ------------------------------------------------------------
    def _backup(self, leaf: _PendingLeaf, v: float, d: float, m: float) -> None:
        path = leaf.path
        node = path[-1]
        update_parent_bounds = (self.sticky_endgames and node.is_terminal
                                and node.n == 0)
        n_to_fix = 0
        v_delta = d_delta = m_delta = 0.0

        for i in range(len(path) - 1, -1, -1):
            n = path[i]
            p = path[i - 1] if i > 0 else None
            if n.is_terminal:
                v, d, m = n.wl, n.d, n.m
            n.finalize_score_update(v, d, m, 1)
            if n_to_fix > 0 and not n.is_terminal:
                n.adjust_for_terminal(v_delta, d_delta, m_delta, n_to_fix)
            if p is None:
                break
            if p.is_terminal:
                n_to_fix = 0
            if update_parent_bounds and p is not self.root and not p.is_terminal:
                update_parent_bounds, n_to_fix, v_delta, d_delta, m_delta = \
                    self._maybe_set_bounds(p, m, n_to_fix, v_delta, d_delta,
                                           m_delta)
            else:
                update_parent_bounds = False
            v = -v
            v_delta = -v_delta
            m += 1
        self.stats.playouts += 1

    def _maybe_set_bounds(self, p: Node, m: float, n_to_fix: int,
                          v_delta: float, d_delta: float, m_delta: float):
        """``SearchWorker::MaybeSetBounds`` in this file's own-frame convention.

        A child's bounds ``(l, u)`` seen from the parent become ``(-u, -l)``;
        the parent's own bounds are the element-wise maximum over its moves.
        """
        lower = BOUND_LOSS
        upper = BOUND_LOSS
        losing_m = 0.0
        for child in p.children:
            if child is None:
                edge_lower, edge_upper = BOUND_LOSS, BOUND_WIN
                child_m = 0.0
            else:
                edge_lower, edge_upper = -child.upper, -child.lower
                child_m = child.m
            lower = max(edge_lower, lower)
            upper = max(edge_upper, upper)
            if edge_lower == BOUND_WIN:
                break  # a proven win is the best possible outcome
            if edge_upper == BOUND_LOSS:
                losing_m = max(losing_m, child_m)

        if lower == BOUND_LOSS and upper == BOUND_WIN:
            return False, n_to_fix, v_delta, d_delta, m_delta
        if lower == upper:
            n_to_fix = p.n
            cur_v, cur_d, cur_m = p.wl, p.d, p.m
            new_m = (max(losing_m, m) if upper == BOUND_LOSS else m) + 1.0
            p.make_terminal(upper, new_m)
            v_delta = -(p.wl - cur_v)
            d_delta = p.d - cur_d
            m_delta = p.m - cur_m
        else:
            p.set_bounds(lower, upper)
        return True, n_to_fix, v_delta, d_delta, m_delta

    # -- root --------------------------------------------------------------
    def _apply_dirichlet_noise(self) -> None:
        """``ApplyDirichletNoise``: P <- (1-eps) P + eps * Dir(alpha)."""
        eps = self.params.noise_epsilon
        if eps <= 0.0 or self.root.policy is None:
            return
        alpha = self.params.noise_alpha
        noise = self.rng.gamma(alpha, 1.0, size=len(self.root.policy))
        total = noise.sum()
        if total < 1e-30:
            return
        self.root.policy = (self.root.policy * (1 - eps)
                            + eps * (noise / total)).astype(np.float32)
        self._root_noise_applied = True

    def prepare_root(self) -> Optional[EvalRequest]:
        """Request needed to expand the root, or ``None`` if there is none.

        Split out of the search loop so a driver running several games can put
        the root expansions of all of them into the same network batch.
        """
        if self.root.is_expanded or self.root.is_terminal:
            return None
        legal_moves = self.history.legal_moves()
        if self._terminal_check(self.root, 0, legal_moves):
            return None
        request, legal_moves, transform, planes = self._make_request(legal_moves)
        self.transform_at_root = transform
        self.root_planes = planes
        self._root_legal_moves = legal_moves
        return request

    def apply_root(self, result: EvalResult) -> None:
        """Expand the root from its evaluation and add the Dirichlet noise."""
        self.root_nn_eval = result
        self.root.expand(self._root_legal_moves, result.p.astype(np.float32))
        self.root.n_in_flight = 1
        self.root.finalize_score_update(result.q, result.d, result.m, 1)
        self.stats.nodes += 1
        self._root_legal_moves = None
        if not self._root_noise_applied:
            self._apply_dirichlet_noise()

    def _ensure_root_expanded(self) -> bool:
        request = self.prepare_root()
        if request is None:
            return self.root.is_expanded or self.root.is_terminal
        self.apply_root(self.backend.evaluate([request])[0])
        return True

    # -- main loop ---------------------------------------------------------
    def run(self, limits: GoParams, deadline_ms: Optional[float] = None,
            stop_check: Optional[Callable[[], bool]] = None) -> None:
        start = time.perf_counter()
        if not self._ensure_root_expanded():
            return
        if self.root.is_terminal:
            return

        target_nodes = limits.nodes
        last_info = 0.0
        while True:
            if stop_check is not None and stop_check():
                break
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if deadline_ms is not None and elapsed_ms >= deadline_ms:
                break
            if limits.movetime is not None and elapsed_ms >= limits.movetime:
                break
            if target_nodes is not None and self.root.n >= target_nodes:
                break
            if (self.root.is_expanded and len(self.root.moves) == 1
                    and self.root.n > 1 and not limits.infinite):
                # Only one legal move: lc0's smart pruning would stop here too.
                break

            budget = None
            if target_nodes is not None:
                budget = max(1, target_nodes - self.root.n)
            batch_limit = self.params.minibatch_size
            if budget is not None:
                batch_limit = min(batch_limit, budget)
            if not self._do_iteration(batch_limit):
                break

            if self.info_callback is not None and elapsed_ms - last_info > 500:
                last_info = elapsed_ms
                self.info_callback(self.build_info(elapsed_ms))

    # -- one iteration, split so several searches can share one NN batch ----
    def gather(self, batch_limit: int) -> _Gathered:
        """Collect leaves to evaluate. Pure CPU: the network is not touched.

        This is the first half of an iteration. Between ``gather`` and
        ``apply`` the in-flight counters keep the collected leaves reserved, so
        another search can safely be gathered in the meantime -- which is what
        lets many games share one NN batch (lc0's multiplexing backend).
        """
        timing = self.timing
        t_iter = time.perf_counter() if timing is not None else 0.0
        pending: List[_PendingLeaf] = []
        collisions: List[_PendingLeaf] = []
        collision_events = 0
        collision_visits = 0
        while len(pending) < batch_limit:
            leaf, kind = self._descend()
            if kind == "collision":
                # The leaf is already queued in this batch. Like lc0's
                # collision nodes, the in-flight visits stay on the path so the
                # next descent is pushed somewhere else; they are cancelled
                # after the batch. Gathering stops once the collision budget
                # (MaxCollisionEvents / MaxCollisionVisits, both 1 during
                # self-play) is used up.
                collisions.append(leaf)
                collision_events += 1
                collision_visits += 1
                self.stats.collisions += 1
                if (collision_events >= self.params.max_collision_events
                        or collision_visits >= self.params.max_collision_visits):
                    break
                continue
            pending.append(leaf)

        if timing is not None:
            timing.select_s += time.perf_counter() - t_iter
        self.reserved += len(pending) + len(collisions)
        return _Gathered(pending=pending, collisions=collisions,
                         started_at=t_iter)

    def requests_of(self, gathered: _Gathered) -> List[EvalRequest]:
        return [leaf.request for leaf in gathered.pending
                if leaf.request is not None]

    def apply(self, gathered: _Gathered,
              results: Sequence[EvalResult]) -> bool:
        """Second half of an iteration: expand the leaves and back the values up.

        ``results`` must line up with :meth:`requests_of`.
        """
        timing = self.timing
        pending = gathered.pending
        collisions = gathered.collisions

        if not pending:
            for leaf in collisions:
                self._unwind(leaf.path)
            self.reserved -= len(collisions)
            return False

        t_apply = time.perf_counter() if timing is not None else 0.0
        index = 0
        for leaf in pending:
            if leaf.request is not None:
                r = results[index]
                index += 1
                leaf.node.expand(leaf.legal_moves, r.p.astype(np.float32))
                self.stats.nodes += 1
                self._backup(leaf, r.q, r.d, r.m)
            else:
                node = leaf.node
                self._backup(leaf, node.wl, node.d, node.m)
        for leaf in collisions:
            self._unwind(leaf.path)
        self.reserved -= len(pending) + len(collisions)
        self.stats.batches += 1
        if timing is not None:
            timing.backup_s += time.perf_counter() - t_apply
        return True

    def _do_iteration(self, batch_limit: int) -> bool:
        """One gather/evaluate/apply cycle for this search alone."""
        timing = self.timing
        gathered = self.gather(batch_limit)
        requests = self.requests_of(gathered)
        t0 = time.perf_counter() if timing is not None else 0.0
        results = self.backend.evaluate(requests) if requests else []
        if timing is not None:
            timing.eval_s += time.perf_counter() - t0
        return self.apply(gathered, results)

    # -- results -----------------------------------------------------------
    def get_best_children_no_temperature(self, parent: Node, count: int,
                                         depth: int) -> List[int]:
        """``GetBestChildrenNoTemperature`` (returns child indices).

        The comparator is lc0's, pairwise (hence ``cmp_to_key``): prefer better
        proven outcomes; among proven wins the shortest, among proven losses
        the longest, among terminal draws the shortest; otherwise most visits,
        then best eval, then largest prior.
        """
        if parent.n == 0 or not parent.is_expanded:
            return []
        child_draw_score = self._draw_score(depth + 1)

        K_TERMINAL_LOSS, K_NON_TERMINAL, K_TERMINAL_WIN = 0, 2, 4

        def edge_rank(child: Optional[Node]) -> int:
            if child is None or child.n == 0 or not child.is_terminal:
                return K_NON_TERMINAL
            wl = -child.wl  # value seen from the parent
            if wl == 0:
                return K_NON_TERMINAL
            return K_TERMINAL_LOSS if wl < 0 else K_TERMINAL_WIN

        def n_of(child):
            return child.n if child is not None else 0

        def m_of(child):
            return child.m if child is not None else 0.0

        def q_of(child):
            return -child.q(child_draw_score) if child is not None else 0.0

        def cmp(ia: int, ib: int) -> int:
            a, b = parent.children[ia], parent.children[ib]
            ra, rb = edge_rank(a), edge_rank(b)
            if ra != rb:
                return -1 if ra > rb else 1
            if (ra == K_NON_TERMINAL and n_of(a) and n_of(b)
                    and a.is_terminal and b.is_terminal):
                # Both are terminal draws: prefer the shorter one.
                return -1 if m_of(a) < m_of(b) else (1 if m_of(a) > m_of(b) else 0)
            if ra == K_NON_TERMINAL:
                if n_of(a) != n_of(b):
                    return -1 if n_of(a) > n_of(b) else 1
                if q_of(a) != q_of(b):
                    return -1 if q_of(a) > q_of(b) else 1
                pa, pb = float(parent.policy[ia]), float(parent.policy[ib])
                return -1 if pa > pb else (1 if pa < pb else 0)
            if ra > K_NON_TERMINAL:  # both winning: shortest win
                return -1 if m_of(a) < m_of(b) else (1 if m_of(a) > m_of(b) else 0)
            # Both losing: longest loss.
            return -1 if m_of(a) > m_of(b) else (1 if m_of(a) < m_of(b) else 0)

        import functools
        indices = list(range(len(parent.moves)))
        indices.sort(key=functools.cmp_to_key(cmp))
        return indices[:count]

    def get_best_child_no_temperature(self, parent: Node, depth: int) -> int:
        res = self.get_best_children_no_temperature(parent, 1, depth)
        return res[0] if res else -1

    def get_best_root_child_with_temperature(self, temperature: float) -> int:
        """``GetBestRootChildWithTemperature``."""
        root = self.root
        draw_score = self._draw_score(0)
        offset = self.params.temperature_visit_offset
        fpu = self._fpu(root, True, draw_score, root.visited_policy())

        max_n = 0.0
        max_eval = -1.0
        for i, child in enumerate(root.children):
            n = child.n if child is not None else 0
            if n + offset > max_n:
                max_n = n + offset
                max_eval = (-child.q(self._draw_score(1))
                            if (child is not None and n) else fpu)
        min_eval = max_eval - self.params.temperature_winpct_cutoff / 50.0

        candidates: List[int] = []
        weights: List[float] = []
        for i, child in enumerate(root.children):
            n = child.n if child is not None else 0
            q = (-child.q(self._draw_score(1))
                 if (child is not None and n) else fpu)
            if q < min_eval:
                continue
            if max_n <= 0.0:
                base = float(root.policy[i])
            else:
                base = (n + offset) / max_n
            weights.append(math.pow(max(0.0, base), 1.0 / temperature))
            candidates.append(i)
        total = sum(weights)
        if total <= 0.0:
            return self.get_best_child_no_temperature(self.root, 0)
        toss = self.rng.random() * total
        acc = 0.0
        for idx, w in zip(candidates, weights):
            acc += w
            if acc >= toss:
                return idx
        return candidates[-1]

    def pick_move(self, game_ply: int) -> int:
        """``EnsureBestMoveKnown`` -- temperature schedule for self-play."""
        p = self.params
        temperature = p.temperature
        moves = game_ply // 2
        if p.temperature_cutoff_move and (moves + 1) >= p.temperature_cutoff_move:
            temperature = p.temperature_endgame
        elif temperature and p.temp_decay_moves:
            if moves >= p.temp_decay_delay_moves + p.temp_decay_moves:
                temperature = 0.0
            elif moves >= p.temp_decay_delay_moves:
                temperature *= float(p.temp_decay_delay_moves
                                     + p.temp_decay_moves - moves) / p.temp_decay_moves
            temperature = max(temperature, p.temperature_endgame)
        if temperature:
            return self.get_best_root_child_with_temperature(temperature)
        return self.get_best_child_no_temperature(self.root, 0)

    def get_best_eval(self):
        """``Search::GetBestEval`` -- (eval from the root player's view, is_terminal, index)."""
        root = self.root
        parent = (root.wl, root.d, root.m)
        if not root.is_expanded:
            return parent, True, -1
        idx = self.get_best_child_no_temperature(root, 0)
        if idx < 0:
            return parent, True, -1
        child = root.children[idx]
        if child is None or child.n == 0:
            return parent, False, idx
        return (-child.wl, child.d, child.m + 1.0), child.is_terminal, idx

    def get_edge_eval(self, idx: int):
        """Eval of one root move, seen from the root player (``played_eval``)."""
        child = self.root.children[idx]
        if child is None or child.n == 0:
            return (self.root.wl, self.root.d, self.root.m)
        return (-child.wl, child.d, child.m + 1.0)

    def get_root_nn_eval(self) -> Optional[EvalResult]:
        """The raw network output for the root, from this search or the cache."""
        if self.root_nn_eval is not None:
            return self.root_nn_eval
        return self.backend.cache.get(self.backend.cache_key(self.history))

    def best_is_proven(self, best_wl: float, best_is_terminal: bool) -> bool:
        """``SelfPlayGame::Play``'s ``best_is_proof`` check."""
        if not best_is_terminal or best_wl >= 1.0:
            return best_is_terminal
        best = BOUND_DRAW if best_wl == 0 else BOUND_LOSS
        upper = best
        for child in self.root.children:
            edge_upper = BOUND_WIN if child is None else -child.lower
            upper = max(upper, edge_upper)
        return not (best < upper)

    # -- reporting ---------------------------------------------------------
    def principal_variation(self, max_len: int = 20) -> List[str]:
        pv: List[str] = []
        node = self.root
        depth = 0
        while node is not None and node.is_expanded and node.n > 0:
            idx = self.get_best_child_no_temperature(node, depth)
            if idx < 0:
                break
            pv.append(node.moves[idx].uci())
            node = node.children[idx]
            depth += 1
            if len(pv) >= max_len or node is None:
                break
        return pv

    def build_info(self, elapsed_ms: float) -> dict:
        q = self.root.q(0.0)
        return {
            "depth": max(1, self.stats.cum_depth // max(1, self.stats.playouts)),
            "seldepth": self.stats.max_depth,
            "nodes": self.root.n,
            "time": int(elapsed_ms),
            "nps": int(self.root.n / max(1e-6, elapsed_ms / 1000.0)),
            "score_cp": q_to_centipawn(q),
            "wdl": q_d_to_wdl_permille(q, self.root.d),
            "pv": self.principal_variation(),
        }


def q_to_centipawn(q: float) -> int:
    """lc0's ``centipawn`` score type: ``90 * tan(1.5637541897 * q)``."""
    q = max(-0.99999, min(0.99999, q))
    return int(round(90.0 * math.tan(1.5637541897 * q)))


def q_d_to_wdl_permille(q: float, d: float) -> Tuple[int, int, int]:
    w = (1.0 + q - d) / 2.0
    l = (1.0 - q - d) / 2.0
    return (int(round(1000 * w)), int(round(1000 * d)), int(round(1000 * l)))
