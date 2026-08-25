"""Search tree nodes, following lc0's ``search/classic/node.{h,cc}``.

Differences in bookkeeping (not in behaviour) from lc0:

* Lc0 splits every node into an ``Edge`` array (move + prior P) owned by the
  parent and lazily created ``Node`` objects. We keep the move list and the
  prior array on the parent too, and create a child ``Node`` on first visit --
  the same lazy structure, just expressed with Python objects.
* Lc0 stores ``wl_`` from the *parent's* point of view. Here ``wl`` is stored
  from the point of view of the side to move **at this node**, so the utility
  of a child seen from its parent is ``-child.wl``. Every formula below is the
  Lc0 one with that single sign convention applied consistently.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

TERMINAL_NONE = 0
TERMINAL_ENDOFGAME = 1
TERMINAL_TWOFOLD = 2

# Bounds, from the node's own point of view: -1 loss, 0 draw, 1 win.
BOUND_LOSS = -1
BOUND_DRAW = 0
BOUND_WIN = 1


class Node:
    __slots__ = ("parent", "index", "n", "n_in_flight", "wl", "d", "m",
                 "moves", "policy", "children", "terminal", "lower", "upper")

    def __init__(self, parent: Optional["Node"] = None, index: int = -1):
        self.parent = parent
        self.index = index
        self.n = 0
        self.n_in_flight = 0
        self.wl = 0.0
        self.d = 1.0
        self.m = 0.0
        self.moves = None          # list[chess.Move] once expanded
        self.policy: Optional[np.ndarray] = None
        self.children: Optional[List[Optional["Node"]]] = None
        self.terminal = TERMINAL_NONE
        self.lower = BOUND_LOSS
        self.upper = BOUND_WIN

    # -- state -------------------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        return self.terminal != TERMINAL_NONE

    @property
    def is_expanded(self) -> bool:
        return self.moves is not None

    def children_visits(self) -> int:
        """``Node::GetChildrenVisits`` -- visits below this node."""
        return self.n - 1 if self.n > 0 else 0

    def q(self, draw_score: float = 0.0) -> float:
        """``Node::GetQ`` in this node's own frame."""
        return self.wl + draw_score * self.d

    def expand(self, moves, policy: np.ndarray) -> None:
        self.moves = moves
        self.policy = policy
        self.children = [None] * len(moves)

    def make_terminal(self, result: int, plies_left: float = 0.0,
                      terminal_type: int = TERMINAL_ENDOFGAME) -> None:
        """``Node::MakeTerminal``; ``result`` is from this node's own frame."""
        self.terminal = terminal_type
        self.m = plies_left
        if result == BOUND_DRAW:
            self.wl = 0.0
            self.d = 1.0
        elif result == BOUND_WIN:
            self.wl = 1.0
            self.d = 0.0
        else:
            self.wl = -1.0
            self.d = 0.0
        self.lower = result
        self.upper = result

    def make_not_terminal(self) -> None:
        """``Node::MakeNotTerminal`` -- recompute from the children."""
        self.terminal = TERMINAL_NONE
        self.lower = BOUND_LOSS
        self.upper = BOUND_WIN
        self.n = 1
        self.wl = 0.0
        self.d = 0.0
        self.m = 0.0
        if not self.children:
            return
        for child in self.children:
            if child is None or child.n <= 0:
                continue
            # Weighted by the child's visits; the child's value is negated
            # because it is stored in the child's own frame.
            self.n += child.n
            self.wl += -child.wl * child.n
            self.d += child.d * child.n
            self.m += (child.m + 1) * child.n
        if self.n > 1:
            self.wl /= (self.n - 1)
            self.d /= (self.n - 1)
            self.m /= (self.n - 1)

    def set_bounds(self, lower: int, upper: int) -> None:
        self.lower = lower
        self.upper = upper

    # -- updates -----------------------------------------------------------
    def finalize_score_update(self, v: float, d: float, m: float,
                              multivisit: int = 1) -> None:
        """``Node::FinalizeScoreUpdate`` (running average)."""
        n = self.n
        denom = n + multivisit
        self.wl += multivisit * (v - self.wl) / denom
        self.d += multivisit * (d - self.d) / denom
        self.m += multivisit * (m - self.m) / denom
        self.n = denom
        self.n_in_flight -= multivisit

    def adjust_for_terminal(self, v_delta: float, d_delta: float,
                            m_delta: float, multivisit: int) -> None:
        """``Node::AdjustForTerminal``."""
        self.wl += multivisit * v_delta / self.n
        self.d += multivisit * d_delta / self.n
        self.m += multivisit * m_delta / self.n

    def revert_terminal_visits(self, v: float, d: float, m: float,
                               multivisit: int) -> None:
        """``Node::RevertTerminalVisits``, used when undoing a two-fold draw."""
        n_new = self.n - multivisit
        if n_new <= 0:
            self.wl = 0.0
            self.d = 1.0
            self.m = 0.0
            self.n = 0
            return
        self.wl -= multivisit * (v - self.wl) / n_new
        self.d -= multivisit * (d - self.d) / n_new
        self.m -= multivisit * (m - self.m) / n_new
        self.n = n_new

    def cancel_in_flight(self, count: int = 1) -> None:
        self.n_in_flight -= count

    # -- introspection -----------------------------------------------------
    def child_n(self, i: int) -> int:
        child = self.children[i]
        return child.n if child is not None else 0

    def visited_policy(self) -> float:
        """``Node::GetVisitedPolicy`` -- prior mass of the visited children."""
        if self.children is None:
            return 0.0
        total = 0.0
        pol = self.policy
        for i, child in enumerate(self.children):
            if child is not None and child.n > 0:
                total += float(pol[i])
        return total

    def total_child_visits(self) -> int:
        if self.children is None:
            return 0
        return sum(c.n for c in self.children if c is not None)


class NodeTree:
    """``classic::NodeTree`` -- a root node plus the history that leads to it."""

    def __init__(self, history):
        self.history = history
        self.root = Node()
        self.moves: List = []

    def make_move(self, move, reuse: bool = True) -> None:
        """Advance the tree, keeping the subtree of ``move`` when asked."""
        child = None
        if reuse and self.root.is_expanded:
            for i, mv in enumerate(self.root.moves):
                if mv == move:
                    child = self.root.children[i]
                    break
        self.history.append(move)
        self.moves.append(move)
        if child is None:
            self.root = Node()
        else:
            child.parent = None
            child.index = -1
            self.root = child

    def trim(self) -> None:
        self.root = Node()
