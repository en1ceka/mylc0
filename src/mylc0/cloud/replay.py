"""Which generations the trainer is allowed to learn from right now.

Self-play never stops, so data keeps arriving for a generation after the next
one has already been published. Those games are not stale -- they were played
by a network only one step behind -- and throwing them away would waste a node
farm's output. What they must not do is drown the newest data.

The policy here is deliberately a single object with a single method, because
the interesting variations all live in ``weights_for``:

    LastN(3)                      the current default: newest 3, sampled evenly
    LastN(5)                      a longer window
    Weighted({0: .70, 1: .25, 2: .05})   70/25/5 by age

``select`` returns the generations in the window and the weight each should
get when a batch is sampled. A weight of ``None`` throughout means "no
preference", which lets the loader keep its existing uniform-over-chunks
behaviour rather than paying for weighted sampling nobody asked for.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ReplayWindow:
    """Base policy: the newest ``generations`` generations, weighted evenly."""

    generations: int = 3

    def select(self, available: Sequence[int]
               ) -> Tuple[List[int], Optional[Dict[int, float]]]:
        chosen = self._window(available)
        return chosen, self.weights_for(chosen)

    def _window(self, available: Sequence[int]) -> List[int]:
        if not available:
            return []
        ordered = sorted(set(int(g) for g in available))
        count = max(1, int(self.generations))
        return ordered[-count:]

    def weights_for(self, chosen: Sequence[int]
                    ) -> Optional[Dict[int, float]]:
        """None means "sample chunks uniformly, ignoring generation"."""
        return None

    def describe(self) -> str:
        return f"last {self.generations} generation(s), sampled evenly"


@dataclass
class WeightedReplayWindow(ReplayWindow):
    """Explicit weight per age, newest first.

    ``by_age={0: 0.7, 1: 0.25, 2: 0.05}`` means the newest generation supplies
    70% of sampled positions. Ages with no data are dropped and the rest are
    renormalised, so the policy still behaves sensibly at the start of a run
    when only one generation exists.
    """

    by_age: Dict[int, float] = None    # type: ignore[assignment]

    def __post_init__(self):
        if not self.by_age:
            self.by_age = {0: 0.70, 1: 0.25, 2: 0.05}
        self.generations = max(self.generations, len(self.by_age))

    def _window(self, available):
        ordered = sorted(set(int(g) for g in available))
        return ordered[-max(1, len(self.by_age)):]

    def weights_for(self, chosen):
        if not chosen:
            return None
        newest = max(chosen)
        raw = {g: self.by_age.get(newest - g, 0.0) for g in chosen}
        total = sum(raw.values())
        if total <= 0:
            return None
        return {g: w / total for g, w in raw.items()}

    def describe(self) -> str:
        parts = ", ".join(f"age {age}: {share:.0%}"
                          for age, share in sorted(self.by_age.items()))
        return f"weighted window ({parts})"


def generation_dirs(roots, policy: "ReplayWindow"):
    """The ``gen_NNNNNN`` directories the policy currently admits.

    Returns ``(paths, generations)``. The directories on disk are the source
    of truth rather than the index, so this keeps working if the index is
    rebuilt or the data was put there by hand.

    Recomputed per generation by the callers: the window moves as new data
    lands, and a loader built once at startup would keep reading the same
    three directories while newer ones piled up beside it.
    """
    found = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            match = re.fullmatch(r"gen_(\d{6})", name)
            path = os.path.join(root, name)
            if match and os.path.isdir(path):
                found.setdefault(int(match.group(1)), []).append(path)
    if not found:
        return [], []
    chosen, _weights = policy.select(sorted(found))
    return [p for gen in chosen for p in found[gen]], chosen


def policy_from_config(generations: int = 3,
                       weights: Optional[Dict[int, float]] = None
                       ) -> ReplayWindow:
    """One entry point so callers do not branch on which class to build."""
    if weights:
        return WeightedReplayWindow(generations=generations, by_age=weights)
    return ReplayWindow(generations=generations)


__all__ = ["ReplayWindow", "WeightedReplayWindow", "policy_from_config",
           "generation_dirs"]
