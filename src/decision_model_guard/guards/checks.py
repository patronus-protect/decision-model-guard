"""Additional decision checks: option-order consistency and confidence."""
from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

from ..models.base import DecisionModel
from ..types import GuardReport, Kind
from ..util import argmax, normalize


class OptionOrder:
    """Asks the model again with the options in a different order and maps the answers back. A decision whose answer
    depends on where an option is listed is not robust; decision-model documentation names option order as a known
    source of error. permutations: "reverse" (1 extra call), "rotate" (n-1 calls), "random" (`samples` calls), "all"
    (every permutation, small option lists only). With `average=True` the guarded probabilities become the mean over
    all orders (debiasing)."""

    name = "OptionOrder"

    def __init__(self, permutations: str = "reverse", samples: int = 2, average: bool = False, seed: int = 0):
        if permutations not in ("reverse", "rotate", "random", "all"):
            raise ValueError("permutations must be reverse, rotate, random or all")
        self.permutations, self.samples, self.average, self.seed = permutations, samples, average, seed

    def _orders(self, n: int) -> list[list[int]]:
        base = list(range(n))
        if n < 2:
            return []
        if self.permutations == "reverse":
            return [base[::-1]]
        if self.permutations == "rotate":
            return [base[k:] + base[:k] for k in range(1, n)]
        if self.permutations == "all":
            return [list(p) for p in itertools.permutations(base) if list(p) != base][:120]
        rng = random.Random(self.seed)
        out = []
        for _ in range(self.samples):
            p = base[:]
            rng.shuffle(p)
            if p != base:
                out.append(p)
        return out or [base[::-1]]

    def check(self, model: DecisionModel, scene: str, question: str, options: Sequence[str], probs: Sequence[float], kind: Kind,
              *, raw_probs: Sequence[float] | None = None, **_ctx) -> GuardReport:
        probs = list(raw_probs if raw_probs is not None else probs)       # the model's own answer on this scene
        orders = self._orders(len(options))
        if not orders:
            return GuardReport(self.name, False)
        mapped = []
        for perm in orders:
            p = normalize(model.decide_batch([scene], question, [options[i] for i in perm])[0])
            back = [0.0] * len(options)
            for pos, orig in enumerate(perm):
                back[orig] = p[pos]
            mapped.append(back)
        base = argmax(list(probs))
        flips = sum(argmax(m) != base for m in mapped)
        detail = {"orders": len(orders), "flips": flips}
        if self.average:
            allp = [list(probs)] + mapped
            detail["averaged_probs"] = [round(sum(c) / len(allp), 6) for c in zip(*allp, strict=True)]
        return GuardReport(self.name, flips > 0, flips / len(orders), len(orders), detail)


class Confidence:
    """Flags decisions whose top probability is below `min_confidence` or whose lead over the second option is below
    `min_margin`. The policy turns that into abstain or escalate."""

    name = "Confidence"

    def __init__(self, min_confidence: float = 0.6, min_margin: float = 0.0):
        self.min_confidence, self.min_margin = min_confidence, min_margin

    def check(self, model: DecisionModel, scene: str, question: str, options: Sequence[str], probs: Sequence[float], kind: Kind) -> GuardReport:
        s = sorted(probs, reverse=True)
        top, margin = s[0], s[0] - (s[1] if len(s) > 1 else 0.0)
        low = top < self.min_confidence or margin < self.min_margin
        return GuardReport(self.name, low, float(1 - top), 0, {"top": round(top, 6), "margin": round(margin, 6)})
