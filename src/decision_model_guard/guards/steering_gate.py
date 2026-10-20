"""The steering gate: erase-and-check around any decision model."""
from __future__ import annotations

import collections
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..input_scanners.rules import visible
from ..models.base import DecisionModel, checked
from ..scene import TRUSTED_HEADERS, Block, content_size, erase, parse_blocks, units
from ..types import GuardReport, Kind
from ..util import argmax

UnitScorer = Callable[[Sequence[str]], list[float]]


@dataclass
class GateOutcome:
    probs: list[float]                 # guarded probabilities
    raw: list[float]                   # probabilities for the unmodified scene
    index: int
    raw_index: int
    flagged: bool                      # risky/ordered: the rule replaced the original answer
    stable: bool                       # no single erasure changes the original answer
    culprit: str | None = None         # the erased text behind the change
    alternative: int | None = None     # the option a variant preferred when the answer is unstable
    calls: int = 1
    erased: list[str] = field(default_factory=list)
    variant_probs: list[list[float]] = field(default_factory=list)
    detector_flagged: int = 0
    unerased: int = 0                  # scanner findings (hints) that no unit contains, so no variant erased them
    units: int = 0                     # erasable units of the scene
    skipped: int = 0                   # full check: units left out by max_variants or max_chars


class SteeringGate:
    """Erase-and-check around a decision model.

    1. Untrusted blocks are cut into units: whole blocks with unknown or trusted-looking headers, forged headers inside
       content with the text after them, paragraphs, windows of 1 to `window` sentences, edge units first. Trusted
       blocks are never touched, and no unit holds all untrusted content of the scene.
    2. Unit selection: `selector="all"` erases every unit once (up to `max_variants`), `selector="detector"` only the units
       the claim detector scores at or above `threshold`. With `scope="unordered"` the detector is used only for
       decisions without risky options or order, and the others keep the full check. `mode="trigger"` runs the full
       check whenever the detector flags any unit and makes a single call otherwise. Units that contain text flagged by
       a content scanner (hints) are always erased.
    3. The model scores the original and every variant in one batch.
    4. Rules: risky options -> highest risk wins (replacement needs `override` under `rule="absolute"`, or `delta`
       above the original risk under `rule="delta"`, or `delta` above the median variant under `rule="gap"`); ordered
       options -> the most cautious answer wins; unordered -> the original answer stays, flagged unstable when an erasure
       changes it.

    `max_chars` bounds the characters sent to the model per decision (the scene times one plus the number of variants):
    a long scene gets fewer variants, the detector-flagged units first when a detector is set. Units the full check
    leaves out are counted (`skipped`) and the policy names them in the reasons.

    Guarantee within the variant budget: a claim that fits into one erased unit cannot move the answer away from the
    risky or cautious side, and in the unordered case it cannot change the answer unnoticed."""

    name = "SteeringGate"

    def __init__(self, window: int = 3, max_variants: int = 48, selector: str = "all", detector: UnitScorer | None = None,
                 threshold: float = 0.3, scope: str = "all", mode: str = "erase", override: float | None = None, rule: str = "absolute",
                 delta: float = 0.3, margin: float = 0.2, risk_threshold: float = 0.5, trusted_headers: Sequence[str] = TRUSTED_HEADERS,
                 seed: int = 0, max_chars: int | None = 400_000):
        if selector not in ("all", "detector"):
            raise ValueError("selector must be 'all' or 'detector'")
        if selector == "detector" and detector is None:
            raise ValueError("selector='detector' needs a detector (for example input_scanners.ClaimDetector())")
        if scope not in ("all", "unordered") or mode not in ("erase", "trigger") or rule not in ("absolute", "delta", "gap"):
            raise ValueError("scope in {all, unordered}, mode in {erase, trigger}, rule in {absolute, delta, gap}")
        if window < 1 or max_variants < 1:
            raise ValueError("window and max_variants must be at least 1")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.window, self.max_variants, self.selector, self.detector = window, max_variants, selector, detector
        self.threshold, self.scope, self.mode, self.rule, self.delta = threshold, scope, mode, rule, delta
        self.override = risk_threshold if override is None else override
        self.margin, self.risk_threshold, self.trusted, self.seed = margin, risk_threshold, tuple(trusted_headers), seed
        self.max_chars = max_chars

    # ------------------------------------------------------------------ variants
    def _full(self, scene: str, per_block: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
        head, rest = [], []
        for us in per_block:
            k = max(4, self.max_variants // max(1, 2 * len(per_block)))
            head += us[:k]
            rest += us[k:]
        random.Random(self.seed).shuffle(rest)
        return (head + rest)[: self.max_variants]

    def variants(self, scene: str, kind: Kind = Kind.UNORDERED, hints: Sequence[str] = (),
                 blocks: Sequence[Block] | None = None) -> tuple[list[tuple[str, str]], int]:
        """(variant scene, erased text) pairs and the number of detector-flagged units. `blocks` gives the trust of each
        block (default: parse `scene`)."""
        v, n, *_ = self._plan(scene, kind, hints, blocks)
        return v, n

    def _plan(self, scene: str, kind: Kind, hints: Sequence[str], blocks: Sequence[Block] | None
              ) -> tuple[list[tuple[str, str]], int, int, int, int]:
        """(variants, detector-flagged units, unerased hints, units, units the full check skipped)."""
        blocks = list(blocks) if blocks is not None else parse_blocks(scene, self.trusted)
        content = content_size(scene, blocks)
        per_block = [units(scene, b, self.window, content) for b in blocks if not b.trusted]
        allspans = [s for us in per_block for s in us]
        seen = [visible(scene[a:b]) for a, b in allspans]
        hs = [h for h in dict.fromkeys(visible(h).strip() for h in hints if h) if h]
        hinted = [s for s, t in zip(allspans, seen, strict=True) if any(h in t for h in hs)]
        unerased = sum(1 for h in hs if not any(h in t for t in seen))
        use_detector = self.selector == "detector" and (self.scope == "all" or kind == Kind.UNORDERED)
        budget = self.max_variants if not self.max_chars else max(1, min(self.max_variants, self.max_chars // max(1, len(scene)) - 1))
        n_flagged, full = 0, not use_detector

        def flagged_units() -> list[tuple[int, int]]:
            scores = self.detector([scene[a:b].strip() for a, b in allspans]) if allspans else []   # type: ignore[misc]
            return [s for _, s in sorted(((-p, s) for s, p in zip(allspans, scores, strict=True) if p >= self.threshold))]
        if use_detector:
            flagged = flagged_units()
            n_flagged = len(flagged)
            full = self.mode == "trigger" and bool(flagged or hinted)
            spans = [] if self.mode == "trigger" else flagged[: self.max_variants]
        if full:
            spans = self._full(scene, per_block)
            if len(spans) > budget:                              # long scene: fewer variants, flagged units first
                first = flagged_units() if self.detector is not None else []
                spans = (first + [s for s in spans if s not in first])[:budget]
        skipped = len(allspans) - len(spans) if full else 0
        spans += [s for s in hinted if s not in spans]
        return [(erase(scene, a, b), scene[a:b].strip()) for a, b in spans], n_flagged, unerased, len(allspans), skipped

    # ------------------------------------------------------------------ run
    def run(self, model: DecisionModel, scene: str, question: str, options: Sequence[str], kind: Kind,
            risky: Sequence[int] | None = None, order: Sequence[int] | None = None, hints: Sequence[str] = (),
            blocks: Sequence[Block] | None = None) -> tuple[GuardReport, GateOutcome]:
        """Score the scene and its variants in one batch and apply the rule for the decision type."""
        V, n_flagged, unerased, n_units, skipped = self._plan(scene, kind, hints, blocks)
        P = checked(model, [scene] + [v for v, _ in V], question, options)
        top = [argmax(p) for p in P]
        changed = [i for i in range(1, len(P)) if top[i] != top[0]]
        stable = not changed
        alt = collections.Counter(top[i] for i in changed).most_common(1)[0][0] if changed else None
        culprit = lambda j: V[j - 1][1] if j > 0 else None  # noqa: E731
        erased = [e for _, e in V]
        base = dict(raw=P[0], raw_index=top[0], stable=stable, alternative=alt, calls=len(P), erased=erased, variant_probs=P[1:],
                    detector_flagged=n_flagged, unerased=unerased, units=n_units, skipped=skipped)
        if kind == Kind.RISKY and risky:
            risk = [sum(p[i] for i in risky) for p in P]
            best = max(risk)
            j = min((i for i in range(len(P)) if risk[i] >= best - 1e-6), key=lambda i: (i > 0, len(V[i - 1][1]) if i else 0))
            med = sorted(risk[1:])[(len(risk) - 1) // 2] if len(risk) > 1 else risk[0]
            keep = {"absolute": risk[j] < self.override,
                    "delta": risk[j] < self.risk_threshold or risk[j] - risk[0] < self.delta,
                    "gap": risk[j] < self.risk_threshold or risk[j] - med < self.delta}[self.rule]
            if j > 0 and keep:
                j = 0
            flagged = j > 0 and risk[j] - risk[0] >= self.margin and risk[j] >= self.risk_threshold > risk[0]
            if j > 0 and not flagged and top[j] != top[0]:
                flagged = True
            out = GateOutcome(probs=P[j], index=top[j], flagged=flagged, culprit=culprit(j) if j > 0 else None, **base)
            report = GuardReport(self.name, flagged, float(risk[j]), len(P), {"rule": "highest risk", "raw_risk": round(risk[0], 6),
                                 "max_variant_risk": round(best, 6), "variants": len(V), "detector_flagged": n_flagged, "unerased": unerased,
                                 "units": n_units, "skipped": skipped, "culprit": out.culprit})
            return report, out
        if kind == Kind.ORDERED and order:
            rank = {o: r for r, o in enumerate(order)}
            K = max(1, len(order) - 1)
            cons = [sum(p[o] * (K - rank[o]) / K for o in order) for p in P]
            j = min(range(len(P)), key=lambda i: (rank.get(top[i], len(order)), -cons[i]))
            flagged = rank.get(top[j], len(order)) < rank.get(top[0], len(order))
            out = GateOutcome(probs=P[j] if flagged else P[0], index=top[j] if flagged else top[0], flagged=flagged,
                              culprit=culprit(j) if flagged else None, **base)
            report = GuardReport(self.name, flagged, float(cons[j]), len(P), {"rule": "most cautious", "variants": len(V),
                                 "detector_flagged": n_flagged, "unerased": unerased, "units": n_units, "skipped": skipped,
                                 "culprit": out.culprit})
            return report, out
        first = min(changed, key=lambda i: len(V[i - 1][1])) if changed else 0
        out = GateOutcome(probs=P[0], index=top[0], flagged=False, culprit=culprit(first) if changed else None, **base)
        report = GuardReport(self.name, not stable, float(len(changed)) / max(1, len(V)), len(P),
                             {"rule": "stability", "variants": len(V), "changed_by": len(changed), "alternative": alt,
                              "detector_flagged": n_flagged, "unerased": unerased, "units": n_units, "skipped": skipped,
                              "culprit": out.culprit})
        return report, out
