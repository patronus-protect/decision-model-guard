"""Coverage: decisions over content longer than the model can read.

Every decision model has a context limit and drops what does not fit: Wolf Defender Jev keeps the first third and the
last two thirds of a long scene, LLM judges keep the end, classifiers the beginning. An attacker who pads the content
can move an instruction into the part the model never sees, and nothing reports it. Coverage cuts the untrusted text of
a long scene into overlapping windows (trusted blocks stay in every window), asks the model once per window and
combines the answers with the gate's rules: risky options -> the highest risk wins, ordered options -> the most cautious
answer wins, unordered -> the decision is flagged when a window disagrees.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from ..models.base import DecisionModel, checked
from ..scene import TRUSTED_HEADERS, Block, neutralize, parse_blocks, sentences
from ..types import GuardReport, Kind
from ..util import argmax


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Coverage:
    """Check long scenes window by window.

    max_tokens: the model's context budget for the scene; default `model.context_tokens` (the built-in adapters set it).
    count_tokens: token counter; default `model.count_tokens`, else about four characters per token.
    overlap_tokens: overlap between windows, so a sentence on a window border is seen whole at least once.
    max_windows: cap on extra calls; when the content needs more windows the report says how much was covered.
    gate: optional SteeringGate run inside every window (erase-and-check per window, more calls).
    margin, risk_threshold: for risky decisions a window replaces the answer when its risk is at least `risk_threshold`
    and `margin` above the current risk, or when it turns the answer to a risky option."""

    name = "Coverage"

    def __init__(self, max_tokens: int | None = None, count_tokens: Callable[[str], int] | None = None, overlap_tokens: int = 64,
                 max_windows: int = 16, gate=None, margin: float = 0.2, risk_threshold: float = 0.5,
                 trusted_headers: Sequence[str] = TRUSTED_HEADERS):
        self.max_tokens, self.count_tokens, self.overlap, self.max_windows = max_tokens, count_tokens, overlap_tokens, max_windows
        self.gate, self.margin, self.risk_threshold, self.trusted = gate, margin, risk_threshold, tuple(trusted_headers)

    def _budget(self, model) -> tuple[int | None, Callable[[str], int]]:
        count = self.count_tokens or getattr(model, "count_tokens", None) or _approx_tokens
        return self.max_tokens or getattr(model, "context_tokens", None), count

    def windows(self, scene: str, budget: int, count: Callable[[str], int], blocks: Sequence[Block] | None = None) -> tuple[list[str], float]:
        """Window scenes and the share of untrusted sentences they cover. `blocks` gives the trust of each block
        (default: parse `scene`)."""
        blocks = list(blocks) if blocks is not None else parse_blocks(scene, self.trusted)
        fixed = sum(count(scene[b.start: b.end]) for b in blocks if b.trusted) + 8 * len(blocks)
        avail = budget - fixed
        units: list[tuple[int, str, int]] = []                       # (block index, sentence, tokens)
        for bi, b in enumerate(blocks):
            if b.trusted:
                continue
            body = b.body(scene)
            for a, z in sentences(body) or [(0, len(body))]:
                s = body[a:z].strip()
                if s:
                    units.append((bi, s, count(s) + 1))
        if not units or avail < 32:
            return [], 0.0
        spans: list[tuple[int, int]] = []
        i = 0
        while i < len(units) and len(spans) < self.max_windows:
            j, used = i, 0
            while j < len(units) and used + units[j][2] <= avail:
                used += units[j][2]
                j += 1
            j = max(j, i + 1)                                          # a sentence longer than the window still gets one
            spans.append((i, j))
            if j >= len(units):
                break
            k, back = j, 0
            while k > i + 1 and back + units[k - 1][2] <= self.overlap:
                back += units[k - 1][2]
                k -= 1
            i = k
        covered = spans[-1][1] / len(units)
        out = []
        for a, z in spans:
            chosen: dict[int, list[str]] = {}
            for bi, s, _ in units[a:z]:
                chosen.setdefault(bi, []).append(s)
            parts = []
            for bi, b in enumerate(blocks):
                if b.trusted:
                    parts.append(scene[b.start: b.end].strip("\n"))
                elif bi in chosen:
                    head = f"[{b.header}]\n" if b.header else ""
                    parts.append(head + neutralize(" ".join(chosen[bi])))
            out.append("\n\n".join(parts))
        return out, covered

    def check(self, model: DecisionModel, scene: str, question: str, options: Sequence[str], probs: Sequence[float], kind: Kind,
              *, risky: Sequence[int] | None = None, order: Sequence[int] | None = None, fallback: int | None = None,
              blocks: Sequence[Block] | None = None) -> GuardReport:
        budget, count = self._budget(model)
        if not budget:                                          # the policy reports it: an unchecked long scene is not "covered"
            return GuardReport(self.name, False, detail={"skipped": "unknown context size: pass Coverage(max_tokens=...)"})
        n = count(scene)
        if n <= budget:
            return GuardReport(self.name, False, detail={"truncated": False, "tokens": n, "budget": budget})
        wins, covered = self.windows(scene, budget, count, blocks)
        if not wins:
            return GuardReport(self.name, False, detail={"truncated": True, "tokens": n, "budget": budget, "windows": 0,
                                                         "skipped": "trusted blocks alone fill the context"})
        if self.gate is not None:
            P = [self.gate.run(model, w, question, options, kind, risky, order)[1].probs for w in wins]
            calls = sum(1 + len(self.gate.variants(w, kind)[0]) for w in wins)
        else:
            P = checked(model, wins, question, options)
            calls = len(wins)
        base = list(probs)
        top = argmax(base)
        best, flagged = None, False
        if kind == Kind.RISKY and risky:
            risk = lambda p: sum(p[i] for i in risky)  # noqa: E731
            j = max(range(len(P)), key=lambda i: risk(P[i]))
            moves = argmax(P[j]) in risky and top not in risky
            flagged = moves or (risk(P[j]) >= self.risk_threshold and risk(P[j]) - risk(base) >= self.margin)
            best = j if flagged else None
        elif kind == Kind.ORDERED and order:
            rank = {o: r for r, o in enumerate(order)}
            j = min(range(len(P)), key=lambda i: rank.get(argmax(P[i]), len(order)))
            flagged = rank.get(argmax(P[j]), len(order)) < rank.get(top, len(order))
            best = j if flagged else None
        else:
            diff = [i for i in range(len(P)) if argmax(P[i]) != top]
            flagged = bool(diff)
            best = diff[0] if diff else None
        detail = {"truncated": True, "tokens": n, "budget": budget, "windows": len(wins), "covered": round(covered, 4)}
        if best is not None:
            detail.update({"window": best, "guarded_index": argmax(P[best]), "guarded_probs": [round(x, 6) for x in P[best]]})
        score = max((sum(p[i] for i in risky) for p in P), default=0.0) if risky else float(flagged)
        return GuardReport(self.name, flagged, float(score), calls, detail)
