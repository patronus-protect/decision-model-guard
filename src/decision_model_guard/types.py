"""Result types shared by scanners, guards and the pipeline."""
from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


class Kind(str, enum.Enum):
    """How the options of a decision relate to each other. It selects the combination rule of the steering gate."""

    RISKY = "risky"            # some options are risky (a security verdict): the highest risk over the variants wins
    ORDERED = "ordered"        # the options have a cautious order (verify < reject < authorize): the most cautious wins
    UNORDERED = "unordered"    # no order (ticket routing): keep the answer, report it as unstable when an erasure changes it


class Action(str, enum.Enum):
    """What the pipeline did with the decision."""

    ALLOW = "allow"            # the model's answer stands; reasons, if any, are informational (scanner_action="report")
    CAUTIOUS = "cautious"      # the answer was moved to the cautious side (gate or policy)
    FALLBACK = "fallback"      # an unstable unordered decision went to the fallback option
    ESCALATE = "escalate"      # the answer is returned but should go to a human
    ABSTAIN = "abstain"        # the model was not confident enough to decide
    BLOCK = "block"            # no answer: the input was refused


@dataclass
class Finding:
    """One thing a scanner or guard noticed, with the character span in the scanned text when it has one."""

    source: str
    score: float
    message: str = ""
    start: int | None = None
    end: int | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The finding as a dict."""
        return asdict(self)


@dataclass
class ScanResult:
    """Output of one content scanner, LLM Guard style: sanitized text, valid flag and a risk score in [0, 1]."""

    scanner: str
    sanitized: str
    valid: bool
    score: float
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The result as a dict (without the sanitized text)."""
        return {"scanner": self.scanner, "valid": self.valid, "score": round(self.score, 6), "findings": [f.to_dict() for f in self.findings]}


@dataclass
class GuardReport:
    """Output of one decision guard (steering gate, option order, confidence)."""

    guard: str
    triggered: bool
    score: float = 0.0
    calls: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The report as a dict."""
        return asdict(self)


@dataclass
class GuardedDecision:
    """The final, guarded decision."""

    options: list[str]
    index: int | None                  # chosen option, None when blocked
    probs: list[float]                 # guarded probabilities (the original ones when no guard changed them)
    raw_index: int
    raw_probs: list[float]
    action: Action
    kind: Kind
    reasons: list[str] = field(default_factory=list)
    scans: list[ScanResult] = field(default_factory=list)
    guards: list[GuardReport] = field(default_factory=list)
    calls: int = 1
    latency_ms: float = 0.0
    culprit: str | None = None         # the erased text that moved the decision, when the gate found one
    sanitized_scene: str | None = None
    id: str = ""

    @property
    def answer(self) -> str | None:
        return None if self.index is None else self.options[self.index]

    @property
    def raw_answer(self) -> str:
        return self.options[self.raw_index]

    @property
    def changed(self) -> bool:
        return self.index != self.raw_index

    @property
    def triggered(self) -> bool:
        return self.action != Action.ALLOW

    def to_dict(self, include_scene: bool = False) -> dict[str, Any]:
        """The decision as a JSON-serialisable dict; include_scene adds the scene the model read."""
        d = {"id": self.id, "answer": self.answer, "index": self.index, "probs": [round(p, 6) for p in self.probs],
             "raw_answer": self.raw_answer, "raw_index": self.raw_index, "raw_probs": [round(p, 6) for p in self.raw_probs],
             "action": self.action.value, "kind": self.kind.value, "changed": self.changed, "reasons": self.reasons,
             "culprit": self.culprit, "calls": self.calls, "latency_ms": round(self.latency_ms, 2),
             "scans": [s.to_dict() for s in self.scans], "guards": [g.to_dict() for g in self.guards]}
        if include_scene:
            d["sanitized_scene"] = self.sanitized_scene
        return d
