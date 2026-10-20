"""Policy: turns gate, scanner and guard results into the final answer and an action."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .guards.steering_gate import GateOutcome
from .types import Action, GuardReport, Kind, ScanResult

PRIORITY = {Action.ALLOW: 0, Action.CAUTIOUS: 1, Action.FALLBACK: 2, Action.ESCALATE: 3, Action.ABSTAIN: 4, Action.BLOCK: 5}


@dataclass
class Policy:
    """What to do when something triggers.

    gate_action:          risky/ordered decision moved by the gate: "apply" (take the gate's cautious answer), "escalate"
                          (take it and mark for review) or "report" (keep the model's answer, only record the reason)
    unstable_action:      unordered decision changed by an erasure: "fallback" (use the fallback option), "escalate", "keep"
    scanner_action:       a content scanner failed: "report", "escalate", "cautious" (most cautious option) or "block"
    block_scanners:       scanners whose failure always blocks, e.g. ["PromptInjection"]
    low_confidence_action: Confidence guard triggered: "abstain" or "escalate"
    inconsistent_action:  OptionOrder or Invariance guard triggered: "escalate" or "report"
    coverage_action:      Coverage guard found a window of long content that moves the decision: "apply" (risky/ordered:
                          take the window's more cautious answer; unordered: fallback), "escalate" or "report"
    fallback_option:      default fallback option index for unordered decisions (a per-decision fallback wins)

    Confidence and Conformal guards use low_confidence_action."""

    gate_action: str = "apply"
    unstable_action: str = "fallback"
    scanner_action: str = "report"
    block_scanners: Sequence[str] = ()
    low_confidence_action: str = "abstain"
    inconsistent_action: str = "escalate"
    coverage_action: str = "apply"
    fallback_option: int | None = None

    def __post_init__(self):
        checks = {"gate_action": ("apply", "escalate", "report"), "unstable_action": ("fallback", "escalate", "keep"),
                  "scanner_action": ("report", "escalate", "cautious", "block"), "low_confidence_action": ("abstain", "escalate"),
                  "inconsistent_action": ("escalate", "report"), "coverage_action": ("apply", "escalate", "report")}
        for k, allowed in checks.items():
            if getattr(self, k) not in allowed:
                raise ValueError(f"{k} must be one of {allowed}")

    @staticmethod
    def cautious_index(kind: Kind, probs: Sequence[float], risky: Sequence[int] | None, order: Sequence[int] | None,
                       fallback: int | None) -> int | None:
        if kind == Kind.RISKY and risky:
            return max(risky, key=lambda i: probs[i])
        if kind == Kind.ORDERED and order:
            return order[0]
        return fallback

    def apply(self, kind: Kind, gate: GateOutcome, scans: Sequence[ScanResult], guards: Sequence[GuardReport],
              risky: Sequence[int] | None = None, order: Sequence[int] | None = None, fallback: int | None = None
              ) -> tuple[int | None, list[float], Action, list[str]]:
        fb = fallback if fallback is not None else self.fallback_option
        index, probs = gate.raw_index, list(gate.raw)
        action, reasons = Action.ALLOW, []

        def raise_to(a: Action):
            nonlocal action
            if PRIORITY[a] > PRIORITY[action]:
                action = a

        if kind in (Kind.RISKY, Kind.ORDERED) and gate.flagged:
            reasons.append(f"steering gate: erasing {gate.culprit!r} moves the decision to the cautious side" if gate.culprit
                           else "steering gate: the decision moves to the cautious side")
            if self.gate_action in ("apply", "escalate"):
                index, probs = gate.index, list(gate.probs)
                raise_to(Action.CAUTIOUS if self.gate_action == "apply" else Action.ESCALATE)
        if kind == Kind.UNORDERED and not gate.stable:
            reasons.append(f"steering gate: erasing {gate.culprit!r} changes the answer" if gate.culprit else "steering gate: answer unstable")
            if self.unstable_action == "fallback" and fb is not None:
                index = fb
                raise_to(Action.FALLBACK)
            elif self.unstable_action in ("fallback", "escalate"):
                raise_to(Action.ESCALATE)
        failed = [s for s in scans if not s.valid]
        if failed:
            names = ", ".join(s.scanner for s in failed)
            reasons.append(f"content scanners: {names}")
            if any(s.scanner in self.block_scanners for s in failed) or self.scanner_action == "block":
                index = None
                raise_to(Action.BLOCK)
            elif self.scanner_action == "escalate":
                raise_to(Action.ESCALATE)
            elif self.scanner_action == "cautious":
                ci = self.cautious_index(kind, probs, risky, order, fb)
                if ci is not None and index is not None:
                    index = ci
                    raise_to(Action.CAUTIOUS)
                else:
                    raise_to(Action.ESCALATE)
        for g in guards:
            if g.guard == "Coverage" and g.detail.get("truncated") and not g.triggered:
                reasons.append(f"long content: {g.detail.get('windows', 0)} windows checked, none moves the decision")
            if not g.triggered:
                continue
            if g.guard == "Coverage":
                reasons.append(f"long content: window {g.detail.get('window')} of {g.detail.get('windows')} moves the decision")
                if self.coverage_action == "report" or index is None:
                    continue
                if kind in (Kind.RISKY, Kind.ORDERED) and "guarded_index" in g.detail:
                    index, probs = g.detail["guarded_index"], list(g.detail["guarded_probs"])
                    raise_to(Action.CAUTIOUS if self.coverage_action == "apply" else Action.ESCALATE)
                elif self.coverage_action == "apply" and fb is not None:
                    index = fb
                    raise_to(Action.FALLBACK)
                else:
                    raise_to(Action.ESCALATE)
            elif g.guard == "Invariance":
                reasons.append(f"answer changes under {', '.join(g.detail.get('changed_under', []))}")
                if self.inconsistent_action == "escalate":
                    raise_to(Action.ESCALATE)
            elif g.guard == "Conformal":
                reasons.append(f"conformal set at {100 * g.detail.get('coverage', 0):.0f} % coverage: {g.detail.get('set')}")
                raise_to(Action.ABSTAIN if self.low_confidence_action == "abstain" else Action.ESCALATE)
            elif g.guard == "Confidence":
                reasons.append(f"low confidence: top {g.detail.get('top')}")
                raise_to(Action.ABSTAIN if self.low_confidence_action == "abstain" else Action.ESCALATE)
            elif g.guard == "OptionOrder":
                reasons.append(f"answer depends on option order ({g.detail.get('flips')} of {g.detail.get('orders')} orders)")
                if self.inconsistent_action == "escalate":
                    raise_to(Action.ESCALATE)
            else:
                reasons.append(f"{g.guard} triggered")
                raise_to(Action.ESCALATE)
        if action == Action.BLOCK:
            index = None
        return index, probs, action, reasons
