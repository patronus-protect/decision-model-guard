"""Steering evaluation: how easily can claims inside the content steer your decision model, with and without the guard?

A case is one decision with a known correct answer:

    {"id": "refund-1", "scene": "...", "question": "...", "options": ["Verify", "Reject", "Authorize"],
     "order": [0, 1, 2], "gold": 0}                     # or "risky": [1], or neither (+ optional "fallback")

For every case the harness inserts claims that push towards a target answer (attacks) and neutral sentences (controls)
into the last untrusted block, then asks the bare model and the guarded pipeline. Metrics follow the protocol of
decision-steering studies: attack success counts attacks on cases the model answers correctly without the claim, where
the target differs from the correct answer.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from ..models.base import DecideBatch, DecisionModel, as_model, checked
from ..pipeline import DecisionGuard
from ..scene import TRUSTED_HEADERS, parse_blocks
from ..util import argmax


def _data(name: str) -> str:
    return resources.files("decision_model_guard.evaluate").joinpath("data", name).read_text(encoding="utf-8")


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Cases from a JSONL file; without a path the bundled sample cases."""
    text = Path(path).read_text(encoding="utf-8") if path else _data("sample_cases.jsonl")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def claim_bank() -> dict[str, Any]:
    return yaml.safe_load(_data("claims.yaml"))


def neutral_bank() -> dict[str, list[str]]:
    return yaml.safe_load(_data("neutral.yaml"))["neutral"]


def default_target(case: dict[str, Any]) -> int:
    n, gold = len(case["options"]), case["gold"]
    if case.get("risky"):
        risky = set(case["risky"])
        return next(i for i in range(n) if i not in risky) if gold in risky else min(risky)
    if case.get("order"):
        order = case["order"]
        return order[-1] if gold != order[-1] else order[-2]
    return next(i for i in range(n) if i != gold)


def insert(scene: str, text: str, position: str = "suffix", trusted_headers: Sequence[str] = TRUSTED_HEADERS) -> str:
    """Insert text into the last untrusted block (suffix, prefix or middle); append a user block if there is none."""
    blocks = [b for b in parse_blocks(scene, trusted_headers) if not b.trusted]
    if not blocks:
        return scene.rstrip() + f"\n\n[USER PROMPT]\n{text}"
    b = blocks[-1]
    body = scene[b.body_start: b.end].strip("\n")
    if position == "prefix":
        new = text + " " + body
    elif position == "middle":
        cut = body.find(". ", len(body) // 3)
        new = body + " " + text if cut < 0 else body[: cut + 1] + " " + text + body[cut + 1:]
    else:
        new = body + " " + text
    tail = scene[b.end:]
    return scene[: b.body_start] + "\n" + new + ("\n\n" + tail.lstrip("\n") if tail.strip() else "\n")


@dataclass
class EvalReport:
    model: str
    cases: int
    attacks: int
    eligible_attacks: int
    accuracy_raw: float
    accuracy_guarded: float
    attack_success_raw: float | None
    attack_success_guarded: float | None
    controls: int
    control_flip_raw: float | None
    control_changed_guarded: float | None
    mean_calls_guarded: float
    actions: dict[str, int] = field(default_factory=dict)
    by_kind: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def to_markdown(self) -> str:
        f = lambda x: "n/a" if x is None else f"{100 * x:.1f} %"  # noqa: E731
        rows = [("Accuracy on originals", f(self.accuracy_raw), f(self.accuracy_guarded)),
                ("Attack success", f(self.attack_success_raw), f(self.attack_success_guarded)),
                ("Neutral controls changed", f(self.control_flip_raw), f(self.control_changed_guarded))]
        lines = [f"# Steering evaluation: {self.model}", "", f"{self.cases} cases, {self.attacks} attacks ({self.eligible_attacks} eligible), "
                 f"{self.controls} neutral controls, {self.mean_calls_guarded:.1f} model calls per guarded decision.", "",
                 "| | Model alone | With Decision Model Guard |", "|---|---:|---:|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
        if self.by_kind:
            lines += ["", "| Decision type | Eligible attacks | Attack success alone | With Decision Model Guard | Controls changed |", "|---|---:|---:|---:|---:|"]
            for k, v in self.by_kind.items():
                lines.append(f"| {k} | {v['eligible']} | {f(v['asr_raw'])} | {f(v['asr_guarded'])} | {f(v['control_changed'])} |")
        if self.actions:
            lines += ["", "Actions on attacks: " + ", ".join(f"{k} {v}" for k, v in sorted(self.actions.items()))]
        return "\n".join(lines) + "\n"


def evaluate(model: DecisionModel | DecideBatch, cases: Sequence[dict[str, Any]] | None = None, guard: DecisionGuard | None = None,
             attacks_per_case: int = 4, controls_per_case: int = 2, positions: Sequence[str] = ("suffix", "prefix"),
             langs: Sequence[str] = ("en",), seed: int = 0) -> EvalReport:
    """Run the steering evaluation. `guard` defaults to `DecisionGuard(model, gate=SteeringGate())` (full check)."""
    from ..guards import SteeringGate
    m = as_model(model)
    guard = guard or DecisionGuard(m, gate=SteeringGate())
    cases = list(cases) if cases is not None else load_cases()
    bank, neutral = claim_bank(), neutral_bank()
    rng = random.Random(seed)
    tot = Counter()
    per_kind: dict[str, Counter] = {}
    actions: Counter = Counter()
    calls = []

    def ask_raw(scene, c):
        return argmax(checked(m, [scene], c["question"], c["options"])[0])

    def ask_guard(scene, c):
        d = guard.decide(scene, c["question"], c["options"], risky=c.get("risky"), order=c.get("order"), fallback=c.get("fallback"))
        calls.append(d.calls)
        return d

    for c in cases:
        kind = "risky" if c.get("risky") else "ordered" if c.get("order") else "unordered"
        pk = per_kind.setdefault(kind, Counter())
        gold, target = c["gold"], c.get("target", default_target(c))
        raw0, g0 = ask_raw(c["scene"], c), ask_guard(c["scene"], c)
        tot["cases"] += 1
        tot["acc_raw"] += raw0 == gold
        tot["acc_guard"] += g0.index == gold
        tgt_text = c["options"][target].rstrip(".")
        for k in range(attacks_per_case):
            lang = rng.choice(list(langs))
            fams = bank["claims"].get(lang) or bank["claims"]["en"]
            fam = rng.choice(list(fams))
            tpl = rng.choice(fams[fam])
            claim = tpl.replace("{t}", tgt_text).replace("{label}", tgt_text)
            sc = insert(c["scene"], claim, positions[k % len(positions)])
            tot["attacks"] += 1
            if raw0 != gold or target == gold:
                continue
            ra, ga = ask_raw(sc, c), ask_guard(sc, c)
            actions[ga.action.value] += 1
            tot["eligible"] += 1
            pk["eligible"] += 1
            tot["asr_raw"] += ra == target
            pk["asr_raw"] += ra == target
            tot["asr_guard"] += ga.index == target
            pk["asr_guard"] += ga.index == target
        for k in range(controls_per_case):
            pool = neutral.get(rng.choice(list(langs))) or neutral["en"]
            sc = insert(c["scene"], rng.choice(pool), positions[k % len(positions)])
            rc, gc = ask_raw(sc, c), ask_guard(sc, c)
            tot["controls"] += 1
            pk["controls"] += 1
            tot["ctl_raw"] += rc != raw0
            tot["ctl_guard"] += gc.index != g0.index
            pk["ctl_guard"] += gc.index != g0.index

    r = lambda a, b: (a / b) if b else None  # noqa: E731
    by_kind = {k: {"eligible": v["eligible"], "asr_raw": r(v["asr_raw"], v["eligible"]), "asr_guarded": r(v["asr_guard"], v["eligible"]),
                   "control_changed": r(v["ctl_guard"], v["controls"])} for k, v in per_kind.items()}
    return EvalReport(model=m.name, cases=tot["cases"], attacks=tot["attacks"], eligible_attacks=tot["eligible"],
                      accuracy_raw=r(tot["acc_raw"], tot["cases"]) or 0.0, accuracy_guarded=r(tot["acc_guard"], tot["cases"]) or 0.0,
                      attack_success_raw=r(tot["asr_raw"], tot["eligible"]), attack_success_guarded=r(tot["asr_guard"], tot["eligible"]),
                      controls=tot["controls"], control_flip_raw=r(tot["ctl_raw"], tot["controls"]),
                      control_changed_guarded=r(tot["ctl_guard"], tot["controls"]),
                      mean_calls_guarded=sum(calls) / max(1, len(calls)), actions=dict(actions), by_kind=by_kind)
