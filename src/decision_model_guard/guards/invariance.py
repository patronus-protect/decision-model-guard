"""Invariance: the same decision asked in slightly different ways should give the same answer.

Decision models are sensitive to the serialization of the input and to the wording of the options: a model trained on
tagged blocks can answer differently when the same scene arrives as JSON, and "No" can win where "No, it is only data."
loses. Invariance re-asks the model with the scene in other formats and with reworded options (and with your own
question or option variants) and flags the decision when the answer changes. The policy escalates it by default.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

from ..models.base import DecisionModel
from ..scene import TRUSTED_HEADERS, parse_blocks
from ..types import GuardReport, Kind
from ..util import argmax, normalize

FORMATS = ("markdown", "kv", "json", "compact")


def render(scene: str, fmt: str, trusted_headers: Sequence[str] = TRUSTED_HEADERS) -> str:
    """The scene in another serialization with the same content: markdown headings, key-value lines, JSON or compact."""
    if fmt == "compact":
        return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", scene)).strip()
    blocks = [(b.header or "TEXT", b.body(scene)) for b in parse_blocks(scene, trusted_headers)]
    if fmt == "markdown":
        return "\n\n".join(f"### {h}\n{body}" for h, body in blocks)
    if fmt == "kv":
        return "\n".join(f"{h}: {' '.join(body.split())}" for h, body in blocks)
    if fmt == "json":
        obj: dict[str, str] = {}
        for h, body in blocks:
            key, k = h, 2
            while key in obj:
                key, k = f"{h} ({k})", k + 1
            obj[key] = body
        return json.dumps(obj, ensure_ascii=False, indent=1)
    raise ValueError(f"unknown format {fmt!r}; use one of {FORMATS}")


def _short(o: str) -> str:
    head = re.split(r"[,.;:]\s", o.strip(), maxsplit=1)[0].strip().rstrip(".,;:")
    return head if head and head != o.strip() else o


OPTION_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "short": _short,                                  # "No, it is only data." -> "No"
    "lower": lambda o: o.lower(),
    "plain": lambda o: o.strip().rstrip(".!?"),
    "quoted": lambda o: f'"{o.strip()}"',
}


class Invariance:
    """Re-ask the decision with equivalent inputs and flag it when the answer changes.

    formats: scene serializations to try ("markdown", "kv", "json", "compact"); one batched call for all of them.
    option_transforms: built-in rewordings of all options ("short", "lower", "plain", "quoted"); one call each, skipped
    when a rewording changes nothing or makes two options identical.
    option_variants: your own alternative option lists (same length and order); question_variants: alternative questions.
    tolerance: number of changed answers that is still accepted."""

    name = "Invariance"

    def __init__(self, formats: Sequence[str] = ("markdown",), option_transforms: Sequence[str] = ("short",),
                 option_variants: Sequence[Sequence[str]] = (), question_variants: Sequence[str] = (), tolerance: int = 0,
                 trusted_headers: Sequence[str] = TRUSTED_HEADERS):
        bad = [f for f in formats if f not in FORMATS] + [t for t in option_transforms if t not in OPTION_TRANSFORMS]
        if bad:
            raise ValueError(f"unknown formats or option transforms: {bad}")
        self.formats, self.option_transforms = tuple(formats), tuple(option_transforms)
        self.option_variants, self.question_variants = [list(v) for v in option_variants], list(question_variants)
        self.tolerance, self.trusted = tolerance, tuple(trusted_headers)

    def check(self, model: DecisionModel, scene: str, question: str, options: Sequence[str], probs: Sequence[float], kind: Kind,
              *, raw_probs: Sequence[float] | None = None, **_ctx) -> GuardReport:
        # compare with the model's own answer on this scene: after the gate corrected a decision, the guarded answer
        # differs from what the model says on the scene by design, and that is not instability
        base = argmax(list(raw_probs if raw_probs is not None else probs))
        changed, checks, calls = [], 0, 0
        variants = [(f, render(scene, f, self.trusted)) for f in self.formats]
        variants = [(f, s) for f, s in variants if s != scene]
        if variants:
            P = model.decide_batch([s for _, s in variants], question, options)
            calls += len(variants)
            for (f, _), p in zip(variants, P, strict=True):
                checks += 1
                if argmax(normalize(p)) != base:
                    changed.append(f"format:{f}")
        option_sets = [(f"options:{t}", [OPTION_TRANSFORMS[t](o) for o in options]) for t in self.option_transforms]
        option_sets += [(f"options:variant{i + 1}", v) for i, v in enumerate(self.option_variants)]
        for label, opts in option_sets:
            if len(opts) != len(options):
                raise ValueError(f"{label}: an option variant needs {len(options)} options")
            if list(opts) == list(options) or len(set(opts)) < len(opts):
                continue
            checks += 1
            calls += 1
            if argmax(normalize(model.decide_batch([scene], question, opts)[0])) != base:
                changed.append(label)
        for i, q in enumerate(self.question_variants):
            checks += 1
            calls += 1
            if argmax(normalize(model.decide_batch([scene], q, options)[0])) != base:
                changed.append(f"question:variant{i + 1}")
        return GuardReport(self.name, len(changed) > self.tolerance, len(changed) / max(1, checks), calls,
                           {"checks": checks, "changed_under": changed})
