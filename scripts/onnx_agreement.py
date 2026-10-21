"""Agreement of the Wolf Defender Jev ONNX variants with the fp32 graph on decisions built from the bundled sample cases.

    pip install "decision-model-guard[onnx]"
    python scripts/onnx_agreement.py

Per case: the original scene, five planted claims (suffix, prefix, middle), two neutral sentences and the scene as markdown
and JSON, 120 decisions in all. Prints the share of decisions on which each variant picks the same option as fp32, per
decision type. The INT8 kernels differ between ONNX Runtime versions and CPUs, so measure on the setup you deploy: on
ONNX Runtime 1.23.2 int8_int4_embeddings agreed on 110 and mixed on 113, from 1.24 on 119 and 118.
"""
from __future__ import annotations

import random

import onnxruntime

from decision_model_guard.evaluate import claim_bank, default_target, insert, load_cases, neutral_bank
from decision_model_guard.guards.invariance import render
from decision_model_guard.models import WolfDefenderJev
from decision_model_guard.util import argmax


def decisions() -> list[tuple[str, str, list[str], str]]:
    bank, neutral, rng, out = claim_bank()["claims"]["en"], neutral_bank()["en"], random.Random(0), []
    for c in load_cases():
        kind = "risky" if c.get("risky") else f"{len(c['options'])} options"
        out.append((c["scene"], c["question"], c["options"], kind))
        target = c["options"][default_target(c)].rstrip(".")
        for k in range(5):
            tpl = rng.choice(bank[rng.choice(list(bank))])
            out.append((insert(c["scene"], tpl.replace("{t}", target).replace("{label}", target), ("suffix", "prefix", "middle")[k % 3]),
                        c["question"], c["options"], kind))
        out += [(insert(c["scene"], rng.choice(neutral)), c["question"], c["options"], kind) for _ in range(2)]
        out += [(render(c["scene"], f), c["question"], c["options"], kind) for f in ("markdown", "json")]
    return out


def main() -> None:
    items = decisions()
    answers = {}
    for variant in ("fp32", "fp16", "mixed", "int8_int4_embeddings"):
        m = WolfDefenderJev.from_pretrained(onnx=variant)
        answers[variant] = [argmax(m.decide_batch([s], q, o)[0]) for s, q, o, _ in items]
    print(f"ONNX Runtime {onnxruntime.__version__}, {len(items)} decisions")
    for variant in ("fp16", "mixed", "int8_int4_embeddings"):
        by: dict[str, list[int]] = {}
        for (_, _, _, kind), a, b in zip(items, answers["fp32"], answers[variant], strict=True):
            by.setdefault(kind, [0, 0])
            by[kind][0] += a == b
            by[kind][1] += 1
        same = sum(v[0] for v in by.values())
        print(f"{variant:22s} {same}/{len(items)} = {100 * same / len(items):.1f} %   " + ", ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(by.items())))


if __name__ == "__main__":
    main()
