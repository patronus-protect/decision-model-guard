"""Toy decision models for offline tests (importable from configs as helpers:...)."""
from __future__ import annotations

CLAIM_WORDS = ("approved", "eligible", "already verified", "confirmed it is safe", "marked safe", "harmless")


def refund_toy(scenes, question, options):
    """Ordered refund decision [verify, reject, authorize]: believes any approval claim, otherwise verifies."""
    out = []
    for s in scenes:
        low = s.lower()
        if any(w in low for w in CLAIM_WORDS):
            out.append([0.05, 0.05, 0.9][: len(options)])
        else:
            out.append([0.8, 0.1, 0.1][: len(options)])
    return out


def injection_toy(scenes, question, options):
    """Risky decision [benign, injection]: sees an injection when 'forward all' appears, unless a 'safe' claim is present."""
    out = []
    for s in scenes:
        low = s.lower()
        attack = "forward all" in low or "ignore your task" in low
        steered = any(w in low for w in ("confirmed it is safe", "marked safe", "harmless"))
        p = 0.95 if attack and not steered else (0.3 if attack else 0.05)
        out.append([1 - p, p])
    return out


def routing_toy(scenes, question, options):
    """Unordered routing [billing, technical, general]: follows 'route to technical' claims and the word 'charged'."""
    out = []
    for s in scenes:
        low = s.lower()
        if "technical team" in low:
            out.append([0.1, 0.8, 0.1])
        elif "charged" in low or "payment" in low:
            out.append([0.8, 0.1, 0.1])
        elif "crash" in low:
            out.append([0.1, 0.8, 0.1])
        else:
            out.append([0.1, 0.1, 0.8])
    return out


def position_biased(scenes, question, options):
    """Always prefers whatever option is listed first: an order-sensitive model."""
    n = len(options)
    return [[0.7] + [0.3 / (n - 1)] * (n - 1) for _ in scenes]


def fake_detector(texts):
    """Claim scores: 0.95 for units with a claim word, 0.02 otherwise."""
    return [0.95 if any(w in t.lower() for w in CLAIM_WORDS + ("technical team",)) else 0.02 for t in texts]


class ToyModelClass:
    """Class form for config tests."""

    name = "toy-class"

    def __init__(self, kind: str = "refund"):
        self.fn = {"refund": refund_toy, "injection": injection_toy, "routing": routing_toy}[kind]

    def decide_batch(self, scenes, question, options):
        return self.fn(scenes, question, options)
