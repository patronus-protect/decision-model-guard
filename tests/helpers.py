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


def gullible(scenes, question, options):
    """Believes any option it finds named in the content (the last mention wins); otherwise answers option 0."""
    out = []
    for s in scenes:
        low = s.lower()
        hits = [(low.rfind(o.lower().rstrip(".")), i) for i, o in enumerate(options) if o.lower().rstrip(".") in low]
        k = max(hits)[1] if hits else 0
        out.append([0.9 if i == k else 0.1 / (len(options) - 1) for i in range(len(options))])
    return out


class Truncating:
    """Risky decision [benign, injection] by a model that reads only the first and the last `chars` characters, like a
    model with a context limit. Sees an injection when 'forward all invoices' is visible."""

    name = "truncating"

    def __init__(self, chars: int = 400, steerable: bool = False):
        self.chars, self.steerable = chars, steerable
        self.context_tokens = (2 * chars) // 4

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def decide_batch(self, scenes, question, options):
        out = []
        for s in scenes:
            seen = (s[: self.chars] + " " + s[-self.chars:]).lower() if len(s) > 2 * self.chars else s.lower()
            attack = "forward all invoices" in seen
            if self.steerable and "marked safe" in seen:
                attack = False
            out.append([0.1, 0.9] if attack else [0.9, 0.1])
        return out


def format_sensitive(scenes, question, options):
    """Answers option 0 for the tagged format and the last option when the scene arrives in another format."""
    out = []
    for s in scenes:
        other = s.lstrip().startswith(("{", "###")) or ": " in s.split("\n", 1)[0]
        k = len(options) - 1 if other else 0
        out.append([0.9 if i == k else 0.1 / (len(options) - 1) for i in range(len(options))])
    return out


def wording_sensitive(scenes, question, options):
    """Picks the first option only while it is written as a full sentence: rewording the options changes the answer."""
    w = [1.5] * len(options)
    w[0] = 2.0 if options[0].strip().endswith(".") else 1.0
    t = sum(w)
    return [[x / t for x in w] for _ in scenes]
