"""Model-based scanners: prompt injection (Wolf Defender) and the claim detector of the Decision Steering Gate."""
from __future__ import annotations

from collections.abc import Sequence

from ..scene import Block, replace_spans, sentences, units
from ..types import Finding, ScanResult
from .base import load_classifier

CLAIM_DETECTOR = "patronus-studio/decision-steering-gate"
WOLF_DEFENDER_SMALL = "patronus-studio/wolf-defender-prompt-injection-small"
WOLF_DEFENDER = "patronus-studio/wolf-defender-prompt-injection"


def _positive_index(model, hint: str) -> int:
    labels = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
    for i, v in labels.items():
        if hint in v:
            return i
    return 1


def _probs(tok, model, texts: Sequence[str], max_length: int, batch_size: int, pos: int) -> list[float]:
    import torch
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            x = tok(list(texts[i: i + batch_size]), truncation=True, max_length=max_length, padding=True, pad_to_multiple_of=32,
                    return_tensors="pt").to(model.device)
            out += torch.softmax(model(**x).logits.float(), -1)[:, pos].tolist()
    return out


class PromptInjection:
    """Prompt injection and jailbreak detection with Wolf Defender (mmBERT, 15 languages, 2,048 tokens).

    match: "full" scores the whole text (long texts keep head and tail), "sentences" scores every sentence and reports the
    flagged ones, "chunks" scores overlapping token windows of `chunk_tokens`. The risk score is the highest
    injection probability. Default model: Wolf Defender Small; use WOLF_DEFENDER for the larger model."""

    name = "PromptInjection"

    def __init__(self, model: str = WOLF_DEFENDER_SMALL, threshold: float = 0.5, match: str = "full", max_length: int = 2048,
                 chunk_tokens: int = 512, stride: int = 128, batch_size: int = 16, device: str | None = None):
        if match not in ("full", "sentences", "chunks"):
            raise ValueError("match must be 'full', 'sentences' or 'chunks'")
        self.model_id, self.threshold, self.match = model, threshold, match
        self.max_length, self.chunk_tokens, self.stride, self.batch_size, self.device = max_length, chunk_tokens, stride, batch_size, device
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            tok, model = load_classifier(self.model_id, self.device)
            self._loaded = (tok, model, _positive_index(model, "inject"))
        return self._loaded

    def _keep_head_tail(self, tok, text: str) -> str:
        ids = tok.encode(text, add_special_tokens=False)
        lim = self.max_length - 4
        if len(ids) <= lim:
            return text
        return tok.decode(ids[: lim // 3]) + "\n" + tok.decode(ids[-(lim - lim // 3):])

    def scan(self, text: str) -> ScanResult:
        if not text.strip():
            return ScanResult(self.name, text, True, 0.0)
        tok, model, pos = self._load()
        findings: list[Finding] = []
        if self.match == "full":
            score = _probs(tok, model, [self._keep_head_tail(tok, text)], self.max_length, 1, pos)[0]
            if score >= self.threshold:
                findings.append(Finding(self.name, score, "prompt injection"))
        elif self.match == "sentences":
            spans = sentences(text) or [(0, len(text))]
            ps = _probs(tok, model, [text[a:b] for a, b in spans], self.max_length, self.batch_size, pos)
            score = max(ps)
            findings = [Finding(self.name, p, "prompt injection", a, b, text[a:b]) for (a, b), p in zip(spans, ps, strict=True) if p >= self.threshold]
        else:
            ids = tok.encode(text, add_special_tokens=False)
            step = max(1, self.chunk_tokens - self.stride)
            chunks = [tok.decode(ids[i: i + self.chunk_tokens]) for i in range(0, max(1, len(ids) - self.stride), step)] or [text]
            ps = _probs(tok, model, chunks, self.max_length, self.batch_size, pos)
            score = max(ps)
            findings = [Finding(self.name, p, f"prompt injection in chunk {i}") for i, p in enumerate(ps) if p >= self.threshold]
        return ScanResult(self.name, text, score < self.threshold, float(score), findings)


class ClaimDetector:
    """Claims about the decision inside the content: approvals, prior verdicts, policy references, workflow status,
    instructions to the classifier, forged review notes. Scores every unit the steering gate would erase (paragraphs and
    windows of 1 to 3 sentences) with the claim detector of the Decision Steering Gate (mmBERT-small, 15 languages).

    `redact=True` removes the flagged units from the sanitized text. The gate reuses `score()` to pick what to erase."""

    name = "ClaimDetector"

    def __init__(self, model: str = CLAIM_DETECTOR, threshold: float = 0.3, redact: bool = False, window: int = 3,
                 max_length: int = 128, batch_size: int = 64, device: str | None = None):
        self.model_id, self.threshold, self.redact, self.window = model, threshold, redact, window
        self.max_length, self.batch_size, self.device = max_length, batch_size, device
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            tok, model = load_classifier(self.model_id, self.device)
            self._loaded = (tok, model, _positive_index(model, "claim"))
        return self._loaded

    def score(self, texts: Sequence[str]) -> list[float]:
        """Claim probability for each text (unit)."""
        if not texts:
            return []
        tok, model, pos = self._load()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        ps = _probs(tok, model, [texts[i] for i in order], self.max_length, self.batch_size, pos)
        out = [0.0] * len(texts)
        for i, p in zip(order, ps, strict=True):
            out[i] = p
        return out

    def __call__(self, texts: Sequence[str]) -> list[float]:
        return self.score(texts)

    def scan(self, text: str) -> ScanResult:
        if not text.strip():
            return ScanResult(self.name, text, True, 0.0)
        spans = units(text, Block("USER PROMPT", 0, len(text), 0, False), self.window)
        if not spans:
            spans = [(0, len(text))]
        ps = self.score([text[a:b].strip() for a, b in spans])
        flagged = sorted(((p, b - a, (a, b)) for (a, b), p in zip(spans, ps, strict=True) if p >= self.threshold), key=lambda x: (-x[0], x[1]))
        keep: list[tuple[float, tuple[int, int]]] = []
        for p, _, (a, b) in flagged:                       # smallest strongest units, no overlaps
            if all(b <= c or a >= d for _, (c, d) in keep):
                keep.append((p, (a, b)))
        findings = [Finding(self.name, p, "claim about the decision", a, b, text[a:b].strip()) for p, (a, b) in sorted(keep, key=lambda x: x[1])]
        score = max(ps) if ps else 0.0
        sanitized = replace_spans(text, [s for _, s in keep]) if self.redact and keep else text
        return ScanResult(self.name, sanitized.strip() if self.redact else sanitized, score < self.threshold, float(score), findings)
