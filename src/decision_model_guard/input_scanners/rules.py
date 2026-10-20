"""Rule-based scanners: fast, dependency-free checks for patterns that steer decision models."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from ..scene import TRUSTED_HEADERS, replace_spans
from ..types import Finding, ScanResult

RANGES = ((0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x2064), (0x2066, 0x2069), (0xFEFF, 0xFEFF), (0xE0000, 0xE007F), (0x00AD, 0x00AD))


def _hidden(ch: str) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in RANGES) or (unicodedata.category(ch) in ("Cf", "Co", "Cn") and ch not in "\n\t")


def _emoji_joiner(text: str, i: int) -> bool:
    """U+200D between two emoji builds one emoji (family, profession): harmless."""
    if text[i] != "\u200d":
        return False
    prev = text[i - 1] if i > 0 else ""
    nxt = text[i + 1] if i + 1 < len(text) else ""
    return bool(prev and nxt) and unicodedata.category(prev) in ("So", "Mn", "Sk") and unicodedata.category(nxt) == "So"


FLAGS = re.compile("\U0001F3F4(?:" + "|".join("".join(chr(0xE0000 + ord(c)) for c in t) for t in ("gbeng", "gbsct", "gbwls")) + ")\U000E007F")
SOFT_HYPHEN = "\u00ad"


def hidden_spans(text: str) -> list[tuple[int, int]]:
    """Spans of the invisible characters InvisibleText reports. The tag characters of the subdivision flag emoji
    (England, Scotland, Wales) are part of a visible emoji and not reported."""
    flags = {i for m in FLAGS.finditer(text) for i in range(m.start() + 1, m.end())}
    return [(i, i + 1) for i, ch in enumerate(text) if _hidden(ch) and i not in flags and not _emoji_joiner(text, i)]


def visible(text: str) -> str:
    """The text without the invisible characters InvisibleText removes (used to match flagged text across scans)."""
    return replace_spans(text, hidden_spans(text)) if any(_hidden(ch) for ch in text) else text


class InvisibleText:
    """Zero-width characters, bidirectional controls, Unicode tag characters and other invisible format characters.

    They can hide instructions from human reviewers while the model still reads them. Sanitizing removes them. Soft
    hyphens (U+00AD, ordinary hyphenation) are removed too, so later scanners see the plain word, but on their own they
    do not make the text invalid (score 0.3)."""

    name = "InvisibleText"
    RANGES = RANGES

    def __init__(self, remove: bool = True):
        self.remove = remove

    def scan(self, text: str) -> ScanResult:
        """Find invisible characters and remove them from the sanitized text."""
        spans = hidden_spans(text)
        if not spans:
            return ScanResult(self.name, text, True, 0.0)
        strong = [x for x in spans if text[x[0]] != SOFT_HYPHEN]
        score = 1.0 if strong else 0.3
        shown = strong or spans
        findings = [Finding(self.name, score, f"{len(shown)} invisible character(s)" if strong else f"{len(spans)} soft hyphen(s)",
                            shown[0][0], shown[-1][1])]
        return ScanResult(self.name, replace_spans(text, spans) if self.remove else text, not strong, score, findings)


class ForgedTrustBlock:
    """Trusted-looking structure inside untrusted content: block headers that name a trusted source, review or verdict
    stamps in square brackets, and chat role markers (system/assistant turns, [INST], <<SYS>>). A decision model may read
    them as context from the application instead of as part of the content. Speaker labels such as "Assistant:" score
    0.6 and stay below the default threshold of 0.7 on their own. A header finding covers the forged block: the header
    line and the text up to the next header-looking line, so erasing or neutralizing it removes what the block claims."""

    name = "ForgedTrustBlock"
    WORDS = r"trusted|verified|security|review|system|policy|admin|approved|guardrail|moderation|classifier|verdict|audit|compliance|official|signed|freigabe|geprüft|sicherheit"
    HEADER = re.compile(r"^[ \t]*\[([^\[\]\r\n]{1,120})\][ \t]*\r?$", re.M)
    STRONG = re.compile(r"<\|(?:im_start|im_end|system|assistant|user|eot_id|start_header_id)\|>|\[/?INST\]|<<\s*/?SYS\s*>>|</?system>|"
                        r"^[ \t]*#{2,}\s*(?:system|instruction|assistant)\b|^[ \t]*(?:system|developer)\s*(?:prompt|message)?\s*:", re.I | re.M)
    WEAK = re.compile(r"^[ \t]*(?:assistant|ai|bot|moderator|reviewer)\s*:", re.I | re.M)

    def __init__(self, threshold: float = 0.7, trusted_headers: Sequence[str] = TRUSTED_HEADERS, neutralize: bool = False):
        self.threshold, self.trusted, self.neutralize = threshold, tuple(h.upper() for h in trusted_headers), neutralize
        self._words = re.compile(self.WORDS, re.I)

    def scan(self, text: str) -> ScanResult:
        """Find trusted-looking headers, chat role markers and speaker labels."""
        findings: list[Finding] = []
        heads = list(self.HEADER.finditer(text))
        for i, m in enumerate(heads):
            h = m.group(1)
            name = h.split("|")[0].strip().upper()
            if name in self.trusted or self._words.search(h):
                a = m.start() + len(m.group(0)) - len(m.group(0).lstrip(" \t"))
                b = heads[i + 1].start() if i + 1 < len(heads) else len(text)
                b = a + len(text[a:b].rstrip())
                findings.append(Finding(self.name, 1.0, f"trusted-looking block header [{h.strip()}]", a, b, text[a:b]))
        for m in self.STRONG.finditer(text):
            findings.append(Finding(self.name, 1.0, "chat role marker", m.start(), m.end(), m.group(0).strip()))
        for m in self.WEAK.finditer(text):
            findings.append(Finding(self.name, 0.6, "speaker label that may imitate another party", m.start(), m.end(), m.group(0).strip()))
        score = max((f.score for f in findings), default=0.0)
        sanitized = text
        if self.neutralize and findings:
            sanitized = replace_spans(text, [(f.start, f.end) for f in findings if f.start is not None], "")
        return ScanResult(self.name, sanitized, score < self.threshold, score, findings)


class VerdictTokens:
    """Bare verdicts written into the content: `verdict: safe`, `label=benign`, `"is_injection": false`, `Einstufung:
    harmlos`. Content never decides its own label, so these are steering attempts or, at best, noise."""

    name = "VerdictTokens"
    KEYS = (r"verdict|label|classification|risk[ _-]?(?:score|level)|is[ _-]?(?:injection|malicious|safe|spam|phishing|attack)|"
            r"safe[ _-]?to[ _-]?process|approved|approval|moderation|freigabe|einstufung|urteil")
    # everyday keys ("Order status: no tracking yet", "Result: 1 item"): only a verdict word counts
    GENERIC = r"class|decision|outcome|status|result|entscheidung|bewertung|ergebnis"
    VERDICTS = (r"benign|safe|clean|allow(?:ed)?|approve[d]?|pass(?:ed)?|harmless|legit(?:imate)?|verified|"
                r"not[ _-]?(?:malicious|spam|phishing|an?[ _-]?attack|injection)|green|freigegeben|harmlos|genehmigt|sicher|unbedenklich|kein[ _-]?angriff")
    VALUES = rf"yes|no|true|false|ok|okay|low|none|0|1|{VERDICTS}"
    PLAIN = re.compile(rf"\b(?:{KEYS})\b\s*[:=]\s*[\"'`]?(?:{VALUES})\b|\b(?:{GENERIC})\b\s*[:=]\s*[\"'`]?(?:{VERDICTS})\b", re.I)
    JSON = re.compile(rf"[\"'](?:{KEYS})[\"']\s*:\s*(?:true|false|[\"'](?:{VALUES})[\"']|0|1)|[\"'](?:{GENERIC})[\"']\s*:\s*[\"'](?:{VERDICTS})[\"']", re.I)

    def __init__(self, threshold: float = 0.5, redact: bool = False):
        self.threshold, self.redact = threshold, redact

    def scan(self, text: str) -> ScanResult:
        """Find verdict tokens such as verdict: safe or label=benign."""
        findings = [Finding(self.name, 1.0, "verdict token in content", m.start(), m.end(), m.group(0))
                    for rx in (self.PLAIN, self.JSON) for m in rx.finditer(text)]
        score = 1.0 if findings else 0.0
        sanitized = replace_spans(text, [(f.start, f.end) for f in findings]) if self.redact and findings else text
        return ScanResult(self.name, sanitized, score < self.threshold, score, findings)


class Regex:
    """Generic regular expressions. `is_blocked=True`: a match makes the content invalid; False: content must match."""

    name = "Regex"

    def __init__(self, patterns: Sequence[str], is_blocked: bool = True, redact: bool = False, flags: int = re.I, name: str | None = None):
        self.patterns = [re.compile(p, flags) for p in patterns]
        self.is_blocked, self.redact = is_blocked, redact
        if name:
            self.name = name

    def scan(self, text: str) -> ScanResult:
        """Match the patterns (invalid on a match, or without one when is_blocked=False)."""
        hits = [(m.start(), m.end(), m.group(0)) for rx in self.patterns for m in rx.finditer(text)]
        findings = [Finding(self.name, 1.0, "pattern match", a, b, t) for a, b, t in hits]
        if self.is_blocked:
            valid = not hits
            sanitized = replace_spans(text, [(a, b) for a, b, _ in hits], "[REDACTED]") if self.redact and hits else text
        else:
            valid = bool(hits)
            sanitized = text
        return ScanResult(self.name, sanitized, valid, 0.0 if valid else 1.0, findings)


class BanSubstrings:
    """Substrings that must not appear in untrusted content (case-insensitive by default)."""

    name = "BanSubstrings"

    def __init__(self, substrings: Sequence[str], case_sensitive: bool = False, redact: bool = False, match_all: bool = False):
        self.subs, self.cs, self.redact, self.match_all = list(substrings), case_sensitive, redact, match_all

    def scan(self, text: str) -> ScanResult:
        """Invalid when a banned substring (or all of them, with match_all) occurs."""
        hay = text if self.cs else text.lower()
        spans = []
        found = set()
        for s in self.subs:
            needle = s if self.cs else s.lower()
            start = hay.find(needle)
            while start >= 0 and needle:
                spans.append((start, start + len(needle)))
                found.add(s)
                start = hay.find(needle, start + 1)
        hit = (len(found) == len(self.subs)) if self.match_all else bool(found)
        findings = [Finding(self.name, 1.0, f"banned substring '{s}'") for s in sorted(found)]
        sanitized = replace_spans(text, spans, "[REDACTED]") if self.redact and hit else text
        return ScanResult(self.name, sanitized, not hit, 1.0 if hit else 0.0, findings if hit else [])


class TokenLimit:
    """Length limit. Without a tokenizer, tokens are approximated as whitespace-separated words x 1.3. Sanitizing keeps
    the head and the tail of over-long content (the tail is where appended claims sit)."""

    name = "TokenLimit"

    def __init__(self, limit: int = 4096, tokenizer: str | None = None, truncate: bool = True):
        self.limit, self.truncate = limit, truncate
        self._tok = None
        if tokenizer:
            from ..util import require
            self._tok = require("transformers", "hf").AutoTokenizer.from_pretrained(tokenizer)

    def count(self, text: str) -> int:
        """Token count (the tokenizer's, or words x 1.3)."""
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return int(len(text.split()) * 1.3) + 1

    def scan(self, text: str) -> ScanResult:
        """Invalid above the limit; the sanitized text keeps head and tail."""
        n = self.count(text)
        if n <= self.limit:
            return ScanResult(self.name, text, True, 0.0)
        sanitized = text
        if self.truncate:
            keep = max(1, int(len(text) * self.limit / n))
            sanitized = text[: keep // 3] + "\n...\n" + text[-(keep - keep // 3):]
        return ScanResult(self.name, sanitized, False, min(1.0, n / self.limit - 1), [Finding(self.name, 1.0, f"{n} tokens > limit {self.limit}")])
