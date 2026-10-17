"""Scenes: the text a decision model reads, split into trusted and untrusted blocks.

Text format (the one Jev-like security models and the steering gate use):

    [SYSTEM CONTEXT]
    Refund desk of an online shop.

    [USER PROMPT]
    My parcel never arrived.

    [TOOL OUTPUT | email.read | untrusted]
    ...

A block starts with a header line in square brackets. Blocks whose header names one of `TRUSTED_HEADERS` are trusted;
everything else is content an attacker may write: the user prompt, tool output, retrieved pages, text before the first
header and any block with an unknown header (a forged "[SECURITY REVIEW | trusted]" included). A header containing the
word "untrusted" is always untrusted, and a trusted-looking header that appears after untrusted content is only accepted
for the action that follows it.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

TRUSTED_HEADERS: tuple[str, ...] = ("SYSTEM CONTEXT", "AVAILABLE TOOLS", "USER GOAL", "TOOL CALL", "PROPOSED ACTION", "SECURITY POLICY")
# untrusted blocks with these headers are split into sentences; any other untrusted header is erased as a whole first
CONTENT_HEADERS: tuple[str, ...] = ("USER PROMPT", "TOOL OUTPUT", "RETRIEVED CONTENT", "EMAIL", "DOCUMENT", "WEB PAGE", "FILE",
                                    "AGENT MEMORY", "CUSTOMER MESSAGE", "MESSAGE", "TICKET")
HEADER = re.compile(r"^\[([^\[\]\n]{1,120})\]\s*$", re.M)
SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


@dataclass
class Block:
    """A block of a scene with character offsets into the scene text."""

    header: str          # text inside the brackets, "" for text before the first header
    start: int           # offset of the block, header line included
    end: int
    body_start: int      # offset of the body
    trusted: bool

    def body(self, scene: str) -> str:
        return scene[self.body_start: self.end].strip("\n")


def parse_blocks(scene: str, trusted_headers: Sequence[str] = TRUSTED_HEADERS) -> list[Block]:
    """Split a scene into blocks and mark which ones are trusted."""
    heads = list(HEADER.finditer(scene))
    blocks: list[Block] = []
    if not heads or heads[0].start() > 0:
        end = heads[0].start() if heads else len(scene)
        if scene[:end].strip():
            blocks.append(Block("", 0, end, 0, False))
    seen_untrusted = False
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(scene)
        h = m.group(1).strip()
        name = h.split("|")[0].strip().upper()
        trusted = name in trusted_headers and "untrusted" not in h.lower()
        if trusted and seen_untrusted and name not in ("PROPOSED ACTION", "TOOL CALL"):
            trusted = False
        seen_untrusted |= not trusted
        blocks.append(Block(h, m.start(), end, m.end(), trusted))
    return blocks


def units(scene: str, blk: Block, window: int = 3) -> list[tuple[int, int]]:
    """Character spans of a content block that the gate may erase, edge units first.

    Units are: the whole block when its header is unknown (plus header line and first paragraph), paragraphs, and windows
    of 1 to `window` consecutive sentences. A unit that covers 90 % or more of the block's content is never returned,
    because erasing all content tells nothing."""
    out: list[tuple[int, int]] = []
    body = scene[blk.body_start: blk.end]
    off = blk.body_start
    paras = [(m.start() + off, m.end() + off) for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*$)", body, re.S)]
    name = blk.header.split("|")[0].strip().upper()
    if blk.header and name not in TRUSTED_HEADERS + CONTENT_HEADERS:
        out.append((blk.start, blk.end))
        if paras:
            out.append((blk.start, paras[0][1]))
    sents: list[tuple[int, int]] = []
    for a, b in paras:
        pos = a
        for m in SENT.finditer(scene, a, b):
            if scene[pos: m.start()].strip():
                sents.append((pos, m.start()))
            pos = m.end()
        if scene[pos:b].strip():
            sents.append((pos, b))
    wins = [(sents[i][0], sents[min(i + w, len(sents)) - 1][1]) for w in range(1, window + 1) for i in range(len(sents) - w + 1)]
    edge = lambda s: min(abs(s[0] - blk.body_start), abs(blk.end - s[1]))  # noqa: E731
    content = len(re.sub(r"\s", "", body))
    whole = lambda s: len(re.sub(r"\s", "", scene[s[0]:s[1]])) >= 0.9 * content  # noqa: E731
    seen = set(out)
    for s in sorted(paras, key=edge) + sorted(wins, key=lambda s: (edge(s), s[1] - s[0])):
        if s not in seen and not whole(s):
            seen.add(s)
            out.append(s)
    return out


def erase(scene: str, a: int, b: int) -> str:
    """The scene with characters a..b removed and the whitespace around the gap tidied."""
    left, right = scene[:a].rstrip(" \t"), scene[b:].lstrip(" \t")
    sep = "" if left.endswith("\n") or right.startswith("\n") or not left else " "
    return re.sub(r"\n{3,}", "\n\n", left + sep + right).strip()


def sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans of a plain text (same splitting rule as the gate)."""
    spans, pos = [], 0
    for m in SENT.finditer(text):
        if text[pos: m.start()].strip():
            spans.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    return spans


@dataclass
class Scene:
    """A scene built from blocks. Use `Scene.from_text` to parse one, or the `add_*` helpers to build one."""

    blocks: list[tuple[str, str, bool]] = field(default_factory=list)   # (header, text, trusted)

    def add(self, header: str, text: str, trusted: bool | None = None) -> Scene:
        name = header.split("|")[0].strip().upper()
        if trusted is None:
            trusted = name in TRUSTED_HEADERS and "untrusted" not in header.lower()
        self.blocks.append((header, text.strip("\n"), trusted))
        return self

    def system(self, text: str) -> Scene:
        return self.add("SYSTEM CONTEXT", text, True)

    def goal(self, text: str) -> Scene:
        return self.add("USER GOAL", text, True)

    def tools(self, text: str) -> Scene:
        return self.add("AVAILABLE TOOLS", text, True)

    def policy(self, text: str) -> Scene:
        return self.add("SECURITY POLICY", text, True)

    def action(self, text: str) -> Scene:
        return self.add("PROPOSED ACTION", text, True)

    def user(self, text: str) -> Scene:
        return self.add("USER PROMPT", text, False)

    def tool_output(self, text: str, tool: str = "tool") -> Scene:
        return self.add(f"TOOL OUTPUT | {tool} | untrusted", text, False)

    def retrieved(self, text: str, source: str = "web") -> Scene:
        return self.add(f"RETRIEVED CONTENT | {source} | untrusted", text, False)

    def untrusted(self, header: str, text: str) -> Scene:
        h = header if "untrusted" in header.lower() else f"{header} | untrusted"
        return self.add(h, text, False)

    def to_text(self) -> str:
        return "\n\n".join(f"[{h}]\n{t}" for h, t, _ in self.blocks)

    def __str__(self) -> str:
        return self.to_text()

    @classmethod
    def from_text(cls, text: str, trusted_headers: Sequence[str] = TRUSTED_HEADERS) -> Scene:
        sc = cls()
        for b in parse_blocks(text, trusted_headers):
            sc.blocks.append((b.header or "USER PROMPT", b.body(text), b.trusted))
        return sc

    @classmethod
    def from_message(cls, message: str, context: str | None = None, header: str = "USER PROMPT") -> Scene:
        """Shortcut for the common case: an optional trusted context plus one untrusted message."""
        sc = cls()
        if context:
            sc.system(context)
        return sc.add(header, message, False)


def as_text(scene: str | Scene) -> str:
    return scene.to_text() if isinstance(scene, Scene) else str(scene)


def untrusted_spans(scene: str, trusted_headers: Sequence[str] = TRUSTED_HEADERS) -> list[tuple[int, int]]:
    """Character spans of the bodies of all untrusted blocks."""
    return [(b.body_start, b.end) for b in parse_blocks(scene, trusted_headers) if not b.trusted]


def replace_spans(text: str, spans: Iterable[tuple[int, int]], repl: str = "") -> str:
    """Replace non-overlapping spans (any order) by `repl`."""
    out, pos = [], 0
    for a, b in sorted(spans):
        if a < pos:
            continue
        out.append(text[pos:a])
        out.append(repl)
        pos = b
    out.append(text[pos:])
    return "".join(out)
