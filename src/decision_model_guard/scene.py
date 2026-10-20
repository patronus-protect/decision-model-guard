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
word "untrusted" is always untrusted. Once untrusted content has appeared, a trusted-looking header is untrusted too;
the only exception is a `PROPOSED ACTION` or `TOOL CALL` block that is the last block of a text that starts with a
header. A text that starts without a header is untrusted content as a whole: no header inside it is trusted.

A `Scene` built with the builder methods keeps the trust of every block as it was added; the pipeline uses that
directly and never re-parses it. The builder writes headers that parse back to the same blocks, and in untrusted text
it indents lines that look like a header, so content cannot open a block of its own.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

TRUSTED_HEADERS: tuple[str, ...] = ("SYSTEM CONTEXT", "AVAILABLE TOOLS", "USER GOAL", "TOOL CALL", "PROPOSED ACTION", "SECURITY POLICY")
# a block with one of these headers after untrusted content is still trusted when it is the last block of a parsed text
ACTION_HEADERS: tuple[str, ...] = ("PROPOSED ACTION", "TOOL CALL")
# untrusted blocks with these headers are split into sentences; any other untrusted header is erased as a whole first
CONTENT_HEADERS: tuple[str, ...] = ("USER PROMPT", "TOOL OUTPUT", "RETRIEVED CONTENT", "EMAIL", "DOCUMENT", "WEB PAGE", "FILE",
                                    "AGENT MEMORY", "CUSTOMER MESSAGE", "MESSAGE", "TICKET")
HEADER = re.compile(r"^\[([^\[\]\n]{1,120})\]\s*$", re.M)
HEADER_LIKE = re.compile(r"^[ \t]*\[[^\[\]\n]{1,120}\][ \t]*$", re.M)   # also indented ones: forged headers inside content
SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_MARK = " | untrusted"


@dataclass
class Block:
    """A block of a scene with character offsets into the scene text."""

    header: str          # text inside the brackets, "" for text before the first header
    start: int           # offset of the block, header line included
    end: int
    body_start: int      # offset of the body
    trusted: bool
    builder: bool = False  # added through Scene: the application wrote the header

    def body(self, scene: str) -> str:
        return scene[self.body_start: self.end].strip("\n")

    @property
    def name(self) -> str:
        return self.header.split("|")[0].strip().upper()


def normalize_newlines(text: str) -> str:
    """CRLF and CR line endings as LF, so every entry point (library, CLI, server) reads the same text."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_blocks(scene: str, trusted_headers: Sequence[str] = TRUSTED_HEADERS, allow_action: bool = True) -> list[Block]:
    """Split a scene text into blocks and mark which ones are trusted. Text before the first header is untrusted, and
    then no later block is trusted. `allow_action=False` drops the exception for a final action block."""
    heads = list(HEADER.finditer(scene))
    blocks: list[Block] = []
    seen_untrusted = False
    if not heads or heads[0].start() > 0:
        end = heads[0].start() if heads else len(scene)
        if scene[:end].strip():
            blocks.append(Block("", 0, end, 0, False))
            seen_untrusted, allow_action = True, False
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(scene)
        h = m.group(1).strip()
        name = h.split("|")[0].strip().upper()
        trusted = name in trusted_headers and "untrusted" not in h.lower()
        if trusted and seen_untrusted and not (allow_action and name in ACTION_HEADERS and i == len(heads) - 1):
            trusted = False
        seen_untrusted |= not trusted
        blocks.append(Block(h, m.start(), end, m.end(), trusted))
    return blocks


def content_size(scene: str, blocks: Iterable[Block]) -> int:
    """Non-space characters in the bodies of the untrusted blocks."""
    return sum(len(re.sub(r"\s", "", scene[b.body_start: b.end])) for b in blocks if not b.trusted)


def _sections(scene: str, a: int, b: int) -> list[tuple[int, int]]:
    """Spans from each header-looking line inside scene[a:b] to the next one (or the end of the body)."""
    heads = [m for m in HEADER_LIKE.finditer(scene, a, b) if m.start() >= a]
    out = []
    for i, m in enumerate(heads):
        s = m.start() + len(m.group(0)) - len(m.group(0).lstrip(" \t"))
        e = heads[i + 1].start() if i + 1 < len(heads) else b
        e = s + len(scene[s:e].rstrip())
        out.append((s, e))
    return out


def units(scene: str, blk: Block, window: int = 3, content: int | None = None) -> list[tuple[int, int]]:
    """Character spans of an untrusted block that the gate may erase, edge units first.

    Units are: the whole block with its header line (and the header line with the first paragraph) when the header is
    not a content header and was not written by the `Scene` builder; for every header-looking line inside the body,
    that line and the text up to the next such line (a forged block); paragraphs; windows of 1 to `window` consecutive
    sentences. A paragraph or window that holds all untrusted content is never returned, because erasing all of it
    tells nothing. `content` is the number of non-space characters of all untrusted blocks of the scene (default: this
    block), so a block that is one untrusted block among others can be erased as a whole."""
    out: list[tuple[int, int]] = []
    body = scene[blk.body_start: blk.end]
    off = blk.body_start
    paras = [(m.start() + off, m.end() + off) for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*$)", body, re.S)]
    if blk.header and blk.name not in CONTENT_HEADERS and not blk.builder:
        out.append((blk.start, blk.end))
        if paras:
            out.append((blk.start, paras[0][1]))
    for s in _sections(scene, blk.body_start, blk.end):
        if s not in out:
            out.append(s)
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
    total = content if content is not None else len(re.sub(r"\s", "", body))
    whole = lambda s: len(re.sub(r"\s", "", scene[s[0]:s[1]])) >= total  # noqa: E731
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


def neutralize(text: str) -> str:
    """Untrusted text with every line that would parse as a block header indented by one space."""
    return HEADER.sub(lambda m: " " + m.group(0), text)


def clean_header(header: str, trusted: bool) -> str:
    """A header that parses back as written: brackets become parentheses, line breaks spaces, at most 120 characters.
    An untrusted block with a trusted-looking name gets " | untrusted", which also survives the length limit."""
    h = " ".join(header.replace("[", "(").replace("]", ")").split())
    if not h:
        raise ValueError("a block needs a header")
    if not trusted and (h.split("|")[0].strip().upper() in TRUSTED_HEADERS or "untrusted" in h.lower()):
        core = re.sub(r"\s*\|\s*untrusted\s*$", "", h, flags=re.I)
        h = core[: 120 - len(_MARK)].rstrip(" |") + _MARK
    return h[:120].rstrip()


@dataclass
class Scene:
    """A scene built from blocks. Use `Scene.from_text` to parse one, or the builder methods to build one.

    The pipeline takes the trust of every block from the builder (`trusted=` of `add`), not from the text. Headers are
    cleaned so the text parses back to the same blocks, and header-looking lines in untrusted text are indented."""

    blocks: list[tuple[str, str, bool]] = field(default_factory=list)   # (header, text, trusted)

    def add(self, header: str, text: str, trusted: bool | None = None) -> Scene:
        """Append a block. Without `trusted`, a block is trusted when its header names one of TRUSTED_HEADERS."""
        if trusted is None:
            trusted = header.split("|")[0].strip().upper() in TRUSTED_HEADERS and "untrusted" not in header.lower()
        text = normalize_newlines(text).strip("\n")
        self.blocks.append((clean_header(header, trusted), text if trusted else neutralize(text), bool(trusted)))
        return self

    def system(self, text: str) -> Scene:
        """Trusted application context."""
        return self.add("SYSTEM CONTEXT", text, True)

    def goal(self, text: str) -> Scene:
        """The user's goal for an agent (trusted)."""
        return self.add("USER GOAL", text, True)

    def tools(self, text: str) -> Scene:
        """The tools an agent may call (trusted)."""
        return self.add("AVAILABLE TOOLS", text, True)

    def policy(self, text: str) -> Scene:
        """A security policy (trusted)."""
        return self.add("SECURITY POLICY", text, True)

    def action(self, text: str) -> Scene:
        """The action an agent proposes (trusted)."""
        return self.add("PROPOSED ACTION", text, True)

    def user(self, text: str) -> Scene:
        """The user's message (untrusted)."""
        return self.add("USER PROMPT", text, False)

    def tool_output(self, text: str, tool: str = "tool") -> Scene:
        """Output of a tool call (untrusted)."""
        return self.add(f"TOOL OUTPUT | {tool} | untrusted", text, False)

    def retrieved(self, text: str, source: str = "web") -> Scene:
        """A retrieved page or document (untrusted)."""
        return self.add(f"RETRIEVED CONTENT | {source} | untrusted", text, False)

    def untrusted(self, header: str, text: str) -> Scene:
        """Any other untrusted block, e.g. untrusted("EMAIL", text)."""
        h = header if "untrusted" in header.lower() else f"{header} | untrusted"
        return self.add(h, text, False)

    def to_text(self) -> str:
        return "\n\n".join(f"[{h}]\n{t}" for h, t, _ in self.blocks)

    def layout(self) -> tuple[str, list[Block]]:
        """The scene text and its blocks with the trust each block was added with (no parsing)."""
        out: list[Block] = []
        pos = 0
        for h, t, trusted in self.blocks:
            if out:
                out[-1].end = pos + 2
                pos += 2
            n = len(h) + 3 + len(t)
            out.append(Block(h, pos, pos + n, pos + len(h) + 2, trusted, builder=True))
            pos += n
        return self.to_text(), out

    def __str__(self) -> str:
        return self.to_text()

    @classmethod
    def from_text(cls, text: str, trusted_headers: Sequence[str] = TRUSTED_HEADERS) -> Scene:
        """Parse a scene text (trust as `parse_blocks` decides it)."""
        sc = cls()
        text = normalize_newlines(text)
        for b in parse_blocks(text, trusted_headers):
            sc.add(b.header or "USER PROMPT", b.body(text), b.trusted)
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


def scene_blocks(scene: str | Scene, trusted_headers: Sequence[str] = TRUSTED_HEADERS, allow_action: bool = True) -> tuple[str, list[Block]]:
    """The text of a scene and its blocks: a `Scene` keeps the trust its builder gave each block, a string is parsed
    (after normalising line endings)."""
    if isinstance(scene, Scene):
        return scene.layout()
    text = normalize_newlines(str(scene))
    return text, parse_blocks(text, trusted_headers, allow_action)


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
