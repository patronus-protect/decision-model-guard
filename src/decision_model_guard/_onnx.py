"""ONNX Runtime backend. The Patronus models (Wolf Defender, the Decision Steering Gate, Wolf Defender Jev) ship four ONNX
variants in one layout, so a variant name is enough to pick a graph:

    fp32                  onnx/onnx_fp32/model.onnx              native export
    fp16                  onnx/onnx_fp16/model_fp16.onnx         half size
    mixed                 onnx/onnx_mixed/model_mixed.onnx       INT8 weights, FP16 embeddings
    int8_int4_embeddings  onnx/int8_int4_embeddings/model.onnx   INT8 weights, INT4 embeddings, the smallest

A path to an .onnx file works as well. Needs the `onnx` extra (onnxruntime, tokenizers, numpy, huggingface_hub), no torch.
"""
from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .util import require

VARIANTS = {"fp32": "onnx/onnx_fp32/model.onnx", "fp16": "onnx/onnx_fp16/model_fp16.onnx",
            "mixed": "onnx/onnx_mixed/model_mixed.onnx", "int8_int4_embeddings": "onnx/int8_int4_embeddings/model.onnx"}
DEFAULT_VARIANT = "int8_int4_embeddings"
INT8_VARIANTS = ("mixed", "int8_int4_embeddings")

_SESSIONS: dict[tuple, Any] = {}
_LOCK = threading.Lock()


def resolve(repo_or_path: str, variant: str, files: Sequence[str] = ("config.json", "tokenizer.json"),
            revision: str | None = None) -> tuple[Path, Path]:
    """(folder with config and tokenizer, graph path). From the hub only the listed files and the chosen graph are fetched.
    A variant that names an existing local file (absolute or relative to the working directory) is used as it is."""
    rel = VARIANTS.get(variant, variant)
    if not rel.endswith(".onnx"):
        raise ValueError(f"onnx must be one of {sorted(VARIANTS)} or a path to an .onnx file, got {variant!r}")
    local = Path(os.path.expanduser(rel)).resolve() if variant not in VARIANTS and os.path.isfile(os.path.expanduser(rel)) else None
    if os.path.isdir(repo_or_path):
        folder = Path(repo_or_path)
        graph = local or (Path(rel) if os.path.isabs(rel) else folder / rel)
    else:
        hub = require("huggingface_hub", "onnx")
        paths = [Path(hub.hf_hub_download(repo_or_path, f, revision=revision)) for f in files]
        folder = paths[0].parent
        graph = local or (Path(rel) if os.path.isabs(rel) else Path(hub.hf_hub_download(repo_or_path, rel, revision=revision)))
    if not graph.is_file():
        raise FileNotFoundError(f"ONNX graph not found: {graph}")
    return folder, graph


def runtime_version() -> tuple[int, int]:
    """(major, minor) of the installed ONNX Runtime."""
    ort = require("onnxruntime", "onnx")
    major, minor = (re.findall(r"\d+", ort.__version__) + ["0", "0"])[:2]
    return int(major), int(minor)


def session(graph: Path, threads: int | None = None, providers: Sequence[str] | None = None):
    """One InferenceSession per graph, thread count and providers, shared within the process."""
    ort = require("onnxruntime", "onnx")
    key = (str(graph), threads, tuple(providers or ()))
    with _LOCK:
        if key not in _SESSIONS:
            opts = ort.SessionOptions()
            if threads:
                opts.intra_op_num_threads = threads
            _SESSIONS[key] = ort.InferenceSession(str(graph), opts, providers=list(providers or ["CPUExecutionProvider"]))
        return _SESSIONS[key]


def clear_sessions() -> None:
    with _LOCK:
        _SESSIONS.clear()


def tokenizer(folder: Path):
    """tokenizers.Tokenizer without the padding and truncation settings a saved tokenizer.json may carry."""
    tk = require("tokenizers", "onnx")
    tok = tk.Tokenizer.from_file(str(folder / "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    return tok


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def pad(seqs: Sequence[Sequence[int]], pad_id: int = 0):
    np = require("numpy", "onnx")
    L = max(1, max((len(s) for s in seqs), default=1))
    ids = np.full((len(seqs), L), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, : len(s)] = 1
    return ids, mask


def softmax(logits):
    np = require("numpy", "onnx")
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class OnnxClassifier:
    """A sequence classifier exported to ONNX (inputs input_ids, attention_mask; output logits), with the tokenizer and
    the label names of its repository. Same interface as the PyTorch backend of the content scanners."""

    def __init__(self, repo_or_path: str, variant: str = DEFAULT_VARIANT, threads: int | None = None,
                 providers: Sequence[str] | None = None, revision: str | None = None):
        folder, graph = resolve(repo_or_path, variant, revision=revision)
        self.session = session(graph, threads, providers)
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tok = tokenizer(folder)
        cfg = read_json(folder / "config.json")
        self.id2label = {int(k): str(v) for k, v in (cfg.get("id2label") or {}).items()}
        pad_token = self.tok.token_to_id("<pad>")
        self.pad_id = cfg.get("pad_token_id", pad_token if pad_token is not None else 0) or 0
        self._max_length: int | None = None
        self._plain = tokenizer(folder)                                       # untruncated, for head/tail and chunking

    def ids(self, text: str) -> list[int]:
        return self._plain.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._plain.decode(list(ids))

    def _encode(self, texts: Sequence[str], max_length: int) -> list[list[int]]:
        """Token ids with the special tokens of the template, truncated like transformers' `truncation=True`."""
        if self._max_length != max_length:
            self.tok.enable_truncation(max_length)
            self._max_length = max_length
        return [e.ids for e in self.tok.encode_batch(list(texts))]

    def probs(self, texts: Sequence[str], max_length: int, batch_size: int, pos: int) -> list[float]:
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            ids, mask = pad(self._encode(texts[i: i + batch_size], max_length), self.pad_id)
            feed = {k: v for k, v in (("input_ids", ids), ("attention_mask", mask)) if k in self.inputs}
            out += softmax(self.session.run(None, feed)[0])[:, pos].tolist()
        return out
