"""Audit log: one JSON line per guarded decision."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from typing import Any

from .types import GuardedDecision
from .util import sha256


class AuditLogger:
    """Appends one JSON line per decision. By default only hashes of the scene, question and culprit are stored, so the
    log holds no customer content; set `include_text=True` to keep the text for debugging.

    `meta` is stored as it is (keep customer data out of it); larger than `max_meta_bytes` it is replaced by a note
    with its size. Plain SHA-256 hashes of short or predictable texts can be confirmed by guessing; `hash_key_env`
    names an environment variable with a secret key, and the log then stores HMAC-SHA256 values (`*_hmac`) instead.
    At `max_bytes` the file is rotated to `.1`, `.2`, ... keeping `backup_count` old files."""

    def __init__(self, path: str, include_text: bool = False, max_bytes: int | None = 50_000_000, backup_count: int = 5,
                 max_meta_bytes: int = 4096, hash_key_env: str | None = None):
        if backup_count < 0 or max_meta_bytes < 0:
            raise ValueError("backup_count and max_meta_bytes must not be negative")
        self.path, self.include_text, self.max_bytes = path, include_text, max_bytes
        self.backup_count, self.max_meta_bytes = backup_count, max_meta_bytes
        self._key = None
        if hash_key_env:
            key = os.environ.get(hash_key_env)
            if not key:
                raise ValueError(f"audit.hash_key_env names {hash_key_env!r}, but that environment variable is empty or not set")
            self._key = key.encode("utf-8")
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def _hash(self, text: str) -> str:
        return hmac.new(self._key, text.encode("utf-8"), hashlib.sha256).hexdigest() if self._key else sha256(text)

    def record(self, d: GuardedDecision, scene: str, question: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """The JSON record of one decision: hashes (or text with include_text), actions, scores, meta."""
        meta = meta or {}
        size = len(json.dumps(meta, ensure_ascii=False, default=str))
        if size > self.max_meta_bytes:
            meta = {"meta_dropped": f"{size} bytes > max_meta_bytes {self.max_meta_bytes}"}
        rec: dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "id": d.id, "action": d.action.value, "kind": d.kind.value,
                               "answer": d.answer, "raw_answer": d.raw_answer, "changed": d.changed, "probs": [round(p, 6) for p in d.probs],
                               "raw_probs": [round(p, 6) for p in d.raw_probs], "reasons": d.reasons, "calls": d.calls,
                               "latency_ms": round(d.latency_ms, 2), "options": d.options,
                               "scans": {s.scanner: {"valid": s.valid, "score": round(s.score, 6)} for s in d.scans},
                               "guards": {g.guard: g.triggered for g in d.guards}, "meta": meta}
        if self.include_text:
            rec.update(scene=scene, question=question, culprit=d.culprit)
        else:
            k = "hmac" if self._key else "sha256"
            rec.update({f"scene_{k}": self._hash(scene), f"question_{k}": self._hash(question),
                        f"culprit_{k}": self._hash(d.culprit) if d.culprit else None})
            if d.culprit:                                   # reasons quote the culprit; keep content out of the log
                rec["reasons"] = [r.replace(repr(d.culprit), "<culprit>").replace(d.culprit, "<culprit>") for r in d.reasons]
        return rec

    def _rotate(self) -> None:
        for i in range(self.backup_count - 1, 0, -1):
            if os.path.exists(f"{self.path}.{i}"):
                os.replace(f"{self.path}.{i}", f"{self.path}.{i + 1}")
        if self.backup_count:
            os.replace(self.path, self.path + ".1")
        else:
            os.remove(self.path)

    def log(self, d: GuardedDecision, scene: str, question: str, meta: dict[str, Any] | None = None) -> None:
        """Append the record of one decision (thread-safe, rotating at max_bytes)."""
        line = json.dumps(self.record(d, scene, question, meta), ensure_ascii=False, default=str)
        with self._lock:
            if self.max_bytes and os.path.exists(self.path) and os.path.getsize(self.path) > self.max_bytes:
                self._rotate()
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
