"""Audit log: one JSON line per guarded decision."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from .types import GuardedDecision
from .util import sha256


class AuditLogger:
    """Appends one JSON line per decision. By default only hashes of the scene, question and culprit are stored, so the
    log holds no customer content; set `include_text=True` to keep the text for debugging. Rotates at `max_bytes`."""

    def __init__(self, path: str, include_text: bool = False, max_bytes: int | None = 50_000_000):
        self.path, self.include_text, self.max_bytes = path, include_text, max_bytes
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def record(self, d: GuardedDecision, scene: str, question: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        rec: dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "id": d.id, "action": d.action.value, "kind": d.kind.value,
                               "answer": d.answer, "raw_answer": d.raw_answer, "changed": d.changed, "probs": [round(p, 6) for p in d.probs],
                               "raw_probs": [round(p, 6) for p in d.raw_probs], "reasons": d.reasons, "calls": d.calls,
                               "latency_ms": round(d.latency_ms, 2), "options": d.options,
                               "scans": {s.scanner: {"valid": s.valid, "score": round(s.score, 6)} for s in d.scans},
                               "guards": {g.guard: g.triggered for g in d.guards}, "meta": meta or {}}
        if self.include_text:
            rec.update(scene=scene, question=question, culprit=d.culprit)
        else:
            rec.update(scene_sha256=sha256(scene), question_sha256=sha256(question), culprit_sha256=sha256(d.culprit) if d.culprit else None)
            if d.culprit:                                   # reasons quote the culprit; keep content out of the log
                rec["reasons"] = [r.replace(repr(d.culprit), "<culprit>").replace(d.culprit, "<culprit>") for r in d.reasons]
        return rec

    def log(self, d: GuardedDecision, scene: str, question: str, meta: dict[str, Any] | None = None) -> None:
        line = json.dumps(self.record(d, scene, question, meta), ensure_ascii=False)
        with self._lock:
            if self.max_bytes and os.path.exists(self.path) and os.path.getsize(self.path) > self.max_bytes:
                os.replace(self.path, self.path + ".1")
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
