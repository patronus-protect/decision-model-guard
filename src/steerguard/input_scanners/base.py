"""Scanner contract (LLM Guard style) and a shared cache for transformer models."""
from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable

from ..types import ScanResult
from ..util import pick_device, require


@runtime_checkable
class Scanner(Protocol):
    """A content scanner looks at untrusted text and returns a ScanResult: sanitized text, valid flag, risk score."""

    name: str

    def scan(self, text: str) -> ScanResult: ...


_MODELS: dict[tuple[str, str, str | None], tuple[Any, Any]] = {}
_LOCK = threading.Lock()


def load_classifier(model_id: str, device: str | None = None, subfolder: str | None = None):
    """(tokenizer, model) for a sequence classifier, loaded once per process and shared between scanners and the gate."""
    dev = pick_device(device)
    key = (model_id, dev, subfolder)
    with _LOCK:
        if key not in _MODELS:
            tf = require("transformers", "hf")
            kw = {"subfolder": subfolder} if subfolder else {}
            tok = tf.AutoTokenizer.from_pretrained(model_id)
            model = tf.AutoModelForSequenceClassification.from_pretrained(model_id, **kw).to(dev).eval()
            _MODELS[key] = (tok, model)
        return _MODELS[key]


def clear_model_cache() -> None:
    with _LOCK:
        _MODELS.clear()
