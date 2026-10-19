"""Small helpers: lazy imports, device selection, hashing, ids."""
from __future__ import annotations

import hashlib
import importlib
import logging
import os
import uuid
from typing import Any

log = logging.getLogger("decision_model_guard")


def require(module: str, extra: str) -> Any:
    """Import an optional dependency or raise a clear error naming the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(f"'{module}' is required for this feature: pip install 'decision-model-guard[{extra}]'") from e


def pick_device(device: str | None = None) -> str:
    """'cpu', 'cuda' or 'mps'. Default: DMGUARD_DEVICE, else cuda if available, else cpu (MPS only on request)."""
    if device:
        return device
    env = os.environ.get("DMGUARD_DEVICE")
    if env:
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def argmax(xs: list[float]) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def normalize(xs: list[float]) -> list[float]:
    s = float(sum(xs))
    return [float(x) / s for x in xs] if s > 0 else [1.0 / len(xs)] * len(xs)
