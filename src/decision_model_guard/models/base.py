"""The one contract every decision model satisfies."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

DecideBatch = Callable[[Sequence[str], str, Sequence[str]], list[list[float]]]


class ModelOutputError(RuntimeError):
    """A model returned something other than the contract allows (wrong shape, NaN, negative scores, all zeros)."""


@runtime_checkable
class DecisionModel(Protocol):
    """A decision model reads scenes and returns one probability per option for each scene.

    `decide_batch(scenes, question, options)` must return a list (one entry per scene) of lists (one probability per
    option, in the order of `options`). Probabilities must be finite and non-negative with a positive sum; they need
    not sum exactly to 1, the pipeline normalises them. Anything else raises `ModelOutputError`."""

    name: str

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]: ...


class FunctionModel:
    """Wrap any callable `fn(scenes, question, options) -> list[list[float]]` as a decision model."""

    def __init__(self, fn: DecideBatch, name: str = "function"):
        self.fn, self.name = fn, name

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        return check_probs(self.fn(list(scenes), question, list(options)), len(scenes), len(options), self.name)


def as_model(model: DecisionModel | DecideBatch, name: str | None = None) -> DecisionModel:
    """Accept a model object or a bare decide_batch callable."""
    if hasattr(model, "decide_batch"):
        return model  # type: ignore[return-value]
    if callable(model):
        return FunctionModel(model, name or getattr(model, "__name__", "function"))
    raise TypeError("model must have decide_batch(scenes, question, options) or be such a callable")


def check_probs(out, n_scenes: int, n_options: int, name: str = "model") -> list[list[float]]:
    """Validate a decide_batch result and normalise every row to sum 1. Raises ModelOutputError instead of turning
    broken output into a uniform distribution (which would pick option 0 with action allow)."""
    try:
        rows = [[float(x) for x in p] for p in out]
    except (TypeError, ValueError):
        raise ModelOutputError(f"{name}: decide_batch must return one list of numbers per scene") from None
    if len(rows) != n_scenes:
        raise ModelOutputError(f"{name}: {len(rows)} rows for {n_scenes} scenes")
    for r in rows:
        if len(r) != n_options:
            raise ModelOutputError(f"{name}: {len(r)} scores for {n_options} options")
        if not all(math.isfinite(x) and x >= 0 for x in r) or sum(r) <= 0:
            raise ModelOutputError(f"{name}: scores must be finite, non-negative and not all zero, got {[round(x, 4) for x in r]}")
    return [[x / sum(r) for x in r] for r in rows]


def checked(model: DecisionModel, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
    """`model.decide_batch` with its output validated and normalised (see `check_probs`)."""
    scenes, options = list(scenes), list(options)
    return check_probs(model.decide_batch(scenes, question, options), len(scenes), len(options), getattr(model, "name", type(model).__name__))
