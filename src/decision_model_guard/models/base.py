"""The one contract every decision model satisfies."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

DecideBatch = Callable[[Sequence[str], str, Sequence[str]], list[list[float]]]


@runtime_checkable
class DecisionModel(Protocol):
    """A decision model reads scenes and returns one probability per option for each scene.

    `decide_batch(scenes, question, options)` must return a list (one entry per scene) of lists (one probability per
    option, in the order of `options`). Probabilities need not sum exactly to 1; the pipeline normalises them."""

    name: str

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]: ...


class FunctionModel:
    """Wrap any callable `fn(scenes, question, options) -> list[list[float]]` as a decision model."""

    def __init__(self, fn: DecideBatch, name: str = "function"):
        self.fn, self.name = fn, name

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        return [list(map(float, p)) for p in self.fn(list(scenes), question, list(options))]


def as_model(model: DecisionModel | DecideBatch, name: str | None = None) -> DecisionModel:
    """Accept a model object or a bare decide_batch callable."""
    if hasattr(model, "decide_batch"):
        return model  # type: ignore[return-value]
    if callable(model):
        return FunctionModel(model, name or getattr(model, "__name__", "function"))
    raise TypeError("model must have decide_batch(scenes, question, options) or be such a callable")
