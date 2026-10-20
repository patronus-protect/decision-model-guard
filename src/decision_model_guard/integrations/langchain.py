"""LangChain integration: the guard as a Runnable and as a tool-friendly function."""
from __future__ import annotations

from typing import Any

from ..pipeline import DecisionGuard
from ..scene import Scene
from ..util import require


def as_runnable(guard: DecisionGuard, include_scene: bool = False):
    """A LangChain Runnable. Input: {"scene", "question", "options", optional "risky"/"order"/"fallback"/"meta"}.
    Output: the guarded decision as a dict (answer, action, reasons, probabilities, ...)."""
    core = require("langchain_core.runnables", "langchain")

    def _run(inp: dict[str, Any]) -> dict[str, Any]:
        d = guard.decide(inp["scene"], inp["question"], inp["options"], risky=inp.get("risky"), order=inp.get("order"),
                         fallback=inp.get("fallback"), meta=inp.get("meta"))
        return d.to_dict(include_scene=include_scene)

    return core.RunnableLambda(_run, name="Decision Model Guard")


def decision_tool(guard: DecisionGuard, question: str, options: list[str], name: str = "guarded_decision", context: str | None = None,
                  **decide_kw):
    """A LangChain tool that takes untrusted text and returns the guarded answer for a fixed question. The input is always
    one untrusted block (header-looking lines in it are neutralized); `context` is trusted application context placed
    before it. To pass a whole scene, use `as_runnable`."""
    tools = require("langchain_core.tools", "langchain")

    def _tool(text: str) -> str:
        sc = Scene().system(context) if context else Scene()
        d = guard.decide(sc.user(text), question, options, **decide_kw)
        return f"{d.answer} (action: {d.action.value})"

    _tool.__doc__ = f"{question} Options: {', '.join(o.rstrip('.') for o in options)}. Input: the untrusted text."
    return tools.StructuredTool.from_function(_tool, name=name, description=_tool.__doc__)
