"""Decision model adapters. Every adapter implements `decide_batch(scenes, question, options) -> list[list[float]]`."""
from .base import DecideBatch, DecisionModel, FunctionModel, as_model
from .http import OpenAICompatible, SystemOneHTTP
from .local import CausalLMJudge, HFClassifier
from .wolf_defender_jev import WolfDefenderJev

__all__ = ["DecideBatch", "DecisionModel", "FunctionModel", "as_model", "SystemOneHTTP", "OpenAICompatible", "CausalLMJudge",
           "HFClassifier", "WolfDefenderJev"]
