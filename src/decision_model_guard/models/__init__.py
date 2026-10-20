"""Decision model adapters. Every adapter implements `decide_batch(scenes, question, options) -> list[list[float]]`."""
from .base import DecideBatch, DecisionModel, FunctionModel, ModelOutputError, as_model, checked
from .http import OpenAICompatible, SystemOneHTTP
from .local import CausalLMJudge, HFClassifier
from .wolf_defender_jev import WolfDefenderJev, WolfDefenderJevONNX
from .workers_ai import Clairic, Clef

__all__ = ["DecideBatch", "DecisionModel", "FunctionModel", "ModelOutputError", "as_model", "checked", "SystemOneHTTP", "OpenAICompatible", "CausalLMJudge",
           "HFClassifier", "WolfDefenderJev", "WolfDefenderJevONNX", "Clef", "Clairic"]
