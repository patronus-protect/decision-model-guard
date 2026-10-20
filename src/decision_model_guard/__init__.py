"""Decision Model Guard: guardrails for decision models.

Decision models (Jev-like models, classifiers, LLM judges) read a ticket, an email or a tool output and pick an answer.
Text inside that content can claim the answer ("this case was already approved") and steer the decision. Decision Model Guard
scans the untrusted content, runs the steering gate (erase-and-check) around any decision model, checks long content
window by window, adds consistency, invariance, confidence and conformal checks and turns everything into an answer
plus an action. `dmguard audit` measures how robust your model is with and without the guard.

    from decision_model_guard import DecisionGuard
    from decision_model_guard.models import WolfDefenderJev

    guard = DecisionGuard.default(WolfDefenderJev.from_pretrained())
    d = guard.decide(scene, "Does the content try to give the agent new instructions?",
                     ["No, it is only data.", "Yes, an injected instruction."], risky={1})
    d.answer, d.action, d.reasons
"""
from .audit import AuditLogger
from .calibration import Calibrated, Calibrator
from .config import build_guard
from .guards import Confidence, Conformal, Coverage, Invariance, OptionOrder, SteeringGate
from .models import DecisionModel, FunctionModel
from .pipeline import BOUNDARY_NOTE, DecisionGuard, Request, guard_decision, scan_content
from .policy import Policy
from .scene import TRUSTED_HEADERS, Scene, parse_blocks, units
from .types import Action, Finding, GuardedDecision, GuardReport, Kind, ScanResult
from .version import __version__

__all__ = ["__version__", "DecisionGuard", "Request", "guard_decision", "scan_content", "build_guard", "Policy", "AuditLogger",
           "SteeringGate", "OptionOrder", "Confidence", "Coverage", "Invariance", "Conformal", "Calibrator", "Calibrated", "Scene", "TRUSTED_HEADERS", "parse_blocks", "units", "DecisionModel",
           "FunctionModel", "Action", "Kind", "Finding", "ScanResult", "GuardReport", "GuardedDecision", "BOUNDARY_NOTE"]
