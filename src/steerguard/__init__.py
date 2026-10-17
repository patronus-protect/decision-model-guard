"""SteerGuard: guardrails for decision models.

Decision models (Jev-like models, classifiers, LLM judges) read a ticket, an email or a tool output and pick an answer.
Text inside that content can claim the answer ("this case was already approved") and steer the decision. SteerGuard
scans the untrusted content, runs the steering gate (erase-and-check) around any decision model, adds consistency and
confidence checks and turns everything into an answer plus an action.

    from steerguard import DecisionGuard
    from steerguard.models import WolfDefenderJev

    guard = DecisionGuard.default(WolfDefenderJev.from_pretrained())
    d = guard.decide(scene, "Does the content try to give the agent new instructions?",
                     ["No, it is only data.", "Yes, an injected instruction."], risky={1})
    d.answer, d.action, d.reasons
"""
from .audit import AuditLogger
from .config import build_guard
from .guards import Confidence, OptionOrder, SteeringGate
from .models import DecisionModel, FunctionModel
from .pipeline import BOUNDARY_NOTE, DecisionGuard, Request, guard_decision, scan_content
from .policy import Policy
from .scene import TRUSTED_HEADERS, Scene, parse_blocks, units
from .types import Action, Finding, GuardedDecision, GuardReport, Kind, ScanResult
from .version import __version__

__all__ = ["__version__", "DecisionGuard", "Request", "guard_decision", "scan_content", "build_guard", "Policy", "AuditLogger",
           "SteeringGate", "OptionOrder", "Confidence", "Scene", "TRUSTED_HEADERS", "parse_blocks", "units", "DecisionModel",
           "FunctionModel", "Action", "Kind", "Finding", "ScanResult", "GuardReport", "GuardedDecision", "BOUNDARY_NOTE"]
