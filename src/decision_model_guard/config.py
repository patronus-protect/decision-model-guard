"""Build a DecisionGuard from a YAML or dict config.

    model:
      type: wolf_defender_jev            # wolf_defender_jev | clef | systemone_http | openai_compatible | causal_lm | hf_classifier | callable
      device: cpu
      # onnx: int8_int4_embeddings       # ONNX Runtime instead of PyTorch: fp32 | fp16 | mixed | int8_int4_embeddings
    scanners:
      - type: InvisibleText
      - type: ForgedTrustBlock
      - type: VerdictTokens
      - type: PromptInjection
        threshold: 0.5
      - type: ClaimDetector
        threshold: 0.3
        # onnx: int8_int4_embeddings     # the 96 MB ONNX variant, no torch needed
    gate:
      selector: detector                 # all | detector
      scope: unordered                   # all | unordered
      threshold: 0.3
    guards:
      - type: Coverage                   # long content: check the parts the model would cut off, window by window
      - type: Invariance                 # same scene as markdown, shorter option wording: the answer must not change
        formats: [markdown]
        option_transforms: [short]
      - type: Confidence
        min_confidence: 0.55
      - type: OptionOrder
        permutations: reverse
      # - type: Conformal                # abstain unless the conformal set has one option (dmguard calibrate)
    # calibration: calibration.json      # temperature + conformal threshold from your labelled cases
    policy:
      unstable_action: fallback
      scanner_action: report
      block_scanners: [PromptInjection]
    audit:
      path: logs/decisions.jsonl
      include_text: false
    use_sanitized: true
    boundary_note: false
"""
from __future__ import annotations

import copy
import importlib
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

from . import input_scanners as S
from .audit import AuditLogger
from .calibration import Calibrated, Calibrator
from .guards import Confidence, Conformal, Coverage, Invariance, OptionOrder, SteeringGate
from .models import CausalLMJudge, Clef, DecisionModel, HFClassifier, OpenAICompatible, SystemOneHTTP, WolfDefenderJev, as_model
from .pipeline import DecisionGuard
from .policy import Policy

SCANNERS = {c.__name__: c for c in (S.InvisibleText, S.ForgedTrustBlock, S.VerdictTokens, S.Regex, S.BanSubstrings, S.TokenLimit,
                                     S.PromptInjection, S.ClaimDetector)}
GUARDS = {"Confidence": Confidence, "OptionOrder": OptionOrder, "Coverage": Coverage, "Invariance": Invariance, "Conformal": Conformal}


def _import(target: str):
    mod, _, attr = target.partition(":")
    if not attr:
        raise ValueError("callable target must look like 'package.module:name'")
    if os.getcwd() not in sys.path:           # like uvicorn's --app-dir: modules next to the config resolve from the CLI too
        sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(mod), attr)


def build_model(cfg: dict[str, Any]) -> DecisionModel:
    cfg = dict(cfg)
    t = cfg.pop("type")
    if t == "callable":                       # your own model: a decide_batch function, or a class/factory returning a model
        obj = _import(cfg.pop("target"))
        name = cfg.pop("name", None)
        if isinstance(obj, type) or cfg:
            obj = obj(**cfg)
        return as_model(obj, name=name)
    if t == "wolf_defender_jev":
        return WolfDefenderJev.from_pretrained(cfg.pop("repo", "patronus-studio/wolf-defender-jev"), **cfg)
    if t == "systemone_http":
        return SystemOneHTTP(**cfg)
    if t == "clef":                           # Cloudflare Workers AI: variant clef | clef-flash, account_id / proxy_url
        return Clef(**cfg)
    if t == "openai_compatible":
        return OpenAICompatible(**cfg)
    if t == "causal_lm":
        return CausalLMJudge(**cfg)
    if t == "hf_classifier":
        return HFClassifier(**cfg)
    raise ValueError(f"unknown model type {t!r}")


def build_scanner(cfg: dict[str, Any]):
    cfg = dict(cfg)
    t = cfg.pop("type")
    if t not in SCANNERS:
        raise ValueError(f"unknown scanner {t!r}; available: {sorted(SCANNERS)}")
    return SCANNERS[t](**cfg)


def build_guard(config: dict[str, Any] | str | Path, model: DecisionModel | None = None) -> DecisionGuard:
    """DecisionGuard from a config dict or a YAML file path. Pass `model` to override the configured model."""
    if isinstance(config, (str, Path)):
        config = yaml.safe_load(Path(config).read_text())
    cfg = copy.deepcopy(config or {})
    m = model if model is not None else build_model(cfg["model"]) if "model" in cfg else None
    if m is None:
        raise ValueError("config needs a 'model' section (or pass model=...)")
    calibrator = Calibrator.load(cfg["calibration"]) if cfg.get("calibration") else None
    if calibrator is not None:                                   # calibrated probabilities everywhere in the pipeline
        m = Calibrated(m, calibrator)
    scanners = [build_scanner(s) for s in cfg.get("scanners", [])]
    gate = None
    if cfg.get("gate") is not False:
        g = dict(cfg.get("gate") or {})
        if g.get("selector") == "detector":
            det = next((s for s in scanners if isinstance(s, S.ClaimDetector)), None)
            if det is None:
                extra = {k: g[k] for k in ("device", "onnx") if k in g}
                det = S.ClaimDetector(threshold=g.get("threshold", 0.3), **extra)
            g["detector"] = det
        g.pop("device", None)
        g.pop("onnx", None)
        gate = SteeringGate(**g)
    guards = []
    for gc in cfg.get("guards", []):
        gc = dict(gc)
        t = gc.pop("type")
        if t not in GUARDS:
            raise ValueError(f"unknown guard {t!r}; available: {sorted(GUARDS)}")
        if t == "Conformal":
            src = gc.pop("calibration", None)
            if src is None and calibrator is None:
                raise ValueError("Conformal needs 'calibration: path.json' (on the guard or at the top level)")
            gc["calibrator"] = Calibrator.load(src) if src else calibrator
        if t == "Coverage" and gc.get("gate") is True:            # erase-and-check inside every window with the configured gate
            gc["gate"] = gate
        guards.append(GUARDS[t](**gc))
    policy = Policy(**cfg.get("policy", {}))
    audit = AuditLogger(**cfg["audit"]) if cfg.get("audit") else None
    return DecisionGuard(m, scanners=scanners, gate=gate, guards=guards, policy=policy, audit=audit,
                         use_sanitized=cfg.get("use_sanitized", True), boundary_note=cfg.get("boundary_note"))


EXAMPLE_CONFIG = textwrap.dedent(__doc__.split("\n\n", 1)[1]).strip("\n") + "\n" if __doc__ else ""
