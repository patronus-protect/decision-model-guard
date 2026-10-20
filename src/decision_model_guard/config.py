"""Build a DecisionGuard from a YAML or dict config.

    model:
      type: wolf_defender_jev            # wolf_defender_jev | clef | systemone_http | openai_compatible | causal_lm | hf_classifier | callable
      onnx: int8_int4_embeddings         # ONNX Runtime, no torch: fp32 | fp16 | mixed | int8_int4_embeddings; remove for PyTorch ([hf] extra)
      # device: cuda                     # PyTorch only: cpu | cuda | mps (default: DMGUARD_DEVICE, else cuda when available)
    scanners:                            # run in this order on the untrusted blocks
      - type: InvisibleText              # zero-width and bidi characters (removed from what the model sees)
      - type: ForgedTrustBlock           # trusted-looking headers and chat role markers inside the content
      - type: VerdictTokens              # "verdict: safe", "label=benign" inside the content
      - type: PromptInjection            # Wolf Defender; a finding blocks (see policy.block_scanners)
        threshold: 0.5
        onnx: int8_int4_embeddings
      - type: ClaimDetector              # claims about the decision; the gate reuses this detector
        threshold: 0.3
        onnx: int8_int4_embeddings
    gate:                                # the steering gate; "gate: false" switches it off
      selector: detector                 # all (erase every unit) | detector (only units the claim detector flags)
      scope: unordered                   # all | unordered (detector for unordered decisions, full check for risky and ordered)
      threshold: 0.3
    guards:
      - type: Coverage                   # long content: check the parts the model would cut off, window by window
      - type: Invariance                 # same scene as markdown, shorter option wording: the answer must not change
        formats: [markdown]              # markdown | kv | json | compact
        option_transforms: [short]       # short | lower | plain | quoted
      - type: Confidence
        min_confidence: 0.55
      - type: OptionOrder
        permutations: reverse            # reverse | rotate | random | all
      # - type: Conformal                # abstain unless the conformal set has one option (dmguard calibrate)
    # calibration: calibration.json      # temperature + conformal threshold from your labelled cases
    policy:
      gate_action: apply                 # apply | escalate | report
      unstable_action: fallback          # fallback | escalate | keep
      scanner_action: report             # report | escalate | cautious | block
      block_scanners: [PromptInjection]  # names of configured scanners whose finding always blocks
      low_confidence_action: abstain     # abstain | escalate (Confidence, Conformal)
      inconsistent_action: escalate      # escalate | report (OptionOrder, Invariance)
      coverage_action: apply             # apply | escalate | report
    audit:
      path: logs/decisions.jsonl         # one JSON line per decision
      include_text: false                # true stores scene and question text instead of hashes
    use_sanitized: true                  # the model reads the sanitized scene (invisible characters removed)
    boundary_note: false                 # true appends a source-trust note to the question; measure per model first
    # server:                            # dmguard serve
    #   auth_token_env: DMGUARD_TOKEN    # require a bearer token; the server does not start while the variable is unset
    #   max_body_bytes: 2000000
    #   max_scene_chars: 100000
    #   max_meta_bytes: 4096
    #   request_timeout: 120
    #   docs: true                       # false removes /docs, /redoc and /openapi.json
"""
from __future__ import annotations

import copy
import difflib
import importlib
import inspect
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
MODELS = ("wolf_defender_jev", "clef", "systemone_http", "openai_compatible", "causal_lm", "hf_classifier", "callable")
TOP_LEVEL = {"model": dict, "scanners": list, "gate": (dict, bool, type(None)), "guards": list, "calibration": (str, type(None)),
             "policy": dict, "audit": (dict, type(None)), "server": (dict, type(None)), "use_sanitized": bool,
             "boundary_note": (bool, str, type(None))}


def _close(name: str, allowed) -> str:
    hit = difflib.get_close_matches(name, list(allowed), n=1)
    return f" (did you mean {hit[0]!r}?)" if hit else ""


def check_config(cfg: Any) -> dict[str, Any]:
    """Raise ValueError for unknown top-level keys and sections of the wrong type: a typo must not remove protection."""
    if not isinstance(cfg, dict):
        raise ValueError("a config must be a mapping of sections (model, scanners, gate, ...)")
    for k, v in cfg.items():
        if k not in TOP_LEVEL:
            raise ValueError(f"unknown config key {k!r}{_close(k, TOP_LEVEL)}; allowed: {list(TOP_LEVEL)}")
        if not isinstance(v, TOP_LEVEL[k]):
            raise ValueError(f"'{k}' has the wrong type ({type(v).__name__})")
        if k in ("scanners", "guards") and not all(isinstance(x, dict) and "type" in x for x in v):
            raise ValueError(f"every entry of '{k}' needs a 'type'")
    from .server.app import server_settings
    server_settings(cfg)
    return cfg


def _load(config: dict[str, Any] | str | Path) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config, (str, Path)):
        base = Path(config).resolve().parent
        cfg = check_config(yaml.safe_load(Path(config).read_text(encoding="utf-8-sig")) or {})
    else:
        base, cfg = None, check_config(copy.deepcopy(config))
    if base is not None:                                         # relative paths in a config file mean: next to the file
        def rel(v: Any, must_exist: bool = False) -> Any:
            if not isinstance(v, str) or not v or os.path.isabs(v) or (must_exist and not (base / v).exists()):
                return v
            return str(base / v)
        cfg["calibration"] = rel(cfg.get("calibration")) if cfg.get("calibration") else cfg.get("calibration")
        if isinstance(cfg.get("audit"), dict) and "path" in cfg["audit"]:
            cfg["audit"]["path"] = rel(cfg["audit"]["path"])
        for g in cfg.get("guards", []):
            if g.get("calibration"):
                g["calibration"] = rel(g["calibration"])
        for section in [cfg.get("model")] + list(cfg.get("scanners", [])) + [cfg.get("gate")]:
            if isinstance(section, dict):                       # local model folders and .onnx files, not hub ids
                for k in ("repo", "model", "onnx"):
                    if k in section:
                        section[k] = rel(section[k], must_exist=True)
        if cfg.get("calibration") is None:
            cfg.pop("calibration", None)
    return cfg, base


def load_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    """A config dict from a dict or a YAML file (UTF-8), checked with `check_config`. In a file, relative paths
    (calibration, audit.path, a Conformal calibration, local model folders and .onnx files) are resolved against the
    folder of the file."""
    return _load(config)[0]


def _make(where: str, cls, kw: dict[str, Any]):
    try:
        return cls(**kw)
    except TypeError as e:                                       # unknown or missing parameters
        raise ValueError(f"{where}: {e}") from None


def _import(target: str, base: Path | None = None):
    mod, _, attr = target.partition(":")
    if not attr:
        raise ValueError("callable target must look like 'package.module:name'")
    for d in [os.getcwd()] + ([str(base)] if base else []):   # modules next to the config file or in the working directory
        if d not in sys.path:
            sys.path.insert(0, d)
    return getattr(importlib.import_module(mod), attr)


def _decide_function(obj) -> bool:
    """True when obj can be called as decide_batch(scenes, question, options)."""
    try:
        inspect.signature(obj).bind(None, None, None)
        return True
    except TypeError:
        return False
    except ValueError:                        # no signature (builtins): assume a decide function
        return True


def build_model(cfg: dict[str, Any], base: Path | None = None) -> DecisionModel:
    """A decision model from the `model:` section. A `callable` target is a decide function, a model object, or a class or
    factory: it is called with the remaining keys when it is a class, when keys remain, when `factory: true` is set, or
    when it cannot take (scenes, question, options)."""
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise ValueError(f"the model section needs a 'type': one of {list(MODELS)}")
    cfg = dict(cfg)
    t = cfg.pop("type")
    if t == "callable":                       # your own model: a decide_batch function, or a class/factory returning a model
        if "target" not in cfg:
            raise ValueError("model type callable needs target: 'package.module:name'")
        obj = _import(cfg.pop("target"), base)
        name, factory = cfg.pop("name", None), cfg.pop("factory", None)
        if isinstance(obj, type) or cfg or factory or (callable(obj) and not hasattr(obj, "decide_batch") and not _decide_function(obj)):
            obj = _make("model", obj, cfg)
        return as_model(obj, name=name)
    if t == "wolf_defender_jev":
        repo = cfg.pop("repo", "patronus-studio/wolf-defender-jev")
        try:
            return WolfDefenderJev.from_pretrained(repo, **cfg)
        except TypeError as e:
            raise ValueError(f"model: {e}") from None
    classes = {"systemone_http": SystemOneHTTP, "clef": Clef, "openai_compatible": OpenAICompatible, "causal_lm": CausalLMJudge,
               "hf_classifier": HFClassifier}
    if t not in classes:
        raise ValueError(f"unknown model type {t!r}{_close(str(t), MODELS)}; available: {list(MODELS)}")
    return _make("model", classes[t], cfg)


def build_scanner(cfg: dict[str, Any]):
    """A content scanner from one entry of `scanners:`."""
    cfg = dict(cfg)
    t = cfg.pop("type")
    if t not in SCANNERS:
        raise ValueError(f"unknown scanner {t!r}{_close(str(t), SCANNERS)}; available: {sorted(SCANNERS)}")
    return _make(f"scanner {t}", SCANNERS[t], cfg)


def build_scanners(config: dict[str, Any] | str | Path) -> list[Any]:
    """Only the scanners of a config (what `dmguard scan` needs), without loading the decision model."""
    return [build_scanner(x) for x in load_config(config).get("scanners", [])]


def build_guard(config: dict[str, Any] | str | Path, model: DecisionModel | None = None) -> DecisionGuard:
    """DecisionGuard from a config dict or a YAML file path. Pass `model` to override the configured model. Unknown keys
    and wrong types raise ValueError before any model is loaded."""
    cfg, base = _load(config)
    if model is None and "model" not in cfg:
        raise ValueError("config needs a 'model' section (or pass model=...)")
    policy = _make("policy", Policy, cfg.get("policy", {}))
    scanners = [build_scanner(s) for s in cfg.get("scanners", [])]
    names = [x.name for x in scanners]
    unknown = [n for n in policy.block_scanners if n not in names]
    if unknown:
        raise ValueError(f"policy.block_scanners names {unknown}, which are not among the configured scanners {names}")
    calibrator = Calibrator.load(cfg["calibration"]) if cfg.get("calibration") else None
    gate = None
    if cfg.get("gate") is not False:
        g = dict(cfg["gate"]) if isinstance(cfg.get("gate"), dict) else {}      # gate: true or no section: the default gate
        if g.get("selector") == "detector":
            det = next((x for x in scanners if isinstance(x, S.ClaimDetector)), None)
            if det is None:
                extra = {k: g[k] for k in ("device", "onnx") if k in g}
                det = S.ClaimDetector(threshold=g.get("threshold", 0.3), **extra)
            g["detector"] = det
        g.pop("device", None)
        g.pop("onnx", None)
        gate = _make("gate", SteeringGate, g)
    guards = []
    for gc in cfg.get("guards", []):
        gc = dict(gc)
        t = gc.pop("type")
        if t not in GUARDS:
            raise ValueError(f"unknown guard {t!r}{_close(str(t), GUARDS)}; available: {sorted(GUARDS)}")
        if t == "Conformal":
            src = gc.pop("calibration", None)
            if src is None and calibrator is None:
                raise ValueError("Conformal needs 'calibration: path.json' (on the guard or at the top level)")
            if src and calibrator is not None and Path(src).resolve() != Path(cfg["calibration"]).resolve():
                raise ValueError("the Conformal guard and the top-level calibration name different files; the model is "
                                 "calibrated with the top-level one, so give the guard the same file or none")
            gc["calibrator"] = calibrator if calibrator is not None else Calibrator.load(src)   # one calibration, applied once
        if t == "Coverage" and gc.get("gate") is True:            # erase-and-check inside every window with the configured gate
            gc["gate"] = gate
        guards.append(_make(f"guard {t}", GUARDS[t], gc))
    audit = _make("audit", AuditLogger, cfg["audit"]) if cfg.get("audit") else None
    m = model if model is not None else build_model(cfg["model"], base)
    if calibrator is not None:                                   # calibrated probabilities everywhere in the pipeline
        m = Calibrated(m, calibrator)
    return DecisionGuard(m, scanners=scanners, gate=gate, guards=guards, policy=policy, audit=audit,
                         use_sanitized=cfg.get("use_sanitized", True), boundary_note=cfg.get("boundary_note"))


EXAMPLE_CONFIG = textwrap.dedent(__doc__.split("\n\n", 1)[1]).strip("\n") + "\n" if __doc__ else ""
