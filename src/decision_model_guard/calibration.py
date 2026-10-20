"""Calibration and conformal abstention from your own labelled decisions.

A probability of 0.7 from a decision model rarely means 70 %: calibration drifts between domains, and a threshold of
0.5 or one copied from a benchmark seldom fits your traffic. With a few hundred labelled cases:

    cal = Calibrator.fit(model, cases, alpha=0.05)   # temperature scaling + split-conformal threshold
    cal.save("calibration.json")
    guard = DecisionGuard(Calibrated(model, cal), guards=[Conformal(cal)], ...)

`Calibrated` rescales the model's probabilities (temperature scaling: one parameter, argmax unchanged). `Conformal`
turns every decision into a prediction set that contains the correct answer with probability at least 1 - alpha
(split conformal prediction; the guarantee holds when your traffic looks like the calibration cases). A set with
exactly one option is a decision; a set with two or more options, or none, means the model cannot decide at that
coverage, and the policy abstains or escalates. `fit` also reports risk thresholds for target false-positive rates.
"""
from __future__ import annotations

import json
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models.base import DecideBatch, DecisionModel, as_model, checked
from .types import GuardReport, Kind
from .util import argmax, normalize

EPS = 1e-12


def _scale(p: Sequence[float], t: float) -> list[float]:
    logits = [math.log(max(x, EPS)) / t for x in p]
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    s = sum(e)
    return [v / s for v in e]


def _nll(P: Sequence[Sequence[float]], y: Sequence[int], t: float) -> float:
    return -sum(math.log(max(_scale(p, t)[g], EPS)) for p, g in zip(P, y, strict=True)) / max(1, len(y))


def _ece(P: Sequence[Sequence[float]], y: Sequence[int], bins: int = 10) -> float:
    acc = [[0, 0.0, 0] for _ in range(bins)]                 # count, confidence sum, correct
    for p, g in zip(P, y, strict=True):
        c, k = max(p), argmax(list(p))
        b = min(bins - 1, int(c * bins))
        acc[b][0] += 1
        acc[b][1] += c
        acc[b][2] += int(k == g)
    n = max(1, len(y))
    return sum(abs(cs / cnt - cor / cnt) * cnt / n for cnt, cs, cor in acc if cnt)


def fit_temperature(P: Sequence[Sequence[float]], y: Sequence[int]) -> float:
    """Temperature minimising the negative log-likelihood (golden-section search over log T in [0.05, 20])."""
    lo, hi = math.log(0.05), math.log(20.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = _nll(P, y, math.exp(a)), _nll(P, y, math.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = _nll(P, y, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = _nll(P, y, math.exp(b))
    return math.exp((lo + hi) / 2)


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Split-conformal threshold: the ceil((n + 1)(1 - alpha))-th smallest nonconformity score (1.0, every option in every
    set, when there are fewer than `min_cases(alpha)` scores)."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha) - 1e-9)
    return 1.0 if k > n else sorted(scores)[k - 1]


def min_cases(alpha: float) -> int:
    """The smallest number of labelled cases for which a conformal threshold at miss rate `alpha` exists."""
    n = 1
    while math.ceil((n + 1) * (1 - alpha) - 1e-9) > n:
        n += 1
    return n


def collect(model: DecisionModel | DecideBatch, cases: Sequence[dict[str, Any]]) -> tuple[list[list[float]], list[int]]:
    """Model probabilities and gold indices for labelled cases ({"scene", "question", "options", "gold"}), batched per task."""
    m = as_model(model)
    groups: dict[tuple, list[int]] = {}
    for i, c in enumerate(cases):
        groups.setdefault((c["question"], tuple(c["options"])), []).append(i)
    P: list[list[float] | None] = [None] * len(cases)
    for (q, opts), idx in groups.items():
        for i, p in zip(idx, checked(m, [cases[i]["scene"] for i in idx], q, opts), strict=True):
            P[i] = p
    return [p for p in P if p is not None], [int(c["gold"]) for c in cases]


class Calibrator:
    """Temperature scaling plus a split-conformal threshold, fitted on labelled decisions."""

    def __init__(self, temperature: float = 1.0, qhat: float | None = None, alpha: float | None = None, n: int = 0,
                 metrics: dict[str, Any] | None = None, operating_points: list[dict[str, float]] | None = None,
                 risky: Sequence[int] | None = None, tasks: Sequence[str] | None = None):
        self.temperature, self.qhat, self.alpha, self.n = temperature, qhat, alpha, n
        self.metrics, self.operating_points = metrics or {}, operating_points or []
        self.risky, self.tasks = list(risky) if risky else None, list(tasks or [])

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, model: DecisionModel | DecideBatch, cases: Sequence[dict[str, Any]], alpha: float = 0.1,
            target_fprs: Sequence[float] = (0.001, 0.01, 0.05, 0.1)) -> Calibrator:
        """Fit on labelled cases. Use cases you did not tune anything else on; 200 or more give stable thresholds."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        P, y = collect(model, cases)
        return cls.fit_probs(P, y, alpha, risky=_common_risky(cases), target_fprs=target_fprs,
                             tasks=sorted({c["question"] for c in cases}))

    @classmethod
    def fit_probs(cls, P: Sequence[Sequence[float]], y: Sequence[int], alpha: float = 0.1, risky: Sequence[int] | None = None,
                  target_fprs: Sequence[float] = (0.001, 0.01, 0.05, 0.1), tasks: Sequence[str] | None = None) -> Calibrator:
        if len(P) != len(y) or not P:
            raise ValueError("need one probability list per gold label")
        t = fit_temperature(P, y)
        Pc = [_scale(p, t) for p in P]
        metrics = {"cases": len(y), "accuracy": sum(argmax(list(p)) == g for p, g in zip(P, y, strict=True)) / len(y),
                   "nll_before": _nll(P, y, 1.0), "nll_after": _nll(P, y, t), "ece_before": _ece(P, y), "ece_after": _ece(Pc, y)}
        qhat: float | None = None
        if len(y) < min_cases(alpha):                        # the threshold would be 1.0: every decision would abstain
            warnings.warn(f"{len(y)} cases are too few for alpha={alpha}: a conformal threshold needs at least {min_cases(alpha)}; "
                          "the calibrator keeps the temperature and has no conformal threshold", stacklevel=2)
        else:
            qhat = conformal_quantile([1 - p[g] for p, g in zip(Pc, y, strict=True)], alpha)
            sets = [[i for i, v in enumerate(p) if 1 - v <= qhat] for p in Pc]
            metrics.update({"set_coverage": sum(g in s for s, g in zip(sets, y, strict=True)) / len(y),
                            "singleton_rate": sum(len(s) == 1 for s in sets) / len(y), "mean_set_size": sum(len(s) for s in sets) / len(y)})
        ops = _operating_points(Pc, y, risky, target_fprs) if risky else []
        return cls(t, qhat, alpha, len(y), metrics, ops, risky, tasks)

    # ------------------------------------------------------------------ use
    def calibrate(self, probs: Sequence[float]) -> list[float]:
        return _scale(normalize(probs), self.temperature)

    def prediction_set(self, probs: Sequence[float], calibrated: bool = False) -> list[int]:
        """Options whose calibrated probability is at least 1 - qhat."""
        if self.qhat is None:
            raise ValueError("this calibrator has no conformal threshold")
        p = list(probs) if calibrated else self.calibrate(probs)
        return [i for i, v in enumerate(p) if 1 - v <= self.qhat]

    def evaluate(self, model: DecisionModel | DecideBatch, cases: Sequence[dict[str, Any]]) -> dict[str, float]:
        """Coverage and set sizes on held-out cases (check that the guarantee holds on your data)."""
        P, y = collect(model, cases)
        sets = [self.prediction_set(p) for p in P]
        single = [(s, g) for s, g in zip(sets, y, strict=True) if len(s) == 1]
        return {"cases": len(y), "coverage": sum(g in s for s, g in zip(sets, y, strict=True)) / len(y),
                "singleton_rate": len(single) / len(y), "singleton_accuracy": sum(s[0] == g for s, g in single) / max(1, len(single)),
                "mean_set_size": sum(len(s) for s in sets) / len(y), "ece": _ece([self.calibrate(p) for p in P], y)}

    def summary(self) -> str:
        m = self.metrics
        lines = [f"Calibration on {self.n} cases: temperature {self.temperature:.3f}, accuracy {100 * m.get('accuracy', 0):.1f} %",
                 f"ECE {100 * m.get('ece_before', 0):.1f} -> {100 * m.get('ece_after', 0):.1f}, NLL {m.get('nll_before', 0):.3f} -> {m.get('nll_after', 0):.3f}"]
        if self.qhat is not None:
            lines.append(f"Conformal sets at {100 * (1 - self.alpha):.0f} % coverage: {100 * m.get('singleton_rate', 0):.1f} % single answers, "
                         f"mean set size {m.get('mean_set_size', 0):.2f}")
        elif self.alpha:
            lines.append(f"No conformal threshold: {self.n} cases are too few for alpha={self.alpha} (at least {min_cases(self.alpha)} "
                         "needed); Conformal cannot use this file")
        for op in self.operating_points:
            lines.append(f"Risk threshold {op['threshold']:.4f}: FPR {100 * op['fpr']:.2f} % (target {100 * op['target_fpr']:.1f} %), "
                         f"TPR {100 * op['tpr']:.1f} %")
        return "\n".join(lines)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "qhat": self.qhat, "alpha": self.alpha, "n": self.n, "metrics": self.metrics,
                "operating_points": self.operating_points, "risky": self.risky, "tasks": self.tasks}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _common_risky(cases: Sequence[dict[str, Any]]) -> list[int] | None:
    rs = {tuple(c["risky"]) for c in cases if c.get("risky")}
    return list(rs.pop()) if len(rs) == 1 and all(c.get("risky") for c in cases) else None


def _operating_points(P, y, risky, targets) -> list[dict[str, float]]:
    rs = set(risky)
    risk = [sum(p[i] for i in rs) for p in P]
    pos = [r for r, g in zip(risk, y, strict=True) if g in rs]
    neg = sorted((r for r, g in zip(risk, y, strict=True) if g not in rs), reverse=True)
    if not pos or not neg:
        return []
    out = []
    for t in targets:
        k = int(math.floor(t * len(neg)))                   # negatives allowed above the threshold
        thr = min(1.0, neg[k] + 1e-9) if k < len(neg) else 0.0
        out.append({"target_fpr": t, "threshold": thr, "fpr": sum(r >= thr for r in neg) / len(neg),
                    "tpr": sum(r >= thr for r in pos) / len(pos)})
    return out


class Calibrated:
    """A decision model with calibrated probabilities (temperature scaling; the answer itself never changes)."""

    def __init__(self, model: DecisionModel | DecideBatch, calibrator: Calibrator):
        self.model, self.calibrator = as_model(model), calibrator
        self.name = f"calibrated:{self.model.name}"
        for attr in ("context_tokens", "count_tokens"):            # keep what Coverage needs
            if hasattr(self.model, attr):
                setattr(self, attr, getattr(self.model, attr))

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        return [self.calibrator.calibrate(p) for p in self.model.decide_batch(scenes, question, options)]


class Conformal:
    """Abstain unless the conformal prediction set holds exactly one option (coverage 1 - alpha of the calibrator).

    The guarantee holds only for the decisions the calibrator was fitted on: questions it has not seen are skipped and
    reported as such (fit one calibrator per decision task, or fit on a mix of all tasks you guard)."""

    name = "Conformal"

    def __init__(self, calibrator: Calibrator | str | Path):
        self.calibrator = c = calibrator if isinstance(calibrator, Calibrator) else Calibrator.load(calibrator)
        if c.qhat is None or c.qhat >= 1.0:                  # 1.0: every option always in the set (too few cases)
            need = f"; alpha={c.alpha} needs at least {min_cases(c.alpha)} labelled cases" if c.alpha else ""
            raise ValueError(f"the calibrator has no usable conformal threshold (fitted on {c.n} cases{need}); refit it with alpha")

    def check(self, model: DecisionModel, scene: str, question: str, options: Sequence[str], probs: Sequence[float], kind: Kind,
              **_ctx) -> GuardReport:
        tasks = self.calibrator.tasks
        if tasks and not any(question.startswith(t) for t in tasks):   # startswith: a boundary note may follow the question
            return GuardReport(self.name, False, 0.0, 0, {"skipped": "not calibrated for this question"})
        already = isinstance(model, Calibrated) and (model.calibrator is self.calibrator
                                                      or model.calibrator.to_dict() == self.calibrator.to_dict())
        s = self.calibrator.prediction_set(probs, calibrated=already)
        return GuardReport(self.name, len(s) != 1, float(len(s) != 1), 0,
                           {"set": [options[i] for i in s], "coverage": round(1 - (self.calibrator.alpha or 0), 4)})
