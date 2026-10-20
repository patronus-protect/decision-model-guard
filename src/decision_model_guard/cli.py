"""Command line: dmguard {version, init-config, scan, decide, eval, audit, calibrate, serve}."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .version import __version__

EXIT_CODES = """exit codes:
  0  success (decide: any action, unless --exit-code is given)
  1  error (invalid config, arguments or input, failed model call)
  2  usage error; scan: content flagged; audit: attack success above --max-attack-success, or no attack ran
  decide --exit-code: 0 allow, 3 cautious or fallback, 4 escalate, 5 abstain, 6 block"""
ACTION_EXIT = {"allow": 0, "cautious": 3, "fallback": 3, "escalate": 4, "abstain": 5, "block": 6}


def _guard(args):
    from .config import build_guard
    return build_guard(args.config)


def _positive(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _indices(v: str) -> list[int]:
    try:
        return [int(x) for x in v.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated option indices such as 0,1,2, got {v!r}") from None


def _alpha(v: str) -> float:
    a = float(v)
    if not 0 < a < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return a


def _read(args) -> str:
    if getattr(args, "text", None) is not None:
        text = args.text
    elif getattr(args, "file", None):
        text = Path(args.file).read_text(encoding="utf-8-sig")
    elif not sys.stdin.isatty():                                 # UTF-8 regardless of the locale (German content on Windows)
        text = sys.stdin.buffer.read().decode("utf-8-sig") if hasattr(sys.stdin, "buffer") else sys.stdin.read()
    else:
        raise ValueError("give --text, --file or pipe text on stdin")
    if not text.strip():
        raise ValueError("the input is empty")
    return text


def cmd_init(args) -> int:
    from .config import EXAMPLE_CONFIG
    path = Path(args.path)
    if path.exists() and not args.force:
        raise ValueError(f"{path} exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    print(f"wrote {path}")
    return 0


def cmd_scan(args) -> int:
    from .config import build_scanners
    from .pipeline import scan_scene, scan_text
    text = _read(args)
    scanners = build_scanners(args.config)                         # the decision model is not needed for scanning
    sanitized, results = scan_scene(text, scanners) if args.scene else scan_text(text, scanners)   # text: one untrusted block
    out = {"valid": all(r.valid for r in results), "results": [r.to_dict() for r in results]}
    if args.sanitized:
        out["sanitized"] = sanitized
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["valid"] else 2


def cmd_decide(args) -> int:
    from .pipeline import check_decision
    scene = _read(args)                                            # arguments and input first, then the models
    check_decision(args.option, args.risky, args.order, args.fallback)
    g = _guard(args)
    d = g.decide(scene, args.question, args.option, risky=args.risky, order=args.order, fallback=args.fallback)
    if args.json:
        print(json.dumps(d.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"answer:     {d.answer if d.answer is not None else '(blocked)'}")
        print(f"action:     {d.action.value}")
        print(f"raw answer: {d.raw_answer}")
        if d.reasons:
            print("reasons:    " + "\n            ".join(d.reasons))
        print(f"calls:      {d.calls}   latency: {d.latency_ms:.0f} ms")
    return ACTION_EXIT[d.action.value] if args.exit_code else 0


def cmd_eval(args) -> int:
    from .evaluate import claim_bank, evaluate, load_cases
    langs = args.lang or ["en"]
    known = sorted(claim_bank()["claims"])
    bad = [x for x in langs if x not in known]
    if bad:
        raise ValueError(f"unknown --lang {bad}; available: {known}")
    cases = load_cases(args.cases)
    g = _guard(args)
    rep = evaluate(g.model, cases, guard=g, attacks_per_case=args.attacks, controls_per_case=args.controls, langs=langs, seed=args.seed,
                   log_decisions=args.audit_log)
    md = rep.to_markdown()
    print(md)
    if args.out:
        Path(args.out).write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
    return 0


def cmd_audit(args) -> int:
    from .evaluate import FAMILIES, audit, load_cases
    cases = load_cases(args.cases)
    g = _guard(args)
    rep = audit(g.model, cases, guard=None if args.no_guard else g, families=args.check or list(FAMILIES),
                per_case=args.per_case, padding_tokens=args.padding_tokens, seed=args.seed, log_decisions=args.audit_log)
    md = rep.to_markdown()
    print(md)
    if args.out:
        Path(args.out).write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
    if args.max_attack_success is not None:                       # for CI: fail when an attack gets through too often
        worst = rep.attack_rates(guarded=not args.no_guard)
        if not worst:                                             # nothing measured is not a pass
            print("no attack check ran on any case (empty case file, cases the model answers wrongly, or no attack checks "
                  "selected), so --max-attack-success cannot pass", file=sys.stderr)
            return 2
        over = {k: r for k, r in worst.items() if r > args.max_attack_success}
        if over:
            print("attack success above the limit: " + ", ".join(f"{k} {100 * r:.1f} %" for k, r in over.items()), file=sys.stderr)
            return 2
    return 0


def cmd_calibrate(args) -> int:
    from .calibration import Calibrator
    from .config import _load, build_model
    from .evaluate import load_cases
    cases, test = load_cases(args.cases), load_cases(args.test) if args.test else None
    cfg, base = _load(args.config)
    if "model" not in cfg:
        raise ValueError("config needs a 'model' section")
    model = build_model(cfg["model"], base)                       # the bare model: calibration is fitted on raw probabilities
    cal = Calibrator.fit(model, cases, alpha=args.alpha)
    print(cal.summary())
    if test and cal.qhat is not None:
        ev = cal.evaluate(model, test)
        print(f"Held-out: coverage {100 * ev['coverage']:.1f} % (target {100 * (1 - args.alpha):.0f} %), "
              f"{100 * ev['singleton_rate']:.1f} % single answers, accuracy of single answers {100 * ev['singleton_accuracy']:.1f} %")
    cal.save(args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_serve(args) -> int:
    import logging

    from .server import create_app
    from .util import require
    uvicorn = require("uvicorn", "server")
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s:     %(name)s: %(message)s")
    app = create_app(config=args.config)
    if not app.state.auth and args.host not in ("127.0.0.1", "localhost", "::1"):
        logging.getLogger("decision_model_guard.server").warning("listening on %s without authentication", args.host)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, timeout_graceful_shutdown=10)
    return 0


class _Help(argparse.RawDescriptionHelpFormatter):
    """Keeps the exit-code epilog as written and shows defaults that are set."""

    def _get_help_string(self, action):
        h = action.help or ""
        if action.option_strings and action.default not in (None, False, argparse.SUPPRESS) and "%(default)" not in h:
            h += " (default: %(default)s)"
        return h


def parser() -> argparse.ArgumentParser:
    """The dmguard argument parser."""
    from .evaluate.audit import FAMILIES
    fmt = _Help
    ap = argparse.ArgumentParser(prog="dmguard", description="Guardrails for decision models.", epilog=EXIT_CODES, formatter_class=fmt)
    ap.add_argument("--version", action="version", version=f"dmguard {__version__}")
    ap.add_argument("--debug", action="store_true", help="show tracebacks instead of one-line errors")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version", help="print the version")
    p = sub.add_parser("init-config", help="write a commented example config", formatter_class=fmt)
    p.add_argument("path", nargs="?", default="dmguard.yml", help="where to write it (folders are created)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    for name, helptext in (("scan", "scan untrusted text with the configured scanners"), ("decide", "one guarded decision"),
                           ("eval", "steering evaluation: planted claims, with and without the guard"),
                           ("audit", "robustness audit: attacks and stress checks, with and without the guard"),
                           ("calibrate", "fit temperature and conformal threshold on labelled cases"), ("serve", "HTTP API")):
        q = sub.add_parser(name, help=helptext, description=helptext, formatter_class=fmt, epilog=EXIT_CODES)
        q.add_argument("--config", required=True, help="YAML config (dmguard init-config writes an example)")
        if name in ("scan", "decide"):
            q.add_argument("--text", help="the input as an argument")
            q.add_argument("--file", help="read the input from a file (UTF-8); without --text and --file, stdin is read")
        if name == "scan":
            q.add_argument("--sanitized", action="store_true", help="also print the sanitized text")
            q.add_argument("--scene", action="store_true", help="parse the input as a scene: trusted blocks are not scanned "
                           "(default: the whole input is untrusted text)")
        if name == "decide":
            q.add_argument("--question", required=True, help="the question the model answers")
            q.add_argument("--option", action="append", required=True, help="one answer option; repeat it for every option")
            q.add_argument("--risky", type=_indices, help="comma-separated indices of the risky options (security verdicts)")
            q.add_argument("--order", type=_indices, help="all option indices, most cautious first (ordered decisions)")
            q.add_argument("--fallback", type=int, help="option index for unstable unordered decisions")
            q.add_argument("--json", action="store_true", help="print the full decision as JSON")
            q.add_argument("--exit-code", action="store_true", help="exit with a code per action (see below)")
        if name in ("eval", "audit"):
            q.add_argument("--cases", help="JSONL cases with scene, question, options, gold (integer index), risky or order "
                           "(default: the bundled samples); see docs/evaluation.md")
            q.add_argument("--seed", type=int, default=0, help="sampling seed")
            q.add_argument("--out", help="write the report as JSON")
            q.add_argument("--audit-log", action="store_true", help="also write the guarded test decisions to the configured audit log")
        if name == "eval":
            q.add_argument("--attacks", type=int, default=4, help="planted claims per case")
            q.add_argument("--controls", type=int, default=2, help="neutral sentences per case")
            q.add_argument("--lang", action="append", help="claim language (repeatable; 15 languages, e.g. en, de, fr); default en")
        if name == "audit":
            q.add_argument("--check", action="append", choices=list(FAMILIES), help="run only these checks (repeatable)")
            q.add_argument("--per-case", type=_positive, default=2, help="variants per check and case")
            q.add_argument("--padding-tokens", type=_positive, help="filler per side for the padding check (default: 1.2 x model context)")
            q.add_argument("--no-guard", action="store_true", help="measure the model alone")
            q.add_argument("--markdown", help="write the scorecard as Markdown")
            q.add_argument("--max-attack-success", type=float, help="exit 2 when an attack check exceeds this rate, or when no attack ran (CI)")
        if name == "calibrate":
            q.add_argument("--cases", required=True, help="labelled JSONL cases for fitting")
            q.add_argument("--test", help="labelled JSONL cases to check coverage on")
            q.add_argument("--alpha", type=_alpha, default=0.1, help="allowed miss rate of the conformal sets")
            q.add_argument("--out", default="calibration.json", help="where to write the calibration")
        if name == "serve":
            q.add_argument("--host", default="127.0.0.1", help="interface to listen on")
            q.add_argument("--port", type=int, default=8000, help="port")
            q.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug", "trace"],
                           help="log level")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.cmd == "version":
        print(__version__)
        return 0
    cmd = {"init-config": cmd_init, "scan": cmd_scan, "decide": cmd_decide, "eval": cmd_eval, "audit": cmd_audit,
           "calibrate": cmd_calibrate, "serve": cmd_serve}[args.cmd]
    if args.debug:
        return cmd(args)
    import yaml
    try:
        return cmd(args)
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError, yaml.YAMLError) as e:
        msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        print(f"dmguard: error: {msg} (--debug shows the traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
