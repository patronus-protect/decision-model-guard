"""Command line: dmguard {version, init-config, scan, decide, eval, audit, calibrate, serve}."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .version import __version__


def _guard(args):
    from .config import build_guard
    return build_guard(args.config)


def _positive(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _read(args) -> str:
    if getattr(args, "text", None):
        return args.text
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("give --text, --file or pipe text on stdin")


def cmd_init(args) -> int:
    from .config import EXAMPLE_CONFIG
    path = Path(args.path)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists (use --force to overwrite)")
    path.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    print(f"wrote {path}")
    return 0


def cmd_scan(args) -> int:
    from .config import build_scanners
    from .pipeline import scan_scene, scan_text
    scanners = build_scanners(args.config)                         # the decision model is not needed for scanning
    text = _read(args)
    sanitized, results = scan_scene(text, scanners) if args.scene else scan_text(text, scanners)   # text: one untrusted block
    out = {"valid": all(r.valid for r in results), "results": [r.to_dict() for r in results]}
    if args.sanitized:
        out["sanitized"] = sanitized
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["valid"] else 2


def cmd_decide(args) -> int:
    g = _guard(args)
    scene = _read(args)
    ints = lambda s: [int(x) for x in s.split(",")] if s else None  # noqa: E731
    d = g.decide(scene, args.question, args.option, risky=ints(args.risky), order=ints(args.order), fallback=args.fallback)
    if args.json:
        print(json.dumps(d.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"answer:   {d.answer}")
        print(f"action:   {d.action.value}")
        print(f"model:    {d.raw_answer}")
        if d.reasons:
            print("reasons:  " + "\n          ".join(d.reasons))
        print(f"calls:    {d.calls}   latency: {d.latency_ms:.0f} ms")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate, load_cases
    g = _guard(args)
    rep = evaluate(g.model, load_cases(args.cases), guard=g, attacks_per_case=args.attacks, controls_per_case=args.controls,
                   langs=args.lang or ["en"], seed=args.seed)
    md = rep.to_markdown()
    print(md)
    if args.out:
        Path(args.out).write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
    return 0


def cmd_audit(args) -> int:
    from .evaluate import FAMILIES, audit, load_cases
    g = _guard(args)
    rep = audit(g.model, load_cases(args.cases), guard=None if args.no_guard else g, families=args.check or list(FAMILIES),
                per_case=args.per_case, padding_tokens=args.padding_tokens, seed=args.seed)
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
    import yaml

    from .calibration import Calibrator
    from .config import build_model
    from .evaluate import load_cases
    cfg = yaml.safe_load(Path(args.config).read_text())
    model = build_model(cfg["model"])                             # the bare model: calibration is fitted on raw probabilities
    cal = Calibrator.fit(model, load_cases(args.cases), alpha=args.alpha)
    print(cal.summary())
    if args.test and cal.qhat is not None:
        ev = cal.evaluate(model, load_cases(args.test))
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dmguard", description="Guardrails for decision models.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    p = sub.add_parser("init-config", help="write an example config")
    p.add_argument("path", nargs="?", default="dmguard.yml")
    p.add_argument("--force", action="store_true")
    for name, helptext in (("scan", "scan untrusted text"), ("decide", "one guarded decision"), ("eval", "steering evaluation"),
                           ("audit", "robustness audit: attacks and stress checks, with and without the guard"),
                           ("calibrate", "fit temperature and conformal threshold on labelled cases"), ("serve", "HTTP API")):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("--config", required=True)
        if name in ("scan", "decide"):
            q.add_argument("--text")
            q.add_argument("--file")
        if name == "scan":
            q.add_argument("--sanitized", action="store_true", help="also print the sanitized text")
            q.add_argument("--scene", action="store_true", help="parse the input as a scene: trusted blocks are not scanned "
                           "(default: the whole input is untrusted text)")
        if name == "decide":
            q.add_argument("--question", required=True)
            q.add_argument("--option", action="append", required=True)
            q.add_argument("--risky", help="comma-separated risky option indices")
            q.add_argument("--order", help="option indices, most cautious first")
            q.add_argument("--fallback", type=int)
            q.add_argument("--json", action="store_true")
        if name == "eval":
            q.add_argument("--cases", help="JSONL cases (default: bundled samples)")
            q.add_argument("--attacks", type=int, default=4)
            q.add_argument("--controls", type=int, default=2)
            q.add_argument("--lang", action="append")
            q.add_argument("--seed", type=int, default=0)
            q.add_argument("--out", help="write the report as JSON")
        if name == "audit":
            q.add_argument("--cases", help="JSONL cases (default: bundled samples)")
            q.add_argument("--check", action="append", help="run only these checks (repeatable)")
            q.add_argument("--per-case", type=_positive, default=2, help="variants per check and case")
            q.add_argument("--padding-tokens", type=int, help="filler per side for the padding check (default: 1.2 x model context)")
            q.add_argument("--no-guard", action="store_true", help="measure the model alone")
            q.add_argument("--seed", type=int, default=0)
            q.add_argument("--out", help="write the report as JSON")
            q.add_argument("--markdown", help="write the scorecard as Markdown")
            q.add_argument("--max-attack-success", type=float, help="exit 2 when an attack check exceeds this rate (CI)")
        if name == "calibrate":
            q.add_argument("--cases", required=True, help="labelled JSONL cases for fitting")
            q.add_argument("--test", help="labelled JSONL cases to check coverage on")
            q.add_argument("--alpha", type=float, default=0.1, help="allowed miss rate of the conformal sets")
            q.add_argument("--out", default="calibration.json")
        if name == "serve":
            q.add_argument("--host", default="127.0.0.1")
            q.add_argument("--port", type=int, default=8000)
            q.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)
    if args.cmd == "version":
        print(__version__)
        return 0
    return {"init-config": cmd_init, "scan": cmd_scan, "decide": cmd_decide, "eval": cmd_eval, "audit": cmd_audit,
            "calibrate": cmd_calibrate, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
