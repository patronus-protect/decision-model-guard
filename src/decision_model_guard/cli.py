"""Command line: dmguard {version, init-config, scan, decide, eval, serve}."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .version import __version__


def _guard(args):
    from .config import build_guard
    return build_guard(args.config)


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
    g = _guard(args)
    text = _read(args)
    if not text.lstrip().startswith("["):
        text = f"[USER PROMPT]\n{text}"
    sanitized, results = g.scan(text)
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


def cmd_serve(args) -> int:
    from .server import create_app
    from .util import require
    uvicorn = require("uvicorn", "server")
    uvicorn.run(create_app(config=args.config), host=args.host, port=args.port, log_level=args.log_level)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dmguard", description="Guardrails for decision models.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    p = sub.add_parser("init-config", help="write an example config")
    p.add_argument("path", nargs="?", default="dmguard.yml")
    p.add_argument("--force", action="store_true")
    for name, helptext in (("scan", "scan untrusted text"), ("decide", "one guarded decision"), ("eval", "steering evaluation"), ("serve", "HTTP API")):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("--config", required=True)
        if name in ("scan", "decide"):
            q.add_argument("--text")
            q.add_argument("--file")
        if name == "scan":
            q.add_argument("--sanitized", action="store_true", help="also print the sanitized text")
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
        if name == "serve":
            q.add_argument("--host", default="127.0.0.1")
            q.add_argument("--port", type=int, default=8000)
            q.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)
    if args.cmd == "version":
        print(__version__)
        return 0
    return {"init-config": cmd_init, "scan": cmd_scan, "decide": cmd_decide, "eval": cmd_eval, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
