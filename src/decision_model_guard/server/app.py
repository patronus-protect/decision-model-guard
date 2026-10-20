"""HTTP API (FastAPI), LLM Guard API style.

    dmguard serve --config config.yml --port 8000

Endpoints:
    GET  /healthz            liveness
    GET  /readyz             the guard is built
    GET  /metrics            Prometheus text format
    POST /v1/scan            {"text": "..."} (all untrusted) or {"scene": "..."} -> scanner results and the sanitized text
    POST /v1/decide          one guarded decision
    POST /v1/decide/batch    {"requests": [...]} -> list of guarded decisions

Authentication: with `server.auth_token_env` in the config the named environment variable must hold the token (the
server refuses to start otherwise); without the key, DMGUARD_TOKEN is used when it is set. Clients send
`Authorization: Bearer <token>`. /healthz, /readyz, /metrics and the docs routes stay public (`server.docs: false`
removes /docs, /redoc and /openapi.json).
"""
# no "from __future__ import annotations": FastAPI resolves the locally defined request models at runtime
import asyncio
import hmac
import json
import logging
import os
import threading
import time
from collections import Counter
from typing import Any

from ..util import require

log = logging.getLogger("decision_model_guard.server")
SERVER_KEYS = {"auth_token_env": str, "max_body_bytes": int, "max_scene_chars": int, "max_meta_bytes": int,
               "request_timeout": (int, float), "docs": bool}


def server_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """The `server:` section of a config, checked: unknown keys and wrong types raise ValueError."""
    srv = cfg.get("server") or {}
    if not isinstance(srv, dict):
        raise ValueError("server must be a mapping")
    bad = sorted(set(srv) - set(SERVER_KEYS))
    if bad:
        raise ValueError(f"unknown server keys {bad}; allowed: {sorted(SERVER_KEYS)}")
    for k, v in srv.items():
        if not isinstance(v, SERVER_KEYS[k]) or (isinstance(v, bool) and SERVER_KEYS[k] is not bool):
            raise ValueError(f"server.{k} has the wrong type: {v!r}")
    return srv


def resolve_token(srv: dict[str, Any], auth_token: str | None = None) -> str | None:
    """The bearer token: `auth_token`, else the variable named in server.auth_token_env (which must be set), else
    DMGUARD_TOKEN when it is set."""
    if auth_token:
        return auth_token
    if "auth_token_env" in srv:
        name = srv["auth_token_env"]
        token = os.environ.get(name) if name else None
        if not token:
            raise ValueError(f"server.auth_token_env names {name!r}, but that environment variable is empty or not set; "
                             "set it, or remove the key to run without authentication")
        return token
    return os.environ.get("DMGUARD_TOKEN") or None


def create_app(guard=None, config: dict[str, Any] | str | None = None, auth_token: str | None = None, max_body_bytes: int = 2_000_000,
               max_scene_chars: int = 100_000, max_meta_bytes: int = 4096, request_timeout: float | None = 120.0, docs: bool = True):
    """The FastAPI app. Settings from the config's `server:` section win over the keyword arguments."""
    fastapi = require("fastapi", "server")
    pydantic = require("pydantic", "server")
    from starlette.concurrency import run_in_threadpool
    from starlette.responses import JSONResponse

    from ..config import build_guard
    from ..pipeline import scan_text
    cfg: dict[str, Any] = {}
    if isinstance(config, str):
        import yaml
        with open(config, encoding="utf-8-sig") as fh:
            cfg = yaml.safe_load(fh) or {}
    elif isinstance(config, dict):
        cfg = config
    srv = server_settings(cfg)
    token = resolve_token(srv, auth_token)
    max_body = int(srv.get("max_body_bytes", max_body_bytes))
    max_scene = int(srv.get("max_scene_chars", max_scene_chars))
    max_meta = int(srv.get("max_meta_bytes", max_meta_bytes))
    timeout = srv.get("request_timeout", request_timeout)
    state: dict[str, Any] = {"guard": guard}
    lock = threading.Lock()
    metrics: Counter = Counter()
    latency = {"sum": 0.0, "count": 0}

    def get_guard():
        with lock:
            if state["guard"] is None:
                state["guard"] = build_guard(cfg)
            return state["guard"]

    BaseModel, Field = pydantic.BaseModel, pydantic.Field

    class ScanIn(BaseModel):
        text: str | None = None
        scene: str | None = None

    class DecideIn(BaseModel):
        scene: str
        question: str
        options: list[str] = Field(min_length=2)
        risky: list[int] | None = None
        order: list[int] | None = None
        fallback: int | None = None
        meta: dict[str, Any] | None = None
        include_scene: bool = False

    class BatchIn(BaseModel):
        requests: list[DecideIn] = Field(min_length=1, max_length=256)

    app = fastapi.FastAPI(title="Decision Model Guard", version=__import__("decision_model_guard").__version__,
                          description="Guardrails for decision models: steering gate, content scanners and decision checks.",
                          **({} if srv.get("docs", docs) else {"docs_url": None, "redoc_url": None, "openapi_url": None}))
    app.state.auth = token is not None
    log.info("authentication %s", "on (bearer token)" if token else "off: set DMGUARD_TOKEN or server.auth_token_env to require a token")

    class Limits:
        """Auth and the body limit for /v1/*. The limit counts the bytes actually received, so chunked requests without a
        Content-Length are capped too."""

        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
                return await self.inner(scope, receive, send)
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            if token:
                scheme, _, cred = headers.get("authorization", "").strip().partition(" ")
                if scheme.lower() != "bearer" or not hmac.compare_digest(cred.strip().encode(), token.encode()):
                    metrics["unauthorized_total"] += 1
                    return await JSONResponse({"detail": "unauthorized"}, 401, {"WWW-Authenticate": "Bearer"})(scope, receive, send)
            too_big = JSONResponse({"detail": "request too large"}, 413)
            size = headers.get("content-length", "")
            if size.isdigit() and int(size) > max_body:
                return await too_big(scope, receive, send)
            body, more = b"", True
            while more:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    return
                body += msg.get("body", b"")
                more = msg.get("more_body", False)
                if len(body) > max_body:
                    return await too_big(scope, receive, send)
            done = False

            async def replay():
                nonlocal done
                if not done:
                    done = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            return await self.inner(scope, replay, send)

    app.add_middleware(Limits)

    def _limit(scene: str | None, meta: dict[str, Any] | None = None) -> None:
        if scene is not None and len(scene) > max_scene:
            raise fastapi.HTTPException(413, f"scene longer than {max_scene} characters (server.max_scene_chars)")
        if meta is not None and len(json.dumps(meta, ensure_ascii=False)) > max_meta:
            raise fastapi.HTTPException(413, f"meta larger than {max_meta} bytes (server.max_meta_bytes)")

    async def _run(fn, *args):
        try:
            return await asyncio.wait_for(run_in_threadpool(fn, *args), timeout) if timeout else await run_in_threadpool(fn, *args)
        except asyncio.TimeoutError:
            raise fastapi.HTTPException(504, f"no decision within {timeout} s (server.request_timeout)") from None

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        get_guard()
        return {"status": "ready"}

    @app.get("/metrics", response_class=fastapi.responses.PlainTextResponse)
    def prom():
        lines = ["# TYPE dmguard_requests_total counter"]
        lines += [f'dmguard_requests_total{{endpoint="{k[9:]}"}} {v}' for k, v in sorted(metrics.items()) if k.startswith("requests:")]
        lines += ["# TYPE dmguard_decisions_total counter"]
        lines += [f'dmguard_decisions_total{{action="{k[7:]}"}} {v}' for k, v in sorted(metrics.items()) if k.startswith("action:")]
        lines += [f"dmguard_unauthorized_total {metrics['unauthorized_total']}",
                  "# TYPE dmguard_decision_latency_ms summary",
                  f"dmguard_decision_latency_ms_sum {latency['sum']:.3f}", f"dmguard_decision_latency_ms_count {latency['count']}"]
        return "\n".join(lines) + "\n"

    def _scan(body: ScanIn) -> dict[str, Any]:
        g = get_guard()
        sanitized, results = g.scan(body.scene) if body.scene is not None else scan_text(body.text, g.scanners)
        return {"sanitized": sanitized, "valid": all(r.valid for r in results), "results_valid": {r.scanner: r.valid for r in results},
                "results_score": {r.scanner: round(r.score, 6) for r in results}, "results": [r.to_dict() for r in results]}

    @app.post("/v1/scan")
    async def scan(body: ScanIn):
        metrics["requests:scan"] += 1
        if (body.text is None) == (body.scene is None):
            raise fastapi.HTTPException(422, "send either 'text' (untrusted text) or 'scene' (a scene with trusted blocks)")
        _limit(body.scene if body.scene is not None else body.text)
        return await _run(_scan, body)

    def _decide(b: DecideIn) -> dict[str, Any]:
        t = time.perf_counter()
        d = get_guard().decide(b.scene, b.question, b.options, risky=b.risky, order=b.order, fallback=b.fallback, meta=b.meta)
        metrics[f"action:{d.action.value}"] += 1
        latency["sum"] += (time.perf_counter() - t) * 1000
        latency["count"] += 1
        return d.to_dict(include_scene=b.include_scene)

    @app.post("/v1/decide")
    async def decide(body: DecideIn):
        metrics["requests:decide"] += 1
        _limit(body.scene, body.meta)
        try:
            return await _run(_decide, body)
        except ValueError as e:
            raise fastapi.HTTPException(422, str(e)) from None

    @app.post("/v1/decide/batch")
    async def decide_batch(body: BatchIn):
        metrics["requests:decide_batch"] += 1
        for r in body.requests:
            _limit(r.scene, r.meta)
        try:
            return {"decisions": await _run(lambda: [_decide(r) for r in body.requests])}
        except ValueError as e:
            raise fastapi.HTTPException(422, str(e)) from None

    return app
