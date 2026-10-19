"""HTTP API (FastAPI), LLM Guard API style.

    dmguard serve --config config.yml --port 8000

Endpoints:
    GET  /healthz            liveness
    GET  /readyz             the guard is built
    GET  /metrics            Prometheus text format
    POST /v1/scan            {"text": "..."} or {"scene": "..."} -> scanner results and the sanitized text
    POST /v1/decide          one guarded decision
    POST /v1/decide/batch    {"requests": [...]} -> list of guarded decisions

Authentication: set `server.auth_token_env` in the config (or DMGUARD_TOKEN) and send `Authorization: Bearer <token>`.
"""
# no "from __future__ import annotations": FastAPI resolves the locally defined request models at runtime
import hmac
import os
import threading
import time
from collections import Counter
from typing import Any

from ..util import require


def create_app(guard=None, config: dict[str, Any] | str | None = None, auth_token: str | None = None, max_body_bytes: int = 2_000_000):
    fastapi = require("fastapi", "server")
    pydantic = require("pydantic", "server")
    import yaml

    from ..config import build_guard
    cfg: dict[str, Any] = {}
    if isinstance(config, str):
        with open(config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    elif isinstance(config, dict):
        cfg = config
    srv = cfg.get("server", {}) or {}
    token = auth_token or os.environ.get(srv.get("auth_token_env", "DMGUARD_TOKEN") or "DMGUARD_TOKEN")
    max_body = int(srv.get("max_body_bytes", max_body_bytes))
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
                          description="Guardrails for decision models: steering gate, content scanners and decision checks.")

    @app.middleware("http")
    async def guard_requests(request, call_next):
        if request.url.path.startswith("/v1/"):
            if token:
                auth = request.headers.get("authorization", "")
                if not hmac.compare_digest(auth, f"Bearer {token}"):
                    metrics["unauthorized_total"] += 1
                    return fastapi.responses.JSONResponse({"detail": "unauthorized"}, status_code=401)
            size = request.headers.get("content-length")
            if size and int(size) > max_body:
                return fastapi.responses.JSONResponse({"detail": "request too large"}, status_code=413)
        return await call_next(request)

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

    @app.post("/v1/scan")
    def scan(body: ScanIn):
        metrics["requests:scan"] += 1
        g = get_guard()
        text = body.scene if body.scene is not None else body.text
        if text is None:
            raise fastapi.HTTPException(422, "send 'text' or 'scene'")
        if body.scene is None and "[" not in text:
            text = f"[USER PROMPT]\n{text}"
        sanitized, results = g.scan(text)
        return {"sanitized": sanitized, "valid": all(r.valid for r in results), "results_valid": {r.scanner: r.valid for r in results},
                "results_score": {r.scanner: round(r.score, 6) for r in results}, "results": [r.to_dict() for r in results]}

    def _decide(b: DecideIn) -> dict[str, Any]:
        t = time.perf_counter()
        d = get_guard().decide(b.scene, b.question, b.options, risky=b.risky, order=b.order, fallback=b.fallback, meta=b.meta)
        metrics[f"action:{d.action.value}"] += 1
        latency["sum"] += (time.perf_counter() - t) * 1000
        latency["count"] += 1
        return d.to_dict(include_scene=b.include_scene)

    @app.post("/v1/decide")
    def decide(body: DecideIn):
        metrics["requests:decide"] += 1
        try:
            return _decide(body)
        except ValueError as e:
            raise fastapi.HTTPException(422, str(e)) from None

    @app.post("/v1/decide/batch")
    def decide_batch(body: BatchIn):
        metrics["requests:decide_batch"] += 1
        try:
            return {"decisions": [_decide(r) for r in body.requests]}
        except ValueError as e:
            raise fastapi.HTTPException(422, str(e)) from None

    return app
