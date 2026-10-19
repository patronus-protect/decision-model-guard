# API server

```bash
pip install "decision-model-guard[server,hf]"
dmguard serve --config dmguard.yml --host 0.0.0.0 --port 8000
```

The guard is built on the first request (`GET /readyz` builds it eagerly).

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/healthz` | | `{"status": "ok"}` |
| GET | `/readyz` | | builds the guard, `{"status": "ready"}` |
| GET | `/metrics` | | Prometheus text: requests, decisions by action, latency sum and count |
| POST | `/v1/scan` | `{"text": ...}` or `{"scene": ...}` | sanitized text, `results_valid`, `results_score`, findings |
| POST | `/v1/decide` | `scene`, `question`, `options`, optional `risky`, `order`, `fallback`, `meta`, `include_scene` | the guarded decision |
| POST | `/v1/decide/batch` | `{"requests": [...]}` (up to 256) | `{"decisions": [...]}` |

Server settings in the config:

```yaml
server:
  auth_token_env: DMGUARD_TOKEN    # require "Authorization: Bearer <token>" when the variable is set
  max_body_bytes: 2000000
```

Invalid requests (fewer than two options, option index out of range) return 422. In Python:
`decision_model_guard.server.create_app(guard=..., config=..., auth_token=...)` returns the FastAPI app for your own ASGI server.
