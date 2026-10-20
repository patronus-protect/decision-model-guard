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
| POST | `/v1/scan` | `{"text": ...}` (all of it untrusted, offsets into the text) or `{"scene": ...}` (trusted blocks are skipped) | sanitized text, `results_valid`, `results_score`, findings |
| POST | `/v1/decide` | `scene`, `question`, `options`, optional `risky`, `order`, `fallback`, `meta`, `include_scene` | the guarded decision |
| POST | `/v1/decide/batch` | `{"requests": [...]}` (up to 256) | `{"decisions": [...]}` |

Server settings in the config (unknown keys are an error):

```yaml
server:
  auth_token_env: DMGUARD_TOKEN    # require "Authorization: Bearer <token>"; the server does not start while it is unset
  max_body_bytes: 2000000          # counted on the bytes received, so chunked requests are capped too (413)
  max_scene_chars: 100000          # longer scenes and texts get 413
  max_meta_bytes: 4096             # meta is written to the audit log as it is; larger meta gets 413
  request_timeout: 120             # seconds until the client gets 504 (the running computation is not interrupted)
  docs: true                       # false removes /docs, /redoc and /openapi.json
```

Without `auth_token_env` the server requires a token only when `DMGUARD_TOKEN` is set; the startup log says whether
authentication is on, and `dmguard serve` warns when it listens on another host than localhost without it. The scheme
is case-insensitive and a 401 carries `WWW-Authenticate: Bearer`. `/healthz`, `/readyz`, `/metrics` and the docs
routes need no token.

Invalid requests (fewer than two options, option index out of range) return 422. In Python:
`decision_model_guard.server.create_app(guard=..., config=..., auth_token=...)` returns the FastAPI app for your own ASGI server.
