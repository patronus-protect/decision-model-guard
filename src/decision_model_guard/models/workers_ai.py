"""Cloudflare Workers AI: Clef (decision model) and Clairic (extraction model).

Two ways to reach Workers AI:
- the REST API: account id from CLOUDFLARE_ACCOUNT_ID, API token from the environment variable named in `api_token_env`
  (default CLOUDFLARE_API_TOKEN; the token needs the Workers AI permission). Credentials are read from the environment
  and never logged.
- `proxy_url`: any endpoint that answers POST <proxy_url>/<model> with the Workers AI response, e.g. a Worker with an AI
  binding (`env.AI.run(model, body)`) or an AI Gateway route.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..util import normalize
from .http import _auth_headers, _post

API = "https://api.cloudflare.com/client/v4"


class _WorkersAI:
    def __init__(self, model: str, account_id: str | None = None, api_token_env: str = "CLOUDFLARE_API_TOKEN",
                 proxy_url: str | None = None, timeout: float = 60.0, retries: int = 2, concurrency: int = 8):
        self.model, self.timeout, self.retries, self.concurrency = model, timeout, retries, concurrency
        if proxy_url:
            self.url, self.headers = f"{proxy_url.rstrip('/')}/{model}", {}
        else:
            account = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
            if not account:
                raise ValueError("set CLOUDFLARE_ACCOUNT_ID (or pass account_id), or use proxy_url")
            if not os.environ.get(api_token_env):
                raise ValueError(f"set the environment variable {api_token_env} to a Workers AI API token, or use proxy_url")
            self.url, self.headers = f"{API}/accounts/{account}/ai/run/{model}", _auth_headers(None, api_token_env)

    def _run(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = _post(self.url, payload, self.headers, self.timeout, self.retries)
        if isinstance(out, dict) and "result" in out:
            if out.get("success") is False:
                raise RuntimeError(f"Workers AI error for {self.model}")      # error bodies are not echoed
            return out["result"]
        return out

    def _map(self, fn, items: Sequence[Any]) -> list[Any]:
        if len(items) <= 1 or self.concurrency <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(fn, items))


class Clef(_WorkersAI):
    """Cloudflare's decision model on Workers AI as a decision model (`decide_batch`). `variant`: "clef" or "clef-flash".
    The options become choice criteria o0, o1, ...; one request per scene, run in parallel (`concurrency`)."""

    def __init__(self, variant: str = "clef", **kw):
        super().__init__(f"@cf/cloudflare/{variant}", **kw)
        self.variant, self.name = variant, f"cloudflare/{variant}"

    def _one(self, scene: str, question: str, options: Sequence[str]) -> list[float]:
        ids = [f"o{i}" for i in range(len(options))]
        res = self._run({"model": self.variant, "state": scene,
                         "questions": {"q": {"type": "choice", "instructions": question, "criteria": dict(zip(ids, options, strict=True))}}})
        probs = res["answers"]["q"].get("probabilities") or {}
        return normalize([float(probs.get(i, 0.0)) for i in ids])

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        return self._map(lambda s: self._one(s, question, options), list(scenes))


class Clairic(_WorkersAI):
    """Cloudflare's extraction model on Workers AI as an extraction model (`extract`): typed fields (text, number, date,
    list, estimate) -> {field: answer} per scene, with spans, confidence, p_null and estimate intervals."""

    name = "cloudflare/clairic"

    def __init__(self, reference_date: str | None = None, **kw):
        super().__init__("@cf/cloudflare/clairic", **kw)
        self.reference_date = reference_date

    def _one(self, scene: str, fields: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {"model": "clairic", "state": scene, "fields": fields}
        if self.reference_date:
            payload["reference_date"] = self.reference_date
        return self._run(payload)["answers"]

    def extract(self, scenes: Sequence[str], fields: dict[str, dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
        return self._map(lambda s: self._one(s, fields), list(scenes))
