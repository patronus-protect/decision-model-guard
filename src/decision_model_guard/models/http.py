"""Decision models behind HTTP: Jev-like named-question endpoints and OpenAI-compatible chat servers."""
from __future__ import annotations

import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

from ..util import normalize

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


RETRY = (408, 409, 429, 500, 502, 503, 504)


def _api_errors(body: bytes) -> str:
    """Code and message of a Workers AI style error envelope ({"success": false, "errors": [...]}), else "". These are
    validation messages of the service; other response bodies are never echoed (they can hold content)."""
    try:
        out = json.loads(body.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return ""
    errs = out.get("errors") if isinstance(out, dict) and out.get("success") is False else None
    if not isinstance(errs, list):
        return ""
    parts = [": ".join(str(e[k]) for k in ("code", "message") if isinstance(e, dict) and e.get(k) is not None) for e in errs]
    return "; ".join(p for p in parts if p)[:300]


def _post(url: str, body: dict, headers: dict, timeout: float, retries: int) -> dict:
    """POST JSON, retry 408/409/429/5xx and transport failures, raise RuntimeError otherwise. Errors name the status or
    the transport reason and, for a service error envelope, its code and message; never credentials or other bodies."""
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            try:
                return json.loads(raw)
            except ValueError:
                raise RuntimeError(f"unexpected response from {url}: not JSON") from None
        except urllib.error.HTTPError as e:
            detail = _api_errors(e.read(65536) if e.fp else b"")
            if e.code not in RETRY or detail or attempt == retries:   # a validation error repeats: no retry
                raise RuntimeError(f"HTTP {e.code} from {url}" + (f": {detail}" if detail else "")) from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            if attempt == retries:
                reason = getattr(e, "reason", None) or type(e).__name__
                raise RuntimeError(f"transport failure for {url}: {reason}") from None
        time.sleep(min(2 ** attempt * 0.5, 8))
    raise RuntimeError("unreachable")


def _get(out: dict, url: str, *keys):
    """out[k1][k2]... or a RuntimeError naming the missing key (no body echo)."""
    for k in keys:
        if not isinstance(out, (dict, list)) or (isinstance(out, dict) and k not in out) or (isinstance(out, list) and not out):
            raise RuntimeError(f"unexpected response from {url}: no {k!r}")
        out = out[k]
    return out


def _map(fn, items: Sequence, concurrency: int) -> list:
    """fn over items in parallel; the first error cancels the requests that have not started and is raised."""
    if len(items) <= 1 or concurrency <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(fn, x) for x in items]
        done, _ = wait(futs, return_when=FIRST_EXCEPTION)
        bad = next((f for f in futs if f in done and f.exception() is not None), None)
        if bad is not None:
            for f in futs:
                f.cancel()
            raise bad.exception()
        return [f.result() for f in futs]


def _auth_headers(headers: dict | None, api_key_env: str | None) -> dict:
    h = dict(headers or {})
    if api_key_env and os.environ.get(api_key_env):
        h["Authorization"] = f"Bearer {os.environ[api_key_env]}"
    return h


def answer_index(text: str, options: Sequence[str]) -> int | None:
    """The option a free-text answer names: the option text itself, or exactly one standalone option letter ("B",
    "Answer: B", "(b)"). None when the answer names none or several."""
    t = text.strip()
    norm = lambda x: " ".join(x.lower().strip().rstrip(".!").split())  # noqa: E731
    hit = [i for i, o in enumerate(options) if norm(t) == norm(o)]
    if hit:
        return hit[0]
    letters = {m.group(1).upper() for m in re.finditer(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", re.sub(r"(?i)^\s*answer\s*[:=-]?", "", t))}
    valid = [c for c in letters if c in LETTERS[: len(options)]]
    return LETTERS.index(valid[0]) if len(valid) == 1 else None


def choice_probs(ans: dict, options: Sequence[str], url: str) -> list[float]:
    """Option probabilities from a named-question answer: `probabilities` (dict or list) or `choice` (an id o<i> or the
    option text) with an optional `confidence`. Raises when the answer names no option."""
    ids = [f"o{i}" for i in range(len(options))]
    probs = ans.get("probabilities") or ans.get("distribution") if isinstance(ans, dict) else None
    if isinstance(probs, list):                                       # [{"value": "o0", "probability": 0.3}, ...]
        probs = {p.get("value"): p.get("probability", 0.0) for p in probs if isinstance(p, dict)}
    if not probs and isinstance(ans, dict) and ans.get("choice") is not None:   # only a choice with a confidence
        ch = ans["choice"]
        k = ids.index(ch) if ch in ids else next((i for i, o in enumerate(options) if str(ch).strip().lower() == o.strip().lower()), None)
        if k is None:
            raise RuntimeError(f"unexpected answer from {url}: the choice is none of the options")
        c = float(ans.get("confidence", 1.0))
        probs = {i: (c if j == k else (1 - c) / max(1, len(ids) - 1)) for j, i in enumerate(ids)}
    if not isinstance(probs, dict) or sum(float(probs.get(i, 0.0)) for i in ids) <= 0:
        raise RuntimeError(f"unexpected answer from {url}: no probabilities for the options")
    return normalize([float(probs.get(i, 0.0)) for i in ids])


class SystemOneHTTP:
    """Any HTTP endpoint with the named-question interface of Jev-like models (a state plus typed questions; answers carry
    a choice and its probabilities), for example a local `kev.serve` server.

    The scene goes into `state[state_key]`; the options become choice criteria with ids o0, o1, ... Requests run in
    parallel (`concurrency`) and retry on 408/429/5xx. Credentials come from the environment variable named in
    `api_key_env` and are never logged."""

    name = "systemone-http"

    def __init__(self, url: str, model: str = "", headers: dict | None = None, api_key_env: str | None = None,
                 timeout: float = 60.0, retries: int = 2, concurrency: int = 4, state_key: str = "content"):
        self.url, self.model, self.timeout, self.retries = url, model, timeout, retries
        self.concurrency, self.state_key = concurrency, state_key
        self.headers = _auth_headers(headers, api_key_env)

    def _one(self, scene: str, question: str, options: Sequence[str]) -> list[float]:
        ids = [f"o{i}" for i in range(len(options))]
        body: dict = {"state": {self.state_key: scene}, "questions": {"q": {"type": "choice", "instructions": question,
                                                                            "criteria": dict(zip(ids, options, strict=True))}}}
        if self.model:
            body["model"] = self.model
        return choice_probs(_get(_post(self.url, body, self.headers, self.timeout, self.retries), self.url, "answers", "q"), options, self.url)

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        """One named-question request per scene, in parallel."""
        return _map(lambda s: self._one(s, question, options), list(scenes), self.concurrency)


class OpenAICompatible:
    """An LLM judge behind an OpenAI-compatible chat completions endpoint (vLLM, llama.cpp server, Ollama, LM Studio,
    hosted APIs). The options are listed with letters; the probabilities come from the top log-probabilities of the
    first answer token. Servers without log-probabilities fall back to a one-hot answer.

    `api_key_env` names the environment variable with the API key. Without it, OPENAI_API_KEY is sent only to
    api.openai.com, never to another host."""

    name = "openai-compatible"
    SYSTEM = "You are a decision model. Read the input and answer the question with the letter of exactly one option."

    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, headers: dict | None = None,
                 timeout: float = 60.0, retries: int = 2, concurrency: int = 4, top_logprobs: int = 20, system_prompt: str | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.timeout, self.retries, self.concurrency, self.top = model, timeout, retries, concurrency, top_logprobs
        if api_key_env is None and urllib.parse.urlsplit(self.url).hostname == "api.openai.com":
            api_key_env = "OPENAI_API_KEY"
        self.headers = _auth_headers(headers, api_key_env)
        self.system = system_prompt or self.SYSTEM

    def _one(self, scene: str, question: str, options: Sequence[str]) -> list[float]:
        letters = LETTERS[: len(options)]
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
        body = {"model": self.model, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": self.top,
                "messages": [{"role": "system", "content": self.system},
                             {"role": "user", "content": f"Input:\n{scene}\n\nQuestion: {question}\nOptions:\n{opts}\n\nAnswer with one letter."}]}
        r = _get(_post(self.url, body, self.headers, self.timeout, self.retries), self.url, "choices", 0)
        scores = [0.0] * len(options)
        content = ((r.get("logprobs") or {}).get("content") or [])
        if content:
            for t in content[0].get("top_logprobs", []):
                tk = t.get("token", "").strip().strip(".:)").upper()
                if len(tk) == 1 and tk in letters:
                    scores[letters.index(tk)] += math.exp(float(t.get("logprob", -100)))
        if sum(scores) == 0:                                          # no log-probabilities: read the answer text
            k = answer_index((r.get("message") or {}).get("content") or "", options)
            if k is None:                                             # never a silent uniform answer (option 0, allow)
                raise RuntimeError(f"no usable answer from {self.url}: expected one of the letters {', '.join(letters)}")
            scores[k] = 1.0
        return normalize(scores)

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        """Letter probabilities from the top log-probabilities, one request per scene."""
        return _map(lambda s: self._one(s, question, options), list(scenes), self.concurrency)
