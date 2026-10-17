"""Decision models behind HTTP: Jev-like named-question endpoints and OpenAI-compatible chat servers."""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from ..util import log, normalize

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _post(url: str, body: dict, headers: dict, timeout: float, retries: int) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"HTTP {e.code} from {url}") from None   # never echo response bodies or credentials
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise RuntimeError(f"transport failure for {url}") from None
        time.sleep(min(2 ** attempt * 0.5, 8))
    raise RuntimeError("unreachable")


def _auth_headers(headers: dict | None, api_key_env: str | None) -> dict:
    h = dict(headers or {})
    if api_key_env and os.environ.get(api_key_env):
        h["Authorization"] = f"Bearer {os.environ[api_key_env]}"
    return h


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
        ans = _post(self.url, body, self.headers, self.timeout, self.retries)["answers"]["q"]
        probs = ans.get("probabilities") or ans.get("distribution")
        if isinstance(probs, list):                                   # [{"value": "o0", "probability": 0.3}, ...]
            probs = {p.get("value"): p.get("probability", 0.0) for p in probs}
        if probs is None:                                             # only a choice with a confidence
            c = float(ans.get("confidence", 1.0))
            probs = {i: (c if i == ans.get("choice") else (1 - c) / max(1, len(ids) - 1)) for i in ids}
        return normalize([float(probs.get(i, 0.0)) for i in ids])

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        if len(scenes) == 1 or self.concurrency <= 1:
            return [self._one(s, question, options) for s in scenes]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(lambda s: self._one(s, question, options), scenes))


class OpenAICompatible:
    """An LLM judge behind an OpenAI-compatible chat completions endpoint (vLLM, llama.cpp server, Ollama, LM Studio,
    hosted APIs). The options are listed with letters; the probabilities come from the top log-probabilities of the
    first answer token. Servers without log-probabilities fall back to a one-hot answer."""

    name = "openai-compatible"
    SYSTEM = "You are a decision model. Read the input and answer the question with the letter of exactly one option."

    def __init__(self, base_url: str, model: str, api_key_env: str | None = "OPENAI_API_KEY", headers: dict | None = None,
                 timeout: float = 60.0, retries: int = 2, concurrency: int = 4, top_logprobs: int = 20, system_prompt: str | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.timeout, self.retries, self.concurrency, self.top = model, timeout, retries, concurrency, top_logprobs
        self.headers = _auth_headers(headers, api_key_env)
        self.system = system_prompt or self.SYSTEM

    def _one(self, scene: str, question: str, options: Sequence[str]) -> list[float]:
        letters = LETTERS[: len(options)]
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
        body = {"model": self.model, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": self.top,
                "messages": [{"role": "system", "content": self.system},
                             {"role": "user", "content": f"Input:\n{scene}\n\nQuestion: {question}\nOptions:\n{opts}\n\nAnswer with one letter."}]}
        r = _post(self.url, body, self.headers, self.timeout, self.retries)["choices"][0]
        scores = [0.0] * len(options)
        content = ((r.get("logprobs") or {}).get("content") or [])
        if content:
            for t in content[0].get("top_logprobs", []):
                tk = t.get("token", "").strip().strip(".:)").upper()
                if len(tk) == 1 and tk in letters:
                    scores[letters.index(tk)] += math.exp(float(t.get("logprob", -100)))
        if sum(scores) == 0:
            txt = (r.get("message", {}).get("content") or "").strip().upper()[:1]
            if txt in letters:
                scores[letters.index(txt)] = 1.0
            else:
                log.warning("openai-compatible: no usable answer, returning a uniform distribution")
                return [1.0 / len(options)] * len(options)
        return normalize(scores)

    def decide_batch(self, scenes: Sequence[str], question: str, options: Sequence[str]) -> list[list[float]]:
        if len(scenes) == 1 or self.concurrency <= 1:
            return [self._one(s, question, options) for s in scenes]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(lambda s: self._one(s, question, options), scenes))
