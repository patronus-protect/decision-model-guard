# Decision models

The only contract: `decide_batch(scenes, question, options) -> list[list[float]]`, one probability per option for each
scene. The gate sends the original and all variants in one call, so batching inside your model pays off.

## Wolf Defender Jev

```python
from decision_model_guard.models import WolfDefenderJev
model = WolfDefenderJev.from_pretrained()                       # patronus-studio/wolf-defender-jev, CPU by default
model = WolfDefenderJev.from_pretrained("/path/to/folder", device="cuda", trust_remote_code=True)
```

The repository ships its own model code (`wolf_defender_jev.py`), which the PyTorch path imports from the downloaded
folder and runs. For `patronus-studio/wolf-defender-jev` the revision is pinned to the one this release was tested with
(`decision_model_guard.models.wolf_defender_jev.REVISION`; pass `revision=` to choose another); any other repository or
a local folder needs `trust_remote_code=True`. The ONNX path below runs no code from the repository. Batches are cut by
a token budget (`max_tokens=8192`, `max_sq=8e6`), so long scenes run alone and memory stays bounded.

On ONNX Runtime, without torch (`pip install "decision-model-guard[onnx]"`):

```python
model = WolfDefenderJev.from_pretrained(onnx="int8_int4_embeddings")   # returns a WolfDefenderJevONNX
model = WolfDefenderJev.from_pretrained("/path/to/folder", onnx="fp32", threads=4)
```

`onnx` is one of `fp32`, `fp16`, `mixed`, `int8_int4_embeddings` (118 MB, the default of `WolfDefenderJevONNX`) or a path
to an `.onnx` file; only that graph is downloaded. FP32 and FP16 give the same answers as the PyTorch weights; the INT8
graphs agree on 99.2 to 99.3 % of held-out stress decisions. Truncation and token-budget batching are the same as on
PyTorch. `threads` and `providers` go to the ONNX Runtime session.

## Jev-like HTTP endpoints

```python
from decision_model_guard.models import SystemOneHTTP
model = SystemOneHTTP("http://127.0.0.1:8008/v1/systemone", model="kev", api_key_env="KEV_TOKEN", concurrency=4)
```

Request: `{"state": {"content": scene}, "questions": {"q": {"type": "choice", "instructions": question, "criteria":
{"o0": option0, ...}}}}`. Answers with `probabilities` (dict or list) or only `choice` + `confidence` are both understood.

## LLM judges

```python
from decision_model_guard.models import OpenAICompatible, CausalLMJudge
judge = OpenAICompatible("http://localhost:8000/v1", model="Qwen/Qwen2.5-7B-Instruct", api_key_env=None)   # vLLM, llama.cpp, Ollama
judge = CausalLMJudge("Qwen/Qwen2.5-0.5B-Instruct")                                                       # local transformers
```

Options are listed with letters; the probabilities are the (log-)probabilities of the answer letters, renormalised over
the options. Servers without log-probabilities fall back to a one-hot answer.

## Classifiers

```python
from decision_model_guard.models import HFClassifier
model = HFClassifier("patronus-studio/wolf-defender-prompt-injection-small")   # options: [benign, injection]
```

## Credentials

HTTP adapters read keys from the environment variable you name (`api_key_env`) and never log them. They retry 408,
409, 429, 5xx and dropped connections (`retries`), not other client errors, and raise `RuntimeError` with the status or
the transport reason. Response bodies are not echoed, except the code and message of a service error envelope
(`{"success": false, "errors": [...]}`, as Workers AI sends it), which are validation messages; such an error is not
retried. In a batch, the first error stops the requests that have not started. An answer that names no option raises
instead of turning into a uniform distribution: `OpenAICompatible` without log-probabilities reads the option text or
exactly one option letter ("B", "Answer: B"), and `SystemOneHTTP` and `Clef` accept a `choice` id or option text. `OpenAICompatible` without `api_key_env` sends `OPENAI_API_KEY` only to `api.openai.com`; for any
other server, name the variable explicitly.

## Cloudflare Workers AI: Clef

```python
from decision_model_guard.models import Clef
model = Clef(variant="clef")                                   # CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN
model = Clef(variant="clef-flash", proxy_url="http://127.0.0.1:8787/run")   # a Worker with an AI binding
```

Options become choice criteria; one request per scene, `concurrency=8` in parallel. The extraction model `Clairic`
lives next to it; see [extraction.md](extraction.md).
