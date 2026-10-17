# Decision models

The only contract: `decide_batch(scenes, question, options) -> list[list[float]]`, one probability per option for each
scene. The gate sends the original and all variants in one call, so batching inside your model pays off.

## Wolf Defender Jev

```python
from steerguard.models import WolfDefenderJev
model = WolfDefenderJev.from_pretrained()                       # patronus-studio/wolf-defender-jev, CPU by default
model = WolfDefenderJev.from_pretrained("/path/to/folder", device="cuda")
```

The repository ships its own model code (`wolf_defender_jev.py`), which is imported from the downloaded folder. Batches
are cut by a token budget (`max_tokens=8192`, `max_sq=8e6`), so long scenes run alone and memory stays bounded.

## Jev-like HTTP endpoints

```python
from steerguard.models import SystemOneHTTP
model = SystemOneHTTP("http://127.0.0.1:8008/v1/systemone", model="kev", api_key_env="KEV_TOKEN", concurrency=4)
```

Request: `{"state": {"content": scene}, "questions": {"q": {"type": "choice", "instructions": question, "criteria":
{"o0": option0, ...}}}}`. Answers with `probabilities` (dict or list) or only `choice` + `confidence` are both understood.

## LLM judges

```python
from steerguard.models import OpenAICompatible, CausalLMJudge
judge = OpenAICompatible("http://localhost:8000/v1", model="Qwen/Qwen2.5-7B-Instruct", api_key_env=None)   # vLLM, llama.cpp, Ollama
judge = CausalLMJudge("Qwen/Qwen2.5-0.5B-Instruct")                                                       # local transformers
```

Options are listed with letters; the probabilities are the (log-)probabilities of the answer letters, renormalised over
the options. Servers without log-probabilities fall back to a one-hot answer.

## Classifiers

```python
from steerguard.models import HFClassifier
model = HFClassifier("patronus-studio/wolf-defender-prompt-injection-small")   # options: [benign, injection]
```

## Credentials

HTTP adapters read keys from the environment variable you name (`api_key_env`) and never log them or echo response
bodies in errors.
