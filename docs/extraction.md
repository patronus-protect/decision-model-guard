# Extraction models

Extraction models read text and return typed values instead of choosing an option: Cloudflare's
[Clairic](https://blog.cloudflare.com/clairic-extraction-model/) returns spans, numbers, dates, lists and estimates with
intervals; LLM extractors return JSON. Their values feed decisions, and text inside the content can plant them. `evaluate_extraction(guard, load_extraction_cases())`
measures how far planted values move a given extractor, with and without the guard.

## ExtractionGuard

```python
from decision_model_guard import ExtractionGuard
from decision_model_guard.models import Clairic

guard = ExtractionGuard.default(Clairic(), onnx="int8_int4_embeddings")      # rule scanners + claim detector
r = guard.extract(scene, fields)                                               # fields in Clairic's format
r.values, r.fields["payout"].action, r.fields["payout"].culprit, r.flagged
```

1. Content scanners and the claim detector flag pieces of untrusted text; trusted blocks are never touched. Only the
   smallest flagged units are erased, so a claim does not take the facts around it along.
2. The model extracts from the original scene and from the scene without the flagged text, in one batch (one extra
   call; scenes without flags cost nothing extra).
3. A field whose value changes when the flagged text is gone takes the cleaned value (`action="cleaned"`). A value whose
   span lies inside flagged text is reported.
4. `cautious={"payout": "low"}` adds erase-and-check over every unit for that field: one unit at a time is erased and
   the most cautious value wins (`"cautious"`). `full_check=True` does this for all fields and marks values as
   `"unstable"` when another value appears after an erasure (a value that disappears with its only evidence is not
   instability).

Any model with `extract(scenes, fields) -> [{field: answer}]` works; answers use Clairic's shape (`value`, `span`,
`median`, `p10`, `p90`, `confidence`, `p_null`). `rel_tol` (default 10 %) decides when two numbers count as equal.

## Cloudflare Workers AI

`Clef` (decision model, `decide_batch`) and `Clairic` (extraction model, `extract`) call Workers AI in two ways:

- REST API: set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (a token with the Workers AI permission). Credentials
  come from the environment only and are never logged.
- `proxy_url`: a Worker with an AI binding that answers `POST /run/<model>`, for example
  [examples/workers_ai_proxy](../examples/workers_ai_proxy). Run it with `npx wrangler dev`; it uses your wrangler login,
  so the SDK needs no token.

In a config: `model: {type: clef, variant: clef-flash, proxy_url: http://127.0.0.1:8787/run}`.
