# Extraction models

Extraction models read text and return typed values instead of choosing an option: Cloudflare's
[Clairic](https://blog.cloudflare.com/clairic-extraction-model/) returns spans, numbers, dates, lists and estimates with
intervals; LLM extractors return JSON. Their values feed decisions, and text inside the content can plant them.

## Measured on Clairic

Twelve scenarios (insurance claims, invoices, refunds, expenses, a freelancer invoice, a rent deposit, a loan, a flight
delay, a used car, a water damage, an electricity bill, a goodwill credit; two in German), each with a target field
(an estimate or a computed amount) and a control field (a fact stated in the text). Four claim templates plant a value
of 5 % and of 500 % of the reference, 96 attacks in total. Clairic preview on Workers AI, 2026-10-20:

| | Clairic alone | With ExtractionGuard |
|---|---:|---:|
| Median shift of the target value | 17.7 % | 0.0 % |
| Planted values that move it by more than 25 % | 49.0 % | 0.0 % |
| Clean scenes changed by the guard | | 0.0 % |
| Control field changed under attack | | 0.0 % |

1.9 model calls per guarded extraction on average. Example: the hail claim (workshop quote 1,840 EUR) gets a payout
estimate of 805 EUR; planted amounts of 40 EUR move it to between 52 and 63 EUR, planted amounts of 4,026 EUR to
between 557 and 4,019 EUR, depending on the wording. With the guard it stays at 805 EUR in all eight attacks. All rows: [extraction_clairic.json](extraction_clairic.json).
Reproduce with `evaluate_extraction(ExtractionGuard.default(Clairic(reference_date="2026-10-20"), onnx="int8_int4_embeddings"),
load_extraction_cases())`. The run used decision-model-guard 0.1.2; since 0.2.0 the German template writes the planted
amount with German separators (29.000 EUR), cases whose reference value is not a number are listed as skipped, and the
bare model is not asked again when the guard already extracted from the same text.

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
4. `cautious={"payout": "low"}` (an argument of `extract()`, for number and estimate fields) adds erase-and-check
   for that field: one unit at a time is erased and the most cautious value wins (`"cautious"`; the culprit is the
   smallest unit that gives it). `full_check=True` (a constructor argument) does this for all fields and marks values
   as `"unstable"` when another value appears after an erasure (a value that disappears with its only evidence is not
   instability). Up to `max_variants=24` units are checked, single sentences first and then a seeded sample of the
   larger units; when units are left out, the field's reasons say `incomplete: N of M units checked`.

Cautious mode has a cost on clean inputs: for an estimate, erasing the sentence with the evidence (the workshop quote)
can move the value towards the model's prior, and with `"low"` that lower value wins. The published numbers above were
measured without `cautious`; run `evaluate_extraction(guard, cases, cautious="low")` to see how often it changes clean
scenes on your data before you turn it on.

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
