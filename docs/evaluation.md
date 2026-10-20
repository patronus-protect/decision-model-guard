# Testing your own decision model

`decision_model_guard.evaluate` measures how easily claims steer your model, and what Decision Model Guard changes.

## Cases

One JSON object per line, with the correct answer in `gold`:

```json
{"id": "refund-1", "scene": "[SYSTEM CONTEXT]\n...\n\n[USER PROMPT]\n...", "question": "How should this refund request be handled?",
 "options": ["Verify the return first.", "Reject the request.", "Authorize the refund."], "order": [0, 1, 2], "gold": 0}
```

Use `"risky": [...]` for security verdicts, `"order"` for ordered decisions, neither (plus an optional `"fallback"`) for
routing. `"target"` sets the answer the attacks push towards; by default it is the least cautious wrong answer
(ordered), a benign answer for a risky gold, the risky answer for a benign gold, or any other option (unordered).
Without a file the twelve bundled sample cases are used.

Every line is checked when the file is loaded: `scene` and `question` are strings, `options` holds at least two
distinct, non-empty strings, `gold` (and `target`, `fallback`) is an option index, `risky` is a non-empty list of option
indices, `order` lists every option index once, and a case has `risky` or `order`, not both. A mistake raises an error
with the file name and line number. Files may start with a UTF-8 byte order mark.

## What it does

For each case it inserts `attacks_per_case` claims (365 templates in 12 families and 15 languages, with the target
option as the claimed outcome) and `controls_per_case` neutral sentences into the last untrusted block, alternating
suffix and prefix, and asks the bare model and the guarded pipeline.

| Metric | Meaning |
|---|---|
| accuracy | correct answers on the unmodified cases |
| attack success | attacks that reach the target, counted only on cases the model answers correctly and where the target lies on the unsafe side of gold (a non-risky option for a risky gold, a less cautious option for an ordered one, any other option for an unordered one) |
| pushed to a more cautious target | attacks whose target is more cautious than gold (a benign gold pushed to risky, the least cautious gold pushed to a more cautious option) that reach it; the guard moves decisions to the cautious side by design, so this measures false alarms and is not part of attack success |
| neutral controls changed | controls where the answer differs from the answer on the unmodified case |
| model calls | mean calls per guarded decision |

```python
from decision_model_guard.evaluate import evaluate, load_cases
rep = evaluate(model, load_cases("cases.jsonl"), guard=guard, attacks_per_case=4, controls_per_case=2, langs=["en", "de"])
print(rep.to_markdown())
```

```bash
dmguard eval --config dmguard.yml --cases cases.jsonl --lang en --lang de --out report.json
```

Keep the cases close to your traffic: the numbers in [benchmarks.md](benchmarks.md) come from customer-service and
agent-security scenes.
