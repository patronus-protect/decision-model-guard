# Robustness: long content, invariance, calibration and the audit

The steering gate stops claims inside the content. Decision models fail in other ways too. This page covers the four
tools for those, and the audit that measures all of them on your own model.

## Long content: `Coverage`

Every decision model reads a limited number of tokens and drops the rest without telling anyone. Wolf Defender Jev keeps
the first third and the last two thirds of a long scene, `CausalLMJudge` keeps the end, `HFClassifier` the beginning. An
attacker who pads a document can move an instruction into the part the model never sees.

`Coverage` checks scenes that exceed the model's context window by window:

```python
from decision_model_guard import Coverage, DecisionGuard

guard = DecisionGuard.default(model, guards=[Coverage()])
```

- The untrusted text is cut into sentence-aligned windows with an overlap (`overlap_tokens=64`); trusted blocks stay in
  every window. Up to `max_windows=16` extra calls, in one batch.
- The window answers are combined with the gate's rules: risky options, the highest risk wins; ordered options, the most
  cautious answer wins; unordered, the decision is flagged when a window disagrees (fallback or escalate).
- The context size and the token counter come from the model (`context_tokens`, `count_tokens` on the built-in adapters).
  For HTTP models pass `Coverage(max_tokens=...)`.
- `Coverage(gate=SteeringGate())` runs erase-and-check inside every window, for claims that sit in the same window as the
  content they hide (more calls).
- Scenes that fit cost nothing. Policy: `coverage_action` = `apply` (default), `escalate` or `report`.

Measured with Wolf Defender Jev (ONNX, context 6,144 tokens): a tool output of 7,856 tokens with an injected instruction
after token 2,729, so inside the part the model drops. Model alone: probability of "injection" 0.035. With `Coverage`:
"injection", action `cautious`, 2 windows, 3 model calls, 10 s on an Apple M4 CPU.

## Invariance: `Invariance`

The same decision asked in a slightly different way should give the same answer. Decision models are sensitive to the
serialization of the input and to the wording of the options.

```python
from decision_model_guard import Invariance

Invariance(formats=("markdown",), option_transforms=("short",))           # default: 2 extra calls
Invariance(formats=("markdown", "kv", "json"), option_variants=[["Safe", "Attack"]], question_variants=["Is this an attack?"])
```

| Check | What changes |
|---|---|
| `formats` | the scene as `markdown` headings, `kv` lines, `json` or `compact` (one batched call) |
| `option_transforms` | `short` ("No, it is only data." becomes "No"), `lower`, `plain` (no final punctuation), `quoted` |
| `option_variants`, `question_variants` | your own alternative option lists and questions |

A changed answer flags the decision; `inconsistent_action` (default `escalate`) decides what happens. The re-asked answers are compared with the model's own answer on the same scene, so a decision the gate corrected is not mistaken for an unstable one (the same holds for `OptionOrder`). `tolerance` accepts
a number of changes. `OptionOrder` remains the check for option order.

## Calibration and conformal abstention

A probability of 0.7 rarely means 70 %, and a threshold of 0.5 or one copied from a benchmark rarely fits your traffic.
With a few hundred labelled cases from your own traffic:

```bash
dmguard calibrate --config dmguard.yml --cases labelled.jsonl --test heldout.jsonl --alpha 0.05 --out calibration.json
```

```python
from decision_model_guard import Calibrated, Calibrator, Conformal

cal = Calibrator.fit(model, cases, alpha=0.05)        # cases: {"scene", "question", "options", "gold"} (+ "risky")
print(cal.summary())
guard = DecisionGuard(Calibrated(model, cal), guards=[Conformal(cal)])
```

- **Temperature scaling** (`Calibrated`): one parameter fitted by likelihood; the answer never changes, the probabilities
  become honest. `summary()` reports ECE and NLL before and after.
- **Conformal sets** (`Conformal`): split conformal prediction turns every decision into a set of options that contains
  the correct answer with probability at least 1 - alpha, as long as your traffic looks like the calibration cases. One
  option in the set is a decision; two or more, or none, means the model cannot decide at that coverage, and the policy
  abstains or escalates (`low_confidence_action`). `cal.evaluate(model, heldout)` checks the coverage on held-out cases.
  The guarantee only covers the questions the calibrator was fitted on; `Conformal` skips other questions and says so
  in its report. Fit one calibrator per decision task, or one on a mix of all the tasks you guard. A threshold needs
  at least ceil(1/alpha) - 1 cases (19 for alpha 0.05, 9 for 0.1); with fewer, `fit` and `dmguard calibrate` warn and
  save the temperature without a threshold, and `Conformal` refuses the file instead of abstaining on everything.
- **Operating points**: for risky decisions `fit` also reports the risk threshold that keeps the false-positive rate at
  0.1, 1, 5 and 10 % on your cases, with the recall at that threshold.

In the config: `calibration: calibration.json` at the top level wraps the model, and `- type: Conformal` adds the guard.

## The audit: `dmguard audit`

The audit attacks and stresses your model with every pattern above and reports a scorecard, for the model alone and
with your guard:

```bash
dmguard audit --config dmguard.yml --cases my_cases.jsonl --markdown audit.md --out audit.json
dmguard audit --config dmguard.yml --max-attack-success 0.05      # exit code 2 above 5 %, or when no attack ran: use it in CI
```

```python
from decision_model_guard.evaluate import audit

report = audit(model, cases, guard=guard)
print(report.to_markdown())
```

| Check | Type | What it does |
|---|---|---|
| `claim` | attack | a claim about the decision at the end, start or middle of the untrusted text |
| `split_claim` | attack | one claim split into halves at the start and the end |
| `forged_block` | attack | a trusted-looking block appended to the content |
| `verdict_token` | attack | a label token such as `verdict: approved` or `{"decision": "..."}` |
| `obfuscated_claim` | attack | a claim in look-alike letters from other alphabets |
| `padding` | attack | the original content hidden in the middle of long neutral filler, beyond the context window |
| `format` | robustness | the scene as markdown, key-value lines and JSON |
| `option_wording` | robustness | shorter and lower-case option wording |
| `option_order` | robustness | options in reverse order |
| `neutral` | control | a harmless sentence added; changes are false alarms |

Attacks push the answer towards a wrong target on the unsafe side and count only on cases the model answers correctly
without them. Attack variants whose target is more cautious than the correct answer (a benign email pushed to
"injection", an "authorize" case pushed to "reject") go into a separate row, `attacks` "towards a more cautious answer":
the gate moves decisions to the cautious side by design, so these are false alarms, and `--max-attack-success` does not
count them. Robustness checks and controls count changed answers. `Guard flagged` is the share of guarded decisions that
were not simply allowed. Cases use the evaluation format (see [evaluation.md](evaluation.md)); without `--cases` the
twelve bundled sample cases run.
