# Guards and policy

## SteeringGate

| Parameter | Default | Meaning |
|---|---|---|
| `window` | 3 | longest sentence window erased as one unit |
| `max_variants` | 48 | budget of erased units per decision (edge units first, then a seeded random sample) |
| `selector` | `"all"` | `"all"`: erase every unit once; `"detector"`: only units the detector flags |
| `detector` | | a unit scorer, usually `ClaimDetector()` (needed for `selector="detector"`) |
| `threshold` | 0.3 | detector threshold |
| `scope` | `"all"` | `"unordered"`: use the detector only for unordered decisions, full check otherwise |
| `mode` | `"erase"` | `"trigger"`: any flagged unit starts the full check, otherwise one call |
| `override` | 0.5 | risky decisions: a variant replaces the original only with at least this risk (`rule="absolute"`) |
| `rule` | `"absolute"` | `"delta"`: risk at least `delta` above the original; `"gap"`: at least `delta` above the median variant |
| `max_chars` | 400,000 | characters sent to the model per decision (scene x (1 + variants)); a longer scene gets fewer variants, detector-flagged units first, and the reasons say how many units were checked; `None` turns the cap off |
| `seed` | 0 | sampling seed (decisions are deterministic) |

Text flagged by other scanners (forged blocks, verdict tokens) is always erased, also in detector mode and outside
the `max_variants` budget. When flagged text lies in no unit (for example the only sentence of the content), the
gate cannot check it; the report counts it as `unerased` and the policy escalates the decision.

## OptionOrder

Asks the model again with the options reordered and maps the answers back. `permutations`: `"reverse"` (1 extra call),
`"rotate"` (n-1), `"random"` (`samples`), `"all"`. `average=True` reports the mean probabilities over all orders.

## Confidence

`min_confidence` (top probability) and `min_margin` (lead over the second option).

## Coverage, Invariance, Conformal

`Coverage` checks scenes longer than the model's context window by window, `Invariance` re-asks with other formats and
reworded options, `Conformal` abstains unless the calibrated prediction set holds one option. Details and examples:
[robustness.md](robustness.md).

Guards receive the decision context: a custom guard whose `check()` accepts `risky`, `order` and `fallback` keyword
arguments gets them; older guards with the six positional parameters keep working.

## Policy

| Field | Values | Default |
|---|---|---|
| `gate_action` | `apply`, `escalate`, `report` | `apply` |
| `unstable_action` | `fallback`, `escalate`, `keep` | `fallback` (escalates when no fallback option is set) |
| `scanner_action` | `report`, `escalate`, `cautious`, `block` | `report` |
| `block_scanners` | scanner names | `()` |
| `low_confidence_action` | `abstain`, `escalate` (Confidence, Conformal) | `abstain` |
| `inconsistent_action` | `escalate`, `report` (OptionOrder, Invariance) | `escalate` |
| `coverage_action` | `apply`, `escalate`, `report` (Coverage: take the more cautious window answer, fallback for unordered decisions) | `apply` |
| `fallback_option` | option index | `None` (a per-decision `fallback=` wins) |

The final action is the most severe one triggered: block > abstain > escalate > fallback > cautious > allow.
