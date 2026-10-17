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
| `seed` | 0 | sampling seed (decisions are deterministic) |

Text flagged by other scanners (forged blocks, verdict tokens) is always erased, also in detector mode.

## OptionOrder

Asks the model again with the options reordered and maps the answers back. `permutations`: `"reverse"` (1 extra call),
`"rotate"` (n-1), `"random"` (`samples`), `"all"`. `average=True` reports the mean probabilities over all orders.

## Confidence

`min_confidence` (top probability) and `min_margin` (lead over the second option).

## Policy

| Field | Values | Default |
|---|---|---|
| `gate_action` | `apply`, `escalate`, `report` | `apply` |
| `unstable_action` | `fallback`, `escalate`, `keep` | `fallback` (escalates when no fallback option is set) |
| `scanner_action` | `report`, `escalate`, `cautious`, `block` | `report` |
| `block_scanners` | scanner names | `()` |
| `low_confidence_action` | `abstain`, `escalate` | `abstain` |
| `inconsistent_action` | `escalate`, `report` | `escalate` |
| `fallback_option` | option index | `None` (a per-decision `fallback=` wins) |

The final action is the most severe one triggered: block > abstain > escalate > fallback > cautious > allow.
