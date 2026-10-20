# Configuration

`build_guard(path_or_dict, model=None)` builds a `DecisionGuard`. `dmguard init-config` writes this example:

```yaml
model:
  type: wolf_defender_jev            # wolf_defender_jev | clef | systemone_http | openai_compatible | causal_lm | hf_classifier | callable
  onnx: int8_int4_embeddings         # ONNX Runtime, no torch: fp32 | fp16 | mixed | int8_int4_embeddings; remove for PyTorch ([hf] extra)
  # device: cuda                     # PyTorch only: cpu | cuda | mps (default: DMGUARD_DEVICE, else cuda when available)
scanners:                            # run in this order on the untrusted blocks
  - type: InvisibleText              # zero-width and bidi characters (removed from what the model sees)
  - type: ForgedTrustBlock           # trusted-looking headers and chat role markers inside the content
  - type: VerdictTokens              # "verdict: safe", "label=benign" inside the content
  - type: PromptInjection            # Wolf Defender; a finding blocks (see policy.block_scanners)
    threshold: 0.5
    onnx: int8_int4_embeddings
  - type: ClaimDetector              # claims about the decision; the gate reuses this detector
    threshold: 0.3
    onnx: int8_int4_embeddings
gate:                                # the steering gate; "gate: false" switches it off
  selector: detector                 # all (erase every unit) | detector (only units the claim detector flags)
  scope: unordered                   # all | unordered (detector for unordered decisions, full check for risky and ordered)
  threshold: 0.3
guards:
  - type: Coverage                   # long content: check the parts the model would cut off, window by window
  - type: Invariance                 # same scene as markdown, shorter option wording: the answer must not change
    formats: [markdown]              # markdown | kv | json | compact
    option_transforms: [short]       # short | lower | plain | quoted
  - type: Confidence
    min_confidence: 0.55
  - type: OptionOrder
    permutations: reverse            # reverse | rotate | random | all
  # - type: Conformal                # abstain unless the conformal set has one option (dmguard calibrate)
# calibration: calibration.json      # temperature + conformal threshold from your labelled cases
policy:
  gate_action: apply                 # apply | escalate | report
  unstable_action: fallback          # fallback | escalate | keep
  scanner_action: report             # report | escalate | cautious | block
  block_scanners: [PromptInjection]  # names of configured scanners whose finding always blocks
  low_confidence_action: abstain     # abstain | escalate (Confidence, Conformal)
  inconsistent_action: escalate      # escalate | report (OptionOrder, Invariance)
  coverage_action: apply             # apply | escalate | report
audit:
  path: logs/decisions.jsonl         # one JSON line per decision
  include_text: false                # true stores scene and question text instead of hashes
use_sanitized: true                  # the model reads the sanitized scene (invisible characters removed)
boundary_note: false                 # true appends a source-trust note to the question; measure per model first
# server:                            # dmguard serve
#   auth_token_env: DMGUARD_TOKEN    # require a bearer token; the server does not start while the variable is unset
#   max_body_bytes: 2000000
#   max_scene_chars: 100000
#   max_meta_bytes: 4096
#   request_timeout: 120
#   docs: true                       # false removes /docs, /redoc and /openapi.json
```

## Model types

| type | Parameters |
|---|---|
| `wolf_defender_jev` | `repo` (default `patronus-studio/wolf-defender-jev`), `device`, `revision`, `max_tokens`, `max_sq`, `onnx` (with `threads`, `providers`) |
| `systemone_http` | `url`, `model`, `headers`, `api_key_env`, `timeout`, `retries`, `concurrency`, `state_key` |
| `openai_compatible` | `base_url`, `model`, `api_key_env`, `headers`, `timeout`, `retries`, `concurrency`, `top_logprobs`, `system_prompt` |
| `causal_lm` | `model_id`, `device`, `dtype`, `max_tokens`, `system_prompt` |
| `hf_classifier` | `model_id`, `option_labels`, `device`, `max_length`, `batch_size`, `subfolder` |
| `callable` | `target: "package.module:name"` (a decide function, a model object, or a class / factory called with the remaining keys), `name` |

Unknown keys, at the top level and inside every section, are an error, and so are `block_scanners` names that are not
among the configured scanners: a typo must not remove protection. `gate: false` disables the gate. With `selector: detector` the gate reuses a configured `ClaimDetector` scanner (or
creates one, taking `threshold`, `device` and `onnx` from the gate section). `PromptInjection` and `ClaimDetector` take
`onnx` as well. `audit` takes `path`, `include_text` and `max_bytes`; `server` takes `auth_token_env` and `max_body_bytes`.

## Guards

`Confidence` (`min_confidence`, `min_margin`), `OptionOrder` (`permutations`, `samples`, `average`), `Coverage`
(`max_tokens`, `overlap_tokens`, `max_windows`, `margin`, `risk_threshold`; `gate: true` runs the configured gate inside
every window), `Invariance` (`formats`, `option_transforms`, `option_variants`, `question_variants`, `tolerance`) and
`Conformal` (`calibration: path.json`, or the top-level `calibration`). A top-level `calibration: path.json` also wraps
the model in `Calibrated`; with both, the guard reuses the top-level calibrator (the temperature is applied once), and
two different files are an error. See [robustness.md](robustness.md).
