# Configuration

`build_guard(path_or_dict, model=None)` builds a `DecisionGuard`. `dmguard init-config` writes this example:

```yaml
model:
  type: wolf_defender_jev            # wolf_defender_jev | systemone_http | openai_compatible | causal_lm | hf_classifier | callable
  device: cpu
scanners:
  - type: InvisibleText
  - type: ForgedTrustBlock
  - type: VerdictTokens
  - type: PromptInjection
    threshold: 0.5
  - type: ClaimDetector
    threshold: 0.3
gate:
  selector: detector                 # all | detector
  scope: unordered                   # all | unordered
  threshold: 0.3
guards:
  - type: Confidence
    min_confidence: 0.55
  - type: OptionOrder
    permutations: reverse
policy:
  unstable_action: fallback
  scanner_action: report
  block_scanners: [PromptInjection]
audit:
  path: logs/decisions.jsonl
  include_text: false
use_sanitized: true
boundary_note: false
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

`gate: false` disables the gate. With `selector: detector` the gate reuses a configured `ClaimDetector` scanner (or
creates one, taking `threshold`, `device` and `onnx` from the gate section). `PromptInjection` and `ClaimDetector` take
`onnx` as well. `audit` takes `path`, `include_text` and `max_bytes`; `server` takes `auth_token_env` and `max_body_bytes`.
