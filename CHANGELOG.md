# Changelog

## 0.1.0

First release.

- `DecisionGuard` pipeline: content scanners on untrusted blocks, steering gate, decision checks, policy, audit log.
- Steering gate (erase-and-check) with risky, ordered and unordered rules, detector-led unit selection, trigger mode and
  scanner hints.
- Content scanners: `InvisibleText`, `ForgedTrustBlock`, `VerdictTokens`, `PromptInjection` (Wolf Defender),
  `ClaimDetector` (Decision Steering Gate), `Regex`, `BanSubstrings`, `TokenLimit`; `scan_content` in LLM Guard style.
- Decision checks: `OptionOrder`, `Confidence`.
- Model adapters: Wolf Defender Jev, Jev-like HTTP endpoints, OpenAI-compatible LLM judges, local causal LMs, Hugging Face
  classifiers, plain functions.
- YAML configuration, CLI (`decide`, `scan`, `eval`, `serve`, `init-config`), FastAPI server with bearer auth and
  Prometheus metrics, LangChain integration.
- Steering evaluation with 365 claim templates in 15 languages and twelve sample cases.
- ONNX Runtime backend without torch (`onnx` extra): `WolfDefenderJev.from_pretrained(onnx=...)` / `WolfDefenderJevONNX`,
  `ClaimDetector(onnx=...)`, `PromptInjection(onnx=...)`, `DecisionGuard.default(onnx=...)` and the `onnx` key in the
  config, for the four variants of the Patronus models or your own `.onnx` file.
- `Coverage` guard: scenes longer than the model's context are checked window by window and combined with the gate's
  rules; the built-in adapters expose `context_tokens` and `count_tokens`.
- `Invariance` guard: flags answers that change with the scene format (markdown, key-value, JSON, compact), reworded
  options or alternative questions.
- Calibration: `Calibrator` (temperature scaling, split-conformal threshold, risk thresholds for target false-positive
  rates), `Calibrated` model wrapper, `Conformal` guard, `dmguard calibrate`.
- Robustness audit: `decision_model_guard.evaluate.audit` and `dmguard audit` with ten checks (claim, split claim, forged
  block, verdict token, look-alike letters, padding, format, option wording, option order, neutral controls), a
  scorecard with and without the guard, and `--max-attack-success` for CI.
