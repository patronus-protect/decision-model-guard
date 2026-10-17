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
