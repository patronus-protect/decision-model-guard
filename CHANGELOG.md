# Changelog

## 0.2.0 (2026-10-20)

This release fixes guard bypasses found in a QA round on 0.1.2 and changes how scenes are parsed, so some decisions
come out differently. Read "Changed" before upgrading.

### Security

- Untrusted text could be treated as trusted application context: a trusted header after text without a header of
  its own, header lines inside text added with `Scene().user()` or `Scene().tool_output()`, special characters or
  overlong values in a tool or source name, and blocks added with `Scene.add(..., trusted=False)`. Such text was
  neither scanned nor erased. Fixed: the pipeline takes block trust from the `Scene` builder, the builder writes
  headers that parse back as written, and the parser trusts less (see Changed).
- A claim that made up nearly all of the untrusted content (behind a short greeting, or in a block of its own) was
  never erased by the steering gate nor scored by the claim detector. Fixed.
- Text flagged by the scanners was not always erased: forged trusted-name blocks after untrusted content, verdict
  tokens with invisible characters inside, and claims found by a `ClaimDetector` passed to `ExtractionGuard` as a
  scanner. CRLF line endings changed scanner results. Fixed; flagged text that cannot be erased now escalates.
- `dmguard scan --text`, `POST /v1/scan` with `text` and the LangChain `decision_tool` parsed input that starts with
  (or contains) a bracket as a scene, so the scanners could be skipped. Text input is now always one untrusted block.
- Invalid model output (NaN, negative scores, all zeros, too few rows or scores) became a uniform distribution, so the
  first option won with action `allow`, or the gate silently checked nothing. It now raises `ModelOutputError`;
  unparseable answers of the HTTP judges raise too.
- The server ran without authentication when `server.auth_token_env` named an unset variable, and its body limit did
  not apply to chunked requests. Fixed; scene size, `meta` size and request time are limited as well.
- `OpenAICompatible` sent `OPENAI_API_KEY` to any `base_url`, and `WolfDefenderJev.from_pretrained` ran the model code
  of any repository without an opt-in. Fixed.

### Changed

- Parsing: no block after text without a header is trusted; after untrusted content only a `PROPOSED ACTION` or
  `TOOL CALL` block that is the last block stays trusted (in `ExtractionGuard` none). The `Scene` builder cleans headers,
  marks untrusted blocks with trusted names as `| untrusted` and indents header-like lines in untrusted text.
- The steering gate never erases a unit that holds all untrusted content of the scene (0.1.2: 90 % of its block), and
  untrusted blocks with trusted names are erased as a whole. Short content gets more variants than before.
- `decide()` checks its arguments: distinct non-empty options, `order` lists every option once, not both `risky` and
  `order`, `kind` must agree with them, `fallback_option` must be in range. `block_scanners` must name configured
  scanners.
- Configs: unknown keys are errors, relative paths are resolved against the config file, files are read as UTF-8.
  The server's request models reject unknown fields.
- Audit and evaluation count attack success only for targets on the unsafe side; pushes towards a more cautious
  answer are reported apart as false alarms and are not part of `--max-attack-success`. The CI gate fails when no
  attack ran, case files are checked line by line, and `eval`/`audit` no longer write to the decision audit log
  (`--audit-log` keeps that).
- The server builds the guard and loads its models before it starts, refuses to start without a configured token
  variable, answers 422 only for invalid requests (502 for invalid model output) and checks a whole batch first.
- `dmguard scan` treats its input as untrusted text (`--scene` parses a scene) and loads only the scanners; CLI errors
  are one line; `decide` prints `raw answer:` and `(blocked)`.
- Calibration with fewer cases than alpha needs keeps the temperature without a conformal threshold and warns;
  `Conformal` refuses such a calibrator instead of abstaining on everything.
- `DecisionGuard.default()` loads the claim detector at construction; `ClaimDetector` and `PromptInjection` use the
  smallest ONNX variant when torch is not installed; `WolfDefenderJev.from_pretrained()` without `onnx` checks for
  torch before downloading, pins the default repository's revision and needs `trust_remote_code=True` for others.
- `OpenAICompatible(api_key_env=None)` is the default; `OPENAI_API_KEY` goes only to api.openai.com.
- HTTP errors name the status, the transport reason or the service's error message (no other response bodies);
  service error envelopes are not retried and a failing batch stops early.
- `VerdictTokens` counts everyday keys (`status`, `result`, `class`, ...) only with a verdict word; soft hyphens no
  longer make `InvisibleText` invalid; headers written by the `Scene` builder are not scanned as forgeries.
- `AuditLogger` keeps `backup_count` rotated files, caps `meta` and can store keyed hashes (`hash_key_env`).
- `ExtractionGuard` checks single sentences first in cautious mode and reports when units are left out; `cautious`
  is for number and estimate fields only.

### Added

- `ModelOutputError`, `models.checked()`, `pipeline.check_decision()`, `pipeline.scan_text()`, `scene_blocks()`,
  `Scene.layout()`, `DecisionGuard.warmup()` and `without_audit()`, `DecisionGuard.default(extra_scanners=...)`,
  `kind` in `decide_many`.
- `SteeringGate(max_chars=400_000)` caps the characters sent per decision; `ClaimDetector` caches scores, so each unit
  is scored once per decision; `Policy.incomplete_coverage_action` and reasons for partial or skipped `Coverage`.
- Server settings `max_scene_chars`, `max_meta_bytes`, `request_timeout`, `docs`; `dmguard_responses_total` in
  `/metrics`; a startup summary in the log.
- CLI: `--version`, `--debug`, `scan --scene`, `decide --exit-code`, `eval/audit --audit-log`, help texts with defaults
  and the exit codes.
- `evaluate_extraction(cautious=...)`, `ExtractionReport.skipped`, `evaluate.check_case`, `ExtractionGuard` accepts
  plain functions, `scripts/onnx_agreement.py`.

### Documentation

- README links point to their files, installs are ONNX-first (`init-config` too), the audit table was measured again
  with this release, and the README covers the cases file, exit codes, long-scene cost and LangChain. The docs describe
  the trust rules that now ship, the server settings, the extraction fields and parameters, and ONNX agreement on the
  bundled cases.

## 0.1.2 (2026-10-20)

- Extraction measurements on Cloudflare's Clairic back in the README and the docs.

## 0.1.1 (2026-10-20)

- Documentation and examples updated; the README installs from PyPI.

## 0.1.0 (2026-10-20)

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
- `ExtractionGuard` for extraction models: erases flagged claims before trusting extracted values, reports values taken
  from flagged text, optional erase-and-check with a cautious direction; `evaluate_extraction` with twelve bundled cases.
- Cloudflare Workers AI adapters: `Clef` (decision model) and `Clairic` (extraction model), REST or proxy; config type
  `clef`.
