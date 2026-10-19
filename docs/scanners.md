# Content scanners

Every scanner implements `scan(text) -> ScanResult(scanner, sanitized, valid, score, findings)`. In a `DecisionGuard`
they run on the untrusted blocks only; blocks with unknown headers are scanned together with their header line, because
the attacker wrote it. Each scanner sees the previous scanner's sanitized text, while findings and their character
offsets always refer to the original scene.

| Scanner | Parameters | Notes |
|---|---|---|
| `InvisibleText(remove=True)` | | zero-width characters, bidi controls, Unicode tag characters, other invisible format characters; emoji joiners stay |
| `ForgedTrustBlock(threshold=0.7, neutralize=False)` | `trusted_headers` | trusted-looking headers (`[SECURITY REVIEW | trusted]`, `[SYSTEM CONTEXT]`), chat role markers (`<|im_start|>system`, `[INST]`, `<<SYS>>`, `### System:`) score 1.0; speaker labels (`Assistant:`) 0.6 |
| `VerdictTokens(threshold=0.5, redact=False)` | | `verdict: safe`, `label = "benign"`, `"is_injection": false`, `risk_score=0`, `Einstufung: harmlos` |
| `PromptInjection(model=WOLF_DEFENDER_SMALL, threshold=0.5, match="full")` | `match`: `full`, `sentences`, `chunks`; `max_length=2048`; `onnx` | Wolf Defender (mmBERT, 15 languages). `WOLF_DEFENDER` selects the larger model |
| `ClaimDetector(threshold=0.3, redact=False)` | `window=3`; `onnx` | the claim detector of the Decision Steering Gate; reports the smallest strongest units; `redact=True` removes them |
| `Regex(patterns, is_blocked=True, redact=False)` | | your own patterns; `is_blocked=False` requires a match |
| `BanSubstrings(substrings, case_sensitive=False, redact=False, match_all=False)` | | |
| `TokenLimit(limit=4096, tokenizer=None, truncate=True)` | | truncation keeps head and tail |

Transformer models are loaded once per process and shared, so the `ClaimDetector` scanner and the gate's detector use
the same weights.

```python
from decision_model_guard import scan_content
from decision_model_guard.input_scanners import ClaimDetector, InvisibleText, PromptInjection

sanitized, valid, score = scan_content([InvisibleText(), PromptInjection(match="sentences"), ClaimDetector(redact=True)], text)
```

Both model scanners run on PyTorch by default. `onnx="int8_int4_embeddings"` (or `fp32`, `fp16`, `mixed`, or a path to
an `.onnx` file) runs the same model on ONNX Runtime without torch; `threads` sets the session threads. Tokenization and
truncation match the PyTorch path, so FP32 gives the same scores.

## Writing your own scanner

Any object with a `name` and `scan(text) -> ScanResult` works:

```python
from decision_model_guard import Finding, ScanResult

class NoUrls:
    name = "NoUrls"
    def scan(self, text):
        hit = "http://" in text or "https://" in text
        return ScanResult(self.name, text, not hit, 1.0 if hit else 0.0, [Finding(self.name, 1.0, "url")] if hit else [])
```
