# Contributing

Thanks for helping. Bug reports, new model adapters, scanners, guards and benchmark results on your own decision model
are all welcome.

## Setup

```bash
git clone https://github.com/patronus-protect/decision-model-guard
cd decision-model-guard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # offline tests, no downloads
ruff check src tests examples
DMGUARD_SLOW=1 pytest        # plus tests with real models (downloads them once)
```

## Where things live

| You want to add | Put it in | Contract |
|---|---|---|
| a decision model adapter | `src/decision_model_guard/models/` | `decide_batch(scenes, question, options) -> list[list[float]]`; set `context_tokens` and `count_tokens` if you know them |
| a content scanner | `src/decision_model_guard/input_scanners/` | `scan(text) -> ScanResult(scanner, sanitized, valid, score, findings)` |
| a decision guard | `src/decision_model_guard/guards/` | `check(model, scene, question, options, probs, kind, *, risky, order, fallback) -> GuardReport` |
| an audit check | `src/decision_model_guard/evaluate/audit.py` | a family in `FAMILIES` and its variants in `_variants` |

Register new scanners and guards in `config.py` so they work from YAML, and document them in `docs/`.

## Pull requests

- One topic per pull request, with tests. Offline tests use the toy models in `tests/helpers.py`; tests that need real
  models go into `tests/test_slow.py`.
- Keep the public API typed and documented in the docstring.
- `pytest` and `ruff check src tests examples` must pass; CI runs them on Python 3.10 to 3.12.
- Numbers in the README or the docs need a reproducible source (a script or a JSON file in the repository).

## Reporting security issues

Please do not open public issues for vulnerabilities; see [SECURITY.md](SECURITY.md).
