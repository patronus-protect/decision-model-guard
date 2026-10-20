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
ruff check src tests examples scripts
DMGUARD_SLOW=1 pytest        # plus tests with real models (downloads them once), including a fresh install of the built wheel
python -m build && python scripts/smoke_installed.py   # run the smoke test against an installed package
```

Tests come in three layers: unit tests per module (`tests/test_units.py` and the module tests), end-to-end runs without downloads (`tests/test_e2e.py`: CLI subprocesses, a real server process, the Python API) and end-to-end runs with real models (`tests/test_slow.py`). CI runs the first two on Python 3.10 to 3.13 and smoke-tests the built wheel in a fresh environment.

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
- `pytest` and `ruff check src tests examples scripts` must pass; CI runs them on Python 3.10 to 3.13.
- Numbers in the README or the docs need a reproducible source (a script or a JSON file in the repository).

## Reporting security issues

Please do not open public issues for vulnerabilities; see [SECURITY.md](SECURITY.md).
