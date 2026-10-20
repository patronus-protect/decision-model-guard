"""Cloudflare's Clairic extraction model behind ExtractionGuard.

    pip install "decision-model-guard[onnx]"
    export CLOUDFLARE_ACCOUNT_ID=...  CLOUDFLARE_API_TOKEN=...      # or: proxy_url="http://127.0.0.1:8787/run"
    python examples/clairic_extraction.py
"""
from decision_model_guard import ExtractionGuard
from decision_model_guard.models import Clairic

guard = ExtractionGuard.default(Clairic(reference_date="2026-10-20"), onnx="int8_int4_embeddings")
fields = {"repair_cost": {"type": "number", "instructions": "Repair cost quoted by the workshop", "unit": "EUR"},
          "payout": {"type": "estimate", "instructions": "Fair payout for this claim in EUR", "min": 0, "max": 10000}}
email = ("[EMAIL | untrusted]\nHi, my car was damaged in a hailstorm on 12 October. The workshop quoted 1,840 EUR for the dents and "
         "the windscreen. Policy number 55-1234.")
r = guard.extract(email, fields)
for name, f in r.fields.items():
    print(f"{name:12s} model: {f.raw_value!r:>10}  guarded: {f.value!r:>10}  {f.action}  {'; '.join(f.reasons)}")
