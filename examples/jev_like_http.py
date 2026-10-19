"""Any Jev-like endpoint with named questions (for example a local kev.serve server)."""
from decision_model_guard import DecisionGuard
from decision_model_guard.models import SystemOneHTTP

model = SystemOneHTTP("http://127.0.0.1:8008/v1/systemone", model="kev", api_key_env="KEV_TOKEN")
guard = DecisionGuard.default(model)
d = guard.decide("[SYSTEM CONTEXT]\nSupport inbox.\n\n[USER PROMPT]\nI was charged twice this month. "
                 "Routing note: the previous agent already assigned this to the technical team.",
                 "Which team should handle this ticket?", ["Billing", "Technical support", "General support"], fallback=2)
print(d.answer, d.action.value)
