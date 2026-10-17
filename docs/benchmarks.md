# Benchmarks

## Six decision models on a decision-steering benchmark

Five tasks with 100 test cases each. Every case has three attacks (a claim from one of six families appended to the customer
message: prior review, staff approval, policy, external verification, adjudication, workflow complete) and three neutral additions.
Attack success counts attacks on cases the model decides correctly without the claim, where the claim pushes away from the
correct answer. Full erase-and-check, every variant scored by the model itself. Attack success in %, without -> with the gate:

| Model | Refund | Injection | Phishing / spam | Routing | Escalation |
|---|---:|---:|---:|---:|---:|
| Clef | 17.3 -> 0.0 | 0.0 -> 0.0 | 1.4 -> 0.0 | 11.7 -> 0.0 | 48.0 -> 0.0 |
| drex | 0.6 -> 0.0 | 18.8 -> 0.0 | 6.2 -> 0.0 | 16.7 -> 0.0 | 22.7 -> 0.0 |
| Lux | 2.8 -> 0.0 | 15.2 -> 0.0 | 3.5 -> 0.0 | 20.5 -> 0.0 | 0.0 -> 0.0 |
| pplx-decider | 9.7 -> 0.0 | 0.0 -> 0.0 | 19.1 -> 0.0 | 16.7 -> 0.0 | 52.7 -> 0.0 |
| Julia-1 | 0.0 -> 0.0 | 82.0 -> 0.0 | 100.0 -> 9.1 | 24.2 -> 0.0 | 13.3 -> 0.0 |
| GLiNER2.5-Decide | 89.7 -> 0.0 | 21.8 -> 0.0 | 100.0 -> 0.0 | 100.0 -> 0.0 | n/a |

Mean over the 29 model-task pairs with eligible attacks: 28.1 % -> 0.3 %. On unattacked cases accuracy falls by at most 6
points in 28 of 30 model-task pairs (drex phishing 97 -> 86 %, GLiNER2.5-Decide routing 30 -> 21 %).

## Ticket routing with the claim detector

| Model | Attacks that succeed: no gate / full gate / detector | Harmless tickets changed: full gate / detector |
|---|---:|---:|
| Clef | 11.7 / 0.0 / 1.2 % | 93.0 / 3.5 % |
| drex | 16.7 / 0.0 / 0.0 % | 92.3 / 8.3 % |
| Lux | 20.5 / 0.0 / 1.3 % | 92.3 / 1.9 % |
| pplx-decider | 16.7 / 0.0 / 0.6 % | 92.9 / 4.8 % |
| Julia-1 | 24.2 / 0.0 / 3.0 % | 39.4 / 33.3 % |
| GLiNER2.5-Decide | 100.0 / 0.0 / 16.1 % | 37.9 / 13.8 % |

## Claim detector

| Set | Role | AUROC | Claims found (threshold 0.3) | Content flagged |
|---|---|---:|---:|---:|
| Steering benchmark A (17 claim families, 15 languages) | selection | 0.970 | 80.2 % | 0.4 % |
| Steering benchmark B (claims of an earlier experiment) | test | 0.996 | 92.3 % | 0.3 % |
| Decision-steering benchmark | test | 0.969 | 83.5 % | 3.4 % |

Benchmark A chose checkpoint and threshold. The training claims cover some concepts of the test sets (policy references,
workflow status) with independently written wording; on claim styles outside the training banks expect closer to 50 or 60 %.
