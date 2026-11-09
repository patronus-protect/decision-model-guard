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

Measured with the previous claim detector (v1); not re-measured with v2 yet.

| Model | Attacks that succeed: no gate / full gate / detector | Harmless tickets changed: full gate / detector |
|---|---:|---:|
| Clef | 11.7 / 0.0 / 1.2 % | 93.0 / 3.5 % |
| drex | 16.7 / 0.0 / 0.0 % | 92.3 / 8.3 % |
| Lux | 20.5 / 0.0 / 1.3 % | 92.3 / 1.9 % |
| pplx-decider | 16.7 / 0.0 / 0.6 % | 92.9 / 4.8 % |
| Julia-1 | 24.2 / 0.0 / 3.0 % | 39.4 / 33.3 % |
| GLiNER2.5-Decide | 100.0 / 0.0 / 16.1 % | 37.9 / 13.8 % |

## Claim detector

The claim detector on Hugging Face is v2 (16 layers, 128k vocabulary, 80.0M parameters; 58 MB as
`int8_int4_embeddings`). Threshold 0.3, PyTorch FP32, three held-out sets in 15 languages, previous (v1) -> v2:

| Set | Claims found | Benign texts flagged | AUROC |
|---|---:|---:|---:|
| Red-team holdout: 4,886 claims written by a different generator than the training data | 61.3 -> 96.5 % | | |
| IFlippedJev: 1,000 claims that flipped a local decision model on a real case | 46.8 -> 96.6 % | | |
| NotClaim: 1,343 claims and 1,343 benign texts with the same claim vocabulary | 77.7 -> 98.1 % | 11.0 -> 2.2 % | 0.928 -> 0.998 |

The three sets are public: [patronus-studio/decision-steering-benchmarks](https://huggingface.co/datasets/patronus-studio/decision-steering-benchmarks);
IFlippedJev includes the cases, so the claims can be replayed against your own decision model. The weakest group is Turkish IFlippedJev claims (84 %). Per-language numbers, the set descriptions and the variants are on
the [model card](https://huggingface.co/patronus-studio/decision-steering-gate#claim-detector).

CPU latency per piece of text (Apple M4, 4 threads, ONNX Runtime 1.31, tokenization included, median over 1,500 real
texts): `int8_int4_embeddings` 5.7 -> 4.1 ms, `fp32` 7.3 -> 5.3 ms. The previous detector stays available under the
`v1` revision on Hugging Face.
