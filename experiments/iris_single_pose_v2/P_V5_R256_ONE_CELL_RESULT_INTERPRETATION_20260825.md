# P-V5 R256 one-cell learner gate — canonical interpretation

Date: 2026-08-25

## Frozen parent authority

- P ontology / observable canonical position remains closed.
- P-V5 native-scale-once analytic reconstruction remains closed.
- P-V5 field representation closure remains closed: full R=256 was positively certified; R/2=128 was not certified.
- This result therefore concerns the current R256 visual learner + optimizer protocol, not P ontology or free-field representability.

## Apparatus

The corrected V1.1 run completed cleanly.

GPU preflight:

- device: Tesla T4;
- input resolution: 256;
- output field: 256×256;
- full-resolution fusion feature: 8×48×256×256;
- direct full-resolution image-stem gradient: nonzero;
- P-depth head gradient: nonzero;
- AMP forward / FP32 objective: enabled as preregistered;
- scientific optimizer steps during GPU preflight: 0.

No TUNE/CAL/DEV/EXTERNAL split or hidden camera metadata was consumed.

## Frozen decision

Frozen cell:

- asset: `asset_76313e4bd82b82fcd1659c70`
- style: `cel_clean`
- split: FIT

Frozen protocol:

- AdamW lr `3e-4`, betas `(0.9, 0.95)`, weight decay `0`;
- 2048 optimizer steps;
- candidate evaluations at 64/128/256/512/1024/2048;
- P/depth objective only;
- decision authority = full reconstructed canonical P Euclidean p95;
- PASS iff `P_p95 <= 0.005`.

Result:

- selected step: `2048`;
- selected P p95: `0.005682396539486942`;
- threshold: `0.005`;
- status: `P_V5_R256_ONE_CELL_OPTIMIZATION_INSUFFICIENT`.

This is a scientific FAIL under the frozen gate. The promotion ladder therefore remains blocked at one asset × one style.

## Candidate trajectory

| step | P p95 |
|---:|---:|
| INIT | 2.729859101772308 |
| 64 | 0.465613028407096 |
| 128 | 0.2629218861460685 |
| 256 | 0.1124968137592076 |
| 512 | 0.04366506431251761 |
| 1024 | 0.028862940426915864 |
| 2048 | 0.005682396539486942 |

All preregistered candidate evaluations improved monotonically. From 1024 to 2048, P p95 fell by ~80.3%. The final point is the best preregistered point; there is no evidence of a completed plateau at 2048.

At step 2048:

- P mean = `0.002254313743345837`;
- P p50 = `0.001217403681948781`;
- P p90 = `0.003827358572743832`;
- P p95 = `0.005682396539486942`;
- P max = `0.2901246249675751`.

Thus 90% of sampled loci are already below the 0.005 gate while the p95 tail remains slightly above it. The miss is absolute `0.0006823965`, about 13.65% above threshold.

Late training is also noisy: e.g. train p95 was ~0.00676 at step 1920, jumped above 0.012 at 1936/1952, then ended at ~0.00572 at 2048. This is consistent with a possible late-stage optimizer/LR floor, but does not prove it.

## Scientific interpretation

What is proven:

1. The true full-resolution R256 neural path is executable on CUDA and receives gradient.
2. The current visual learner reduced one-cell P p95 by ~480× from initialization.
3. The frozen `3e-4 / 2048-step` protocol did not satisfy the 0.005 gate.
4. The failure is narrow and tail-dominated, not a gross inability to fit the cell.
5. The preregistered candidate trajectory was still materially improving at the final candidate.

What is **not** proven:

- learner capacity insufficiency;
- an information limit;
- P ontology failure;
- V5 analytic geometry failure;
- R256 field representability failure;
- that simply adding more steps would necessarily pass;
- that lowering LR would necessarily pass;
- that a new tail-weighted objective is required.

## Next authorized action

Run `P_V5_R256_ONE_CELL_OPTIMIZER_LOCALIZATION_V1` on the same frozen FIT cell and the exact selected 2048-step checkpoint.

Purpose:

> Distinguish a finite-budget / late-stage LR issue from a remaining learner/objective/tail problem without broadening corpus scope.

No 1×2-style, 8×2-style or unseen-family experiment is authorized until this localization is interpreted.
