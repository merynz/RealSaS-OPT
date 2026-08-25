# IRIS Single-Pose V2 — CI202 Mini Extractability / Generalization Canonical Interpretation

Date: 2026-08-24  
Status: `LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT__GENERALIZATION_GAP_NOT_PRIMARY`

## Authority

This interpretation is bound to the completed CI202 run under the frozen Mini Extractability / Generalization V1 preregistration.

- Representation gate entering learner study: `P_GEOMETRY_SUFFICIENT`.
- Input resolution: 256.
- FIT_TRAIN=128; FIT_SELECT=32; TUNE_FINAL=26.
- Selected checkpoint: epoch 16 / optimizer step 512.
- TUNE was not used for checkpoint selection.
- TUNE staging began only after checkpoint freeze.
- CI104 representation seed was provenance-only and not consumed by learner runtime.
- CAL/DEV/EXTERNAL_HOLDOUT remained unopened.
- Normal correspondence authority remained false; N was diagnostic/local-orientation supervision only.

## Frozen outcome label

`LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT`

This is an absolute extractability/precision failure under the frozen V1 engineering promotion criteria. It is **not** a representation-information failure and it is **not primarily a generalization-gap failure**.

## Random-init -> selected checkpoint learning signal

FIT_SELECT changed substantially from random initialization to epoch 16:

| Metric | Random init | Epoch 16 | Change |
|---|---:|---:|---:|
| P p95 | 3.868741 | 0.344555 | 91.1% reduction |
| Zc Recall@8 | 0.358507 | 0.789931 | +0.431424 |
| P-basin top8 | 0.345486 | 0.649306 | +0.303819 |
| oracle Zf top1 p95 native px | 51.3399 | 35.9471 | 30.0% reduction |
| end-to-end top8 hit <=16px | 0.165799 | 0.827257 | +0.661458 |
| N p95 deg (diagnostic) | 147.630 | 67.848 | 54.0% reduction |

Therefore the model is demonstrably learning the intended evidence directions. The failure label means the learned evidence is not yet precise enough for promotion.

## Frozen thresholds vs result

FIT_SELECT core:

- P p95 <= 0.005 required; observed **0.344555** -> FAIL.
- Zc Recall@8 >= 0.90 required; observed **0.789931** -> FAIL.
- oracle Zf top1 p95 <=16 native px required; observed **35.9471 px** -> FAIL.

TUNE absolute tail/style criteria:

- family Zc@8 p10 >=0.75 required; observed **0.59375** -> FAIL.
- min-style Zc@8 >=0.85 required; observed **0.761218** -> FAIL.
- max-style P p95 <=0.0065 required; observed **0.416006** -> FAIL.

## Generalization diagnosis

FIT_SELECT -> TUNE_FINAL:

| Metric | FIT_SELECT | TUNE_FINAL | Gap |
|---|---:|---:|---:|
| P p95 | 0.344555 | 0.413841 | ratio 1.2011 |
| Zc Recall@8 | 0.789931 | 0.762019 | drop 0.02791 |
| oracle Zf p95 px | 35.9471 | 37.7536 | ratio 1.0503 |
| end-to-end top8 <=16px | 0.827257 | 0.823718 | drop 0.00354 |

The frozen generalization-gap conditions themselves are satisfied descriptively:

- P TUNE/FIT ratio 1.2011 <=1.5.
- Zc top8 drop 0.02791 <=0.10.

Thus the dominant blocker is **absolute extractability/precision**, not unseen-TUNE collapse.

## Style diagnosis

Style is not the main blocker. On TUNE:

- pooled Zc@8=0.762019; min-style Zc@8=0.761218.
- pooled P p95=0.413841; max-style P p95=0.416006.

The two style branches are nearly identical at the aggregate level. The hard tail is substantially asset/family-driven rather than an ink-vs-clean style failure.

## Optimization-state diagnosis

The 16-epoch run did not show a clean convergence plateau.

Training metrics continued improving through epoch 16. Between epochs 12 and 16:

- P training loss 0.059744 -> 0.053148.
- mean P Euclidean error 0.137010 -> 0.125261.
- Zc dual 0.287659 -> 0.253919.
- Zc multi-positive 0.621365 -> 0.539655.
- Zf local loss 1.833869 -> 1.685245.
- Zf local top1 0.335908 -> 0.354972.
- total loss 0.341854 -> 0.300262.

Frozen checkpoint scores improved monotonically:

- epoch4: 223.8967
- epoch8: 202.8393
- epoch12: 198.6633
- epoch16: 189.6269 (selected)

Therefore this run does **not** justify the claim that the current architecture has reached its attainable precision ceiling. It also does not justify simply authorizing a larger full training run: first isolate optimization sufficiency vs architecture/objective limitation.

## Canonical interpretation

1. `P_GEOMETRY_SUFFICIENT` remains valid; this learner failure does not reopen information existence.
2. The current V2 learner can extract substantial P/persistence/local-refinement signal from legal 8-view images.
3. Generalization from FIT_SELECT to TUNE_FINAL is comparatively healthy; there is no catastrophic unseen-family/style collapse in this mini.
4. Absolute P precision is the largest blocker. Z_coarse is promising but below promotion; Z_fine improves but remains above the local-error target.
5. N remains diagnostic only; nothing in CI202 authorizes N as correspondence authority.
6. Because epoch16 is the best and training trends remain improving, the next scientific question is `OPTIMIZATION_SUFFICIENCY_VS_REPRESENTATION_EXTRACTION_CEILING`, not a return to SOI-2 or immediate downstream SurfaceBuilder work.

## Next gate

Before architecture redesign or full-scale training, run a preregistered FIT-only optimization diagnostic with no further TUNE use:

- a tiny same-asset overfit arm to test whether current model/loss can drive P and local Zf near their target on a deliberately small legal subset;
- a longer FIT-only learning-curve arm to distinguish truncated optimization from a stable architecture/objective plateau;
- preserve current P/Zc/Zf/N authority roles and do not reopen TUNE, CAL, DEV or EXTERNAL during diagnostic selection.

Possible conclusions of that diagnostic must distinguish:

- `OPTIMIZATION_BUDGET_LIMITED` — current architecture can fit the target and longer/open-FIT optimization materially continues improvement;
- `CURRENT_LEARNER_OBJECTIVE_OR_ARCHITECTURE_LIMITED` — even tiny overfit / longer FIT-only optimization plateaus far above the target;
- `MIXED_P_LIMIT__CORRESPONDENCE_HEALTHY` — P remains the dominant failure while persistence/local correspondence reaches acceptable behavior, motivating targeted P representation/head/geometry changes rather than a wholesale architecture reset.

No final product/1024, autonomous dense evidence, SurfaceBuilder, Geppetto, Arachne, deformation, or artist-domain claims are opened by CI202.