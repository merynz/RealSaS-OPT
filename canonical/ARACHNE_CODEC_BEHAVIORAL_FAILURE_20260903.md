# RealSaS — Arachne / SkinFieldCodec behavioral failure and correction record

**Date:** 2026-09-03  
**Status:** `V1_PANEL_EVIDENCE_PARTIALLY_INVALID__SURFACE_ROW_BINDING_BUG_CONFIRMED__BOUND_V2_RUNNING`  
**Preregistration:** `canonical/ARACHNE_CODEC_BEHAVIORAL_PANEL_PREREG_20260903.md`  
**Original workflow:** `33752571671`  
**Original job:** `100639284377`  
**Original runner:** `eastus`

## 1. Epistemic rule

The preregistered witness definitions, seeds, model sizes, optimizer settings, horizons, thresholds and three-consecutive-PASS rule remain frozen.

Historical runs are not deleted or rewritten. When a later causal diagnostic proves an evaluator defect, the old observations remain historical facts but claims that depended on the defective measurement are explicitly reclassified.

## 2. Original V1 panel observation

Original V1 harness result:

`2 failed, 20 passed`

- `chain_blend_3`: full A0 -> A1 -> `Arachne.propose -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS` PASS;
- `branch_blend_4`: reported `FAIL_A0`;
- `sharp_fork_5`: reported `FAIL_A0`.

Original V1 final numbers:

### `chain_blend_3`
- A0 sustained PASS at step `640`;
- A0 row-L1 p95 `0.0473073684`;
- A0 deformation ratio `0.0124993324`;
- A1 sustained PASS at step `320`;
- qualified row-L1 p95 `0.0597252958`;
- qualified deformation ratio `0.0191426221`.

### `branch_blend_4`
- final row-L1 p95 `0.0938273296`;
- final deformation ratio `0.0368942656`;
- no A1 run.

### `sharp_fork_5`
- final row-L1 p95 `0.743847549`;
- final deformation ratio `0.240943387`;
- no A1 run.

These numbers were truly produced by the V1 harness. Their interpretation as evidence about Codec capacity is corrected below.

## 3. Independent Codec objective bug — VALID finding

A separate family-independent exact-truth probe showed that historical active-weighted cross entropy multiplied classes inside one simplex row by different constants.

For teacher row:

`[0.6000, 0.3991, 0.0009]`

with `active_threshold=1e-3`, `active_weight=2.0`, exact teacher probabilities produced pre-softmax gradient:

`[-0.00036, -0.00023946, +0.00059946]`

max absolute gradient `5.9946e-4`.

Therefore:

`EXACT_TEACHER_W_STATIONARY_UNDER_OLD_CODEC_CE = FALSE`

This causal finding does not depend on the behavioral-panel surface-row ordering and remains valid.

Repair:
- `f0fe52ab625695d46bed7007acba39fe4cdfb248`: active emphasis changed to a teacher-only row scalar, preserving relative within-row teacher probabilities;
- `9692ac12a44769212906616b6bca13861022d42c`: permanent regression requires exact teacher W to be stationary.

Post-repair run `33755765286`, job `100649682500`, measured max exact-truth gradient approximately `4.43e-17`: regression PASS.

This closes the independent truth-stationarity contract violation only.

## 4. Critical evaluator defect — CONFIRMED

The V1 behavioral harness generated synthetic teacher W in witness creation order:

`S:0, S:1, S:2, ...`

but `ArachneConditioningAdapter` obtains surface tensors through `_surface_features()`, which sorts `SurfaceNode` objects lexicographically by `surface_id`.

For `N < 10`, the orders coincide. For `N >= 10`, they do not.

Observed canonical adapter orders:

### `branch_blend_4`
Creation order:

`0,1,2,3,4,5,6,7,8,9,10,11`

Conditioning order:

`0,1,10,11,2,3,4,5,6,7,8,9`

Permutation:

`[0,1,10,11,2,3,4,5,6,7,8,9]`

### `sharp_fork_5`
Creation order:

`0,1,2,3,4,5,6,7,8,9,10,11,12,13,14`

Conditioning order:

`0,1,10,11,12,13,14,2,3,4,5,6,7,8,9`

Permutation:

`[0,1,10,11,12,13,14,2,3,4,5,6,7,8,9]`

The V1 harness passed teacher row `i` directly against conditioning tensor row `i`. Thus for the two `N>=10` witnesses, dense skin truth and rest-point rows were attached to the wrong canonical `surface_id` after row 1.

`chain_blend_3` has `N=9`; its identity order was unaffected. This exactly matches the V1 PASS/FAIL split.

## 5. Causal row-binding replay

Workflow `33756078567`, job `100650698374`, runner `mexicocentral` retained the same repaired Codec source, same witness definitions, seeds, optimizer, learning rate, weight decay, 1536-step horizon, thresholds and three-consecutive-PASS rule. The only intervention was exact surface-ID rebinding of teacher W and rest rows into `conditioning.surface_ids` order.

### `branch_blend_4`
After correct ID binding:
- sustained A0 PASS at step `544`;
- three consecutive PASS checks;
- final checked row-L1 p95 `0.0287867505`;
- deformation ratio `0.0104174139`;
- max simplex residual `5.96e-08`;
- negative weights `0`.

This causally falsifies the original claim that branch V1 failure demonstrated a Codec representation bottleneck.

### `sharp_fork_5`
After correct ID binding:
- final checked row-L1 p95 `0.0390132964` — individual threshold PASS;
- deformation ratio `0.00743764965` — individual threshold PASS;
- max simplex residual `1.19e-07`;
- negative weights `0`;
- only `2` consecutive PASS checks were accumulated by the frozen 1536-step horizon;
- therefore authoritative preregistered status remains `NOT YET PASS` until the corrected full panel is run.

The intervention improves the original sharp row-L1 p95 from approximately `0.744` to `0.039`, and deformation ratio from approximately `0.241` to `0.0074`, without changing any acceptance threshold or optimizer setting.

Therefore the V1 sharp plateau is overwhelmingly explained by target-row misbinding. Whether a small residual stability/horizon failure remains under the authoritative corrected harness is still an open question.

## 6. Reclassification of intermediate representation diagnostics

Free-latent and explicit point-control-geometry diagnostics executed before the row-binding defect was known consumed the same misbound teacher/rest rows.

Their numerical outputs remain preserved, but representation conclusions drawn from them are reclassified:

`CONTAMINATED_BY_V1_SURFACE_ROW_BINDING_BUG`

They must not justify a Codec architecture change.

In particular:
- do not claim teacher-encoder compression is a confirmed root cause from those runs;
- do not add explicit pair geometry to Codec because of those runs;
- do not increase Codec width/latent capacity because of those runs.

The exact pair-logit oracle remains useful only as a mathematical metric-feasibility sanity check; it did not validate correct canonical row binding.

## 7. Successor behavioral authority

The historical V1 harness is preserved at:

`experiments/geppetto_arachne_r6_20260901/test_arachne_v2_behavioral_panel.py`

It is no longer active behavioral authority because its target axis is known invalid for `N>=10`.

The successor harness is:

`experiments/geppetto_arachne_r6_20260901/test_arachne_v2_behavioral_panel_bound_v2.py`

V2 changes only target binding:
- synthetic teacher W remains generated by the preregistered formula;
- witness geometry unchanged;
- seeds unchanged;
- architecture unchanged;
- A0/A1 optimizer settings unchanged;
- horizons unchanged;
- thresholds unchanged;
- three-consecutive-PASS rule unchanged;
- teacher W and rest rows are bound to the exact canonical `conditioning.surface_ids` order before any optimization or evaluation.

It includes a cause-level regression proving correct row-to-surface-ID binding for both identity-order and `N>=10` non-identity-order witnesses.

Active Arachne CI now runs:
1. exact-truth Codec objective stationarity regression;
2. bound V2 frozen behavioral panel;
3. source/generic-strength gates.

Historical contaminated capacity/pair-geometry diagnostics remain in the repository for provenance but are removed from active PASS authority.

## 8. Current status

`ARACHNE_CODEC_V1_PANEL_IMPLEMENTATION_EVIDENCE = INVALID_FOR_N_GE_10`

`SURFACE_ROW_BINDING_BUG = CONFIRMED`

`INDEPENDENT_CODEC_CE_TRUTH_STATIONARITY_BUG = REPAIRED`

`ARACHNE_CODEC_BOUND_V2_PANEL = RUNNING`

Until Bound V2 closes:
- no Arachne A1/source architecture repair based on the contaminated V1 evidence;
- no threshold/seed/horizon changes;
- no real-family repair selection;
- no architecture refreeze;
- no formal Family-1 authorization.
