# Arachne Mage A0 FIT1 — V7-C2 SkinTokens Top-4 Production-Parity Preregistration

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Parent blocker audit commit: `db9139044daa3291594d9920b155a9a1b9c9ac02`
Diagnostic source commit: `cbc940432fdaef4adb2cdbf783f0f54142b6f79e`
Upstream SkinTokens audit commit: `273b691d35989d71cd17ff2895fdc735097b92d1`

## Question

Does the exact production sparsification contract used by upstream SkinTokens explain a material part of the residual V7-C2 FIT1 error without any new training or learned mechanism?

## Frozen authority

- Model: exact V7-C2 importance-corrected treatment final checkpoint.
- Expected checkpoint SHA-256: `280d126ecb3177dfd718b651a956a65ca8a96bede1952b0b18f7d9a719bad7d0`.
- Architecture/config unchanged: `RealSaS.Arachne.SkinFieldCodec.v7`, config hash `e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1`.
- Mage cache SHA-256: `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`.
- Target binding SHA-256: `ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf`.
- 934 supervised rows, 22 joints.
- No optimizer, backward, gradient update, architecture change, sampler change, threshold sweep, or teacher-dependent support selection.

## Upstream production treatment

Apply exactly one deterministic mapping to the frozen scalar-field logits:

`p_j = sigmoid(z_j)`

For each row independently:
1. sort the 22 positive scalar predictions descending;
2. keep the largest **4**;
3. set the other 18 to exactly zero;
4. renormalize the retained four by their retained sum.

`K=4` is not tuned. The official SkinTokens demo passes `group_per_vertex=4`; its Blender exporter sorts skin weights, keeps the top group count, and renormalizes the retained values. Mage ground-truth supervised support cardinality at `>1e-8` is exactly `{1:444, 2:149, 3:330, 4:11}`, so maximum truth support is 4 and this parity rule cannot remove a fifth true Mage influence.

The optional SkinTokens voxel postprocess is **not** part of this diagnostic.

## Baseline replay requirement

Before applying top-4, the notebook must load and SHA-validate the frozen C2 treatment model and reproduce the sealed C2 final baseline within strict tolerances:

- mean row-L1 `0.056295882424983505`
- p95 row-L1 `0.22197738558673968`
- CVaR10 `0.2754472310858055`
- deformation-error ratio `0.08492327481508255`
- dominant accuracy `0.9946466809421841`
- top3 inclusion `1.0`
- pairwise variation ratio `0.9963325015519967`
- raw sigmoid mass mean `1.0176131891969074`

A replay mismatch is infrastructure failure and must stop before the scientific mapping is evaluated.

## Required top-4 telemetry

- mean / p95 / CVaR10 row-L1
- deformation-error ratio and RMS
- simplex residual
- discarded predicted mass per row: mean / p95 / p95-tail mean
- inactive predicted mass mean / p95 / p95-tail mean after top-4
- dominant accuracy / teacher-dominant top3 / mean rank
- predicted/teacher pairwise row-L1 variation ratio
- predicted/teacher mean joint-std ratio
- predicted nonzero support histogram and mean
- exact true-support recall inside predicted top-4, globally and separately for truth support size 1, 2, 3, and 4
- number of supervised rows whose full true support is contained inside predicted top-4
- number of rows where a false joint displaces at least one true support joint

## Product gates

Existing FIT1 gates are not silently changed:
- p95 row-L1 `<= 0.05`
- deformation-error ratio `<= 0.05`

This diagnostic may show that a SkinTokens-compatible max-4 product contract is useful, but adopting that contract into RealSaS requires an explicit later product-contract decision. Scientific metrics for this diagnostic are reported both baseline and top-4.

## Decision tree

1. **Both gates pass under top-4**: classify `SKINTOKENS_TOP4_PRODUCTION_CONTRACT_CLOSES_C2_FROZEN_FIT1`; do not start C3 training. Separately decide whether RealSaS should version its product contract to max four influences.
2. **Top-4 materially reduces leakage and p95 approaches the teacher-support-oracle floor (~0.079) but p95 remains above 0.05**: classify `TOP4_SOLVES_MOST_SUPPORT_LEAKAGE__WITHIN_SUPPORT_BLEND_REMAINS`; next compare/port SkinTokens boundary-aware dense sampling before inventing a custom blend-ratio objective.
3. **Top-4 remains near the C2 plateau because true weak supports are displaced by false joints**: classify `BLEND_BOUNDARY_SUPPORT_ORDERING_REMAINS`; next treatment is upstream boundary-aware dense sampling, not an arbitrary new loss.
4. **Top-4 does not materially help**: classify `TOP4_PRODUCTION_PARITY_NOT_CAUSAL`; retain the current all-influence product gate and continue diagnosis from boundary sampling / calibration evidence.

## Guardrails

- no K sweep; only K=4
- no threshold tuning
- no teacher oracle used in prediction mapping
- no optional voxel postprocess
- no new training
- no C3 ratio loss
- no 4k continuation
- no architecture/FSQ/token-count/extra-character change
- scientific FAIL is a valid result
