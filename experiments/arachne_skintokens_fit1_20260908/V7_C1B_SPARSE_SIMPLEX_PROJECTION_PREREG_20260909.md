# Arachne Mage A0 FIT1 — V7-C1B Sparse-Simplex Frozen Output Autopsy Preregistration

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Parent result: `V7_C1_FROZEN_TAIL_CALIBRATION_RESULT_20260909.md`

## Question

C1 localized most residual FIT1 product error to accumulated positive mass on teacher-inactive joints. The tested frozen maps (`sigmoid->normalize`, softmax, softplus->normalize) are strictly positive and therefore cannot create exact zero support. Does a teacher-free sparse simplex projection of the already well-calibrated scalar probabilities remove this residual leakage sufficiently to close or nearly close the unchanged FIT1 gates without any training?

## Frozen controls

- Exact V7 C0 2000-step final model SHA-256: `ad90cc0287963d338703c667f132a3073e034d8c3b6e49e8c290d3e5b0daeaf9`.
- Architecture/config unchanged: `RealSaS.Arachne.SkinFieldCodec.v7`, config hash `e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1`, 278,010,880 parameters.
- Mage cache SHA-256: `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`.
- Target binding SHA-256: `ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf`.
- 934 supervised rows / 22 joints.
- No optimizer, backward pass, architecture change, FSQ, extra character or teacher-dependent inference operation.

## C1 evidence motivating this test

- Baseline: p95 `0.3240030049`, deformation `0.1052859798`.
- Best strictly-positive frozen remap (`sigmoid-normalize alpha=1.25`): p95 `0.2306346565`, deformation `0.0956621021`; insufficient.
- Mean inactive predicted mass: `0.0282344623`; mean inactive mass in the baseline p95 tail: `0.2484124955`.
- Teacher-support diagnostic oracle: p95 `0.0791619746`, deformation `0.0299723297`.
- Pure one-joint rows already have p95 `0.0125286955`; residual error is concentrated in blend/hard rows.

## Preregistered teacher-free mappings

Let `P = sigmoid(Z)` from the exact frozen final raw logits.

1. **Primary parameter-free mapping:** Euclidean projection of each `P_i` onto the probability simplex:
   `argmin_w ||w - P_i||_2^2` subject to `w >= 0`, `sum(w)=1`.
   This is equivalent to an adaptive row threshold `w_j=max(P_j-tau,0)` with `tau` chosen only from that prediction row. No teacher support or target values enter the mapping.
2. Sigmoid-temperature + simplex projection for preregistered `alpha = {0.75, 1.0, 1.25}`: project `sigmoid(alpha*Z)` onto the simplex. Alpha=1 is identical to the primary mapping and is included as an identity check.
3. Logit sparsemax (`simplex_projection(alpha*Z)`) for preregistered `alpha = {0.125, 0.25, 0.5, 0.75, 1.0}`. This is secondary because the V7 scalar logits were trained as Bernoulli/weight logits, so probability-space projection is the causally preferred treatment.

No threshold sweep, teacher-selected top-k, teacher support, validation fitting, per-row target cardinality, or learned calibration parameter is permitted.

## Required telemetry

For every mapping:
- mean / p95 / CVaR10 row-L1;
- deformation-error ratio;
- dominant accuracy, top3 and mean rank;
- predicted/teacher pairwise variation and mean-joint-std ratios;
- predicted nonzero support-cardinality histogram;
- teacher support precision/recall/F1 for diagnosis only;
- false-positive predicted mass outside teacher support for evaluation only;
- pure / 2-joint / 3+ joint product slices;
- simplex residual and nonnegativity.

The existing C1 teacher-support oracle remains an evaluation-only upper-bound comparator and is not a candidate mapping.

## Decision logic

1. If the parameter-free probability-simplex projection reaches both unchanged gates (`p95 <= 0.05`, deformation <= `0.05`) with valid simplex semantics, treat it as a FIT1 output-coordinate closure candidate; then run the existing three-stable/product proof checks before any new training.
2. If a preregistered sparse map materially approaches the teacher-support oracle but remains above p95 gate, support leakage is confirmed as the dominant residual and the remaining gap is within-support blend calibration.
3. If sparse projection does not materially improve the product tail, proceed to the separately planned importance-corrected active-heavy warm-start training test.
4. Mean-only improvement is not rescue. p95/CVaR/deformation are authoritative.
5. A1/generalization remains unauthorized.
