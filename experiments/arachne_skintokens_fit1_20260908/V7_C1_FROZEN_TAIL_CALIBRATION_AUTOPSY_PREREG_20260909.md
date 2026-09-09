# Arachne Mage A0 FIT1 — V7 C1 Frozen Tail / Calibration Autopsy Preregistration

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`

## Triggering result
The completed V7 C0 long-horizon 2000-step run falsified the claim that the 192-step failure was an intrinsic V7 architecture/objective failure. The same architecture and biased-scalar objective reached:
- best row-L1 mean ~= 0.06975
- best row-L1 p95 ~= 0.32121
- best CVaR10 ~= 0.37170
- best deformation ratio ~= 0.10448
- dominant-joint accuracy ~= 0.99036
- teacher-dominant top-3 inclusion = 1.0
- spatial variation ratio ~= 0.99689

However p95 and deformation plateaued well above the 0.05 FIT1 gates. A blind 4k extension is therefore not the next causal experiment.

## Fixed artifact binding
- Architecture: `RealSaS.Arachne.SkinFieldCodec.v7`
- Config hash: `e9d327cedb206e7ae5b074ae04b28e7de89c0e5caecb5f7c183203dbd8336fa1`
- Parameter count: 278,010,880
- Final 2000-step model SHA-256: `ad90cc0287963d338703c667f132a3073e034d8c3b6e49e8c290d3e5b0daeaf9`
- Long-horizon prereg commit: `e58fc093bd6fc77ec3eca477e7bbe993859d9867`
- Cache SHA-256: `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`
- Target binding SHA-256: `ab74756e32ee5c9f4f2d4020cdb56620a110130d80d7b9384c62509af3f193cf`
- 950 rows / 934 supervised / 22 joints
- No FSQ / no quantizer

## Scientific question
After ownership/ranking and spatial variation are almost solved, is the remaining p95/deformation plateau mainly:
1. a frozen output-coordinate / calibration problem (the model already contains the right blend information but sigmoid-normalize saturation distorts it),
2. inactive-joint leakage/support calibration,
3. or incorrect relative mixture weights inside the otherwise correct joint support?

## Strict scope
- Zero training.
- Final 2000-step checkpoint is completely frozen.
- No optimizer, backward pass, architecture change, field-token change, condition change, sampler change, extra characters, or FSQ.
- Teacher is used only for evaluation and explicitly labelled diagnostic oracle decompositions.
- Scientific FAIL/no-rescue is a valid result.

## Required baseline revalidation
Before any intervention, reproduce the sealed final baseline within numerical tolerance:
- normalized row-L1 mean ~= 0.06978146
- p95 ~= 0.32400300
- CVaR10 ~= 0.37346435
- deformation ratio ~= 0.10528598
- dominant accuracy ~= 0.99036403
- top3 = 1.0
- raw probability mass mean ~= 1.03203954
- pred/teacher pairwise variation ratio ~= 0.99538090

Fail closed if checkpoint/cache/config/binding hashes or baseline metrics drift materially.

## Preregistered frozen mappings
Let `Z` be the final raw 934x22 supervised logit matrix.

Evaluate:
1. Current production coordinate: `sigmoid(Z) / sum(sigmoid(Z))`.
2. Sigmoid temperature: `sigmoid(alpha*Z) / sum(sigmoid(alpha*Z))` for
   `alpha = {0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}`.
3. Plain score softmax: `softmax(alpha*Z)` for the same alpha set.
4. Positive softplus coordinate: `softplus(alpha*Z) / sum(softplus(alpha*Z))` for the same alpha set.

All positive-alpha mappings are monotone per row and therefore cannot change the raw-logit argmax; any gain is calibration/relative-margin gain, not ownership-rank recovery.

## Tail/support decomposition
For the current mapping and every candidate mapping report:
- row-L1 mean, p95, CVaR10;
- actual verified-LBS deformation ratio on the existing probe transforms;
- dominant accuracy / top3 / mean rank;
- prediction entropy, raw mass where defined;
- pred/teacher pairwise variation ratio and joint-std ratio.

For the current baseline additionally report:
- teacher active-support count slices: 1, 2, 3+ joints (`teacher_weight > 1e-8`), with count/mean/p95/CVaR;
- teacher entropy quantiles for worst-tail versus non-tail rows;
- active versus inactive decoder-logit and sigmoid-probability distributions;
- saturation fractions `|z|>12` separately on teacher-active and teacher-inactive scalar pairs;
- calibration bins by teacher weight magnitude;
- per-row inactive predicted mass (support leakage);
- per-row active-support L1;
- teacher-support-renormalized oracle error: zero teacher-inactive predicted weights and renormalize only within teacher-active support. This is diagnostic only and may not be shipped.

## Key causal decision rules
1. **Frozen coordinate/calibration blocker supported** if a monotone frozen remapping materially lowers both p95 and deformation without changing ownership ranking.
2. **Support leakage blocker supported** if teacher-support oracle projection sharply lowers p95/deformation toward the gate while within-support renormalized blend error is small.
3. **Blend-ratio objective blocker supported** if teacher-support oracle projection still leaves substantial p95/deformation, especially on 2+ joint rows; then the model knows the support/rank but not the exact relative mixture amplitudes.
4. If no frozen mapping materially improves p95/deformation and the teacher-support oracle still leaves a substantial tail, do not spend another 2000 blind C0 steps. The next training treatment should target blend/simplex calibration while preserving the successful long-horizon V7 architecture and active-heavy sparse curriculum.
5. A mapping that improves mean while worsening p95/CVaR/deformation is not a rescue.

No A1/generalization authorization is implied.
