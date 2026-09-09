# Arachne Mage A0 FIT1 — V7-C1 Frozen Tail / Calibration Autopsy Result

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Preregister: `149376408dba94e41fff50f190dae80e8573576b`
Diagnostic source: `3adffff36daeaf3ded07e3158701a49b13e3b00a`
Long-horizon final model SHA-256: `ad90cc0287963d338703c667f132a3073e034d8c3b6e49e8c290d3e5b0daeaf9`

## Integrity / preflight

- A100 preflight: PASS.
- Baseline replay reproduced the sealed long-horizon final within preregistered tolerances.
- Training performed: NO.
- Architecture, checkpoint, cache, target binding, token count, FSQ state and optimizer state were not changed.

Revalidated baseline:
- mean row-L1: `0.0697814627`
- p95 row-L1: `0.3240030049`
- CVaR10: `0.3734643500`
- deformation-error ratio: `0.1052859798`
- dominant accuracy: `0.9903640257`
- teacher-dominant top-3 inclusion: `1.0`
- pairwise spatial variation ratio: `0.9953808967`
- predicted/teacher mean joint-std ratio: `0.9295199563`

## Frozen monotone remapping

Best joint p95/deformation frozen mapping was `sigmoid_normalize_alpha_1.25`:
- mean row-L1: `0.0718965842`
- p95: `0.2306346565`
- CVaR10: `0.3067544148`
- deformation: `0.0956621021`
- dominant accuracy unchanged: `0.9903640257`

This is a real calibration effect but does not reach either unchanged FIT1 gate. Therefore a frozen output remap is insufficient.

## Support structure

Baseline error by teacher support cardinality:
- 1-joint rows: `444`, mean `0.0187764221`, p95 `0.0125286955`, dominant accuracy `0.9932432432`.
- 2-joint rows: `149`, mean `0.1135544587`, p95 `0.3284336895`, dominant accuracy `1.0`.
- 3+ joint rows: `341`, mean `0.1170661010`, p95 `0.3638783324`, dominant accuracy `0.9824046921`.

The residual product tail is therefore concentrated in blend rows rather than pure one-joint rows.

## Active / inactive logits

- Active scalar fraction with `|logit| > 12`: `0.0242117117`.
- Inactive scalar fraction with `|logit| > 12`: `0.7547411038`.
- Mean inactive predicted probability across all zero teacher entries is only `0.0017645389`, but the p95 product tail has much larger accumulated inactive mass.
- Mean inactive predicted mass over all supervised rows: `0.0282344623`.
- Mean inactive predicted mass inside the p95 tail: `0.2484124955`; median `0.2042496731`; max `0.6020559879`.

Thus the remaining tail is not broad under-saturation of every inactive score. It is sparse row/joint false-positive leakage whose accumulated normalized mass becomes large on hard rows.

## Teacher-support oracle — diagnostic only

Set prediction outside the exact teacher support to zero, then renormalize the model's existing probabilities inside teacher support. This does not retrain or alter logits.

Oracle product metrics:
- mean row-L1: `0.0189198921`
- p95: `0.0791619746`
- CVaR10: `0.0884698232`
- deformation-error ratio: `0.0299723297`
- dominant accuracy: `0.9957173448`
- pairwise variation ratio: `0.9996177407`
- joint-std ratio: `0.9996567714`

Causal gains versus baseline:
- p95: `0.3240030049 -> 0.0791619746` (`-0.2448410303`)
- deformation: `0.1052859798 -> 0.0299723297` (`-0.0753136501`)

The deformation gate (`<=0.05`) passes under this diagnostic oracle. The p95 gate (`<=0.05`) does not: `0.07916` remains.

Within the original p95 tail, after support projection:
- mean oracle row-L1: `0.0228175644`
- p95 oracle row-L1: `0.0574549818`
- max: `0.1139325938`

Only 5 of the 47 p95-tail rows have wrong dominant joint; total dominant-wrong rows are 9/934. Hence most residual tail error is not ownership identity failure.

## Causal conclusion

Primary classification: `MIXED_SUPPORT_LEAKAGE_AND_WITHIN_SUPPORT_BLEND_RATIO_ERROR`.

1. The 2000-step horizon already solved geometry/ownership/ranking to near-teacher scale. Blind 4k extension is not authorized.
2. Frozen temperature/output remapping helps but is not sufficient.
3. The largest remaining product error is inactive-joint leakage on a small set of hard rows. Removing only that leakage makes deformation pass and reduces p95 by ~75%.
4. A smaller second blocker remains after perfect support: relative weights among the correct active joints in blend rows. That is why the support oracle still has p95 `0.07916 > 0.05`.
5. Pure one-joint rows are essentially closed already (`p95 ~0.01253`).

## Next experiment decision

Do not change architecture, FSQ, token count, geometry path, or add characters.
Do not bundle leakage correction with a new coupled blend objective in the same first treatment.

Next causal treatment should isolate the known uncorrected active-heavy sampling prior while retaining its sparse-positive curriculum:

**V7-C2 — importance-corrected active-heavy warm-start control/treatment.**

- Start both arms from the exact 2000-step final model and identical optimizer reinitialization / continuation LR schedule.
- Control arm: existing biased 50% active-support + 50% global scalar objective.
- Treatment arm: sample the exact same active-heavy proposal distribution, but apply analytically correct importance weights so the expected scalar BCE/MSE contribution corresponds to the target uniform-supervised row distribution. Preserve the existing active-heavy sample coverage; do not replace it with uniform sampling.
- Keep Dice handling identical unless an exact importance-weighted Dice estimator is separately derived and preflighted; do not silently change its semantics.
- Primary causal telemetry: p95/CVaR/deformation, inactive predicted mass overall and in p95 tail, pure/2-joint/3+ slices, dominant/top3/variation, active scalar calibration.

If importance correction materially removes inactive leakage but p95 stalls near the teacher-support-oracle floor (~0.08), then the following separate experiment may add a simplex-aware blend-ratio auxiliary under the already-good V7 score-space representation.
