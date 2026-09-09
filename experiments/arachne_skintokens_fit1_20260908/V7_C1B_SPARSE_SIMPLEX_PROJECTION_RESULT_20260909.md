# Arachne Mage A0 FIT1 — V7-C1B Sparse-Simplex Frozen Diagnostic Result

Date: 2026-09-09
Branch: `exp/arachne-skintokens-cleanroom-fit1-20260908`
Parent C1 prereg: `149376408dba94e41fff50f190dae80e8573576b`
C1B prereg: `f3d18249eba51a1f646b0c6330d94e5f6256f25d`
Diagnostic source: `d7dcb159a67160080fc8c3fa0c8989d12eba8979`
Bound final model SHA-256: `ad90cc0287963d338703c667f132a3073e034d8c3b6e49e8c290d3e5b0daeaf9`

## Integrity
- Training performed: NO.
- Exact sealed V7 C0 2000-step checkpoint, Mage cache/binding and current product metrics were revalidated before mapping sweeps.
- Architecture, logits, field tokens, condition path, FSQ state and teacher target were not changed.

## Baseline
- mean row-L1: `0.0697814627`
- p95 row-L1: `0.3240030049`
- CVaR10: `0.3734643500`
- deformation-error ratio: `0.1052859798`
- dominant accuracy: `0.9903640257`
- teacher-dominant top-3: `1.0`
- pairwise spatial variation ratio: `0.9953808967`

## Primary parameter-free probability-simplex projection
Primary mapping: Euclidean projection of `sigmoid(Z)` onto the probability simplex, no teacher support and no learned threshold.

Best point: `probability_simplex_alpha_1`
- mean row-L1: `0.0531002863`
- p95: `0.1929114778`
- CVaR10: `0.2598532511`
- deformation: `0.0930516124`
- dominant accuracy: `0.9903640257`
- pairwise variation ratio: `0.9980277611`
- joint-std ratio: `0.9645238368`

Gains versus baseline:
- p95: `-0.1310915272`
- CVaR10: `-0.1136110989`
- deformation: `-0.0122343674`

This is a real frozen improvement but neither unchanged FIT1 gate is reached.

## Logit sparsemax controls
Best logit-sparsemax p95 point was `alpha=0.25`:
- mean row-L1: `0.0611181123`
- p95: `0.2115928356`
- deformation: `0.1101480573`
- support micro precision: `0.9753086420`
- support micro recall: `0.8006756757`

Interpretation: aggressive sparse score projection can remove false support with very high precision, but it also deletes true low-weight blend support. Increasing sparsity further worsens blend rows. There is no single fixed sparse projection/temperature that recovers the correct support-cardinality tradeoff across pure, 2-joint and 3+-joint rows.

The probability-simplex projection has the opposite problem: it preserves most true support (recall ~`0.9583` at alpha 1) but leaves many false-positive nonzero entries because rows whose positive-score mass is below one receive a common positive simplex shift. Thus it cannot act as a teacher-free approximation to the exact teacher-support oracle.

## Oracle reference from C1
Teacher-support oracle upper bound remains:
- mean row-L1 `0.0189198921`
- p95 `0.0791619746`
- deformation `0.0299723297`

Therefore the gap between C1B primary mapping and the oracle is still substantial:
- p95 gap: `0.1137495031`
- deformation gap: `0.0630792826`

## Causal conclusion
Classification: `PARAMETER_FREE_SPARSE_PROJECTION_IMPROVES_BOTH_BUT_REMAINS_ABOVE_GATE`.

1. Post-hoc positive remapping was already insufficient in C1.
2. Post-hoc teacher-free sparse projection is also insufficient in C1B.
3. The model contains strong ownership/support information, but fixed output projection cannot simultaneously preserve weak true blend weights and remove rare high inactive false positives.
4. Blind 4k continuation remains unjustified.
5. No architecture, FSQ, token-count, geometry-path or extra-character change is justified.

## Next experiment
Proceed to the preregistered causal direction from C1: **V7-C2 importance-corrected active-heavy warm-start control/treatment**.

The treatment must retain the exact active-heavy sample proposal so sparse-positive coverage is unchanged, but analytically importance-correct the separable BCE/MSE terms toward the uniform-supervised row target. Dice semantics must remain identical between control and treatment in this first causal test; do not bundle a blend-ratio/simplex auxiliary yet.

If C2 materially reduces inactive leakage and deformation while p95 approaches the teacher-support-oracle floor (~0.08), then a separate later treatment may target the remaining within-support blend ratio error.