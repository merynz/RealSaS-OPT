# RealSaS N1D — Functional Quotient Existence Audit V1 — Canonical Result

**Date:** 2026-08-19  
**Verdict:** `FUNCTIONAL_NON_SINGLETON_MATERIAL__RETURN_TO_RESEARCH`  
**Prereg commit:** `5327b8a879d4e4ec38841fe850ad5f9ea5eaf282`  
**Scope:** frozen GFDR-V2 mechanical-functional contract only; M5 dense skin-weight truth is absent from this broad e00 panel.

## Question

Does the geometric hard tail left by the frozen M256 + R_REL_DIS + two-round cavity solver merely represent alternative geometries that are functionally equivalent, or does it contain mechanically distinct states?

The test intentionally distinguishes `|Q_geometry(O)|` from `|Q_functional(O)|`.

## Validity

Parent reconstruction is exact under the frozen behavioral guards:

```text
round1 reference candidate-index parity   512/512
round2 reference candidate-index parity   512/512
primary n                                 261
round1 primary contain2                   .7777777778
round2 primary contain2                   .8160919540
round2 worst-family contain2              .5200000000
M256 reliable denominator                 492
M256 pooled contain2                      .9756097561
M256 worst-family contain2                .9322033898
geometric hard-tail witnesses             48
```

A parity bug in the first local rebuild was caught before any functional result was interpreted: local surface scale had accidentally used k=6. Canonical source uses k=4. After restoring k=4, all eight family P1 metrics and 512/512 round1/round2 candidate indices reproduced exactly.

## Counterfactual

For each of the 48 primary round2 hard-tail witnesses, compare:

```text
x_round2
vs
x_swap = same complete round2 configuration
         with only that node replaced by the teacher-nearest feasible M256 candidate
```

The comparison is local on the frozen graph motif `{i} U N_graph(i)` using the exact canonical GFDR-V2 implementation. Candidate N_B/V_B are attached by nearest exact dense Pose-B surface sample as preregistered.

## Main result

```text
hard-tail witnesses             48
GFDR-functionally equivalent      0
GFDR-functionally distinct       48
pooled equivalence fraction     0.000
hard-family pooled fraction     0.000
```

Per family:

```text
family   hard n   equivalent
09908       3        0
11032      12        0
12772       3        0
13203      12        0
14404       8        0
14702       1        0
14758       0        —
15290       9        0
```

The preregistered B gate is therefore exceeded by a very large margin. In particular, 11032 and 13203 each have 12/12 non-equivalent hard-tail witnesses, 14404 has 8/8, and 15290 has 9/9.

## Which mechanical consequences differ?

Block pass fractions across the 48 witnesses:

```text
F_delta_NRMS             0 /48    .000
F_response_NRMS          4 /48    .083
F_kernel_RMS            44 /48    .917

D_surface_action_NRMS    8 /48    .167
D_residual_NRMS         12 /48    .250
D_gradient_NRMS         25 /48    .521

R_relB_NRMS              0 /48    .000
R_motion_NRMS            4 /48    .083
R_transfer_NRMS         25 /48    .521

G_direction_disagree    25 /48    .521
G_axis_point_norm_RMS    7 /48    .146
G_support_RMS           46 /48    .958
```

This pattern matters. The alternative geometry often preserves coarse structural support/co-response (`F_kernel`, `G_support`) while changing the actual finite displacement, relative motion realization, differential action, and articulation locus. The hard tail is therefore not merely a cosmetic teacher-coordinate mismatch under the frozen mechanical contract.

## Geometry / belief context

```text
round2 selected normalized teacher error
  median   3.3333
  p90      5.3532

teacher-nearest feasible M256 normalized error
  median    .7526
  p90      1.4605

teacher-nearest candidate normalized rank in round2 belief
  median    .2392
  p90       .5600
```

So the functionally different teacher-near alternative is frequently still present within the bounded M256 domain; the current observation-native objective simply does not collapse to the correct functional class.

## Frozen decision

Preregistered decision tree:

- A (`GEOMETRIC_SINGLETON_NOT_REQUIRED...`) required pooled equivalence >= .80, hard-family >= .75, and family-tail safeguards. **FAIL.**
- B (`FUNCTIONAL_NON_SINGLETON_MATERIAL__RETURN_TO_RESEARCH`) triggers if pooled equivalence <= .50 or a >=5-witness family <= .30. **PASS decisively: pooled=.00.**

Canonical verdict:

`FUNCTIONAL_NON_SINGLETON_MATERIAL__RETURN_TO_RESEARCH`

## What this proves

On the tested truth-open broad-e00 hard tail, the current M256/R_REL_DIS/two-round solution set contains geometrically different choices that also induce materially different GFDR-V2 mechanical behavior. Therefore the present representation+objective has **not** established `|Q_functional(O)| = 1`.

This means the remaining endpoint hard tail cannot be dismissed as harmless canonical-coordinate freedom. Research must resume.

## What this does NOT prove

It does **not** prove that exact teacher/geometric singleton is universally necessary. There are still two logically distinct possibilities:

1. the mechanically distinct classes are separable by raster-observable typed evidence that is not yet correctly represented/used by the solver (for example candidate-conditioned F/D/G evidence);
2. the paired raster observations themselves do not distinguish the mechanically distinct classes, in which case richer observations, explicit ambiguity/set-valued compilation, or another input assumption is required.

The next research question is therefore **functional-class observability**, not another solver-depth sweep and not automatically “reconstruct exact teacher geometry.”

## Dense-skinning limitation

The broad e00 sidecars used here contain surface geometry, normals and visibility but no M5 dense skin-weight truth. This audit is therefore scoped to the frozen GFDR-V2 mechanical-functional contract. Because the result is already a decisive functional non-singleton under GFDR, missing M5 truth cannot turn the current contract into a proven singleton; however a future full rig-functional theorem still requires dense deformation/skinning consequences on an appropriate panel.

## Provenance

- prereg: `5327b8a879d4e4ec38841fe850ad5f9ea5eaf282`
- canonical P1 source blob: `2ab6e9a93594b359a605871d5018b07628e8ec2a`
- canonical current-H source blob: `95a8f713a87f926523d2bdbbebc03e511477f1ea`
- reconstructed parent runner SHA256: `506039a5b02e59035bc79c3635f3896022887bf66a8a3b609640cce219ac25ef`
- functional audit SHA256: `1553d52f1d2206485e1fd394f7a94f5dd187782afe9d2ae5271321e16d498967`
- family worker SHA256: `ba233b5edd961a28a8e50c73f769646f1f14446896d99f5d32ba304df02a853a`
- full local result SHA256: `7dac39ad90b4c9e41e41b855f6c021a8bf0c5659541a7891136fe0453f433e27`

## Authorization

Large/end-to-end training remains forbidden. Stage-B authority remains immutable FAIL/no-retune. `sealed21` / `external10` remain closed.

No further BP depth, graph retune, R_REL_DIS retune, free XYZ, or arbitrary F/D injection is authorized from this result.

Next authorized scientific task: preregister and run a **functional-class observability audit** on these mechanically non-equivalent hard-tail alternatives, testing whether frozen raster-observable typed evidence can distinguish the functional classes without teacher-conditioned selection.
