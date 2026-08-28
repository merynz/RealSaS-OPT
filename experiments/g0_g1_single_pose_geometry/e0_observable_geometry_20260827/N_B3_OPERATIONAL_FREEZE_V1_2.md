# RealSaS N-B3 — Operational Freeze V1.2

**Date:** 2026-08-28  
**Status:** `FROZEN_BEFORE_N_B3_METHOD_SELECTION_OR_DOWNSTREAM_OUTCOMES__FIT_ONLY__PROXY27_UNUSED__DEV32_CLOSED`

## Parent authority

- `N_B3_EPSILON_PREREG_V1.md` Git blob: `c926328098b7cc0918c56bb900befc18e7136156`
- parent prereg byte SHA-256 recorded by V1.1: `c5836e6f2bbb03aa1963601130e8a835932c84192e889d21300bb6ef560efb90`
- `N_B3_EPSILON_PREREG_AMENDMENT_V1_1.md` Git blob: `35ef4ea5d595302840413169844e801537db8b28`
- `CURRENT_STATE.md` Git blob at operational freeze: `89a767335c4a87c67d32f399162bc69401ebcce8`
- E0 scientific builder SHA-256: `0f599e8f717d4c5070c04e224d90e52d1dc6e76a6e2068e64ed7c9d9d2b950b4`
- E0 camera/geometry authority SHA-256: `88872577354055e8559e5e343834355068d2cc5bfd2633f7196a6ae45324fa40`
- E0 SurfaceBuilder SHA-256: `83819fc8869ff26fbfc928fbeda09e59115a0554c94cbddc623b97d1c790c1cb`

This file does not change the scientific question, frozen epsilon values, or arm meanings. It closes the operational degrees of freedom that were not specified in the parent prereg before any N-B3 method-selection score or downstream score is opened.

## Scope

N-B3 remains a **FIT-only information-value experiment**. It is not E0 product qualification and it cannot modify or reinterpret the completed Proxy27 PASS.

No Proxy27 member is used by N-B3. DEV32 remains closed.

## Population

The already-frozen E0 FIT split is reused. The original 374-member E0 training list is partitioned without outcome inspection by preserving its frozen order:

- B3 train: first 310 of frozen E0 `train_ids`
- B3 qualification: final 64 of frozen E0 `train_ids`
- B3 selection: the existing frozen 59 `selection_ids`
- historical calibration4: unused

Hashes (newline-delimited ordered IDs):

- B3 train310: `639534e4eadad4048baff35cdaa7ff27a018d045e35815d0d715c82fcff27ffb`
- B3 selection59: `d3cf11be8d0a25c557c7ece7f8c913e1cc674dcf6fd225014ab1607bce08777c`
- B3 qualification64: `dffee4dd926eef5f08f80c1cc956d2284bb6d8dfb418f2bf88a99cefdbe93644`

The 27-member Proxy27 population is disjoint from all 437 frozen FIT prep records and is unused here.

## Primary carrier

The primary observable substrate is the frozen **D2** 512-anchor carrier from `prep_v1_2`.

Reason: after E0 closure, D2 is the deployable teacher-free persistence arm. D1 and D2 share the same physical anchor P; D2 differs in observation-derived support/provenance admission. B3 asks for marginal N value beyond the product-relevant teacher-free substrate.

For every asset, evaluator-only exact N is reconstructed from the same deterministic anchor carrier and must satisfy both:

- max-abs P parity to `D1_X[:,0:3] <= 2e-6`
- max-abs P parity to `D2_X[:,0:3] <= 2e-6`

Exact mesh normal/topology authority is never exposed to N0/N3a/N3v forward construction.

## Common 39D point slot

All B3 arms use the same input dimension and therefore the same learned model parameter count:

```text
X39 = concat(X36_epsilon, N_slot3)

N0  : N_slot3 = 0
N3a : N_slot3 = historical structured deterministic N(P_epsilon)
N3v : N_slot3 = selected visibility/depth-constrained deterministic N
N3b : N_slot3 = component-masked deterministic N(P_epsilon) [oracle diagnostic]
N2  : N_slot3 = exact clean geometric N [upper bound]
```

The original X36 field ordering is untouched. N is appended at indices 36:39.

Arachne therefore uses a 60D point/joint pair feature (`39D point + frozen 21D relation`). Geppetto receives 39D point features. Every arm/epsilon model is initialized from exactly the same seeded initial state for its consumer.

## Frozen epsilon construction

Epsilon remains exactly:

```text
{0.000, 0.001, 0.003, 0.010}
```

For each `(asset, epsilon)` a PCG64 stream is seeded by the first 64 bits of:

```text
SHA256("RealSaS.NB3.v1.2|<asset_id>|D2_CARRIER|<epsilon with 6 decimals>")
```

and emits one fixed `(512,3)` standard-normal field in anchor order.

```text
P_epsilon = P + epsilon * Z
```

The same P_epsilon is shared by N0/N3a/N3v/N3b/N2 at that epsilon.

In X36:
- indices 0:3 are replaced by P_epsilon;
- support 3:11 is unchanged;
- raster XY 11:27 is unchanged;
- support fraction 27 is unchanged;
- camera-forward depths 28:36 are recomputed analytically from P_epsilon and frozen camera forward vectors.

## N3a — historical bridge

For each P_epsilon point:

1. take 12 nearest other anchors in common-frame Euclidean P;
2. form the distance-weighted covariance with `exp(-(d/median(d))^2)`;
3. use the minimum-eigenvalue eigenvector;
4. orient sign away from a Gaussian-weighted 40-NN local mass centroid;
5. normalize.

This is the historical S0-B2 structured deterministic construction and is not allowed to decide the learned-N question by itself.

## N3b — oracle-component diagnostic

N3b repeats N3a but all 12-NN covariance and 40-NN sign neighborhoods are restricted to the exact connected mesh component of the anchor carrier triangle.

Component identity is evaluator-only oracle topology. N3b is diagnostic only and cannot become product authority or rescue N3v.

## N3v — observation-only challenger

N3v receives only P_epsilon, frozen D2 raster coordinates/support and frozen camera bases.

For each view and each supported anchor:

1. query the K nearest supported anchors in 2D raster-grid coordinates;
2. fit local tangents from `delta_grid -> delta_P_epsilon` by Gaussian distance-weighted linear least squares;
3. perform one deterministic Huber reweight/refit pass;
4. form the unsigned local normal from the tangent cross product;
5. fuse all valid per-view normals by the principal eigenvector of the weighted projective normal matrix `sum(n n^T)`.

The fused unsigned normal sign uses one of two preregistered observation-only rules:
- `VISIBILITY_THEN_MASS`: orient toward the negative support-weighted sum of frozen camera-forward vectors when non-degenerate, otherwise use the P-only local-mass rule;
- `MASS_ONLY`: use only the P-only local-mass rule.

Method-development candidates are frozen to:

```text
K in {8, 12, 20}
sign_mode in {VISIBILITY_THEN_MASS, MASS_ONLY}
```

No other N3v hyperparameter may be introduced after method-selection outcomes are opened.

### N3v method selection

Only the already-open B3 selection59 population is used. Candidate geometric normal fidelity is measured at the two frozen endpoint perturbations `epsilon={0.000,0.010}`.

For each candidate, pool per-asset:
- oriented angular median;
- oriented angular p90;
- sign-invariant angular median.

Selection is lexicographic:
1. lowest mean oriented median across the two endpoint epsilons;
2. then lowest mean oriented p90;
3. then lowest mean sign-invariant median;
4. then smaller K;
5. then `VISIBILITY_THEN_MASS`.

The chosen method and all candidate scores are written and hash-sealed **before any B3 qualification64 downstream metric is evaluated**.

## Downstream training regime

B3 is an information-sufficiency experiment, not a frozen-consumer robustness experiment.

Therefore every `(epsilon, arm)` receives its own matched-capacity downstream probe training:

- Arachne: exact existing E0 schedule, seed 1862, AdamW `lr=2e-3`, `wd=1e-4`, 6 epochs, batch 24 assets, 96 deterministic valid points/asset/epoch; checkpoint selected only by selection59 CE.
- Geppetto: exact existing E0 schedule, seed 1862, AdamW `lr=1.5e-3`, `wd=1e-4`, 8 epochs, batch 16 assets, 256 deterministic points/asset/epoch; checkpoint selected only by selection59 mean matched joint distance.

This isolates information value under matched fixed-capacity consumers. The later predicted-P bridge, not B3, measures robustness of frozen consumers to inference-time P error.

All 40 checkpoints (`5 arms x 4 eps x 2 consumers`) must be trained/selected and hash-sealed before qualification64 opens.

Within a given asset/epoch, the deterministic Arachne point subset and Geppetto point subset are **identical across every arm and epsilon**; the arm/epsilon tag is forbidden from the sampling seed. Only the exposed N/P condition may differ.

## Decision metrics and margins

No new outcome-tuned margin is introduced. N3v is compared to N2 by inheriting the already-frozen E0 downstream practical non-inferiority margins:

Primary lower-is-better:
- Arachne CE: `N3v/N2 <= 1.05`
- Arachne influence displacement: `N3v/N2 <= 1.05`
- Geppetto joint mean: `N3v/N2 <= 1.05`

Tail:
- Geppetto family-p95: `N3v/N2 <= 1.10`

Catastrophe veto:
- Geppetto PCK@.05: `N3v - N2 >= -0.10`
- Geppetto PCK@.08: `N3v - N2 >= -0.10`

All six checks are evaluated separately at every frozen epsilon.

Secondary descriptive metrics include Arachne weight MAE/top1/top4-mass, Geppetto joint-p95 mean, N0/N3a/N3b deltas, and normal angular fidelity. They cannot rescue a primary failure.

## Interpretation

- If N0 is non-inferior to N2 at all four epsilons, B3 supports **no explicit normal feature** under the tested ladder.
- Else if N3v is non-inferior to N2 at all four epsilons, B3 supports **deterministic/compiler-side N; no learned direct-N head justified by information value**.
- If N3v fails N2 at one or more epsilons, B3 records the first failing epsilon and the exact metric(s). A learned N head remains a candidate robustness channel, but the final head decision is deferred until the subsequent ray-aligned predicted-P tolerance bridge determines whether those failing epsilons lie inside the product-relevant P-error envelope.
- N3a is bridge/control only.
- N3b is oracle diagnostic only.

## Firewalls

- Proxy27 B3 outcomes: not opened / not used.
- DEV32: closed.
- calibration4: unused.
- no corpus product-hygiene filtering is introduced into B3.
- no official 437 pack is rebuilt or mutated.
- no E0 Proxy27 result, margin, or checkpoint is modified.
- no B3 qualification64 downstream metric may be evaluated before N3v method selection and all 40 checkpoint hashes are sealed.
