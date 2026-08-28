# RealSaS-OPT — Current State

**Date:** 2026-08-28  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS__N_B3_NEXT__PREDICTED_P_BRIDGE_REQUIRED__DEV32_CLOSED`

This file is the single continuation authority.

## Frozen E0 downstream question — CLOSED PASS

Question:

> Does the exact observable A×8 common-frame substrate preserve enough rigging-relevant information for matched downstream information-isolation proxies?

Arms:

```text
D0 = fixed-budget full-mesh reference + oracle observation provenance
D1 = exact visible-union substrate + oracle physical persistence
D2 = exact visible-union substrate + frozen teacher-free MUTUAL_P003 persistence
```

D0 is not an unconstrained information ceiling: it spends 512 area-uniform full-mesh slots while D1/D2 pool visible rows and use common-frame FPS for 512 anchors. D0→D1 therefore mixes information access and fixed-budget representation/allocation.

The one-shot frozen 27-member truth-capable Proxy32 intersection qualification is now complete and PASS.

## Reusable FIT substrate authority

`prep_v1_2` remains immutable authority for the existing 437 FIT/selection/calibration packs:

- 437 packs complete;
- geometry source SHA-256 `88872577354055e8559e5e343834355068d2cc5bfd2633f7196a6ae45324fa40`;
- downstream builder SHA-256 `0f599e8f717d4c5070c04e224d90e52d1dc6e76a6e2068e64ed7c9d9d2b950b4`;
- D2 remains `MUTUAL_P003`, threshold `0.003`, teacher identity forbidden.

Do not rebuild these 437 packs for downstream model/evaluator changes unless the substrate/pack scientific contract itself changes.

## Calibration V1.3 and frozen checkpoints

Frozen FIT population:

- train 374;
- Arachne common-target train 371;
- selection 59;
- truth-capable historical calibration 4;
- Geppetto train 374.

Calibration result SHA-256:
`82bbb1b56266742242bee5995209df6d83ecfec177ff6a431d13553491ae36b9`.

Sealed checkpoint source: `checkpoints_v1_3`.
Qualification used the six sealed checkpoint bytes unchanged and performed no training or retuning.

## Frozen qualification margins

Primary lower-is-better metrics Arachne CE, Arachne influence displacement, Geppetto joint mean:

```text
D1/D0 <= 1.05
D2/D1 <= 1.05
D2/D0 <= 1.05
```

Geppetto `family_p95`: same comparisons at `<=1.10`.

Catastrophe veto:

```text
PCK@0.05(D2) >= PCK@0.05(D0)-0.10
PCK@0.08(D2) >= PCK@0.08(D0)-0.10
```

These margins were frozen before qualification and were not changed after downstream open.

## Pre-Proxy32 diagnostics — COMPLETE

Result SHA-256:
`906e6e686aa2e82e3e27851b4d5f10dc5566b8c55b583a73f0b6e379dcf4370d`.

### Implicit-N finding

On 30,208 pooled evaluation points:

| probe | mean cosine ↑ | oriented median ↓ | p90 ↓ | p95 ↓ |
|---|---:|---:|---:|---:|
| P3 | 0.5283 | 44.50° | 104.42° | 126.51° |
| X36 | **0.7874** | **24.95°** | **64.27°** | **77.91°** |

Therefore X36 already carries substantial decodable orientation information. `N0` means no explicit N, not zero orientation information.

### D0 budget finding and caveat

D0@512→D0@2048 greatly improved geometric coverage while frozen fixed-capacity Arachne/Geppetto probes moved negligibly.

Frozen interpretation:

> Under the frozen fixed-capacity downstream probes, D0 spatial-coverage scarcity alone does not explain the observed D0@512–D1@512 difference within this fixed-probe regime.

This does not distinguish D0-specific representation/allocation effects from probe-capacity saturation. The specific contribution of hidden-surface information was not isolated.

Authority: `E0_PRE_PROXY32_DIAGNOSTIC_RESULT_NOTE_V1.md`.

## Proxy27 qualification — PASS

Original FIT_PROXY32 membership contained 32 families. Metadata-only downstream-truth capability intersection produced exactly 27 members.

- ordered population SHA-256: `b9120bbd3f603cee6bf80110e170309b9fe8b135ae56fbbd6d8d1bda4fc11817`;
- qualification result SHA-256: `86153066f487bc16fdfd4e817494fa3bd052def076ef97cd998c013455ee74e5`;
- qualification seal SHA-256: `0bbd00843436e78c4cc38cb50765295ba3ba7844c57b7ca7cb23fe3f7c57fe78`;
- status: `E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS`;
- all frozen gate checks: **14/14 PASS**;
- DEV32 opened: **false**;
- qualification margins changed: **false**;
- expectation used for decision: **false**.

Canonical report:
`experiments/g0_g1_single_pose_geometry/e0_observable_geometry_20260827/E0_PROXY27_QUALIFICATION_CANONICAL_REPORT_V1.md`.

### Proxy27 aggregates

| Metric | D0 | D1 | D2 |
|---|---:|---:|---:|
| Arachne CE ↓ | 2.630374 | 2.405529 | 2.501573 |
| Arachne influence displacement ↓ | 0.0118685 | 0.0120806 | 0.0122102 |
| Geppetto joint mean ↓ | 0.227764 | 0.233639 | 0.226213 |
| Geppetto family-p95 ↓ | 0.409818 | 0.422785 | 0.426079 |
| Geppetto PCK@.05 ↑ | 0.102829 | 0.081119 | 0.087815 |
| Geppetto PCK@.08 ↑ | 0.220206 | 0.198208 | 0.209494 |

Key ratios:

```text
Arachne CE:
  D1/D0 = 0.914520 PASS
  D2/D1 = 1.039926 PASS   <- narrowest primary headroom
  D2/D0 = 0.951033 PASS

Arachne influence:
  D1/D0 = 1.017873 PASS
  D2/D1 = 1.010727 PASS
  D2/D0 = 1.028791 PASS

Geppetto joint mean:
  D1/D0 = 1.025794 PASS
  D2/D1 = 0.968215 PASS
  D2/D0 = 0.993190 PASS

Geppetto family-p95:
  D1/D0 = 1.031641 PASS
  D2/D1 = 1.007792 PASS
  D2/D0 = 1.039679 PASS

PCK catastrophe veto:
  @.05 D2-D0 = -0.015014 PASS
  @.08 D2-D0 = -0.010712 PASS
```

The closest primary gate is Arachne CE D2/D1: a `+3.9926%` persistence tax against a frozen `+5%` maximum degradation. This is the most sensitive rung for the later predicted-P/depth-noise bridge.

### Expectation audit

The non-binding calibration expectation only partially generalized:

- Arachne CE direction held, though D2/D1 tax was larger than calibration4;
- Arachne influence did not preserve the calibration improvement direction, but remained inside the 5% margin;
- Geppetto joint-mean direction held;
- Geppetto family-p95 did not show the calibration4 D2 recovery, but remained comfortably inside the 10% tail guard.

Therefore calibration4 magnitude/tail behavior must not be treated as a population law. The PASS rests solely on the preregistered qualification rule.

## Boundary of the E0 PASS

`E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS = TRUE` means:

> Exact observable P plus the frozen observation-derived support/provenance and teacher-free MUTUAL_P003 persistence preserve enough rigging-relevant information under the matched information-isolation consumers and frozen non-inferiority margins.

It does **not** mean:

- product Geppetto/Arachne are qualified;
- a trained IRIS already predicts P accurately enough;
- predicted-P MUTUAL_P003 is qualified;
- N is decided;
- E0 product pass is claimed.

## N-B3 prereg amendment — NEXT

Authority:
`N_B3_EPSILON_PREREG_AMENDMENT_V1_1.md`.

Frozen B3 arms:

```text
N0  = X36, no explicit N
N3a = X36 + historical structured deterministic N(P_epsilon) [bridge]
N3v = X36 + visibility/depth-constrained deterministic
      N_det(P_epsilon, raster, depth, support, frozen cameras)
N3b = oracle-component diagnostic only
N2  = X36 + exact geometric N upper bound
```

Epsilon remains `{0.000, 0.001, 0.003, 0.010}`.

## Required predicted-P bridge after B3

Product IRIS will not receive exact P. Under known cameras it is expected to infer primarily forward depth and compile:

`P_hat = O + d_hat * F`.

Therefore exact-P E0 closure must be followed by a ray-aligned forward-depth noise / predicted-P bridge:

```text
controlled depth error
  -> analytic P_hat
  -> MUTUAL_P003
  -> frozen downstream consumers
```

This bridge should derive the allowed IRIS depth-error envelope from the downstream margins, with special attention to the narrow Arachne CE D2/D1 headroom.

Only after a trained depth model satisfies that envelope may the product IRIS contract be frozen.

## Next executable sequence

1. N-B3 V1.1.
2. Ray-aligned forward-depth tolerance ladder / predicted-P bridge.
3. Backbone bake-off and native C-path adaptation study.
4. Train depth-only IRIS candidate under known-camera constraints.
5. Qualify `P_hat -> MUTUAL_P003 -> downstream`.
6. Freeze final IRIS product contract only if the bridge passes.

## Authorization / firewall state

`E0_CALIBRATION8_V1_3 = COMPLETE`

`E0_B_ADMISSION = MUTUAL_P003_FROZEN`

`E0_DOWNSTREAM_PROXY_PREP_V1_2 = COMPLETE_437_REUSABLE_PACKS`

`E0_DOWNSTREAM_PROXY_CALIBRATION_V1_3 = COMPLETE_FIT_ONLY`

`DOWNSTREAM_NONINFERIORITY_MARGINS = FROZEN_V1`

`E0_PRE_PROXY32_DIAGNOSTICS = COMPLETE__INTERPRETATION_FROZEN`

`E0_PROXY27_NONBINDING_EXPECTATION = FROZEN__NON_BINDING`

`E0_PROXY27_QUALIFICATION = PASS__14_OF_14__SEALED`

`E0_DOWNSTREAM_INFORMATION_SUFFICIENCY_PROXY_PASS = TRUE`

`N_B3_EPSILON = PREREG_AMENDED_V1_1__RUN_NEXT`

`PREDICTED_P_DEPTH_BRIDGE = REQUIRED_AFTER_B3`

`DEV32_EXTERNAL = CLOSED`

`E0_PRODUCT_PASS = NOT_CLAIMED`

`PRODUCT_DOMAIN_V1 = DEFERRED_PROSPECTIVE_AFTER_E0`
