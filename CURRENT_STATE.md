# RealSaS-OPT — Current State

**Date:** 2026-08-28  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_PRE_PROXY32_DIAGNOSTICS_COMPLETE__PROXY27_QUALIFICATION_SEALED_READY__DOWNSTREAM_CLOSED_UNTIL_ONE_SHOT_RUN__DEV32_CLOSED`

This file is the single continuation authority.

## Frozen E0 downstream question

Does the exact observable A×8 common-frame substrate preserve enough rigging-relevant information for matched downstream information-isolation proxies?

```text
D0 = fixed-budget full-mesh reference + oracle observation provenance
D1 = exact visible-union substrate + oracle physical persistence
D2 = exact visible-union substrate + frozen teacher-free MUTUAL_P003 persistence
```

D0 is not an unconstrained information ceiling: it spends 512 area-uniform full-mesh slots while D1/D2 pool visible rows and use common-frame FPS for 512 anchors. D0→D1 therefore mixes information access and fixed-budget representation/allocation.

## Reusable FIT substrate authority

`prep_v1_2` remains immutable authority for the existing 437 FIT/selection/calibration packs:

- 437 packs complete;
- **do not rebuild them** for Proxy27;
- geometry source SHA-256 `88872577354055e8559e5e343834355068d2cc5bfd2633f7196a6ae45324fa40`;
- downstream builder SHA-256 `0f599e8f717d4c5070c04e224d90e52d1dc6e76a6e2068e64ed7c9d9d2b950b4`;
- D2 remains `MUTUAL_P003`, threshold `0.003`, teacher identity forbidden.

## Calibration V1.3 and checkpoints

Frozen FIT population:

- train 374;
- Arachne common-target train 371;
- selection 59;
- truth-capable historical calibration 4;
- Geppetto train 374.

Calibration result SHA-256:
`82bbb1b56266742242bee5995209df6d83ecfec177ff6a431d13553491ae36b9`.

Calibration4 aggregates, lower is better:

| Metric | D0 | D1 | D2 | D0→D1 | D1→D2 | D0→D2 |
|---|---:|---:|---:|---:|---:|---:|
| Arachne CE | 1.195226 | 1.131971 | 1.146689 | -5.29% | +1.30% | -4.06% |
| Arachne influence displacement | 0.0125986 | 0.0117482 | 0.0119394 | -6.75% | +1.63% | -5.23% |
| Geppetto joint mean | 0.137285 | 0.143399 | 0.136530 | +4.45% | -4.79% | -0.55% |
| Geppetto family-p95 | 0.171975 | 0.179822 | 0.164098 | +4.56% | -8.74% | -4.58% |

Sealed V1.3 checkpoint source: `checkpoints_v1_3`. Qualification performs **no training** and loads these checkpoint bytes unchanged.

## Frozen qualification margins

Authority: `E0_DOWNSTREAM_PROXY_MARGIN_DECISION_V1.md/json`.

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

All checks must pass. Margins, metric roles, population rules and D2 admission cannot change after downstream qualification opens.

## Pre-Proxy32 diagnostics — COMPLETE

Result:
`E0_PRE_PROXY32_DIAGNOSTICS_RESULT_V1.json`  
SHA-256: `906e6e686aa2e82e3e27851b4d5f10dc5566b8c55b583a73f0b6e379dcf4370d`.

Execution firewalls passed:

- train64 / selection59;
- official pack rebuild count 0;
- Proxy32/Proxy27 downstream remained closed;
- DEV32 remained closed;
- qualification margins unchanged.

### Implicit-N finding

On 30,208 pooled evaluation points:

| probe | mean cosine ↑ | oriented median ↓ | p90 ↓ | p95 ↓ |
|---|---:|---:|---:|---:|
| P3 | 0.5283 | 44.50° | 104.42° | 126.51° |
| X36 | **0.7874** | **24.95°** | **64.27°** | **77.91°** |

Therefore X36 already carries substantial decodable orientation information. `N0` in the future B3 experiment means **no explicit N**, not zero orientation information.

### D0 budget finding and caveat

D0@512→D0@2048 greatly improved geometric coverage (NN median `0.02126→0.01108`, coverage@.02 `0.4897→0.8776`) while the frozen fixed-capacity Arachne/Geppetto probes moved negligibly.

Frozen interpretation:

> Under the frozen fixed-capacity downstream probes, D0 spatial-coverage scarcity alone does not explain the observed D0@512–D1@512 difference within this fixed-probe regime.

This does **not** distinguish D0-specific representation/allocation effects from probe-capacity saturation. D1@2048 was not run and is not required before qualification.

The specific contribution of hidden-surface information was **not isolated**; D0@2048 increased both visible and hidden full-mesh density. Do not cite this diagnostic as proof that hidden surfaces are useless.

Authority: `E0_PRE_PROXY32_DIAGNOSTIC_RESULT_NOTE_V1.md`.

## N-B3 prereg amendment

Parent epsilon prereg remains independent of Proxy27. Post-diagnostic amendment:
`N_B3_EPSILON_PREREG_AMENDMENT_V1_1.md`.

Frozen future B3 arms:

```text
N0  = X36, no explicit N
N3a = X36 + historical structured deterministic N(P_epsilon) [bridge]
N3v = X36 + visibility/depth-constrained deterministic
      N_det(P_epsilon, raster, depth, support, frozen cameras)
N3b = oracle-component diagnostic only
N2  = X36 + exact geometric N upper bound
```

Epsilon remains `{0.000, 0.001, 0.003, 0.010}`. B3 is **not executed yet** and cannot retroactively change Proxy27 qualification.

## Proxy27 qualification — SEALED READY, NOT OPENED

Original frozen FIT_PROXY32 membership is 32 families. Metadata-only capability intersection produced exactly **27** truth-capable families.

- intersection bytes SHA-256: `fcbdd90d585a3845b8f799fbd1ba6dff526d4714aa581a7f6d301da86d2ecbd9`;
- ordered-ID SHA-256: `b9120bbd3f603cee6bf80110e170309b9fe8b135ae56fbbd6d8d1bda4fc11817`;
- downstream outcomes remain unopened at this state.

Qualification authority:

- `E0_PROXY27_QUALIFICATION_PREREG_V1.md`
- `E0_PROXY27_NONBINDING_EXPECTATION_V1.md/json`
- `E0_PROXY27_QUALIFICATION_PREFLIGHT_V1.md/json`
- `E0_PROXY27_QUALIFICATION_PACKAGE_V1.json`
- `run_e0_proxy27_qualification_v1.py`
- `RealSaS_E0_PROXY27_QUALIFICATION_V1.ipynb`

Exact notebook SHA-256:
`583790ccb53858c11b48ab4f3f433f5e6a01495b9cadeac9d7f79c16e4815975`.

Exact executable runner SHA-256 (embedded in the notebook):
`505420160174ad236355ab62106273dde90a72bbe3d4eac498e4292d9df80d58`.

Preflight status: **PASS before Proxy27 open**.

- embedded source integrity PASS;
- six sealed checkpoint SHA checks 6/6 PASS;
- recovered exact evaluator replayed on real calibration asset across Arachne/Geppetto × D0/D1/D2 with max numeric drift `1.0431e-7`;
- no Proxy27 metric evaluated in preflight;
- no scientific training executed.

The non-binding expectation is frozen only to test generalization of the calibration pattern. It cannot affect the gate.

### Qualification target-availability rule

All 27 packs are built first. If any member has zero Arachne target support in any matched arm, the run stops **before downstream metrics** with:

`BLOCKED_TARGET_AVAILABILITY__NO_DOWNSTREAM_METRICS_EVALUATED`.

No family is silently removed.

### One-shot guard

If a sealed qualification result already exists, the notebook refuses a second open. A crash before final result publication may be resumed only with the exact frozen notebook/source/checkpoint hashes and no retuning based on partial observations.

## Next executable action

Run `RealSaS_E0_PROXY27_QUALIFICATION_V1.ipynb` on Colab GPU with **Run all**.

It must:

- rebuild **0/437** official FIT packs;
- build only 27 separate qualification compact packs;
- load the six sealed V1.3 checkpoints without training;
- apply the already-frozen margin decision exactly once;
- preserve per-family metrics in the result;
- keep DEV32 closed;
- output `E0_PROXY27_QUALIFICATION_RESULT_V1.json` and `E0_PROXY27_QUALIFICATION_SEAL_V1.json`.

No additional diagnostic is authorized before this one-shot run.

## Authorization / firewall state

`E0_CALIBRATION8_V1_3 = COMPLETE`

`E0_B_ADMISSION = MUTUAL_P003_FROZEN`

`E0_DOWNSTREAM_PROXY_PREP_V1_2 = COMPLETE_437_REUSABLE_PACKS`

`E0_DOWNSTREAM_PROXY_CALIBRATION_V1_3 = COMPLETE_FIT_ONLY`

`DOWNSTREAM_NONINFERIORITY_MARGINS = FROZEN_V1`

`E0_PRE_PROXY32_DIAGNOSTICS = COMPLETE__INTERPRETATION_FROZEN`

`E0_PROXY27_NONBINDING_EXPECTATION = FROZEN__NON_BINDING`

`E0_PROXY27_QUALIFICATION_PACKAGE = SEALED_READY__NOT_OPENED`

`E0_PROXY27_DOWNSTREAM_EVALUATION = CLOSED_UNTIL_EXACT_NOTEBOOK_RUN`

`N_B3_EPSILON = PREREG_AMENDED_V1_1__NOT_EXECUTED`

`DEV32_EXTERNAL = CLOSED`

`E0_PRODUCT_PASS = NOT_CLAIMED`

`PRODUCT_DOMAIN_V1 = DEFERRED_PROSPECTIVE_AFTER_E0`
