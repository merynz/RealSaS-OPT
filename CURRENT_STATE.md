# RealSaS-OPT — Current State

**Date:** 2026-08-26  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `P_SCALE_SIGNAL_NOT_STRONG__GAUGE_PATHOLOGY_LOCALIZED__D2_D5_LOCALIZES_CROSSVIEW_CORRESPONDENCE_EVIDENCE_BOTTLENECK__SEALED_CLOSED`

## Read this first

This file is the single continuation authority.

The active question is the **single-pose observable common-frame P hard tail**, specifically camera-forward depth under known cameras. Asset semantic orientation is not part of the P contract: lying, upright, plant, animal, etc. are all valid if observation/camera/teacher frames are consistent.

## Current P formulation

P-V5/R256 uses analytic screen-plane coordinates from raster XY + known camera and learns only camera-forward depth:

```text
raster x/y + known camera -> analytic screen-plane P
image evidence            -> learned forward depth
                                  |
                                  v
                              full P
```

## Frozen FIT-scale ladder

Same architecture, no augmentation change, 7168 successful optimizer updates per rung:

| FIT families | aggregate P95 | cell median P95 | worst cell P95 |
|---:|---:|---:|---:|
| 32 | 0.29040165 | 0.17066082 | 0.51614741 |
| 128 | 0.21184100 | 0.09814133 | 0.51423088 |
| 512 | 0.19979690 | 0.06592983 | 0.51396067 |

Verdict: `SCALE_SIGNAL_BUT_NOT_STRONG`.

512/32 aggregate ratio = `0.68800194`; median ratio = `0.38632081`.

Frozen rung512 checkpoint SHA-256:
`a92fa18c46975578f2705ee3baa426cce573d4a767d42490e7599ed905f6e48e`.

## Gauge localization V3 — COMPLETE

Decision: `TARGET_GAUGE_PATHOLOGY_ESTABLISHED__CORRECT_BEFORE_OPTIMIZATION_OR_PATCHMATCH`

- PROXY32 analytically impossible: `2/64`
- PROXY32 gauge-safe: `62/64`
- TRAIN512 analytically impossible: `6/1024`
- repeated micro-P95 `28.4803` spikes share `asset_90b5f2cc531696040da996a9`

Interpretation: severe target/gauge pathologies exist but explain only a small minority of the proxy hard tail.

## D1 multi-asset raw-RGB discriminability — COMPLETE

Three gauge-safe witnesses:

- `f089 = asset_f089abadcd071194617d640b`: low-texture giant-plane positive control; detached giant quad is real source geometry but semantic role unresolved.
- `ea593 = asset_ea593d044e14f20abe6d2818`: severe foreshortening/correspondence hard tail. Its horizontal/lying orientation remains a **valid P sample**, not an exclusion or excuse.
- `662ed = asset_662ed7f1e328bd85959157cf`: deer/antler, thin/multisurface/repeated-appearance hard tail.

D1 falsified `raw RGB/alpha reprojection + fixed view aggregation` as a universal solution. Silhouette strongly retained truth on ea593/662ed but left broad depth corridors.

## D2→D5 P hard-tail oracle — COMPLETE EXPLORATORY LOCALIZATION

Canonical report:
`experiments/g0_g1_single_pose_geometry/hardtail_forensics_20260826/P_HARDTAIL_ORACLE_D2_D5_20260826.md`

Execution used the exact frozen rung512 checkpoint, no IRIS optimizer steps, and no TUNE/CAL/DEV/EXTERNAL opening.

### Checkpoint representation caveat

Rung512 was a fresh model initialization. Trained P-side modules include `encoder`, `within`, `cross`, `context_fuse`, decoder stages, `p_s1_stem`, `p_s1_fuse`, and `p_depth_head`. Auxiliary `n_head`, `u_geo_head`, `fine_head`, and `coarse_head` were not trainable in this run and therefore are not treated as learned descriptors.

### D2 — frozen learned-feature reprojection

321 depth candidates, 5600 loci/asset, known-camera seven-view reprojection, teacher geometry evaluation-only, no teacher visibility in scoring.

| asset | matched direct P95 | best D2 P95 | best D2 mode |
|---|---:|---:|---|
| f089 | 0.416423 | **0.123106** | `encoder.s2 / mean7` |
| ea593 | **0.471771** | 0.574262 | `encoder.s2 / median7` |
| 662ed | **0.271590** | 0.304407 | `p_stem / median7` |

Thus the frozen learned evidence is highly discriminative on f089 but does **not** solve the representative ea593/662ed tails.

Matched direct cardinal P95 exposes strong view structure:

- ea593: V0 `0.649064`, V2 `0.062547`, V4 `0.618926`, V6 `0.065106`
- 662ed: V0 `0.339891`, V2 `0.054731`, V4 `0.287579`, V6 `0.057863`

This is morphology/view-dependent hard-tail structure, not uniform regression noise.

### D3 — exact-normal plane-warp oracle

To avoid using the untrained N head, D3 grants exact teacher face normal only for the local plane warp.

Hard V0/V4 aggregate P95:

- ea593: direct `0.676355`; point feature `0.685487`; exact-normal plane `0.656420`
- 662ed: direct `0.329607`; point feature `0.362542`; exact-normal plane `0.358451`

Conclusion: normal/plane warp gives at most modest local rescue and does not close the representative tail.

### D4 — oracle visibility/source-view selection

Frozen `encoder.s2` scoring plus teacher visibility used only to oracle-select visible cardinal source views:

- f089: baseline `0.391596` -> oracle `0.815709`
- ea593: baseline `0.701307` -> oracle `0.689757`
- 662ed: baseline `0.370358` -> oracle `0.412018`

Conclusion: occlusion/view-selection is not the sole missing mechanism.

### D5 — multi-scale feature consistency

ACMM-like frozen hierarchy:

```text
s8@32 top64 -> s4@64 top16 -> s2@128 final
```

- f089: direct `0.418936`; single s2 mean `0.130134`; hierarchy mean `0.132531`
- ea593: direct `0.472422`; single s2 median `0.576046`; hierarchy mean `0.602338`
- 662ed: direct `0.267765`; single s2 median `0.328551`; hierarchy mean `0.343820`

Truth retention is the key failure:

- ea593: s8 top64 `0.618` -> after s4 top16 `0.253`
- 662ed: s8 top64 `0.840` -> after s4 top16 `0.338`

The search hierarchy prunes truth too early; propagation cannot recover evidence that has already ranked the correct candidate away.

### D2b — tiny frozen-feature learned metric probe

A 3-asset leave-one-out logistic correspondence metric over frozen `encoder.s2` pair descriptors was run as a follow-up. IRIS stayed frozen; positives/negatives came from teacher correspondences on the other two open FIT witnesses.

- held-out f089: direct P95 `0.414225` -> metric `0.397693`
- held-out ea593: direct `0.464965` -> metric `0.798937`
- held-out 662ed: direct `0.275291` -> metric `0.410630`

This tiny probe is negative on the two representative tails. It is **not sufficient to prove the frozen encoder lacks the information**, but it falsifies the trivial “just learn a tiny pair metric on these three assets” explanation.

## Current localization

Survives:

```text
known camera + candidate common-frame P + cross-view verification
```

Not established as a general solution:

```text
raw RGB/alpha                    [D1]
current frozen feature cosine    [D2]
exact normal / plane warp        [D3]
oracle visibility selection      [D4]
simple multi-scale pruning       [D5]
tiny 3-asset pair metric         [D2b]
```

**Strongest current diagnosis:** representative gauge-safe failures are bottlenecked **before or at cross-view surface correspondence evidence**. Under severe foreshortening, thin/overlapping structures, repeated color and view-dependent cel appearance, current evidence does not rank truth reliably enough. This is not yet an information-limit claim and not yet proof that the encoder itself must change.

## Canonical experiment archive

`experiments/g0_g1_single_pose_geometry/hardtail_forensics_20260826/`

Chronology now includes:

1. gauge hard-tail forensics
2. f089 preliminary and teacher-free propagation positive control
3. multi-asset D1 raw-RGB comparison
4. D2→D5 frozen-feature / plane / visibility / multi-scale localization
5. D2b tiny leave-one-asset-out learned metric probe

Failed intermediate hypotheses are intentionally preserved.

## NEXT EXECUTABLE STEP

`BROADER FIT-DISJOINT CROSS-VIEW CORRESPONDENCE PROBE — METRIC VS REPRESENTATION`

Use many **other open FIT families** for teacher-derived positive correspondences and hard epipolar negatives; keep ea593/662ed frozen as held-out witness assets for that probe.

Required causal fork:

```text
frozen encoder features
   + learned cross-view correspondence metric
            |
            +-- succeeds -> metric/evidence-consumer problem
            |
            +-- fails    -> correspondence-aware representation adaptation/training
```

Only once truth ranking is demonstrated should PatchMatch-style propagation/search be reconsidered.

## Authorization state

`GAUGE_LOCALIZATION = COMPLETE`

`MULTI_ASSET_D1_RGB_ORACLE = COMPLETE`

`D2_LEARNED_FEATURE_REPROJECTION = COMPLETE_NEGATIVE_REPRESENTATIVE__POSITIVE_F089`

`D3_EXACT_NORMAL_PLANE = COMPLETE_NO_CLOSURE`

`D4_VISIBILITY_ORACLE = COMPLETE_NO_CLOSURE`

`D5_MULTISCALE = COMPLETE_NO_CLOSURE`

`D2B_3ASSET_METRIC = EXPLORATORY_NEGATIVE_REPRESENTATIVE`

`FULL_PATCHMATCH = NOT_AUTHORIZED`

`LONGER_P_TRAINING = NOT_NEXT`

`CAL_DEV_EXTERNAL = CLOSED`

`PRODUCT_SUBSTRATE_CLOSURE = NOT_CLAIMED`

## Research rule

```text
apparatus/target gauge
 -> observation discriminability
 -> representation
 -> learner/optimizer
 -> evidence consumer / geometric verification
 -> only then information limit
```
