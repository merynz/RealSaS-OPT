# RealSaS N1D — Canonical Research Status

**Date:** 2026-08-19  
**Current canonical state:** `REPRESENTATION_V2_SUPPORTED__ADAPTIVE_H_V1_FAIL__F16_SAFE_REFERENCE__UNCERTAINTY_TRIGGER_REDESIGN_REQUIRED__FACTOR_HEAD_EXECUTION_PAUSED__BIG_TRAINING_FORBIDDEN`  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Qualification boundary

- Stage A remains **PASS**.
- Frozen Stage B remains **FAIL/no-retune**; its primary F-activity was `0.433160 < 0.50` while the other principal geometry/mechanics metrics passed.
- Stage-B truth is truth-open development evidence only and cannot be reused as blind qualification.
- sealed21 / external10 remain **CLOSED**.
- Any future blind qualification requires a **new untouched preregistered panel**.

## Representation sufficiency remains supported

Canonical sufficiency decision:

`REPRESENTATION_SUFFICIENCY_SUPPORTED_FOR_REVISED_SET_VALUED_FACTORIZED_CONTRACT`

The fixed early top4 contract is permanently falsified on the truth-open broad e00 panel. The revised set-valued representation remains supported:

- fixed top4 H: pooled primary-2x `.93902`, worst family `.84746`, only `6/8` families >= `.90` — **RED**;
- same-pool top16 H: pooled `.98171`, worst `.94915`, `8/8` >= `.90`, gap `5.08pp` — **GREEN development reference**;
- representation-conditioned oracle constrained to revised H: F activity `.94033`, F kernel `.93813`, D `.15481`, R `.11745`, G direction `.96670`, G line `.04074` — **6/6 principal gates PASS**;
- carrier collision audit found no both-moving >3x amplitude or direction-cos<.5 collision in the closest tail; remaining collision class is active/non-active and is localized to planned `p_active`;
- family-disjoint raster-derived activity/amplitude ordering signal remains present.

The supported representation remains:

```text
P_A
bounded set-valued H_i / P_B_geom
N_A, N_B
V_A, V_B
Z
p_active
log_amp
dir
U_pred
+ typed U_obs
```

Authority split:

```text
bounded H_i                         -> physical XYZ authority
p_active                            -> absolute motion/non-motion authority
log_amp                             -> conditional magnitude/ranking evidence
dir                                 -> direction evidence
U / margin / multimodality/support -> retention / expansion / abstention
compiler/global solver              -> final collapse after global consistency
```

## Representation Contract V2

Frozen open-development interface:

- file: `REPRESENTATION_CONTRACT_V2.md`
- commit: `9fa5484f162ba863f4522e4e5afb851c8fedabe2`

Fixed early top4, free XYZ motion authority, truth-conditioned retention and unrestricted factor fusion remain forbidden.

`K=16` is a demonstrated safety reference, **not** a permanent product constant.

## Adaptive Hypothesis Retention V1

Prereg:

- file: `ADAPTIVE_HYPOTHESIS_RETENTION_V1_PREREG.md`
- commit: `3953c7c34540e3fc37386fd8c11a39fad0505d52`

Canonical result:

`FAIL__NO_COMPACT_ADAPTIVE_CONTRACT_PASSES__KEEP_F16_REFERENCE`

### Arm summary

| Arm | Pooled 2x | Worst family | 8-family >=.90 | Gap | Mean K | Median K | K16 frac | Coverage | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| F4 | .93902 | .84746 | 6/8 | 15.25pp | 4.00 | 4 | 0 | FAIL | PASS |
| F8 | .95528 | .88136 | 6/8 | 11.86pp | 8.00 | 8 | 0 | FAIL | PASS |
| F16 | **.98171** | **.94915** | **8/8** | **5.08pp** | 16.00 | 16 | 1.00 | **PASS** | FAIL |
| A1 descriptor adaptive | .96951 | .89831 | 7/8 | 10.17pp | 9.38 | 8 | .331 | FAIL | **PASS** |
| A2 descriptor + geometry | .97358 | .91525 | **8/8** | 8.47pp | 11.93 | 16 | .566 | FAIL | FAIL |

### Interpretation

- `K=8` is not a safe compact replacement for F16.
- descriptor margin/entropy alone is too weak on hard-tail family `11032`;
- raw multiview reprojection/pair-support escalation improves the tail but is not selective enough: it still misses the `.94` worst-family gate while expanding many easy carriers to K16;
- Adaptive V1 therefore fails **compression**, not representation sufficiency.

Current safety contract is fixed F16 while a better observation-only uncertainty trigger is identified.

## Current required experiment — Expansion-Need Separability Audit

Before factor-head training, define evaluator-only:

```text
NEEDS_EXPANSION = F8 misses primary-2x AND F16 contains primary-2x
```

Truth is allowed only to create this truth-open diagnostic label. Candidate trigger features remain observation-only.

Audit strict family-disjoint separability of at least:

```text
margin4
margin8
entropy16
F4/F8 reprojection residual
relative reprojection improvement F4 -> F8
selected endpoint shift F4 -> F8
normalized pair/support count
H width / multimodality
cross-view disagreement
```

The goal is a self-normalized uncertainty signal that detects top16-rescuable carriers without expanding most easy carriers. Then preregister **Adaptive Hypothesis Retention V2** before testing its coverage.

## Training authorization

The representation sufficiency decision itself is **not revoked**.

However, Adaptive V1 prereg required a bounded H contract to be frozen before the small factor heads are executed. Because no compact adaptive contract passed:

- **small `p_active + log_amp` head execution is PAUSED** pending Adaptive H V2/freeze;
- `dir` remains separate;
- **large/end-to-end training remains FORBIDDEN**;
- no Stage-B requalification is allowed.

## Canonical current artifacts

- `REPRESENTATION_CONTRACT_V2.md`
- `ADAPTIVE_HYPOTHESIS_RETENTION_V1_PREREG.md`
- `ADAPTIVE_HYPOTHESIS_RETENTION_V1_RESULT.json`
- `ADAPTIVE_HYPOTHESIS_RETENTION_V1_REPORT.md`
- `adaptive_hypothesis_retention_v1.py`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_CANONICAL_DECISION.md`
- `POST_STAGEB_REPRESENTATION_SUFFICIENCY_BATTERY_V1_CANONICAL_DECISION.json`

Current frontier: **diagnose expansion need; do not train through an unfrozen candidate-retention contract.**
