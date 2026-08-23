# IRIS Research → Executable Role Matrix — 2026-08-24

Status: `AUDIT_AUTHORITY__TRAINING_STILL_FORBIDDEN`

This document prevents paper/experiment conclusions from drifting when transferred into executable code. It does not authorize an optimizer step.

## Evidence authorities reconciled

1. `DESCRIPTOR_TRANSFER_REPORT_20260821.md` — correspondence architecture transfer: objective alignment, coarse/fine separation, local refinement, calibration.
2. `POST_CORPUS_80_CHECK_CLOSURE_MATRIX_V1.md` — post-study experiment verdicts and apparatus gates.
3. `VERY_IMPORTANT_AUDIT_M4_IDENTITY_AMBIGUITY_EQUIVALENT_SUBSTRATE_20260823.md` — target reframing, set-valued ambiguity, E0–E5 downstream sufficiency.
4. `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md` and `canonical/PRODUCT_CONTRACT_V1.md` — product/problem boundaries.
5. `experiments/g0_g1_single_pose_geometry/FRONTEND_NATIVE_1024_CORPUS_CONTRACT_V1.md` — native-resolution/data authority.

## Frozen transfer table

| Research statement | Evidence status | V2 executable binding | Forbidden drift |
|---|---|---|---|
| Observation-level correspondence must be supervised before view pooling | D1 `PASS_DIRECTIONAL` | `losses.py::observation_supcon`, pairwise bidirectional multi-positive CE | returning to one pooled descriptor/identity vector as primary supervision |
| Coarse persistence is a high-recall basin mechanism | supported | `model.py::Z_coarse @ R/8`; `matcher.py` global corridor search | using fine descriptor as global admission authority |
| Fine descriptor is local precision evidence | D2 `SUPPORTED_LOCAL`; global ranking `FAIL_CLOSED` | `model.py::Z_fine @ R/2`; `losses.py::fine_local_loss`; `matcher.py` searches Zf only inside admitted basins | global Zf RRF/top1/cross-basin ranking |
| Reciprocal/cycle contains causal information | G2 `PASS_MECHANISM` | weak ambiguity-aware reciprocal loss + deterministic qualification gate | treating one arbitrary diagonal identity as the only legal cycle return |
| Exact hidden owner identity is not the IRIS target | M4 interpretation superseded | P/N/support/persistence/provenance + set-valued hypotheses | authored joint/owner/parent/skin targets entering IRIS |
| Ambiguity may terminate set-valued | required policy | matcher outputs top-k hypotheses; no singleton authority | premature top1/singleton before calibration |
| P is core geometric substrate | `PASS` scoped | direct P field at R/2 + cross-view consistency | replacing P with opaque identity-only representation |
| Direct N remains materially useful | `PASS_CURRENT` | direct unit N field at R/2 | deleting N before the exact representation/substrate ablation closes |
| U for geometry and match ambiguity are different quantities | audit correction | `U_geo` is trained only from detached P error | reusing U_geo as singleton/match confidence |
| Native 1024 is primary observation authority | canonical corpus contract | architecture is resolution-parametric; 1024→Zc128/Zf/P/N/U512 | fixed 128 candidate lattice presented as native-1024 evidence |
| Fine errors must be measured in explicit native pixels | transfer + post-corpus correction | `metrics.py::native_pixel_error`; exact coarse-cell containment | normalized magic tolerances such as old `TRUTH_TOL=.018` |
| Raster is a surface/visibility witness, not a reason to destroy subpixel truth | apparatus discipline | accepted `track_xy` is exact continuous projection | re-quantizing accepted tracks to nearest raster pixel center |
| Teacher correspondence may supervise but may not enumerate inference candidates | corpus contract | truth cache carries physical loci; matcher candidate universe is alpha-supported field cells | GT track IDs/track nodes becoming candidate universe |
| Geometry-only IRIS source must be physical | corpus firewall | `stage_source.py` emits only `vertices/faces` geometry plus legal observation authority | relying on learner code to ignore rig fields still physically present |
| Resume/reuse must preserve exact experimental semantics | persistence policy | stage/cache source and builder fingerprints | silently reusing stale stage/cache after source/settings/semantic changes |

## D3 boundary

D3 is **not implemented and not currently authorized**. It may be opened only if all of the following are shown prospectively:

1. coarse truth containment is high enough that a local-only refiner has a valid basin;
2. oracle-basin Zf/local metrics demonstrate a reproducible residual hard tail;
3. the residual is local/query-conditioned rather than a missing global candidate;
4. D3 is measured as a local intervention, never as a silent replacement of Zc containment.

## What the current V2 mini would be allowed to claim if later preregistered

Only a **controlled geometry-observation learner test**:

`A_geometry_control x 8 -> P/N/U_geo/Zc/Zf -> set-valued correspondence evidence`.

It may test whether the repaired architecture can learn the intended observable quantities on open FIT/TUNE data. It may **not** be called final/product-domain IRIS while appearance Gate 3 remains open.

## Apparatus blockers inherited from the 80-check matrix

Training remains blocked by the unresolved pre-optimizer gates, not merely by code readiness:

- Gate 1: exhaustive blank/tiny/useful-coverage census and quarantine policy;
- Gate 2: evaluated `.blend`/modifier parity, triangulation risk, authored/custom normal authority on the affected subset;
- Gate 3: source-appearance recoverability and decision/build of appearance-preserving observation arm B;
- Gate 4: exact representation authority comparison, including richer/set-valued alternatives;
- Gate 5: product-relevant exact information ceiling / SOI-2 before any impossibility claim.

A geometry-control mini may only be considered before complete product-domain closure if a **new prereg explicitly scopes it as apparatus/learner development and does not use it to authorize final IRIS or an information-limit claim**. No such prereg exists yet.

## E0–E5 downstream boundary

Even a successful V2 frontend is not product closure. The M4 audit keeps the decisive downstream chain open:

- E0 exact observable SurfaceBuilder ceiling;
- E1 Geppetto exact-substrate sufficiency and ablations;
- E2 extractability from legal A×8 observations;
- E3 exact-vs-predicted substrate consequence gap;
- E4 ambiguity stress: forced top1 vs abstain vs calibrated set-valued;
- E5 functionally equivalent 2.5D rigging-substrate qualification.

The target remains a clean, editable, functionally equivalent rigging substrate, not exact reconstruction of one source author's hidden rig.

## Audit rule

Every future architecture edit must identify the row above that authorizes it. If no row or new preregistered evidence authorizes the edit, the change is exploratory and may not silently enter the canonical executable.
