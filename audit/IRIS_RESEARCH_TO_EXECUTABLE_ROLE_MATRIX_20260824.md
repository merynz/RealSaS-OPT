# IRIS Research → Executable Role Matrix — 2026-08-24

Status: `AUDIT_AUTHORITY__EXACT_REPRESENTATION_STUDY_NEXT__TRAINING_STILL_FORBIDDEN`

This document prevents paper/experiment conclusions from drifting when transferred into executable code. It does not authorize an optimizer step.

## Evidence authorities reconciled

1. 2026-08-21 Descriptor Upgrade Transfer Report — objective alignment, coarse/fine separation, local refinement, calibration.
2. `experiments/post_corpus_audit/POST_CORPUS_80_CHECK_CLOSURE_MATRIX_V1.md` — post-study experiment verdicts and original apparatus ladder.
3. `experiments/post_corpus_audit/POST_CORPUS_STAGE_A_CLOSURE_DECISION_V2.md` + Stage-B/B3/B4 evidence — later localization of apparatus risks.
4. `experiments/post_corpus_audit/IRIS_USABLE_CONTROLLED_CORPUS_FREEZE_V1.md` — latest operational authority for the immediate 3930 controlled path.
5. `experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_STUDY_V1_PREREG.md` — immediate exact/noise representation gate before a new learner.
6. `experiments/m4_identity_audit/VERY_IMPORTANT_AUDIT_M4_IDENTITY_AMBIGUITY_EQUIVALENT_SUBSTRATE_20260823.md` — target reframing, set-valued ambiguity, E0–E5 downstream sufficiency.
7. `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md` and `canonical/PRODUCT_CONTRACT_V1.md` — product/problem boundaries.
8. `experiments/g0_g1_single_pose_geometry/FRONTEND_NATIVE_1024_CORPUS_CONTRACT_V1.md` — native-resolution/data authority.

## Frozen transfer table

| Research statement | Evidence status | V2 executable binding | Forbidden drift |
|---|---|---|---|
| Observation-level correspondence must be supervised before view pooling | D1 `PASS_DIRECTIONAL` | `losses.py::observation_supcon`, pairwise bidirectional multi-positive CE | returning to one pooled descriptor/identity vector as primary supervision |
| Multiview observation sampling must not bias a view prefix | audit correction | deterministic uniform thinning across the full view-major observation list | `idx[:max_obs]` or equivalent prefix truncation |
| Coarse persistence is a high-recall basin mechanism | supported | `model.py::Z_coarse @ R/8`; `matcher.py` global corridor search | using fine descriptor as global admission authority |
| Fine descriptor is local precision evidence | D2 `SUPPORTED_LOCAL`; global ranking `FAIL_CLOSED` | `model.py::Z_fine @ R/2`; `losses.py::fine_local_loss`; `matcher.py` searches Zf only inside admitted basins | global Zf RRF/top1/cross-basin ranking |
| Reciprocal/cycle contains causal information | G2 `PASS_MECHANISM` | ambiguity-aware reciprocal training + `qualification.py` deterministic reciprocal/cycle support | treating one arbitrary diagonal identity as the only legal cycle return; hard-filtering candidates without a new gate |
| Exact hidden owner identity is not the IRIS target | M4 interpretation superseded | P/N/support/persistence/provenance + set-valued hypotheses | authored joint/owner/parent/skin targets entering IRIS |
| Ambiguity may terminate set-valued | required policy | matcher outputs top-k hypotheses; no singleton authority | premature top1/singleton before calibration |
| P is core geometric substrate | `PASS` scoped; exact new-corpus ceiling still pending | direct P field at R/2 + cross-view consistency | replacing P with opaque identity-only representation before representation study |
| Direct N remains materially useful | `PASS_CURRENT`; exact new-corpus ablation still pending | direct unit N field at R/2 | deleting N before exact representation/downstream ablation closes |
| U for geometry and match ambiguity are different quantities | audit correction | `U_geo` is trained only from detached P error | reusing U_geo as singleton/match confidence |
| Native 1024 is primary observation authority | canonical corpus contract | architecture is resolution-parametric; 1024→Zc128/Zf/P/N/U512 | fixed 128 candidate lattice presented as native-1024 evidence |
| Fine errors must be measured in explicit native pixels | transfer + post-corpus correction | `metrics.py::native_pixel_error`; exact coarse-cell containment | normalized magic tolerances such as old `TRUTH_TOL=.018` |
| Raster is a surface/visibility witness, not a reason to destroy subpixel truth | apparatus discipline | accepted `track_xy` is exact continuous projection | re-quantizing accepted tracks to nearest raster pixel center |
| Teacher correspondence may supervise but may not enumerate inference candidates | corpus contract | truth cache carries physical loci; matcher candidate universe is alpha-supported field cells | GT track IDs/track nodes becoming candidate universe |
| Geometry-only IRIS source must be physical | corpus firewall | `stage_source.py` emits only `vertices/faces` geometry plus legal observation authority | relying on learner code to ignore rig fields still physically present |
| Resume/reuse must preserve exact experimental semantics | persistence policy | stage/cache source, settings and builder-semantic fingerprints | silently reusing stale stage/cache after source/settings/semantic changes |
| Checkpoint selection must be observable, not loss-only | transfer + audit | `evaluate_v2.py` produces panel but intentionally defines no key yet | silently selecting by `val['total']` before preregistration |

## Chronological apparatus authority — important correction

The early 80-check matrix listed Gates 1–5 as broad pre-optimizer blockers. Later same-day evidence narrowed that for the **controlled geometry research path**:

### Gate 1 — closed for controlled path

Stage-A found exactly one all-eight-view blank asset. It is excluded.

### Gate 2 — bounded for controlled path

Exhaustive `.blend` consequence work localized the material repair set. Stage-B6 froze corrected artifacts but did not publish/rerender them. The later controlled-corpus freeze deliberately excludes:

- 56 repair-pending assets whose current A render does not match the frozen repaired geometry;
- 6 active non-Basis shape-key assets.

So Gate 2 remains a production-publication task, but it does not invalidate the frozen **3930 clean controlled assets**.

### Gate 3 — product-domain open, controlled geometry deferred by decision

A is explicitly a geometry-isolation observation arm. Appearance recovery/dependency closure is still required before final/natural/product-domain IRIS qualification. However `IRIS_USABLE_CONTROLLED_CORPUS_FREEZE_V1.md` deliberately returns the immediate critical path to the strongest controlled geometry conditions and schedules appearance robustness after controlled success.

Therefore a future mini may be a **geometry-control learner experiment only**. It may not be rebranded as final/product IRIS.

### Gate 4 — immediate blocker

`REPRESENTATION_AUTHORITY_STUDY_V1_PREREG.md` is the next scientific executable. It must determine the exact/noise envelope for P, P+N and any justified uncertainty-aware relational address before V2 learner targets/checkpoint semantics are frozen.

### Gate 5 — conditional / impossibility boundary

SOI-2 is mandatory if the legal representation remains insufficient and is always mandatory before a product-level information-impossibility claim. It is not automatically required before a controlled learner run if the exact representation study supports the target.

## D3 boundary

D3 is **not implemented and not currently authorized**. It may be opened only if all of the following are shown prospectively:

1. coarse truth containment is high enough that a local-only refiner has a valid basin;
2. oracle-basin Zf/local metrics demonstrate a reproducible residual hard tail;
3. the residual is local/query-conditioned rather than a missing global candidate;
4. D3 is measured as a local intervention, never as a silent replacement of Zc containment.

## What a future V2 mini may claim

Only after the exact representation study and a new prereg:

`A_geometry_control x 8 -> P/N/U_geo/Zc/Zf -> reciprocal/cycle support -> set-valued correspondence evidence`.

It may test whether the repaired architecture can learn the intended observable quantities on open FIT/TUNE data. It may **not** be called final/product-domain IRIS while appearance Gate 3 and E0–E5 remain open.

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

Every future architecture edit must identify the row above or a new preregistered result that authorizes it. If no authority exists, the change is exploratory and may not silently enter the canonical executable.
