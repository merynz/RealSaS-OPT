# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__CI202_RAW_XYZ_DIAGNOSTIC_COMPLETE__P_FORMULATION_V3_IMPLEMENTED__OPTIMIZER_ZERO_CLOSURE_REQUIRED__TRAINING_FORBIDDEN`

## Single continuation authority

The sole active candidate implementation remains `experiments/iris_single_pose_v2/`. Controlled V1/M256 is historical/no-run. CI202 remains a completed historical learner diagnostic under its tested raw free-XYZ P architecture; its checkpoint is **not compatible with and not authoritative for P Formulation V3**.

Current executable scientific gate:

- `experiments/iris_single_pose_v2/P_FORMULATION_V3_CLOSURE_PREREG_20260824.md`
- `experiments/iris_single_pose_v2/P_FORMULATION_V3_PANEL_V1.json`
- `experiments/iris_single_pose_v2/p_formulation_corpus_audit_v1.py`

No optimizer is authorized until P-V3 synthetic CI and real-corpus optimizer-zero closure pass.

## Representation gate remains CLOSED/PASS

CI104 canonical label remains `P_GEOMETRY_SUFFICIENT`. Exact P on the frozen representation panel was sufficient for physical same-locus correspondence. The CI202 learner failure does **not** reopen R4/SOI-2 or information existence.

The current correction is an **extractor parameterization correction**, not a change to P's ontology or to the representation result.

## CI202 completed diagnostic — historical raw-XYZ learner

CI202/V4 completed cleanly with FIT_TRAIN128 / FIT_SELECT32 / TUNE_FINAL26, selected epoch16 / optimizer step512, no TUNE checkpoint-selection leakage and no sealed split opening.

Result: `LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT`.

FIT_SELECT selected-epoch metrics included:

- P p95 0.344555 vs frozen <=0.005 target;
- Zc@8 0.789931;
- P-basin top8 0.649306;
- oracle Zf p95 35.947 native px;
- end-to-end top8 <=16px 0.827257.

TUNE P p95 0.413841 and Zc@8 0.762019 showed no primary generalization collapse; the dominant failure was absolute extraction precision.

Canonical historical interpretation remains:

`experiments/iris_single_pose_v2/MINI_EXTRACTABILITY_CANONICAL_INTERPRETATION_CI202_20260824.md`

## Post-CI202 P audit

The full P geometry chain was re-audited rather than attributing the failure to optimization budget.

Findings:

1. The 128x128 P field used by the R=256 mini is **not itself an adequate explanation** for P p95 ~0.34. A real native-1024 oracle check showed an exact-style 128 bilinear P field can represent the current <=0.005 p95 target on the sampled asset (p95 ~0.0017).
2. Real `triangle+barycentric -> canonical P -> camera projection` checks support the existing barycentric convention; p95 raster projection residual was near numerical zero on the inspected asset. Sparse outliers were far too rare to explain CI202.
3. Native RGB/raster alignment checked on a real view; alpha/raster foreground agreed. Staging uses validated native1024 authority and the intended canonical512->256 derivative for the mini.
4. Sampled master assets were centered/unit-scaled, but the executable learner cache audit previously did not enforce the canonical P gauge/envelope. This audit omission is now closed in code.
5. The main formulation defect was the free 3-channel absolute XYZ P head. Under the known orthographic camera, two coordinates are already analytic from observation position; only view-depth must be learned.

## P Formulation V3 — implemented, not yet promoted

Controlled camera contract:

```text
right(theta)   = (cos theta, -sin theta, 0)
forward(theta) = (sin theta,  cos theta, 0)
up             = (0,0,1)
half_extent    = 0.54
```

For normalized image/grid coordinate `(gx,gy)`:

```text
P = 0.54*gx*right(theta)
  - 0.54*gy*up
  + depth*forward(theta)
```

Implementation changes:

- free `Conv2d(...,3)` P head removed;
- one scalar camera-forward depth head at R/2;
- depth head receives explicit normalized `(gx,gy)` and `(sin yaw,cos yaw)` conditioning;
- public 3D P reconstructed deterministically from camera geometry + depth;
- no hard depth clip;
- P learned loss is scalar depth SmoothL1, not an average across three XYZ coordinates;
- Euclidean reconstructed P remains evaluation and U_geo authority;
- N/Zc/Zf roles are unchanged.

## New executable gauge / geometry firewall

`audit_staged_cache_v2.py` now fails closed on:

- wrong authority resolution;
- non-finite or materially noncanonical geometry;
- max canonical |coordinate| >0.55;
- bbox center infinity norm >0.01;
- largest bbox extent outside [0.98,1.02];
- camera contract / half-extent / yaw / right-up-forward basis mismatch;
- raster authority not 1024;
- `geom_p` or `track_p` outside canonical envelope;
- `geom_p -> camera projection` p95 mismatch;
- excessive projection-outlier fraction;
- track continuous-projection or yaw drift.

## CURRENT GATE — P Formulation V3 optimizer-zero closure

G0 — synthetic executable closure:

- camera basis and coordinate formula;
- R=256/512/1024 P formulation;
- R/2 field shapes;
- arbitrary bilinear screen-plane preservation;
- full finite loss/backward;
- finite nonzero depth-head gradient;
- existing matcher role separation and AMP numerics.

G1 — real-corpus geometry closure:

- frozen **16 FIT-only** sentinel assets;
- geometry + camera + raster only;
- no RGB required;
- no TUNE;
- no CAL/DEV/EXTERNAL;
- optimizer steps 0;
- canonical gauge + camera + raster/P alignment + exact-depth reconstruction gates frozen before result.

PASS label: `P_V3_FORMULATION_GEOMETRY_CLOSED`.

FAIL label: `P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`.

## Training policy

**TRAINING FORBIDDEN.**

Do not reuse the old CI202 Mini V1 prereg with P-V3, and do not release a new training bundle until G0+G1 close. The current GitHub workflow must release only the optimizer-zero P-V3 closure artifact from the latest architecture; old mini V4 remains historical and immutable.

If G0+G1 PASS, the next prereg must be a FIT-only depth/P overfit/optimization diagnostic. TUNE must not be reopened merely to debug the reformulation.

## Normal policy remains frozen

`NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED`.

`geom_n` remains legal observation-local orientation supervision and angular diagnostic only. Production correspondence remains Z_coarse + P global admission and Z_fine local refinement. N stays forbidden from correspondence admission/ranking and checkpoint selection.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
