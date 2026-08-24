# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__CI202_RAW_XYZ_DIAGNOSTIC_COMPLETE__P_FORMULATION_V3_G0_CI237_PASS__REAL_G1_NEXT__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

> **Execution-source rule:** P Formulation V3 closure bundle V1 is bound to code-bearing head `cad879f591be245632e8c29eaa0c8e107fa87fb4`, GitHub Actions run #237 / run ID `32743886600`, artifact ID `9526277144`, ZIP SHA-256 `0b336d4aeae2cea21802439edfaa0ea2d0b0d7f037de4944b1373391fea07371`. Later docs commits do not change that execution authority.

## Single continuation authority

The sole active candidate implementation remains `experiments/iris_single_pose_v2/`. Controlled V1/M256 is historical/no-run. CI202 remains a completed historical learner diagnostic under its tested raw free-XYZ P architecture; its checkpoint is **not compatible with and not authoritative for P Formulation V3**.

Current scientific authority:

- `experiments/iris_single_pose_v2/P_FORMULATION_V3_CLOSURE_PREREG_20260824.md`
- `experiments/iris_single_pose_v2/P_FORMULATION_V3_PANEL_V1.json`
- `experiments/iris_single_pose_v2/p_formulation_corpus_audit_v1.py`
- CI237 immutable closure bundle V1.

Drive mirror:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_P_FORMULATION_V3_CLOSURE_BUNDLE_CI237.zip`

No optimizer is authorized until real-corpus G1 closes.

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

## P Formulation V3 — implemented

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

## G0 — CLOSED/PASS under CI237

CI237 passed on exact source head `cad879...`:

- exact committed-source compilation;
- camera basis and P formula;
- R=256/512/1024 P formulation;
- P field at R/2 (native1024 -> 512x512);
- arbitrary bilinear screen-plane preservation;
- finite full loss/backward;
- finite nonzero P depth-head gradient;
- learner cache and partial-visibility regressions;
- historical TUNE-order/firewall regression;
- AMP precision regression;
- evaluator and representation regressions;
- P-V3 panel firewall;
- isolated uploadable closure bundle dependency/content replay.

The independently downloaded artifact was replayed outside the repo checkout; its SHA and all internal SHA256SUMS matched and isolated `preflight.py` passed.

## CURRENT GATE — G1 real-corpus geometry closure

Frozen **16 FIT-only** sentinel assets:

- 8 frozen FIT_SELECT sentinels;
- 8 frozen FIT_TRAIN sentinels;
- geometry + camera + raster only;
- no RGB required;
- no TUNE;
- no CAL/DEV/EXTERNAL;
- optimizer steps 0.

Frozen gates include canonical gauge, camera basis/half extent, raster1024 authority, raster-P projection agreement, and exact-depth V3 reconstruction.

PASS label: `P_V3_FORMULATION_GEOMETRY_CLOSED`.

FAIL label: `P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`.

Canonical Colab runner:

`RealSaS_IRIS_P_Formulation_V3_Closure_CI237.ipynb`

Notebook SHA-256:

`20865b953b5af82a0db3c6229a89e1b5dde0f0f932876f2180278cbed1498ef0`

Persistent result destination:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_P_FORMULATION_V3_CLOSURE_CI237_RESULT`

## Training policy

**TRAINING FORBIDDEN.**

Do not reuse the old CI202 Mini V1 prereg with P-V3. The latest workflow no longer builds a training bundle; it releases only the optimizer-zero P-V3 closure artifact.

If G1 PASS, the next prereg must be a FIT-only depth/P overfit/optimization diagnostic. TUNE must not be reopened merely to debug the reformulation.

## Normal policy remains frozen

`NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED`.

`geom_n` remains legal observation-local orientation supervision and angular diagnostic only. Production correspondence remains Z_coarse + P global admission and Z_fine local refinement. N stays forbidden from correspondence admission/ranking and checkpoint selection.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
