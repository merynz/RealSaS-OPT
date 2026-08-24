# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__CI202_RAW_XYZ_DIAGNOSTIC_COMPLETE__P_V3_G0_PASS__CI237_G1_V1_APPARATUS_FALSE_REJECT__CI244_G1_V2_NEXT__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

> **Execution-source rule:** corrected P Formulation V3 G1 authority is bundle V2 from code-bearing head `985997ba87faa92cc004ad9ab70efeacc400886f`, GitHub Actions run #244 / run ID `32746353085`, artifact ID `9527242584`, ZIP SHA-256 `e04bc7c9638038688304b558cb816f8bccaf5e253bc2195977c0c2328f693667`. Later docs commits do not change that authority.

## Single continuation authority

The active implementation remains `experiments/iris_single_pose_v2/`.

Current P-V3 authority:

- `P_FORMULATION_V3_CLOSURE_PREREG_20260824.md`
- `P_FORMULATION_V3_PANEL_V1.json`
- `p_formulation_corpus_audit_v1.py`
- `p_formulation_master_geometry_firewall_preflight_v1.py`
- `P_FORMULATION_V3_CI237_APPARATUS_SUPERSESSION_20260824.md`
- CI244 immutable closure bundle V2.

Drive mirror:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_P_FORMULATION_V3_CLOSURE_BUNDLE_CI244_V2.zip`

Canonical corrected G1 notebook:

`RealSaS_IRIS_P_Formulation_V3_Closure_CI244_V2.ipynb`

Notebook SHA-256:

`f05b6a420045d9184880915d7454b852250e1ad8c6920bba44025f6f81402b3e`

Persistent corrected result destination:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_P_FORMULATION_V3_CLOSURE_CI244_V2_RESULT`

## Representation gate remains CLOSED/PASS

CI104 label remains `P_GEOMETRY_SUFFICIENT`. P ontology is unchanged: canonical/object-frame physical surface position of the observed physical locus.

CI202 is a completed historical learner diagnostic under the old free-XYZ P extractor. It is not checkpoint-compatible with P-V3.

## P Formulation V3

Known orthographic camera gives two P coordinates analytically:

```text
P = 0.54*gx*right(theta)
  - 0.54*gy*up
  + depth*forward(theta)
```

Only `depth = P·forward(theta)` is learned.

The depth head receives explicit `(gx,gy,sin(yaw),cos(yaw))`. P supervision is scalar depth SmoothL1; reconstructed 3D Euclidean P remains evaluation/U_geo authority. N/Zc/Zf authority roles are unchanged.

## G0 — CLOSED/PASS

Synthetic/executable P-V3 closure passes at 256/512/1024, including native1024 -> 512x512 P field, analytic screen-plane invariants, finite backward, nonzero depth-head gradient, AMP loss boundary, matcher role separation and evaluator/cache regressions.

## CI237 G1 V1 — preserved apparatus false reject

CI237 result root is preserved:

`runs/IRIS_SINGLE_POSE_V2_P_FORMULATION_V3_CLOSURE_CI237_RESULT`

It persisted `P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`, but **no scientific geometry metric opened**. All 16 assets were rejected solely because V1 required the master `primary_geometry.npz` field set to equal exactly `{vertices,faces}`.

That condition contradicted the frozen preregistration. The master source is intentionally a superset; the G1 consumer is required to **read only** `vertices` and `faces`, not require the source container to contain only those names.

CI237 V1 aggregate gauge/projection/reconstruction summaries were therefore `n=0`. It does not support a geometry/formulation failure claim.

Canonical classification:

`APPARATUS_FALSE_REJECT__MASTER_NPZ_SUPERSET_MISTAKEN_FOR_CONSUMPTION_LEAK`

## CI244 correction — CLOSED at apparatus level

V2 semantics:

- master NPZ may be a superset;
- only `vertices` and `faces` values are loaded;
- hidden field names may be recorded, hidden values are not consumed;
- missing `vertices` or `faces` remains fatal;
- object-dtype hidden-field tripwire proves accidental hidden-value consumption would fail under `allow_pickle=False`.

CI244 passed:

- all prior P-V3/model/cache/matcher/evaluator regressions;
- master geometry superset firewall regression;
- hidden object-dtype tripwire;
- panel firewall;
- isolated uploadable V2 bundle dependency/content replay.

The independently downloaded V2 artifact also matched GitHub SHA and passed internal SHA256SUMS + isolated preflights.

## CURRENT GATE — corrected G1 real-corpus geometry closure

Same frozen 16 FIT-only sentinel assets and same preregistered thresholds as CI237:

- 8 FIT_SELECT sentinels;
- 8 FIT_TRAIN sentinels;
- geometry + camera + raster only;
- no RGB;
- no TUNE;
- no CAL/DEV/EXTERNAL;
- optimizer steps 0.

Allowed labels remain exactly:

- `P_V3_FORMULATION_GEOMETRY_CLOSED`
- `P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`

CI237 result must not be overwritten. Corrected V2 uses the distinct CI244 result root above.

## Training policy

**TRAINING FORBIDDEN.**

Do not reuse the CI202 mini prereg or checkpoint. If corrected G1 PASSes, next gate is a separately frozen FIT-only depth/P overfit diagnostic. TUNE stays closed during reformulation debugging.

## Normal policy

`NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED`.

`geom_n` remains observation-local orientation supervision/diagnostic only. N is forbidden from correspondence admission/ranking and checkpoint selection.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
