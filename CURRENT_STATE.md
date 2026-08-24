# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `R0_R3_REPRESENTATION_ONLY_CI104_FROZEN__REAL_RUN_NEXT__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

## Single continuation authority

The sole active candidate implementation is `experiments/iris_single_pose_v2/`. Controlled V1/M256 is historical/no-run. No optimizer or learner training is authorized.

Immediate scientific executable: the frozen 256-asset optimizer-zero R0-R3 Representation Authority measurement, followed by a separate canonical interpretation.

Current freeze:

`experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_EXECUTION_FREEZE_CI104_20260824.md`

## Current exact execution authority — CI104

- exact code-bearing head: `7d99eef46d2c071eca1d883e0c916bf4adecda35`
- GitHub Actions: `IRIS V2 Preflight #104`
- run ID: `32679394221`
- result: `SUCCESS`
- artifact ID: `9503678775`
- artifact: `iris-v2-r0-r3-execution-bundle-v3`
- ZIP SHA-256: `bc1e733049fafcd01f933d26c309f7fb5a9cff0d2abd04124219c4ea9ba39473`
- Drive mirror: `RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_R0_R3_REPRESENTATION_ONLY_BUNDLE_CI104.zip`
- Drive file ID: `1lW7pmAjjGO-M1OWvkJn9vOphqhv29QNP`
- launcher: `RealSaS_IRIS_V2_R0_R3_Representation_Only_CI104.ipynb`
- launcher SHA-256: `c7cd5cf48b2273df8dba82b4dc6717e55555c6c8942927581670e53cbd033680`

## Apparatus-failure history — no scientific result opened

- CI69: RGB-heavy staging was broader than this gate required; interrupted before result.
- CI86: representation-only stage reached `256/256`; execution ZIP omitted `coords.py`; failed before cache.
- CI96: representation-only stage reached `256/256`; partial visibility exposed `track_err[:,v] = err[ok]` broadcast bug; failed before cache completion/audit/study.
- CI104: fixes partial-visibility indexing and adds a regression that exercises visible+hidden tracks in the same view. All preflights and isolated artifact tests pass.

None of CI69/CI86/CI96 produced an R0-R3 scientific result.

## CI104 regression closure

The cache builder now preserves full track-axis semantics:

`track_err[ok, v] = err[ok]`

with hidden/non-witnessed entries left at `+inf`.

CI104 requires, both in repo CI and again inside the isolated uploadable bundle:

- genuine partial visibility;
- `track_visible.shape == track_surface_error.shape`;
- visible witness errors finite;
- hidden witness errors `+inf`;
- `coords.py` included in cache-builder semantic fingerprint;
- complete local dependency closure;
- isolated compile/import;
- representation-only synthetic stage/cache/audit smoke;
- exact SHA256SUMS and bundle file-set verification.

## Representation-only boundary

Stage profile: `representation_authority_geometry_only`.

R0-R3 stages/consumes only:

- sanitized `vertices/faces`;
- native-1024 `raster_authority.npz`;
- `camera.json`.

No `cel_clean` / `ink_cel` RGB is staged, decoded, hashed, or consumed. Hidden rig/mechanics fields are excluded from the legal representation fingerprint.

## Controlled corpus / panel

Controlled corpus: 3930 assets.

- FIT 2935 OPEN
- TUNE 313 OPEN
- CAL 246 SEALED
- DEV 270 SEALED
- EXTERNAL_HOLDOUT 166 SEALED

Frozen source authorities:

- split freeze SHA-256 `9e766ac61126c9b4787eef24146e36aac40cbeac166d67ba898f8b79133e9d66`
- canonical selection SHA-256 `af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9`

Frozen confirmatory panel:

- 256 OPEN assets
- 230 FIT / 26 TUNE
- ordered asset-ID-list SHA-256 `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`
- no substitution after apparatus failure

## Frozen R0-R3 semantics

Unchanged across the apparatus fixes:

- SAME-locus tolerance `0.003` canonical units
- R0 exact P
- R1 exact P+N
- R2 P sigma `{0,.0005,.001,.0025,.005,.01}`
- R3 full 6x5 P/N grid with N `{0,5,10,20,40}` degrees
- fixed P+N score `dP + 0.05*(1-cos N)`
- deterministic Philox source/target perturbations
- physical set-containment top1/top4/top8
- reciprocal + three-view cycle
- family median/p90/p95 tails
- `nearest_non_equivalent_physical_gap` is descriptive geometry-only confusability, never semantic symmetry truth or a promotion threshold
- mean-only promotion forbidden

R4 remains undefined. If R0-R3 motivates it, freeze a separate R4 formulation prereg before any R4 result.

## NEXT EXECUTABLE STEP

Run `RealSaS_IRIS_V2_R0_R3_Representation_Only_CI104.ipynb` with **Run all**.

CPU is sufficient; GPU is not used.

If the same Colab runtime still contains the completed CI96 `256/256` local stage, CI104 may reuse it only after full local seed/panel/marker/file-set/file-hash/firewall/no-RGB verification. Otherwise it stages fresh.

A successful measurement may emit only:

`R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED`

with `optimizer_steps=0`, `training_authorized=false`, and sealed splits unopened.

After that: inspect compact handoff + full result + hard-tail and write one canonical R0-R3 interpretation. Only after representation closure may we freeze the mini learner experiment and consider optimizer use.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
