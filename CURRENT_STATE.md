# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__CI104_INTERPRETED__NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED__MINI_PREREG_PREP_NEXT__TRAINING_STILL_FORBIDDEN`

## Single continuation authority

The sole active candidate implementation is `experiments/iris_single_pose_v2/`. Controlled V1/M256 is historical/no-run.

The frozen 256-asset CI104 optimizer-zero R0-R3 Representation Authority measurement has completed and has been canonically interpreted.

Canonical interpretation:

`experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_CANONICAL_INTERPRETATION_CI104_20260824.md`

Execution freeze:

`experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_EXECUTION_FREEZE_CI104_20260824.md`

## Canonical representation result

### P authority — CLOSED PASS

R0 exact P on 12,288 confirmatory queries:

- top1 = `1.0`
- top4 = `1.0`
- top8 = `1.0`
- reciprocal = `1.0`
- cycle = `1.0`
- pooled physical-error median/p90/p95/max = `0 / 0 / 0 / 0`
- family top1/top4/top8 p5/p10/median/p90/p95/max all `1.0` across 256 assets

Canonical label:

`P_GEOMETRY_SUFFICIENT`

Therefore R4 is not opened, SOI-2 is not opened, and no information-limit branch is justified by this gate.

### P robustness — R2 diagnostic

Pooled top8 under controlled isotropic P perturbation:

- sigma 0.0000 -> `1.0000`
- sigma 0.0005 -> `1.0000`
- sigma 0.0010 -> `1.0000`
- sigma 0.0025 -> `0.9976`
- sigma 0.0050 -> `0.9390`
- sigma 0.0100 -> `0.6848`

This is a descriptive robustness curve, not a post-hoc learner promotion threshold.

### N correspondence authority — NOT ESTABLISHED

R1/R3 must not be interpreted as an exact-N representation failure/sufficiency result.

Implementation inspection shows that `track_n_view` is reconstructed per view from the nearby raster visibility/surface witness `row[ok]`, while `track_p` is the exact persistent common-frame locus. The frozen P+N score `dP + 0.05*(1-cos N)` can therefore let view-specific witness-normal differences perturb a correspondence that exact P alone resolves perfectly.

Observed R1 nominal P+N-exact top1/top4/top8 = `0.96696 / 0.97884 / 0.98543`; reciprocal `0.95980`; cycle `0.94784`.

Canonical label for this sub-claim:

`NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED`

This does not block V2 because the active matcher already uses `Z_coarse + P` for global basin admission and `Z_fine` for local refinement; N is geometry/orientation evidence for the downstream substrate, not global correspondence admission authority.

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
- persisted real result: `RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_REPRESENTATION_AUTHORITY_V2_CI104_RESULT`
- frozen panel asset-ID digest: `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`
- optimizer steps: `0`
- sealed splits opened: `false`

## Apparatus-failure history

- CI69: RGB-heavy staging was broader than this gate required; interrupted before result.
- CI86: representation-only stage reached `256/256`; execution ZIP omitted `coords.py`; failed before cache.
- CI96: representation-only stage reached `256/256`; partial visibility exposed `track_err[:,v] = err[ok]` broadcast bug; failed before cache completion/audit/study.
- CI104: fixed partial-visibility indexing, added a visible+hidden regression, passed all preflights and isolated artifact tests, and produced the first valid frozen R0-R3 measurement.

None of CI69/CI86/CI96 produced a scientific R0-R3 result.

## Representation-only boundary

Stage profile: `representation_authority_geometry_only`.

R0-R3 stages/consumes only sanitized `vertices/faces`, native-1024 `raster_authority.npz`, and `camera.json`. No RGB is staged or consumed. Hidden rig/mechanics fields are excluded from the legal representation fingerprint.

## Controlled corpus / panel

Controlled corpus: 3930 assets.

- FIT 2935 OPEN
- TUNE 313 OPEN
- CAL 246 SEALED
- DEV 270 SEALED
- EXTERNAL_HOLDOUT 166 SEALED

Frozen confirmatory panel:

- 256 OPEN assets
- 230 FIT / 26 TUNE
- ordered asset-ID-list SHA-256 `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`

## NEXT EXECUTABLE STEP

Representation feature search is closed for the current P target. Do **not** open R4 or SOI-2.

Before any optimizer run:

1. run production-width GPU memory/throughput preflight with optimizer=0;
2. freeze mini FIT/TUNE membership;
3. freeze observable checkpoint-selection key + evaluator;
4. freeze mini style/appearance policy;
5. write the mini learner prereg;
6. only then open one learner optimizer runner.

No CAL, DEV, or EXTERNAL_HOLDOUT split may be opened during mini preparation.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
