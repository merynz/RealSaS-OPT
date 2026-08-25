# Representation Authority Execution Freeze — CI86 — 2026-08-24

Status: `FROZEN_PRE_RESULT__REPRESENTATION_ONLY__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

This authority supersedes CI69 **only at the apparatus/data-staging layer**. No R0-R3 scientific definition, panel membership, score, noise arm, SAME tolerance, query policy, candidate universe, or interpretation threshold changed.

## Why CI69 was reopened

CI69 staged and hashed native RGB images even though the exact R0-R3 Representation Authority study consumes only geometry/raster/camera-derived P/N/visibility/correspondence truth. The real run was interrupted during staging before cache/audit/study result generation. No R0-R3 scientific result was opened.

CI86 removes that unnecessary dependency rather than carrying it forward as hidden apparatus cost/risk.

## Exact code authority

- branch: `audit/iris-architecture-discipline-20260824`
- exact code-bearing head: `e32089498054948aa53b74e728d6ad8d7946693f`
- workflow: `IRIS V2 Preflight`
- run: `#86`
- run ID: `32675574055`
- conclusion: `SUCCESS`
- artifact ID: `9502554757`
- artifact name: `iris-v2-r0-r3-execution-bundle-v2`
- artifact ZIP SHA-256: `aa6bba667d1b53e5634b5ed9e6274cceca4437130be9d00ef0f72d3f970470df`

Drive mirror:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_R0_R3_REPRESENTATION_ONLY_BUNDLE_CI86.zip`

Drive file ID remains:

`1lW7pmAjjGO-M1OWvkJn9vOphqhv29QNP`

## Representation-only stage boundary

Stage profile:

`representation_authority_geometry_only`

Admitted source evidence per selected asset:

- `primary_geometry.npz` -> only canonicalized `vertices` and `faces` arrays survive the physical firewall;
- `renders/V0..V7/raster_authority.npz` exact bytes;
- `renders/V0..V7/camera.json` exact bytes.

Explicitly excluded from this gate:

- `cel_clean` RGB;
- `ink_cel` RGB;
- 512/256 RGB derivatives;
- authored rig/mechanics fields in `primary_geometry.npz`;
- any learner/model output.

RGB is still required later for the actual IRIS learner. It is excluded here because R0-R3 is an exact legal representation-ceiling measurement, not a learner run.

## Source fingerprint discipline

The representation-stage fingerprint binds exactly the legal evidence used by this gate:

- canonical `vertices` array content;
- canonical `faces` array content;
- raster-authority bytes;
- camera bytes.

RGB mutation and hidden rig/mechanics mutation must not change the fingerprint. Raster/camera/legal geometry mutation must change it.

## CI86 dedicated preflight

`representation_data_preflight_v1.py` passed on the exact committed head and proves:

1. deliberately invalid RGB decoy files are never read;
2. RGB mutation leaves the representation-stage fingerprint unchanged;
3. hidden rig/mechanics mutation leaves the legal representation fingerprint unchanged;
4. raster-authority mutation changes the fingerprint;
5. physical stage contains only `vertices/faces` plus `camera.json` and `raster_authority.npz`;
6. exact truth cache builds without RGB;
7. continuous projected track truth survives;
8. the dedicated auditor rejects any staged image/unexpected-file leak;
9. optimizer steps remain zero.

All prior architecture/evaluator/R0-R3 semantic preflights also passed in CI86.

## Frozen scientific semantics unchanged

- exact 256 OPEN asset panel;
- ordered panel-ID SHA-256 `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`;
- R0 exact P;
- R1 exact P+N;
- R2 six P-noise arms;
- R3 full 6x5 P/N noise grid;
- SAME tolerance `0.003` canonical units;
- fixed P+N coefficient `0.05`;
- deterministic Philox perturbations;
- physical set-containment top1/top4/top8;
- reciprocal + three-view cycle;
- family median/p90/p95 tails;
- geometry-only structural-confusability diagnostic;
- R4 undefined until separately preregistered if needed.

## Active execution

Entrypoint:

`run_representation_authority_v2.py --mode all`

The V2 orchestrator hash-binds itself plus:

- `build_representation_seed_v1.py`;
- `stage_representation_authority_v1.py`;
- `prepare_representation_cache_v1.py`;
- `audit_representation_stage_cache_v1.py`;
- `representation_authority_study_v1.py`;
- `compact_representation_handoff_v1.py`;
- `geometry.py`.

A successful real run may emit only:

`R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED`

with `optimizer_steps=0`, `training_authorized=false`, `sealed_splits_opened=false`, `rgb_staged=false`, and `rgb_consumed=false`.

No training is authorized by this freeze.
