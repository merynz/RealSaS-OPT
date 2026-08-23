# Representation Authority — Pre-Result Execution Freeze CI69

**Date:** 2026-08-24  
**Status:** `FROZEN_FOR_REAL_R0_R3_RUN__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

This document freezes the executable apparatus for the first real 256-asset R0-R3 Representation Authority measurement. It is downstream of the parent preregistration, execution addendum and exact panel lock. It does not change scientific definitions and does not authorize an optimizer.

## Exact source authority

Active branch:

`audit/iris-architecture-discipline-20260824`

Exact execution source head:

`e110014e79f3a9686bfd411e41454a627914fe14`

The final pre-result integrity amendment at this head adds `run_representation_authority_v1.py` itself to `RUN_AUTHORITY_V1.json`'s script-hash chain. Therefore interrupted/resumed runs fail closed if the orchestrator changes as well as if any stage/cache/study/compact script changes.

## Exact CI authority

Workflow:

`.github/workflows/iris_v2_preflight.yml`

GitHub Actions:

- run: `#69`
- run ID: `32672422405`
- conclusion: `SUCCESS`

The workflow checks out the exact PR head rather than GitHub's synthetic merge commit, then passes:

- exact committed-source compilation;
- architecture / coordinate / matcher preflight;
- physical firewall / truth-cache semantic-invalidation preflight;
- observable evaluator preflight;
- Representation Authority semantic preflight;
- execution-bundle construction;
- execution-bundle artifact upload.

## Exact execution bundle

GitHub artifact:

- artifact ID: `9501738915`
- artifact name: `iris-v2-r0-r3-execution-bundle`
- ZIP SHA-256: `31b8bee0ebae9d91b45a58f41134da7d20d479aa3d855d9012f7837c84d4770f`

`BUNDLE_INFO.json` must state:

- schema `RealSaS.IRISSinglePoseV2.ExecutionBundle.v1`;
- `source_head_sha = e110014e79f3a9686bfd411e41454a627914fe14`;
- `optimizer_steps = 0`;
- `training_authorized = false`;
- entrypoint `run_representation_authority_v1.py --mode all`;
- corpus root `/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3`.

The ZIP contains exactly these execution scripts plus `BUNDLE_INFO.json` and `SHA256SUMS.txt`:

- `build_representation_seed_v1.py`
- `stage_source.py`
- `prepare_cache.py`
- `audit_staged_cache_v2.py`
- `representation_authority_study_v1.py`
- `compact_representation_handoff_v1.py`
- `geometry.py`
- `run_representation_authority_v1.py`

All eight internal SHA-256 entries were independently reverified after artifact download before this freeze.

## Drive mirror

The exact CI69 ZIP is mirrored into the Master Corpus reports tree at:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_R0_R3_EXECUTION_BUNDLE_CI69.zip`

Drive file ID:

`1lW7pmAjjGO-M1OWvkJn9vOphqhv29QNP`

The mirror must hash to the exact ZIP SHA above before execution.

## Colab launcher authority

Prepared launcher attachment:

`RealSaS_IRIS_V2_R0_R3_Representation_Authority_CI69.ipynb`

Notebook SHA-256:

`dd901089c31650cf5a69d79b15a594994717b1d7397e65fa4a91a7f0d41c8a13`

The launcher:

1. mounts Drive;
2. verifies the exact CI69 ZIP SHA;
3. verifies `BUNDLE_INFO.source_head_sha`;
4. verifies every internal `SHA256SUMS.txt` entry;
5. verifies the orchestrator self-hash lock exists;
6. runs the exact frozen command with scratch stage/cache under `/content`;
7. retries only recognized transient Google Drive I/O after remount, without changing the scientific apparatus;
8. persists final evidence/manifests back to Drive;
9. prints only a compact measurement summary and never emits a scientific promotion/training decision.

Expected persistent result directory after success:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_REPRESENTATION_AUTHORITY_V1_CI69_RESULT`

## Frozen panel identity

Ordered 256-asset panel ID-list SHA-256:

`366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`

No substitution is authorized if one locked asset fails legal staging/cache/audit.

## Freeze rule

**Do not modify the R0-R3 execution scripts, panel selection, thresholds, scoring, noise semantics, candidate universe or evidence-consumer semantics before the real result is opened.**

A genuine apparatus failure may reopen implementation only if:

1. the failure is documented as apparatus/infrastructure rather than interpreted as a scientific result;
2. scientific definitions remain unchanged unless a new preregistration is explicitly created;
3. the changed source receives a new commit, exact-head CI run, artifact digest, Drive mirror and launcher authority;
4. the old CI69 run remains immutable provenance.

Until such a failure occurs, the only next executable is the real optimizer-zero R0-R3 run.

## Post-run boundary

A successful run may emit only:

`R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED`

with:

- `optimizer_steps = 0`;
- `training_authorized = false`;
- `sealed_splits_opened = false`.

The subsequent canonical interpretation may decide whether P/P+N closes the legal representation gate or whether R4/SOI-2 work is justified. The measurement apparatus itself may not make that decision.
