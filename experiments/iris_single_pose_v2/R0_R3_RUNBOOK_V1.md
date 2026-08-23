# IRIS V2 — R0-R3 Representation Authority Runbook

**Status:** `CI69_FINAL_PRE_RESULT_FREEZE__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`  
**Date:** 2026-08-24

Final freeze authority:

`experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_EXECUTION_FREEZE_CI69_20260824.md`

## Exact execution authority

Source head:

`e110014e79f3a9686bfd411e41454a627914fe14`

GitHub Actions:

- `IRIS V2 Preflight #69`
- run ID `32672422405`
- SUCCESS

Execution bundle SHA-256:

`31b8bee0ebae9d91b45a58f41134da7d20d479aa3d855d9012f7837c84d4770f`

Drive bundle:

`/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_R0_R3_EXECUTION_BUNDLE_CI69.zip`

Prepared Colab launcher:

`RealSaS_IRIS_V2_R0_R3_Representation_Authority_CI69.ipynb`

Notebook SHA-256:

`dd901089c31650cf5a69d79b15a594994717b1d7397e65fa4a91a7f0d41c8a13`

## Preconditions

Corpus root:

`/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3`

Frozen source SHA-256:

- split freeze: `9e766ac61126c9b4787eef24146e36aac40cbeac166d67ba898f8b79133e9d66`
- canonical selection: `af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9`

Frozen 256-asset ordered panel ID-list SHA-256:

`366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`

No substitution is allowed.

## Preferred execution

Open the CI69 notebook and **Run all**.

The notebook:

1. mounts Drive;
2. verifies the exact bundle ZIP SHA;
3. verifies `BUNDLE_INFO.source_head_sha`;
4. verifies all eight internal script SHA values;
5. verifies the orchestrator self-hash lock;
6. keeps scratch stage/cache under `/content`;
7. runs the exact frozen optimizer-zero command;
8. remounts/retries only on recognized transient Drive I/O, without scientific changes;
9. persists final evidence to:
   `RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_REPRESENTATION_AUTHORITY_V1_CI69_RESULT`.

Equivalent direct entrypoint from the verified bundle directory:

```bash
python run_representation_authority_v1.py \
  --root /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3 \
  --work-dir /content/IRIS_SINGLE_POSE_V2_REPRESENTATION_AUTHORITY_V1_CI69 \
  --mode all
```

## What the runner does

```text
frozen authority SHA verification
 -> exact locked 256 seed
 -> pre-stage panel digest verification
 -> physical geometry-only staging
 -> exact/dense truth cache
 -> read-only stage/cache audit
 -> 38 frozen R0-R3 arms
 -> post-result panel digest verification
 -> compact measurement-only handoff
 -> RUN_COMPLETE_V1.json
```

`RUN_AUTHORITY_V1.json` hashes all execution dependencies **including the orchestrator itself**. A resumed run cannot silently change the runner.

## Expected evidence

- `RUN_AUTHORITY_V1.json`
- `REPRESENTATION_SEED_V1.json`
- `stage_R1024/STAGE_MANIFEST.json`
- `cache_v2/CACHE_MANIFEST.json`
- `STAGE_CACHE_AUDIT_V1.json`
- `REPRESENTATION_AUTHORITY_PANEL_V1.json`
- `REPRESENTATION_AUTHORITY_RESULT_R0_R3_V1.json`
- `REPRESENTATION_AUTHORITY_HARD_TAIL_V1.jsonl`
- `REPRESENTATION_AUTHORITY_COMPACT_HANDOFF_V1.json`
- `RUN_COMPLETE_V1.json`

A completed legal measurement must report:

`R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED`

and:

- `optimizer_steps = 0`
- `training_authorized = false`
- `sealed_splits_opened = false`

## Interpretation boundary

The runner/notebook may not decide:

- P geometry sufficiency;
- whether P+N is necessary/sufficient;
- R4 necessity;
- SOI-2 necessity;
- checkpoint selection;
- training authorization.

Those are subsequent canonical interpretation decisions after the full stratified/tail evidence is inspected.

## Freeze rule

No further execution-code changes are authorized before the first real result unless a genuine apparatus failure is documented. Any apparatus fix requires a new exact-head CI run, bundle digest, Drive mirror and launcher authority; do not patch CI69 in place and call it the same experiment.
