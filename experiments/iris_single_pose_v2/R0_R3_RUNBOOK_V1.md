# IRIS V2 — R0-R3 Representation Authority Runbook

**Status:** `OPTIMIZER_ZERO_ONLY__TRAINING_FORBIDDEN`  
**Date:** 2026-08-24

This runbook executes the frozen confirmatory Representation Authority apparatus. It does not train IRIS.

## Preconditions

Target corpus root:

`/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3`

The runner refuses to start unless these exact frozen authorities are present:

- `reports/iris_controlled_v1/IRIS_CONTROLLED_V1_SPLIT_FREEZE.json`
- `metadata/CANONICAL_VARIANT_SELECTION.json`

Expected SHA-256:

- split freeze: `9e766ac61126c9b4787eef24146e36aac40cbeac166d67ba898f8b79133e9d66`
- canonical selection: `af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9`

The exact 256-asset panel ID-list digest must be:

`366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`

## One command

From `experiments/iris_single_pose_v2/` on the audit branch:

```bash
python run_representation_authority_v1.py --mode all
```

Default staging input is native 1024. The exact correspondence/geometry truth is always derived from native-1024 raster authority.

## What the command does

```text
frozen authority SHA verification
 -> exact locked 256 seed
 -> pre-stage panel digest verification
 -> physical geometry-only staging
 -> V2 exact/dense truth cache
 -> read-only stage/cache audit
 -> 38 R0-R3 arms
 -> post-result panel digest verification
 -> RUN_COMPLETE_V1.json
```

No substitution is allowed. If one of the 256 frozen assets fails staging/cache/audit, the run stops instead of selecting a replacement.

## Expected output directory

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_REPRESENTATION_AUTHORITY_V1/`

Important outputs:

- `RUN_AUTHORITY_V1.json`
- `REPRESENTATION_SEED_V1.json`
- `stage_R1024/STAGE_MANIFEST.json`
- `cache_v2/CACHE_MANIFEST.json`
- `STAGE_CACHE_AUDIT_V1.json`
- `REPRESENTATION_AUTHORITY_PANEL_V1.json`
- `REPRESENTATION_AUTHORITY_RESULT_R0_R3_V1.json`
- `REPRESENTATION_AUTHORITY_HARD_TAIL_V1.jsonl`
- `RUN_COMPLETE_V1.json`

A completed legal measurement must report:

`R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED`

and:

- `optimizer_steps = 0`
- `training_authorized = false`
- `sealed_splits_opened = false`

## Resume behavior

Staging and cache builders have content/settings/builder fingerprints and may reuse only exact matching artifacts.

The run directory also freezes script/settings/source-authority hashes in `RUN_AUTHORITY_V1.json`. If those semantics change, the runner refuses to mutate the opened run and requires a new work directory.

For explicit stage-by-stage continuation after an interrupted process, use the same work directory and one of:

```bash
python run_representation_authority_v1.py --mode stage
python run_representation_authority_v1.py --mode cache
python run_representation_authority_v1.py --mode audit
python run_representation_authority_v1.py --mode study
```

Each continuation mode requires the preceding artifact to exist and match the frozen run authority.

## Interpretation boundary

The measurement runner may not decide:

- `P_GEOMETRY_SUFFICIENT`;
- `P_PLUS_R_REQUIRED`;
- whether R4 is needed;
- whether SOI-2 is needed;
- whether training may start.

Those are subsequent canonical interpretation decisions. If R4 is justified, its exact formulation must be preregistered before any R4 result is opened.
