# IRIS V2 — R0-R3 Representation Authority Runbook

**Status:** `CI86_REPRESENTATION_ONLY_PRE_RESULT_FREEZE__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`  
**Date:** 2026-08-24

Current freeze authority:

`experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_EXECUTION_FREEZE_CI86_20260824.md`

CI69 is superseded at the staging/apparatus layer. Its run was interrupted during unnecessary RGB staging before R0-R3 result generation.

## Exact execution authority

Code-bearing source head:

`e32089498054948aa53b74e728d6ad8d7946693f`

GitHub Actions:

- `IRIS V2 Preflight #86`
- run ID `32675574055`
- SUCCESS

Artifact:

- ID `9502554757`
- name `iris-v2-r0-r3-execution-bundle-v2`
- ZIP SHA-256 `aa6bba667d1b53e5634b5ed9e6274cceca4437130be9d00ef0f72d3f970470df`

Drive bundle:

`/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_R0_R3_REPRESENTATION_ONLY_BUNDLE_CI86.zip`

Prepared Colab launcher:

`RealSaS_IRIS_V2_R0_R3_Representation_Only_CI86.ipynb`

Notebook SHA-256:

`5870a4fc37cfa854c11dd118f492ad1244007f845a34be30a280d1b4338fedf0`

## Representation-only source boundary

This exact R0-R3 gate consumes no RGB.

Stage profile:

`representation_authority_geometry_only`

Per selected asset it stages only:

- sanitized `vertices/faces` from `primary_geometry.npz`;
- `renders/V0..V7/raster_authority.npz`;
- `renders/V0..V7/camera.json`.

It does not stage/decode/hash `cel_clean`, `ink_cel`, 512/256 image derivatives, or hidden rig/mechanics fields.

RGB remains required by the future image-to-evidence learner; this run is only the exact legal representation-ceiling measurement.

## Dedicated CI86 data-path proof

`representation_data_preflight_v1.py` passed and verifies:

- invalid RGB decoys are never read;
- RGB mutation does not alter the representation-stage fingerprint;
- hidden rig/mechanics mutation does not alter the legal fingerprint;
- raster-authority mutation does alter it;
- stage contains only legal geometry/raster/camera files;
- representation truth cache builds with `rgb_consumed=false`;
- exact continuous projected track truth is preserved;
- staged image/unexpected-file leakage is rejected.

## Frozen preconditions

Corpus root:

`/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3`

Source SHA-256 locks:

- split freeze: `9e766ac61126c9b4787eef24146e36aac40cbeac166d67ba898f8b79133e9d66`
- canonical selection: `af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9`

Frozen ordered 256-asset panel SHA-256:

`366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`

No asset substitution is allowed.

## Preferred execution

Open `RealSaS_IRIS_V2_R0_R3_Representation_Only_CI86.ipynb` and **Run all**.

CPU runtime is sufficient; GPU is not used.

Equivalent direct entrypoint from the verified bundle directory:

```bash
python run_representation_authority_v2.py \
  --root /content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3 \
  --work-dir /content/IRIS_SINGLE_POSE_V2_REPRESENTATION_AUTHORITY_V2_CI86 \
  --mode all
```

## What V2 does

```text
frozen authority SHA verification
 -> exact locked 256 seed
 -> pre-stage panel digest verification
 -> representation-only geometry+raster+camera staging
 -> exact/dense truth cache (RGB absent)
 -> dedicated representation stage/cache audit
 -> 38 frozen R0-R3 arms
 -> post-result panel digest verification
 -> compact measurement-only handoff
 -> RUN_COMPLETE_V2.json
```

`RUN_AUTHORITY_V2.json` hash-binds the orchestrator itself and every execution dependency.

## Expected evidence

- `RUN_AUTHORITY_V2.json`
- `REPRESENTATION_SEED_V1.json`
- `stage_representation_v1/STAGE_MANIFEST.json`
- `representation_cache_v1/CACHE_MANIFEST.json`
- `REPRESENTATION_STAGE_CACHE_AUDIT_V1.json`
- `REPRESENTATION_AUTHORITY_PANEL_V1.json`
- `REPRESENTATION_AUTHORITY_RESULT_R0_R3_V1.json`
- `REPRESENTATION_AUTHORITY_HARD_TAIL_V1.jsonl`
- `REPRESENTATION_AUTHORITY_COMPACT_HANDOFF_V1.json`
- `RUN_COMPLETE_V2.json`

A completed legal measurement must report:

`R0_R3_MEASURED__CANONICAL_INTERPRETATION_REQUIRED`

with:

- `optimizer_steps = 0`
- `training_authorized = false`
- `sealed_splits_opened = false`
- `rgb_staged = false`
- `rgb_consumed = false`

## Scientific semantics unchanged

CI86 changes no R0-R3 scientific definition. R0/R1/R2/R3 arms, P+N coefficient, SAME tolerance, perturbations, panel, query policy, candidate universe, reciprocal/cycle, family tails and hard-tail diagnostics remain frozen exactly as preregistered.

## Interpretation boundary

The runner/notebook may not decide P sufficiency, P+N necessity/sufficiency, R4, SOI-2, checkpoint selection or training authorization. Those are separate canonical interpretation decisions after the full evidence is persisted.
