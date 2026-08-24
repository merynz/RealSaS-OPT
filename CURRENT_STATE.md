# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__MINI_EXTRACTABILITY_V1_FROZEN__CI174_V3_RELEASE_PASS__NOTEBOOK_READY__GPU_RELEASE_THEN_OPTIMIZER`

## Single continuation authority

The sole active candidate implementation is `experiments/iris_single_pose_v2/`. Controlled V1/M256 is historical/no-run.

CI104 Representation Authority is complete and canonically interpreted in `experiments/post_corpus_audit/REPRESENTATION_AUTHORITY_CANONICAL_INTERPRETATION_CI104_20260824.md`.

Canonical representation result: `P_GEOMETRY_SUFFICIENT`. R4 and SOI-2 are not opened.

## CI104 result that authorizes the learner target

Exact P on the frozen 256 OPEN panel / 12,288 queries: top1/top4/top8=1/1/1; reciprocal=1; cycle=1; physical-error median/p90/p95/max=0/0/0/0; family tails perfect. R2 P-noise remains the descriptive learner-accuracy reference.

## Normal policy after CI104

`NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED`: `track_n_view` is a nearby view-specific raster witness normal, not exact differential N at the same persistent P locus. Mini V1 keeps legal `geom_n` as observation-local dense orientation target and reports N angular error, but forbids N from correspondence admission/ranking and checkpoint selection. Production correspondence stays Z_coarse+P global admission, Z_fine local refinement.

## Mini Extractability / Generalization V1 — FROZEN BEFORE OPTIMIZER

Prereg: `experiments/iris_single_pose_v2/MINI_EXTRACTABILITY_GENERALIZATION_PREREG_V1.md`

Membership: `experiments/iris_single_pose_v2/MINI_EXTRACTABILITY_MEMBERSHIP_V1.json`

Membership provenance: `experiments/iris_single_pose_v2/MINI_MEMBERSHIP_PROVENANCE_V1.json`

Frozen authorities:

- CI104 representation seed byte SHA-256 `f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af` — provenance only, not learner runtime input
- source panel ordered-ID digest `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`
- mini membership canonical-JSON SHA-256 `4e223c799cf479a210716a86701ab96459673fe21852142e7cc7bd3a9d30e055`
- role proof: FIT_TRAIN128 all source FIT; FIT_SELECT32 all source FIT; TUNE_FINAL26 all source TUNE; roles disjoint

Frozen roles: FIT_TRAIN128; FIT_SELECT32; TUNE_FINAL26; unused FIT70; CAL/DEV/EXTERNAL_HOLDOUT unopened.

Input resolution=256. This mini tests extractability/generalization feasibility, not final1024 product precision.

Training: 16 epochs, AdamW3e-4, microbatch1, grad accumulation4, FP16, geometry warmup epochs1-3, correspondence/local refinement from epoch4, candidate checkpoints4/8/12/16. Checkpoint selection uses FIT_SELECT only. cel_clean and ink_cel are evaluated separately and equal-macro.

Allowed labels: `EXTRACTABILITY_GENERALIZATION_PASS`; `EXTRACTABLE_ON_FIT_SELECT__GENERALIZATION_GAP`; `LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT`.

## Hard TUNE / runtime firewall — CLOSED IN APPARATUS

The learner runtime seed is membership-only and contains exactly `asset_id`, `split`, `mini_role`. CI104 `REPRESENTATION_SEED_V1.json` is not opened by the learner runtime and source-registry/capability metadata is not carried forward.

Execution order is enforced in code:

`minimal membership seed -> FIT-only stage/cache -> FIT_TRAIN optimizer -> FIT_SELECT checkpoint freeze -> CHECKPOINT_SELECTION_FROZEN.json -> TUNE-only stage/cache -> one frozen-checkpoint TUNE_FINAL evaluation`.

`train_mini_v2.py` exposes no TUNE CLI path. `finalize_mini_v2.py` is the separate post-freeze TUNE consumer. A restart removes local TUNE stage/cache before a new scientific attempt.

## Training apparatus fixes before release

Image-bearing learner cache uses `track_err[ok,v] = err[ok]`; hidden entries remain +inf; cache fingerprint binds prepare_cache.py+geometry.py+coords.py. Z_fine local loss uses deterministic position-uniform track thinning rather than first-64 prefix. Both are mandatory CI regressions.

## Current exact learner execution authority — CI174 / bundle V3

Code-bearing head: `53c7275c5f6162c0fcefda6600e837479f708699`.

GitHub Actions `IRIS V2 Preflight` run #174, run ID `32685436284`: **SUCCESS**.

Mini artifact:

- artifact ID `9505516028`
- name `iris-v2-mini-extractability-bundle-v3`
- ZIP SHA-256 `c00b0866f3bffccd6e68ed10a8e7a897754b1b8d98a09b3104fd325203f8de29`
- isolated bundle compile/dependency/semantic replay PASS
- learner partial-visibility regression PASS
- membership provenance PASS
- `runtime_representation_seed_consumed=false`
- `tune_staging_before_checkpoint_freeze=false`
- sealed splits opened=false

Drive immutable mirror:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_MINI_EXTRACTABILITY_BUNDLE_CI174.zip`

Drive file ID: `1MSk1S_FJPQ61MXIysu6I_Ep_cYgTmkYg`.

## Colab launcher authority

Launcher: `RealSaS_IRIS_V2_Mini_Extractability_Generalization_CI174.ipynb`

Notebook SHA-256: `903f40d5942ee57fd6ffbad6d2ac9600de7ef2fcf237155d8e3dfbdddd67b560`.

Launcher validation completed before handoff:

- strict nbformat validation PASS
- all code cells compile PASS
- real CI174/V3 bundle authority cell replay PASS
- bundle SHA256SUMS PASS
- membership-only 186-record runtime seed PASS
- runtime representation-seed CLI absent PASS
- FIT-only training TUNE CLI absent PASS
- post-freeze TUNE finalizer present PASS
- restart symlink semantics PASS
- existing-completion authority verification / fail-closed incomplete marker PASS

Persistent result root when run:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_MINI_EXTRACTABILITY_CI174_RESULT`

Completion marker is persisted last. An incomplete persistent attempt is archived and restarted as a fresh scientific run; completed CI174 evidence is never silently rerun.

## GPU release gate

The closed 256/512/1024 architecture executable preflight is not repeated. Notebook first runs real CUDA R=256 production-width full post-warmup loss forward/backward, finite gradients and AdamW two-moment memory accounting, with zero scientific optimizer steps. OOM/non-finite => no optimizer. Only PASS opens the frozen learner optimizer.

## NEXT EXECUTABLE STEP

Open `RealSaS_IRIS_V2_Mini_Extractability_Generalization_CI174.ipynb` in a GPU Colab runtime and Run All. Do not edit membership, thresholds, checkpoint key, N policy, FIT/TUNE ordering or bundle authority.

After the notebook persists `RUN_COMPLETE_MINI_V1.json`, inspect the frozen result and assign only one preregistered outcome label. Do not alter thresholds after TUNE is opened.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
