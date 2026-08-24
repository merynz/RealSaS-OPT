# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__MINI_EXTRACTABILITY_V1_FROZEN__CI183_RELEASE_PASS__GPU_DIAGNOSTIC_GATE_READY__OPTIMIZER_NOT_YET_OPENED`

## Single continuation authority

The sole active candidate implementation is `experiments/iris_single_pose_v2/`. Controlled V1/M256 is historical/no-run.

CI104 Representation Authority is complete. Canonical representation label: `P_GEOMETRY_SUFFICIENT`; R4/SOI-2 are not opened.

Exact P on the frozen 256 OPEN panel / 12,288 queries: top1/top4/top8=1/1/1; reciprocal=1; cycle=1; physical-error tails all zero.

## Normal policy after CI104

`NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED`: N remains legal observation-local `geom_n` supervision and an angular diagnostic, but is forbidden from correspondence admission/ranking and checkpoint selection. Production correspondence remains Z_coarse+P global admission and Z_fine local refinement.

## Mini Extractability / Generalization V1 — frozen before optimizer

Prereg: `experiments/iris_single_pose_v2/MINI_EXTRACTABILITY_GENERALIZATION_PREREG_V1.md`

Membership: `experiments/iris_single_pose_v2/MINI_EXTRACTABILITY_MEMBERSHIP_V1.json`

Membership provenance: `experiments/iris_single_pose_v2/MINI_MEMBERSHIP_PROVENANCE_V1.json`

Frozen authorities:

- CI104 representation seed SHA-256 `f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af` — provenance only; never learner runtime input
- source panel ordered-ID digest `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`
- mini membership canonical-JSON SHA-256 `4e223c799cf479a210716a86701ab96459673fe21852142e7cc7bd3a9d30e055`
- FIT_TRAIN128 all source FIT; FIT_SELECT32 all source FIT; TUNE_FINAL26 all source TUNE; roles disjoint
- CAL/DEV/EXTERNAL_HOLDOUT unopened

Training remains frozen at input256, 16 epochs, AdamW3e-4, microbatch1, grad accumulation4, FP16, geometry warmup epochs1-3, correspondence/local refinement from epoch4, candidate checkpoints4/8/12/16. Checkpoint selection uses FIT_SELECT only.

Allowed result labels remain only: `EXTRACTABILITY_GENERALIZATION_PASS`; `EXTRACTABLE_ON_FIT_SELECT__GENERALIZATION_GAP`; `LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT`.

## Hard TUNE/runtime firewall

Runtime seed contains only `asset_id`, `split`, `mini_role`. CI104 source seed is not consumed at runtime.

Enforced execution order:

`minimal membership seed -> FIT-only stage/cache -> FIT_TRAIN optimizer -> FIT_SELECT checkpoint freeze -> CHECKPOINT_SELECTION_FROZEN.json -> TUNE-only stage/cache -> one frozen-checkpoint TUNE_FINAL evaluation`.

`train_mini_v2.py` has no TUNE CLI. `finalize_mini_v2.py` is the only post-freeze TUNE consumer.

## Apparatus fixes carried into CI183

- learner cache partial visibility: `track_err[ok,v] = err[ok]`; hidden entries remain +inf
- cache fingerprint binds prepare_cache.py+geometry.py+coords.py
- Z_fine local loss uses deterministic position-uniform track thinning, never prefix truncation
- GPU capacity probe always writes structured JSON with environment, diagnostic stage, error type/message and traceback; generic CUDA failures can no longer collapse into blind `CalledProcessError`
- forced NO_CUDA regression requires exit2 + `status=NO_CUDA` + traceback + optimizer_steps0
- exact synthetic R=256 / geom_samples1024 / tracks128 full post-warmup loss forward+backward executes in CI CPU semantic replay
- model parameter count in that replay: 7,394,135
- evaluator model forward may use CUDA FP16 autocast, but every exported P/N/U/Z evidence field is explicitly cast to FP32 before geometry/grid_sample/matcher arithmetic; half-output regression is mandatory

## CI174 failure status

CI174 is superseded. A user run failed inside the old GPU preflight with returncode1. The user reports T4 had been selected. The exact cause cannot be recovered from CI174 because that script checked CUDA outside its report path and caught only OOM inside the capacity probe; generic CUDA/AMP exceptions produced no structured JSON. Therefore **do not retroactively label the CI174 failure NO_CUDA**.

CI183 exists specifically to measure the actual attached device/PyTorch CUDA state and exact failure stage before scientific state is opened.

## Current exact learner execution authority — CI183 / bundle V3

Code-bearing head: `84954fb56d72a380e4c668983002fb1b8bb5f84e`.

GitHub Actions `IRIS V2 Preflight` run #183, run ID `32687119316`: **SUCCESS**.

Mini artifact:

- artifact ID `9506049384`
- name `iris-v2-mini-extractability-bundle-v3`
- ZIP SHA-256 `acb01cca1db063892cd59cf674ae8442f7beee166f6a38324b449af07f7829bb`
- independent ZIP SHA replay PASS
- internal SHA256SUMS PASS
- isolated bundle compile/dependency/semantic replay PASS
- learner partial-visibility regression PASS
- membership provenance PASS
- R256/geom1024/tracks128 CPU full-loss/backward regression PASS
- forced NO_CUDA structured-report regression PASS
- evaluator AMP-export->FP32 regression PASS
- runtime representation seed consumed=false
- TUNE staging before checkpoint freeze=false
- sealed splits opened=false

Bundle schema remains V3 because the scientific/runtime data contract is unchanged; CI183 exact head+artifact SHA supersede the known-bad CI174 V3 bytes. Do not use CI174 V3.

Drive immutable mirror:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_MINI_EXTRACTABILITY_BUNDLE_CI183.zip`

Drive file ID: `13shqQYNzvQz7XchFZxwPzMGXtSCmQAzt`.

## Colab launcher authority

Launcher: `RealSaS_IRIS_V2_Mini_Extractability_Generalization_CI183.ipynb`

Notebook SHA-256: `eef9cc1c8b2fde684c8309222153d9af9303f1d3d578ffc3763a0df9f9437940`.

Launcher checks completed before handoff:

- strict nbformat validation PASS
- all code cells compile PASS
- first executable cell contains no Drive or persistent-result access; validates `nvidia-smi` + `torch.cuda.is_available()` first
- GPU failure occurs before persistent scientific state creation
- real CI183 bundle SHA/content/internal hashes replay PASS
- exact artifact semantic regressions replay PASS
- minimal 186-record runtime seed PASS
- runtime representation-seed CLI absent PASS
- FIT-only training TUNE CLI absent PASS
- GPU preflight requires structured report even on failure
- run process streams output and persists a failure record with last 500 lines if the frozen run exits nonzero
- completion marker remains last-write authority

Persistent result root when run:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_MINI_EXTRACTABILITY_CI183_RESULT`

## NEXT EXECUTABLE STEP

Open `RealSaS_IRIS_V2_Mini_Extractability_Generalization_CI183.ipynb` in Colab with a GPU runtime and Run All.

The notebook first reports the actual NVIDIA device, torch version, torch CUDA build and CUDA visibility. Only an actual environment PASS reaches the zero-scientific-step CUDA capacity probe. Only capacity PASS creates the persistent run authority and opens the frozen optimizer.

If the GPU gate fails, interpret the structured environment/capacity report; do not alter membership, thresholds, checkpoint key, N policy or FIT/TUNE ordering.

After `RUN_COMPLETE_MINI_V1.json` is persisted, interpret only the preregistered outcome labels.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
