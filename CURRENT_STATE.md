# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__MINI_EXTRACTABILITY_V1_FROZEN__CI202_V4_RELEASE_PASS__REAL_T4_CAPACITY_GATE_NEXT__OPTIMIZER_ZERO`

## Single continuation authority

The sole active candidate implementation is `experiments/iris_single_pose_v2/`. Controlled V1/M256 is historical/no-run.

CI104 Representation Authority is complete. Canonical label: `P_GEOMETRY_SUFFICIENT`; R4/SOI-2 remain closed. Exact P on the frozen 256 OPEN panel / 12,288 queries: top1/top4/top8=1/1/1; reciprocal=1; cycle=1; physical-error tails all zero.

## Frozen learner target / normal policy

Primary learner evidence remains P + Z_coarse persistence + local Z_fine refinement. `geom_n` remains legal observation-local orientation supervision and angular diagnostic. `NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED`: N is forbidden from correspondence admission/ranking and checkpoint selection.

Prereg: `experiments/iris_single_pose_v2/MINI_EXTRACTABILITY_GENERALIZATION_PREREG_V1.md`  
Membership: `experiments/iris_single_pose_v2/MINI_EXTRACTABILITY_MEMBERSHIP_V1.json`  
Membership provenance: `experiments/iris_single_pose_v2/MINI_MEMBERSHIP_PROVENANCE_V1.json`

Frozen membership/authority:

- source panel ordered-ID digest `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`
- mini membership canonical JSON SHA-256 `4e223c799cf479a210716a86701ab96459673fe21852142e7cc7bd3a9d30e055`
- CI104 representation-seed SHA-256 `f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af` is provenance only; never learner runtime input
- FIT_TRAIN128 / FIT_SELECT32 / TUNE_FINAL26 / unused FIT70
- CAL/DEV/EXTERNAL_HOLDOUT unopened

Frozen training: input256; 16 epochs; AdamW3e-4; microbatch1; grad accumulation4; clip1; train tracks128; eval tracks512; geometry warmup epochs1-3; correspondence/local refinement from epoch4; candidate checkpoints4/8/12/16; FIT_SELECT-only checkpoint selection; TUNE evaluated once after checkpoint freeze.

Allowed labels remain only: `EXTRACTABILITY_GENERALIZATION_PASS`; `EXTRACTABLE_ON_FIT_SELECT__GENERALIZATION_GAP`; `LEARNER_EXTRACTABILITY_NOT_YET_SUFFICIENT`.

## Hard TUNE/runtime firewall

Runtime seed contains only `asset_id`, `split`, `mini_role`; source registry/capability metadata and CI104 source seed are not consumed at runtime.

Enforced execution order:

`minimal membership seed -> FIT-only stage/cache -> FIT_TRAIN optimizer -> FIT_SELECT checkpoint freeze -> CHECKPOINT_SELECTION_FROZEN.json -> TUNE-only stage/cache -> one frozen-checkpoint TUNE_FINAL evaluation`.

`train_mini_v2.py` has no TUNE CLI. `finalize_mini_v2.py` is the only post-freeze TUNE consumer.

## CI183 T4 failure — apparatus only, scientific result unopened

The CI183 notebook reached a real Tesla T4 under PyTorch `2.11.0+cu128`. CUDA was available. The zero-step capacity probe failed before optimizer step1 at coarse correspondence loss:

`sim.masked_fill(~pos, -1e9)`

with:

`RuntimeError: value cannot be converted to type c10::Half without overflow`.

Cause: `total_loss` was still executed inside the CUDA autocast region; similarity became FP16 and `-1e9` is outside Half range. Scientific optimizer steps remained `0`, so no learner result or TUNE evidence was opened.

CI183/V3 is superseded and must not be rerun.

## Mixed-precision numeric policy — fixed before optimizer

The frozen objectives/weights/thresholds are unchanged. Execution numeric policy is now explicit:

`CUDA autocast FP16 -> model forward only`  
`autocast OFF -> total_loss with explicit FP32 supervision/matching numerics`

FP32 loss domains include P/N/U_geo sampling and loss math; Z_coarse normalization/similarity/logsumexp/softmax/margin/cycle; Z_fine local correlation/cross-entropy; and P-consistency. Evaluator evidence is also converted to FP32 before geometry/grid_sample/matcher arithmetic.

Mandatory regression `amp_loss_precision_preflight_v1.py`:

- reproduces the prior FP16 `masked_fill(...,-1e9)` failure class
- feeds Half P/N/U/Zc/Zf evidence through all loss branches
- requires finite FP32 total loss and backward
- AST-enforces model forward inside autocast and `total_loss` outside autocast in both trainer and GPU capacity probe
- optimizer steps=0

## Current exact learner execution authority — CI202 / bundle V4

**Code-bearing source head:** `83a856679a3bc4b4c0c894602f55ac1ffd94aa73`.

GitHub Actions `IRIS V2 Preflight` run **#202**, run ID `32688230184`: **SUCCESS**.

Mini artifact:

- artifact ID `9506401956`
- name `iris-v2-mini-extractability-bundle-v4`
- schema `RealSaS.IRISSinglePoseV2.MiniExtractabilityExecutionBundle.v4`
- ZIP SHA-256 `9dcb53a5c8447b0c0fcf8f31d618143fe619262204bbd44c5fd9ae7de911612c`
- independent ZIP SHA replay PASS
- all internal SHA256SUMS PASS
- isolated bundle compile/dependency/semantic replay PASS
- learner partial-visibility regression PASS
- membership/TUNE/runtime firewall PASS
- CPU R256/geom1024/tracks128 full-loss forward/backward PASS
- structured NO_CUDA regression PASS
- evaluator AMP-export -> FP32 evidence regression PASS
- AMP Half-overflow regression PASS
- trainer/GPU-preflight autocast-boundary invariant PASS
- runtime representation seed consumed=false
- TUNE staging before checkpoint freeze=false
- sealed splits opened=false

Drive immutable mirror:

`RealSaS_MASTER_CORPUS_1024_V3/reports/iris_single_pose_v2/IRIS_V2_MINI_EXTRACTABILITY_BUNDLE_CI202.zip`

Drive file ID: `18dm4eCcaexwzJ14Ip-6aUhS9y7vPOt2G`.

## Colab launcher authority

Launcher: `RealSaS_IRIS_V2_Mini_Extractability_Generalization_CI202.ipynb`

Notebook SHA-256: `eeb3d6aa15961e3ea6a9cab826d1b08e802f8e73eca2798d83c9ef77b3681ebe`.

Launcher pre-handoff validation:

- strict nbformat validation PASS
- all code cells compile PASS
- first executable cell checks real NVIDIA/PyTorch CUDA state before Drive/scientific state
- exact CI202/V4 bundle SHA/content/internal hashes replay PASS
- exact artifact partial-visibility, mini-contract and AMP-loss regressions replay PASS
- minimal 186-record runtime seed PASS
- runtime representation-seed CLI absent PASS
- FIT-only training TUNE CLI absent PASS
- capacity PASS additionally requires `diagnostic_stage=complete`, `loss_dtype=torch.float32`, finite gradients and AdamW moment-memory accounting with scientific optimizer steps=0
- run output streams live; nonzero scientific run writes a failure record with the last 500 lines
- completion marker remains last-write authority

Persistent result root when run:

`RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_MINI_EXTRACTABILITY_CI202_RESULT`

## NEXT EXECUTABLE STEP

Run `RealSaS_IRIS_V2_Mini_Extractability_Generalization_CI202.ipynb` on the T4 runtime from the top.

The next scientifically meaningful event is a **real T4 zero-step capacity PASS** under the new boundary. Only that PASS opens optimizer step1. If the capacity gate fails, use its structured stage/error/traceback and keep optimizer at0; do not alter membership, thresholds, checkpoint key, N policy or FIT/TUNE ordering.

After `RUN_COMPLETE_MINI_V1.json` is persisted, interpret only the preregistered outcome labels.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
