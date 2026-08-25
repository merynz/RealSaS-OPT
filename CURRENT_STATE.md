# RealSaS-OPT — Current State

**Date:** 2026-08-25  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_CLOSED__P_V5_CLOSED__R256_FIELD_CLOSED__ONE_CELL_PASS__TWO_STYLE_PASS__EIGHT_BY_TWO_PREREG_FROZEN__A100_RUN_NEXT`

## Single continuation authority
Active implementation: `experiments/iris_single_pose_v2/`. Read this file and root `README.md` before continuing.

## Canonical architecture — evidence controlled

```text
8 ordered neutral-pose views
        ↓
      IRIS
        ↓
rigging-sufficient observable 2.5D substrate
        ↓
    Geppetto
        ↓
editable skeleton / hierarchy proposal
        ↓
     Arachne
        ↓
editable skinning / weight proposal
        ↓
     Compiler
        ↓
verified editable puppet
```

IRIS stops at the observable 2.5D substrate. Geppetto/Arachne training organization remains intentionally unfrozen while IRIS is being closed.

**Architecture change control:** architecture/responsibility boundaries may change only under recorded controlled evidence. Convenience, analogy, intuition, implementation ease or conversational drift are not authority.

## IMPORTANT external precedent — preserve, do not silently promote
`audit/IMPORTANT_EXTERNAL_PRECEDENT_PATCHMATCH_RL_20260825.md`

PatchMatch-RL (ICCV 2021) remains the closest open-code working precedent identified for the current IRIS geometric formulation. Its geometry-in-the-loop cross-view hypothesis verification is an important future intervention candidate if controlled multi-asset/family-disjoint evidence localizes a persistent extraction/coherence hard tail. It is **not admitted into the current 8×2 gate** and is not architecture authority by analogy alone.

## Closed P authority
- `P_GEOMETRY_SUFFICIENT`: CLOSED/PASS.
- P-V5 native-scale-once analytic reconstruction: CLOSED/PASS.
- legal factorization: native1024 RGBA -> estimate `h_native` once -> known yaw -> learn camera-forward scalar depth -> analytic canonical P.
- `camera.json` / teacher camera half extent remain forbidden learner inputs.
- R256 field representation: CLOSED/PASS; 256×256 is the smallest tested certified field.

## Frozen learner promotion ladder
```text
R256 field representation                       PASS
        ↓
1 asset × 1 style learner/optimizer sufficiency PASS
        ↓
1 asset × 2 styles joint fit                    PASS
        ↓
8 assets × 2 styles joint fit                   ← CURRENT / PREREG FROZEN
        ↓ PASS
unseen-family generalization
```

No rung may be skipped without a preregistered evidence-backed revision.

## One asset × two styles — CLOSED/PASS
Canonical result: `experiments/iris_single_pose_v2/P_V5_R256_TWO_STYLE_RESULT_20260825.md`.

One shared fresh model passed both cells at `TAIL_0512`:
- `cel_clean P_p95 = 0.003733412444125855`;
- `ink_cel P_p95 = 0.0038324856432154623`;
- worst-cell `0.0038324856432154623 <= 0.005`.

The `3e-5` fresh-moment tail crossed the two-style gate by tail step 128, independently supporting the prior late-stage LR localization.

## CURRENT GATE — R256 8 assets × 2 styles V1
Preregistration: `experiments/iris_single_pose_v2/P_V5_R256_8X2_PREREG_20260825.md`  
Membership: `experiments/iris_single_pose_v2/P_V5_R256_8X2_MEMBERSHIP_V1.json`

Membership is the same eight frozen FIT sentinels used by the P-V5 field-representation closure, each in `cel_clean` and `ink_cel`: 8 assets / 16 cells. No post-result asset selection.

Scientific question: can **one shared fresh R256 model** jointly fit all 16 cells to `P_p95 <= 0.005` at one preregistered checkpoint?

### Training protocol
- fresh deterministic model; no prior checkpoint initialization;
- same proven schedule: MAIN AdamW `3e-4 × 2048`, then fresh-moment TAIL AdamW `3e-5 × 512`;
- authority checkpoints: INIT; MAIN 512/1024/2048; TAIL 64/128/256/512;
- no LR search or adaptive schedule from 8×2 results;
- every optimizer step represents all 16 cells;
- fixed production decomposition: **8 cells × 2 gradient-accumulation microbatches**;
- each cell therefore participates in all 2560 optimizer steps;
- GroupNorm + zero-dropout architecture makes the decomposition batch-stat independent; reduced deterministic gradient-equivalence preflight PASS with max absolute difference `2.086162567138672e-07`.

### A100 execution
Target runtime: A100/high-memory CUDA. GPU preflight requires >=35 GiB device memory and a successful 8-cell / 64-view R256 forward-backward at optimizer step 0. Failure is apparatus-only and blocks scientific training.

### PASS rule
PASS iff the **same preregistered checkpoint** has all 16 asset-style cells individually `P_p95 <= 0.005`. Aggregate P95 cannot pass or mask a cell.

PASS -> preregister unseen-family generalization only.  
FAIL -> stay at 8×2 and localize the residual pattern before architecture change. PatchMatch-style refinement becomes eligible only if controlled evidence localizes an extraction/coherence blocker.

## Preparation validation
- syntax PASS;
- CPU full-R contract PASS;
- no BatchNorm / nonzero dropout PASS;
- full-batch vs accumulated-gradient equivalence PASS;
- fake 8-asset / 16-cell stage-cache-dataset PASS;
- scientific optimizer steps during preparation: `0`;
- production A100 GPU preflight: notebook step-0 gate, not yet run.

## Research rule
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
