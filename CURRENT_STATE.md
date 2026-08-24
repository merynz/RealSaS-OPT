# RealSaS-OPT — Current State

**Date:** 2026-08-25  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__P_V5_FORMULATION_CLOSED__FIELD_REPRESENTATION_CLOSED__R256_ONE_CELL_PREREG_FROZEN__ATTEMPT1_APPARATUS_SUPERSEDED_OPT0__V1_1_GPU_RUN_NEXT`

## Single continuation authority

Active implementation: `experiments/iris_single_pose_v2/`.

Before continuing in a new session, read this file and the root `README.md`.

## Canonical architecture pointer

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

IRIS stops at the 2.5D observable substrate. Downstream Geppetto/Arachne training organization is intentionally not frozen during current IRIS work.

**Architecture change control:** architecture and responsibility boundaries may change only under recorded controlled evidence. Convenience, analogy, intuition, implementation ease or conversational drift are not authority.

## Closed P authority

- CI104: `P_GEOMETRY_SUFFICIENT` CLOSED/PASS.
- P-V5 / CI283: `P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED` CLOSED/PASS.
- Legal P factorization remains:

```text
native1024 ordered RGBA
  -> estimate h_native once
  -> transport h_native unchanged
  -> canonical yaw from V0..V7 ordering
  -> learn camera-forward scalar depth d
  -> analytic canonical P
```

- `camera.json` / teacher camera half extent remain forbidden learner inputs.
- Historical R/2 8-asset × 2-style learner overfit was insufficient, but is not pure learner-capacity evidence because R/2 output representation was not independently certified.

## P-V5 field representation closure — CLOSED

| field | interpretation | worst-cell P p95 |
|---|---|---:|
| 64×64 (R/4) | NOT_CERTIFIED | `0.03889907157958461` |
| 128×128 (R/2) | NOT_CERTIFIED | `0.01284720621837844` |
| 256×256 (R) | CERTIFIED 16/16 | `0.0008174655519194024` |

R256 global P p95 = `0.0005262544635931412`. Smallest tested certified field = **256×256**.

Canonical label: `P_V5_FIELD_REPRESENTATION_CLOSED`.

## Frozen R256 learner promotion ladder

```text
R256 field representation
        ↓
1 asset × 1 style overfit
        ↓ PASS
1 asset × 2 styles overfit
        ↓ PASS
8 assets × 2 styles overfit
        ↓ PASS
unseen-family generalization
```

No stage may be skipped without a preregistered evidence-backed revision.

## Current gate — R256 one asset × one style

Frozen cell:

- asset `asset_76313e4bd82b82fcd1659c70`
- style `cel_clean`
- split `FIT`

Scientific preregistration and membership remain unchanged.

True full-R P path:

```text
R256 RGBA -> full-res image stem ---------┐
R256 RGBA -> encoder -> y2@128 -> upsample ├-> R256 fusion -> depth@256 -> analytic P@256
```

A bare resized R/2 prediction is forbidden.

Frozen protocol:

- seed `20260824`
- AdamW `3e-4`, betas `(0.9,0.95)`, weight decay `0`
- 2048 optimizer steps
- candidate steps `64/128/256/512/1024/2048`
- P/depth objective only
- PASS iff selected canonical `P_p95 <= 0.005`
- no augmentation
- no TUNE/CAL/DEV/EXTERNAL
- no hidden camera metadata.

## Attempt 1 apparatus supersession — NO SCIENTIFIC RESULT

The first Colab attempt ended with a non-zero child-process exit before any authorized scientific optimizer step.

Root cause was found in `pv5_r256_onecell_gpu_preflight.py`: the diagnostic forward hook used `captured.setdefault(...)` as a lambda return value. PyTorch interprets a non-`None` forward-hook return as a replacement module output, so the hook replaced the R256 feature tensor with a shape tuple and broke the optimizer-zero GPU preflight.

Classification:

`APPARATUS_PREFLIGHT_IMPLEMENTATION_BUG__SCIENTIFIC_OPTIMIZER_STEPS_0`

This attempt does **not** count as PASS, FAIL, learner evidence, representation evidence or optimizer evidence. It does not change the preregistration, membership, objective, architecture hypothesis or promotion ladder.

Patch commit: `0d1e8f1a3466834ea4272c6b9f1cd6151bb9cd0f`.

Fixed hook semantics: capture shape, explicitly return `None`.

Release metadata supersession commit: `3f3493e3dab2d4f1f485cea3be5e955ed96c89df`.

## NEXT EXECUTABLE STEP

Use only:

`RealSaS_IRIS_PV5_R256_OneCell_Overfit_V1_1.ipynb`

The previous `..._V1.ipynb` is superseded and must not be rerun.

Runtime: **CUDA GPU required.**

V1.1 additionally persists apparatus diagnostics to Drive on any non-zero runner exit so a closed notebook cannot erase the failure location.

On PASS: preregister **1 asset × 2 styles R256 overfit only**.  
On scientific FAIL: localize R256 learner/optimizer; do not reopen closed P ontology, V5 analytic geometry or certified free-R256 representation without contradictory evidence.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
