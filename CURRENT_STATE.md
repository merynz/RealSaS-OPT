# RealSaS-OPT — Current State

**Date:** 2026-08-25  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_CLOSED__P_V5_CLOSED__R256_FIELD_CLOSED__ONE_CELL_LEARNER_OPTIMIZER_SUFFICIENT__TWO_STYLE_PREREG_FROZEN__GPU_RUN_NEXT`

## Single continuation authority
Active implementation: `experiments/iris_single_pose_v2/`.

Before continuing, read this file and root `README.md`.

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

## Closed P authority
- `P_GEOMETRY_SUFFICIENT`: CLOSED/PASS.
- P-V5 native-scale-once analytic reconstruction: CLOSED/PASS.
- legal P factorization: native1024 RGBA -> estimate `h_native` once -> canonical yaw -> learn camera-forward scalar depth -> analytic canonical P.
- `camera.json` / teacher camera half extent remain forbidden learner inputs.

## R256 field representation — CLOSED
| field | interpretation | worst-cell P p95 |
|---|---|---:|
| 64×64 | NOT_CERTIFIED | `0.03889907157958461` |
| 128×128 | NOT_CERTIFIED | `0.01284720621837844` |
| 256×256 | CERTIFIED 16/16 | `0.0008174655519194024` |

Smallest tested certified field = **R256**.

## Frozen learner promotion ladder
```text
R256 field representation
        ↓
1 asset × 1 style learner/optimizer sufficiency
        ↓ PASS
1 asset × 2 styles joint fit          ← CURRENT
        ↓ PASS
8 assets × 2 styles
        ↓ PASS
unseen-family generalization
```

No rung may be skipped without a preregistered evidence-backed revision.

## One asset × one style — learner/optimizer sufficiency CLOSED
The original frozen fixed-lr gate remains an immutable FAIL:
- `cel_clean`, 2048 steps, AdamW `3e-4`;
- P p95 `0.005682396539486942` vs threshold `0.005`.

Optimizer localization reproduced the exact checkpoint at step 0 with absolute P-p95 difference `0.0` and then ran three fresh-AdamW restarts:

| arm | lr | first PASS | min P p95 |
|---|---:|---:|---:|
| A control | `3e-4` | — | `0.005420877947472036` |
| B | `1e-4` | 256 | `0.00420133795123547` |
| C | `3e-5` | 64 | `0.003946938854642211` |

Canonical recovery status: `P_V5_R256_ONE_CELL_RECOVERY_PASS`.
Localization label: `LATE_STAGE_LR_FLOOR_SUPPORTED`.

Interpretation authority:
`experiments/iris_single_pose_v2/P_V5_R256_OPTIMIZER_LOCALIZATION_RESULT_20260825.md`

The prior miss is localized to late-stage optimizer/LR behavior; P ontology, V5 geometry and R256 representation remain closed.

## CURRENT GATE — R256 one asset × two styles V1
Preregistration:
`experiments/iris_single_pose_v2/P_V5_R256_TWO_STYLE_PREREG_20260825.md`

Membership:
`experiments/iris_single_pose_v2/P_V5_R256_TWO_STYLE_MEMBERSHIP_V1.json`

Frozen cells:
- `asset_76313e4bd82b82fcd1659c70 / cel_clean / FIT`;
- `asset_76313e4bd82b82fcd1659c70 / ink_cel / FIT`.

Scientific question: can **one shared fresh R256 model** jointly fit both render styles of the same asset to the canonical P threshold?

### Fresh initialization
The recovered one-style checkpoint is not used as initialization. The two-style gate starts from a fresh deterministic model so this is joint fit, not adaptation from a memorized style.

### Frozen repaired optimizer schedule
MAIN: fresh AdamW `3e-4`, betas `(0.9,0.95)`, wd `0`, 2048 steps.  
TAIL: fresh AdamW moments on MAIN-2048 weights, `3e-5`, same betas/wd, 512 steps.

Both style cells are present in every optimizer step (`B=2`, 8 views/cell). No new LR search is authorized.

### Representation / objective
- input/output P field R256;
- true full-resolution image-conditioned P/depth branch;
- camera-forward scalar depth learned; canonical P analytic;
- P/depth objective only; N/U/Z frozen;
- 4096 shared truth loci/view across styles;
- no augmentation;
- no TUNE/CAL/DEV/EXTERNAL;
- no hidden camera metadata.

### PASS rule
Authority candidates: INIT; MAIN `512/1024/2048`; TAIL `64/128/256/512`.

PASS iff the **same preregistered checkpoint** has:
- `cel_clean P_p95 <= 0.005`, and
- `ink_cel P_p95 <= 0.005`.

Aggregate p95 alone cannot pass. Checkpoint selection minimizes worst-cell p95, then aggregate p95, then total steps.

PASS -> preregister `8 assets × 2 styles R256` only.  
FAIL -> remain at same-asset/two-style and localize style interference/joint capacity/optimizer evidence.

## Preparation / executable handoff
Local validated source checks:
- syntax PASS;
- CPU B=2 true-full-R P-only forward/backward PASS;
- fake-master two-style stage/cache/dataset PASS;
- shared truth loci across styles PASS;
- preparation scientific optimizer steps `0`.

Next notebook:
`RealSaS_IRIS_PV5_R256_TwoStyle_Overfit_V1.ipynb`

Notebook SHA-256:
`e4e565a88405bdd948d06da65e6e95cd42403ef1798b7414e139f8ea2bab24c7`

Source bundle:
`IRIS_PV5_R256_TWO_STYLE_OVERFIT_V1_BUNDLE.zip`

Bundle SHA-256:
`7163f2636bd642c5112877e3d90633182876b825db03986b700d8141c19ab041`

Runtime: **CUDA GPU required**. Notebook runs a production B=2 R256 GPU forward/backward preflight at optimizer step 0 and fails closed before training if the runtime cannot carry the gate.

## Research rule
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
