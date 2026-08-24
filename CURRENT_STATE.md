# RealSaS-OPT — Current State

**Date:** 2026-08-25  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__P_V5_FORMULATION_CLOSED__FIELD_REPRESENTATION_CLOSED__R256_ONE_CELL_FROZEN_FAIL_0P005682__OPTIMIZER_LOCALIZATION_PREREG_FROZEN__GPU_RUN_NEXT`

## Single continuation authority

Active implementation: `experiments/iris_single_pose_v2/`.

Before continuing in a new session, read this file and root `README.md`.

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

IRIS stops at the observable 2.5D substrate. Downstream Geppetto/Arachne training organization is intentionally not frozen during current IRIS work.

**Architecture change control:** architecture and responsibility boundaries may change only under recorded controlled evidence. Convenience, analogy, intuition, implementation ease or conversational drift are not authority.

## Closed P authority

- CI104: `P_GEOMETRY_SUFFICIENT` CLOSED/PASS.
- P-V5 / CI283: `P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED` CLOSED/PASS.
- Legal P factorization:

```text
native1024 ordered RGBA
  -> estimate h_native once
  -> transport h_native unchanged
  -> canonical yaw from V0..V7 ordering
  -> learn camera-forward scalar depth d
  -> analytic canonical P
```

- `camera.json` / teacher camera half extent remain forbidden learner inputs.

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
1 asset × 1 style learner/optimizer sufficiency
        ↓ PASS
1 asset × 2 styles
        ↓ PASS
8 assets × 2 styles
        ↓ PASS
unseen-family generalization
```

No rung may be skipped without a preregistered evidence-backed revision.

## R256 one asset × one style — COMPLETED / FROZEN FAIL

Frozen cell:

- asset `asset_76313e4bd82b82fcd1659c70`;
- style `cel_clean`;
- split `FIT`.

True full-R path:

```text
R256 RGBA -> full-res image stem ---------┐
R256 RGBA -> encoder -> y2@128 -> upsample ├-> R256 fusion -> depth@256 -> analytic P@256
```

Corrected V1.1 GPU preflight PASS:

- Tesla T4;
- full-resolution fuse `8×48×256×256`;
- direct image-stem and P-depth gradients nonzero;
- optimizer steps at preflight = 0.

Frozen scientific protocol:

- AdamW lr `3e-4`, betas `(0.9,0.95)`, weight decay `0`;
- 2048 optimizer steps;
- candidate steps `64/128/256/512/1024/2048`;
- P/depth objective only;
- PASS iff canonical `P_p95 <= 0.005`.

Frozen result:

- selected step `2048`;
- selected P p95 `0.005682396539486942`;
- threshold `0.005`;
- status `P_V5_R256_ONE_CELL_OPTIMIZATION_INSUFFICIENT`;
- TUNE consumed false;
- sealed splits opened false;
- hidden camera metadata consumed false.

Candidate P p95:

```text
INIT   2.7298591
64     0.4656130
128    0.2629219
256    0.1124968
512    0.0436651
1024   0.0288629
2048   0.0056824
```

Interpretation:

- this is a real scientific FAIL under the frozen gate;
- it is not evidence to reopen P ontology, V5 geometry or R256 free-field closure;
- all preregistered candidate evaluations improved monotonically;
- 1024→2048 p95 improved by ~80.3%; no completed plateau is demonstrated;
- at 2048 P90=`0.00382736` while P95=`0.00568240`, so the miss is a narrow residual tail;
- late training remains noisy, consistent with a possible late-stage LR/optimizer floor but not proving it.

Canonical interpretation:

`experiments/iris_single_pose_v2/P_V5_R256_ONE_CELL_RESULT_INTERPRETATION_20260825.md`

## NEXT GATE — R256 one-cell optimizer localization V1

Preregistered diagnostic:

`experiments/iris_single_pose_v2/P_V5_R256_ONE_CELL_OPTIMIZER_LOCALIZATION_PREREG_20260825.md`

Purpose:

> Starting from the exact selected 2048-step checkpoint, distinguish finite-budget / late-stage learning-rate behavior from a remaining learner/objective/tail problem without broadening corpus scope.

Frozen parent checkpoint SHA-256:

`cad4421ae728848bbf9181c87e0d4912c38ec6f94d61c86cd114ad0a251c79ee`

Zero-step verification must reproduce parent P p95 `0.005682396539486942` within `1e-7` and persist per-view residual-tail metrics.

Three independent 512-step fresh-AdamW restart arms from the exact same checkpoint:

- A: lr `3e-4` control;
- B: lr `1e-4`;
- C: lr `3e-5`.

Common betas `(0.9,0.95)`, weight decay `0`, same P/depth objective, same evaluator, no augmentation, no TUNE/CAL/DEV/EXTERNAL, no hidden camera metadata.

Any arm reaching `P_p95 <= 0.005` closes one-cell learner/optimizer sufficiency and authorizes preregistration of **1 asset × 2 styles R256 only**. If all fail, remain at one-cell and inspect residual-tail/objective/feature-capacity evidence.

### Executable handoff

Validated source runner:

`experiments/iris_single_pose_v2/pv5_r256_optimizer_localization_v1.py`

Repo source-binding commit:

`924664896240effbd83deb737da2ad67b26032cd`

Next notebook:

`RealSaS_IRIS_PV5_R256_Optimizer_Localization_V1.ipynb`

Notebook SHA-256:

`4f482bd9e4a7c4d54386bb62bea3e56fe8ae40f00facb14a94c7f1927a41328b`

Bundle SHA-256:

`53c34fd494fc9a8dd258f90220c1f2db5511931d43892c07dca79e2243e1d93f`

Preparation scientific optimizer steps: `0`.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
