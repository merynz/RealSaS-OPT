# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__P_V5_FORMULATION_CLOSED__FIELD_REPRESENTATION_CLOSED__R256_ONE_CELL_OVERFIT_PREREG_NEXT`

## Single continuation authority

Active implementation: `experiments/iris_single_pose_v2/`.

Before continuing in a new session, read this file and the root `README.md`.

## Canonical architecture pointer

The visible product architecture is frozen in `README.md` and authoritative in `canonical/PRODUCT_CONTRACT_V1.md`:

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

IRIS stops at the 2.5D observable substrate. Downstream physical training organization for Geppetto/Arachne is not frozen yet and must not be invented during IRIS work.

**Architecture change control:** this plan may change only under recorded controlled evidence as defined in `canonical/PRODUCT_CONTRACT_V1.md`. Convenience, analogy, intuition or conversational drift are not authority.

## Closed P authority

- CI104: `P_GEOMETRY_SUFFICIENT` remains CLOSED/PASS.
- CI202 free-XYZ learner is historical and insufficiently precise.
- P-V3: analytic screen-plane + scalar depth survives; fixed `h=0.54` falsified.
- P-V4: native image-derived scale survives; post-resize scale re-estimation falsified.
- P-V5 / CI283: `P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED`.
  - 16 FIT sentinels × 2 styles × 3 resolution conditions = 96/96 P-p95 cells <= 0.005.
  - max P p95 `0.0013928374974057078`.
  - optimizer 0; TUNE/sealed closed; teacher camera half-extent not a model input.

P-V5 formulation remains authoritative:

```text
native1024 ordered RGBA
  -> estimate h_native once
  -> transport h_native unchanged
  -> canonical yaw from V0..V7 ordering
  -> learn camera-forward scalar depth d
  -> analytic canonical P
```

`camera.json` remains forbidden in the learner/extractor path.

## Historical diagnostic — R/2 P-V5 depth overfit

The preregistered 8 FIT asset × 2 style run completed cleanly:

- 1024 optimizer steps;
- selected epoch 64;
- global P p95 `0.08759939931333059`;
- worst-cell P p95 `0.1367238707840442`;
- threshold `0.005`;
- result `P_V5_DEPTH_OPTIMIZATION_INSUFFICIENT`;
- TUNE consumed false;
- sealed splits opened false;
- `camera.json` consumed false.

This is **not** pure learner-capacity evidence because the experiment emitted P/depth on an R/2=128×128 field. That representation question has now been tested separately.

## P-V5 field representation closure — CLOSED

Canonical persisted result:

`IRIS_SINGLE_POSE_V2_P_V5_FIELD_REPRESENTATION_CLOSURE_V1/P_V5_FIELD_REPRESENTATION_CLOSURE.json`

Frozen gate:

- same 8 pre-result FIT_TRAIN sentinels;
- both styles = 16 cells;
- 4096 deterministic visible raster-authority samples/view;
- no CNN/model weights;
- neural optimizer steps 0;
- no TUNE/CAL/DEV/EXTERNAL;
- no `camera.json` / teacher camera half extent;
- candidate scalar depth fields 64×64, 128×128, 256×256;
- full canonical P Euclidean p95 threshold `<=0.005` per cell.

Result:

| field | interpretation | worst-cell P p95 |
|---|---|---:|
| 64×64 (R/4) | NOT_CERTIFIED | `0.03889907157958461` |
| 128×128 (R/2) | NOT_CERTIFIED | `0.01284720621837844` |
| 256×256 (R) | CERTIFIED 16/16 | `0.0008174655519194024` |

R256 global P p95 by style is `0.0005262544635931412`. The smallest tested certified field is therefore **256×256**.

Canonical label:

`P_V5_FIELD_REPRESENTATION_CLOSED`

Important interpretation: failed L2-oracle resolutions are `NOT_CERTIFIED`, not formal mathematical impossibility proofs. Full R is positively certified and is the current legal neural output-field target.

## R256 learner promotion ladder — CONDITIONAL / FROZEN ORDER

The learner scale-up order is fixed as follows unless controlled evidence requires a preregistered revision:

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

Promotion rule: a stage must be completed and interpreted as PASS before the next stage is authorized. A FAIL localizes work at the current stage under the research-order rule; it does not authorize skipping ahead to a broader corpus or generalization experiment.

This ladder controls the current P learner work only. It does not freeze later Geppetto/Arachne training organization.

## NEXT GATE — R256 one-asset / one-style learner overfit

The next authorized research action is to preregister a **single FIT asset × single style R256 learner overfit** before any broader learner run.

Purpose:

> With P ontology, V5 analytic reconstruction and R256 output representation already closed/certified, can the current visual learner optimize one fixed observation cell to P p95 <= 0.005?

Required discipline:

- R256 **input and P/depth output field**;
- same V5 native-scale-once/yaw formulation;
- P/depth objective only;
- no TUNE/CAL/DEV/EXTERNAL;
- no hidden camera metadata;
- preflight must verify exact 256×256 output shape and evaluator sampling semantics before optimizer step 1;
- no broader 2-style/8-asset/generalization experiment until the one-cell result is interpreted.

Candidate sentinel selection must be frozen in the preregistration before training; selection rationale must be recorded and no post-result switching is permitted.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
