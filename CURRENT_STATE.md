# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__P_V5_FORMULATION_CLOSED__FIELD_REPRESENTATION_CLOSED__R256_ONE_CELL_PREREG_FROZEN__CPU_PREFLIGHT_PASS__GPU_RUN_NEXT`

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

Result:

| field | interpretation | worst-cell P p95 |
|---|---|---:|
| 64×64 (R/4) | NOT_CERTIFIED | `0.03889907157958461` |
| 128×128 (R/2) | NOT_CERTIFIED | `0.01284720621837844` |
| 256×256 (R) | CERTIFIED 16/16 | `0.0008174655519194024` |

R256 global P p95 by style is `0.0005262544635931412`. The smallest tested certified field is **256×256**. Canonical label: `P_V5_FIELD_REPRESENTATION_CLOSED`.

Failed L2-oracle resolutions are `NOT_CERTIFIED`, not formal mathematical impossibility proofs. Full R is positively certified and is the current legal neural output-field target.

## R256 learner promotion ladder — CONDITIONAL / FROZEN ORDER

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

Promotion rule: a stage must complete and be interpreted as PASS before the next stage is authorized. A FAIL localizes work at the current stage and does not authorize skipping ahead.

This ladder controls current P learner work only. It does not freeze later Geppetto/Arachne training organization.

## CURRENT GATE — R256 one-asset / one-style learner overfit

Preregistration is frozen:

- `P_V5_R256_ONE_CELL_PREREG_20260824.md`
- `P_V5_R256_ONE_CELL_MEMBERSHIP_V1.json`
- source authority commit: `5fed9ed277f5937fb2eba4db6c4dfeabc2796d4a`

Frozen cell:

- asset `asset_76313e4bd82b82fcd1659c70`
- style `cel_clean`
- split `FIT`
- selected before learner optimization from completed field-closure evidence; no post-result switching.

Purpose:

> With P ontology, V5 analytic reconstruction and R256 output representation already closed/certified, can the visual learner optimize one fixed observation cell to canonical `P_p95 <= 0.005`?

### True full-R intervention

A bare `R128 -> bilinear upsample -> depth` is forbidden. The R256 gate adds a direct full-resolution image-conditioned P branch:

```text
R256 RGBA ─────────────→ full-resolution image stem ─┐
R256 RGBA → encoder → y2@R128 → upsample to R256 ──┤
                                                     ↓
                                            R256 feature fusion
                                                     ↓
                                            scalar depth d@R256
                                                     ↓
                                             analytic P@R256
```

N/U/Z diagnostic heads remain frozen and outside the objective.

### Frozen optimizer protocol

- seed `20260824`
- AdamW `lr=3e-4`, betas `(0.9,0.95)`, weight decay `0`
- one fixed cell / all 8 views per step
- 2048 planned optimizer steps
- checkpoints: `64,128,256,512,1024,2048`
- FP16 AMP neural forward; FP32 P/depth objective/evaluation
- SmoothL1 camera-forward depth, beta `0.01`
- full canonical P Euclidean p95 is decision authority
- PASS iff selected `P_p95 <= 0.005`
- no augmentation
- no TUNE/CAL/DEV/EXTERNAL
- no `camera.json` / teacher camera half extent.

### Preparation/preflight authority

Local final-source CPU preflight: PASS at scientific optimizer step 0.

Verified before release:

- syntax compile PASS;
- exact frozen one-cell membership PASS;
- upstream P-V5 byte identity PASS;
- native-scale regression PASS;
- metadata/split firewall PASS;
- P output equals input resolution rather than R/2 PASS;
- direct full-resolution image branch present PASS;
- nonzero finite gradients in both full-resolution image stem and P depth head PASS;
- N/U/Z frozen heads receive no gradients PASS.

Local machine has no CUDA; production R256 GPU preflight is intentionally **not claimed**. The notebook must execute a zero-optimizer-step CUDA R256 forward/backward preflight before staging/training and fail closed if it does not pass.

Scientific optimizer steps during preparation: `0`.

## NEXT EXECUTABLE STEP

Notebook:

`RealSaS_IRIS_PV5_R256_OneCell_Overfit_V1.ipynb`

Runtime: **CUDA GPU required.**

Run All will:

1. mount Drive and require CUDA;
2. SHA-verify the embedded exact source bundle;
3. rerun CPU contract preflight at optimizer 0;
4. verify persisted field-closure parent authority;
5. run production R256 CUDA forward/backward preflight at optimizer 0;
6. stage/cache only the frozen FIT/cel_clean cell;
7. freeze `PREOPT_AUTHORITY.json`;
8. train/evaluate the 2048-step R256 one-cell gate;
9. persist canonical output to:

`MyDrive/RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_P_V5_R256_ONE_CELL_OVERFIT_V1`

Decision policy:

- PASS → preregister **1 asset × 2 styles R256 overfit only**;
- FAIL → localize R256 learner/optimizer; do not reopen closed P ontology, V5 analytic geometry or free-R256 representation without contradictory evidence.

No broader learner/generalization training is authorized before this result is interpreted.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
