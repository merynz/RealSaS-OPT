# IRIS H-GEO-P2-SCENE-FIELD Decision Ledger — 2026-10-20

## Question

Can the **existing IRIS V8 architecture** reach the locked metric-geometry court if exact authored-triangle supervision is allowed to update both the upstream triplane scene decoder (`scene_core.scene`) and the local carrier body (`scene_core.field`), without adding a new geometry branch?

## Why P2 exists

H-GEO-P1 was a valid scientific FAIL, not an apparatus failure. Exact triangle truth materially improved H over frozen V8, but field-only retraining missed all four absolute metric gates and missed the preregistered endpoint improvement gate. P1 left `scene_core.scene` frozen, so it did not establish that the existing V8 architecture lacks sufficient geometry capacity.

P2 is the missing control before an explicit geometry branch can be justified.

## Fixed comparison law

P2 changes one scientific variable relative to P1:

- P1: train `scene_core.field` only;
- P2: train `scene_core.scene + scene_core.field`.

Everything else is controlled as tightly as possible:

- exact sealed V8 initialization;
- **P1 checkpoint is forbidden as P2 initialization**;
- exact same P1 train/holdout triangle-dataset bytes;
- no architecture expansion;
- same metric target semantics;
- no subject-specific parameters;
- no TESSA;
- no Compiler repair;
- teacher geometry is target-only and never inference conditioning.

## Runtime law

Before optimizer step 1:

1. exact clean source SHA;
2. exact A100-class GPU with >=39 GiB;
3. exact V8 checkpoint;
4. exact prepared manifest/shards;
5. exact P1 dataset receipt/payload hashes;
6. exact trainable census;
7. a real backward-only smoke must prove finite non-zero gradients reach **both** `scene_core.scene` and `scene_core.field`;
8. frozen module hashes are captured.

During every P2 optimizer step, scene planes are recomputed by `model.encode_scene(...)`; cached/detached P1-style planes are forbidden for the live P2 training forward.

## Training

- 4,800 carrier steps;
- deterministic Fit8 round-robin;
- absolute-step seeded sampler;
- batch 2,048;
- scene LR 3e-5 peak / 3e-6 minimum;
- field+aux LR 1e-4 peak / 1e-5 minimum;
- exact resumable checkpoints;
- frozen-module hashes must remain unchanged after training.

## Independent court

Fresh equal-capacity probes are trained independently on:

- `P2_H_XYZ`;
- `P1_H_XYZ`;
- `V8_H_XYZ`;
- `XYZ_ONLY`;
- `P2_SHUFFLED_H_XYZ`.

The absolute metric gates remain unchanged from P1:

- mean distance MAE <= 0.025 IRIS;
- worst-subject endpoint p95 <= 0.075 IRIS;
- worst-subject normal alignment >= 0.70;
- worst-subject near balanced accuracy >= 0.80.

P2 must also materially beat P1: >=15% distance reduction, >=15% endpoint reduction, >=0.03 worst-normal gain, and >=0.04 mean near-BA gain, while retaining local-alignment and bounded V8 representation drift.

Thresholds may not change after results.

## Decision

### PASS

`DESIGN_NEW_HIGH_RES_DIRECT_SURFACE_EXPERIMENT_FROM_EXISTING_IRIS_ARCHITECTURE__DO_NOT_TRAIN_TESSA_YET`

Interpretation: the existing IRIS architecture has sufficient metric-geometry capacity when upstream scene and local field are jointly supervised. Do not add an explicit geometry branch merely because P1 failed.

### FAIL

`PREREGISTER_EXPLICIT_GEOMETRY_CARRIER_BRANCH_EXPERIMENT__DO_NOT_TRAIN_TESSA`

Interpretation: field-only and scene+field controls of the existing architecture both failed the locked metric court; a dedicated explicit geometry carrier/branch is then scientifically justified for a **new preregistration**, not automatically authorized for training.

Neither P2 outcome directly authorizes SURF-A2 or TESSA.
