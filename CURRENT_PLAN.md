# RealSaS-OPT — Current Plan

**Date:** 2026-10-19  
**Status:** **POST-V8 SURFACE/TESSA PROGRAM — SURF-A1 FROZEN-H RESCUE ACTIVE**  
**Active source:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Review vehicle:** draft PR #122  
**Production authority:** `main` until reviewed merge  
**Active lane:** `SURFACE_TESSA`  
**Parked lane:** `RELATION_V10`  
**Optimizer policy:** **SURF-A1 only after exact zero-step A100 admission. No TESSA or Relation V10 optimizer step.**

## 0. Program objective

Build one generic visual + geometric + mechanical carrier that is faithful enough to source observations and structurally useful enough for animation:

```text
source observations
   ↓
IRIS continuous learned evidence / H
   ↓
qualified high-fidelity surface
   ↓
optional TESSA structuralization if it proves value
   ↓
Compiler qualification
   ↓
ONE canonical animation-ready M
   ↓
AXIS / MIRA / graphics-motion-runtime consumers
```

The intended synthesis is:

- **StdGEN++-class strength:** high-fidelity learned surface, raster supervision, explicit thin/high-resolution continuity work;
- **MeshAnythingV2-class strength:** explicit structural topology when a dense/extracted witness is not the best production/animation mesh;
- **RealSaS authority law:** neither external system is copied as authority. Compiler remains canonical-M authority, and TESSA is optional until an A/B court earns it.

GSA, AXIS and MIRA must eventually position themselves around the qualified carrier rather than force the carrier to preserve historical module boundaries. GSA may shrink/change/disappear if IRIS/TESSA/Compiler assume its useful responsibilities with stronger typed authority; no such removal is authorized yet.

## 1. Non-negotiable rules

- one canonical deformable `M`; no second visual deformation mesh;
- V8 `scene_core/H` stays sealed during SURF-A1;
- teacher topology/rig/skin never enters model inference conditioning;
- Compiler may qualify but may not rescue model incapacity and turn it into a model PASS;
- source CI is not scientific evidence;
- failed courts are not relaxed after results;
- unknown/unsupported truth abstains;
- Fit8 uses one shared architecture and shared contracts: no subject-specific trainable parameters;
- Fit8 is not an unseen-generalization claim, but it is the current system-genericity forcing function before future large unseen campaigns.

## 2. Historical anchors

### IRIS-V8 — immutable

- source `5083c9915533ec8ec6530c7d229c635887abfea7`;
- step `12000`;
- checkpoint `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`;
- shared scene core sealed after M1.

### IRIS-V9-R — completed scoped relation experiment

Boundary evidence learned; full ordering/visibility did not. Keep as history, not final relation architecture.

### SURF-A0-RAWFIELD — falsified

Exact V8 raw field + historical distance-like T512→sparse-T1024 selector failed closed on Snake. Preserve the failure as evidence; do not repair it into success.

### SURF-C0-NATIVE-R24 — measured

Exact V8 trained FlexiCubes path materialized 8/8. Global mask/geometry and fragmentation measurements show native surface debt; because the control had no preregistered PASS/FAIL threshold it remains diagnostic rather than a formal scientific FAIL.

Decision from A0 + C0:

> Do not jump to TESSA and do not immediately retrain H. First test whether sealed V8 H supports a better extraction decoder.

## 3. Active experiment — SURF-A1-FROZEN-H

### 3.1 Question

Can a new extraction-only decoder, with exact V8 scene/H frozen, recover substantially better surface fidelity when given resolution-aware deformation semantics and source-defensible raster supervision?

This is a representation-capacity separation experiment:

```text
A1 PASS
→ H contains useful geometry that V8 native extraction did not fully exploit
→ proceed to true high-resolution direct A-arm design

A1 FAIL after apparatus/admission gates pass
→ do not train TESSA
→ open dedicated H representation investigation
→ carrier retraining still requires a separate preregistered decision
```

### 3.2 Fixed model boundary

Trainable: `FrozenHSurfaceRescueV1` only.

Frozen: V8 scene core/scene planes, V8 appearance/relation/mechanical/structure domains, and all teacher data from inference forward.

Decoder inputs:

- sealed local H;
- raw V8 field;
- inherited V8 extraction delta expressed in native-cell units;
- normalized XYZ;
- explicit resolution ratio.

Cube output starts as exact V8 21-D raw FlexiCubes parameters plus zero-initialized residual. At zero step R24 the new decoder must reproduce native V8 faces exactly and vertices within `2e-6` IRIS units.

### 3.3 Resolution semantics

No blind reuse of V8 absolute `0.03` extraction delta.

```text
cell_step = 2 / resolution
base_delta_cells = V8_delta / (2/24)
new_absolute_delta = bounded_cell_relative_delta × current_cell_step
```

A1 schedule is fixed before runtime:

- R24 — 400 steps;
- R32 — 600 steps;
- R48 — 1400 steps;
- total — 2400 optimizer steps.

A1 R48 is a representation-rescue court, **not** the final high-resolution direct A-arm.

## 4. SURF-A1 truth and differentiable raster apparatus

Exact immutable inputs are bound by `canonical/SURF_A1_FROZEN_H_INPUT_BINDING_V1.json`.

Training truth sidecar:

- Stage01 source alpha masks → silhouette truth;
- authored rest teacher mesh + exact Stage01 camera → pixel-center teacher geometry coverage;
- teacher depth normalized by common half extent;
- teacher face normal;
- depth/normal valid only on `teacher geometry coverage ∩ source foreground`.

Important separation:

- source mask is **not** used to validate the renderer implementation;
- independent first-party teacher-geometry center coverage is the raster apparatus parity authority;
- PyTorch3D hard raster must achieve precision/recall ≥ `0.98` for all 64 views before optimizer step 1.

PyTorch3D is pinned to commit `978cd99221b9e0a6a568f1d427854d73363265cf`. FlexiCubes remains pinned to `4cc7d6c3d0cee83c011ce36721b81adff0dd7db6`.

## 5. SURF-A1 objective

Fixed objective:

```text
L =
    1.00 * surface_chamfer
  + 1.00 * source_silhouette
  + 0.50 * teacher_depth
  + 0.25 * teacher_normal
  + 0.01 * flexicubes_deviation
  + 0.01 * cell_offset_residual_l2
  + 0.001 * field_residual_l2
  + 0.0001 * cube_parameter_residual_l2
```

Component count is measured independently, not used as a hidden topology gradient in A1.

Optimizer: AdamW `1e-4`, betas `0.9/0.95`, weight decay `0.01`, gradient clip `1.0`, 120-step warmup then cosine to `1e-5`.

## 6. A100 admission — zero optimizer steps

Before SURF-A1 step 1:

1. exact reviewed source SHA and clean checkout;
2. A100-class GPU with ≥39 GiB;
3. exact V8 checkpoint and sealed V4 prepared bytes;
4. exact Stage01 bundle;
5. exact authored-teacher batch;
6. exact pinned FlexiCubes checkout;
7. exact pinned PyTorch3D checkout/import source;
8. raster-truth sidecar hash/shape validation;
9. zero-step R24 native V8 parity for all eight subjects;
10. all 64 teacher-geometry raster parity views pass;
11. full V8 scene-core hash unchanged;
12. optimizer steps executed = 0.

Only a typed `PASS_SURF_A1_A100_PREFLIGHT__TRAINING_AUTHORIZED` receipt may open the trainer.

## 7. Training execution

Use one shared decoder, deterministic subject round-robin, two deterministic rotating views per step. No subject-specific parameters.

Operational rules:

- cache exact frozen V8 scene planes once per subject;
- query frozen V8 local evidence in FP32/no-autocast;
- new decoder may use BF16 autocast;
- FlexiCubes extraction remains FP32;
- log every 10 steps: resolution, subject/views, LR, total + component losses, overlap, grad norm, step seconds and GPU memory;
- checkpoint every 200 steps;
- maintain exactly one hash-bound resumable `checkpoint_latest.pt` + `LATEST.json` in mounted Drive;
- validate source/prereg/input/scene hashes before resume;
- fail on non-finite loss/gradient or frozen-scene mutation.

## 8. Independent A1 scientific court

After step 2400, an evaluator independent of the training objective materializes R48 surfaces for all eight subjects and uses exact Stage01 source masks plus sealed geometry truth.

Preregistered gates:

- all 8 subjects complete and finite;
- non-manifold edges per subject = 0;
- zero-area faces per subject = 0;
- mean symmetric Chamfer ≤ `0.04` IRIS;
- every subject teacher→candidate p95 ≤ `0.075` IRIS;
- global mean subject mask IoU ≥ `0.80`;
- minimum subject mean mask IoU ≥ `0.65`;
- minimum view IoU ≥ `0.55`;
- maximum global mask gap ≤ `128 px`;
- minimum subject mean absolute normal alignment ≥ `0.70`.

Component fragmentation is recorded in A1 but is not an A1 gate; topology authority remains a later direct-A/TESSA decision.

## 9. Next branch only if A1 passes — SURF-A2-HIRES

A1 PASS does not mean final surface capability. It only shows the sealed carrier can support a better decoder.

SURF-A2 must then define a serious high-resolution direct witness with explicit sparse/active-resolution semantics and stronger final admission: high-resolution geometry, mask/depth/normal/source fidelity, topology diagnostics, Compiler static qualification, support binding, G5 multiview and G3 dynamic/mechanical behavior. The A2 thresholds and exact resolution strategy must be preregistered before evaluation.

## 10. TESSA-V2-B1 remains blocked

TESSA trains only after a qualified SURF-A2 direct A-arm exists.

TESSA's job is **structuralization**, not rescue:

```text
A = qualified high-resolution direct IRIS surface → Compiler
B = same admissible evidence/surface → TESSA V2 → Compiler
```

B must preserve fidelity/mechanics/support while providing meaningful animation/production topology or face economy. If it does not, prefer A and remove TESSA from the required product path.

## 11. Relation V10 is parked, not lost

The V10 relation program remains valid but is intentionally parked to stop context switching. When SURFACE_TESSA reaches a clean boundary, resume from Truth V3/all-hit/ray-stack readiness rather than from another ad-hoc optimizer run.

## 12. Fit8 generic-system criterion

Every new SURF/TESSA stage must answer both:

1. does it improve the eight measured subjects? and
2. did it make the **same system** more subject-agnostic?

Required genericity checks include one architecture/config across all subjects, no subject-name or semantic-role branching, exact shared input/output contracts, shared Compiler consumers, generic failure/abstention semantics, source+data-bound receipts and execution resumable without chat context.

This is the bridge from Fit8 to future large unseen evaluation.

## 13. Notebook contract

Notebook generation occurs only from an exact source SHA whose dedicated source gate is green.

Operator UX must be:

```text
Open → Run all → one Drive approval → autonomous execution
```

No second OAuth UI, no `auth.authenticate_user()`, no interactive rclone/gcloud configuration, no manual sequencing. The notebook must precompile current source, run source sanity gates, pin external dependencies, hash-check all scientific inputs, print meaningful phase/progress logs, and resume from exact receipts/checkpoints.

## 14. Immediate execution order

```text
P0. synchronize CURRENT_STATE / CURRENT_PLAN / decision ledger
P1. dedicated SURF-A1 source gate GREEN
P2. freeze exact green SHA
P3. generate single-consent A100 Run-All notebook pinned to that SHA
P4. build/verify raster truth
P5. zero-step A100 preflight
P6. only if PASS → 2400-step SURF-A1 fit
P7. independent R48 court
P8a. PASS → design/preregister SURF-A2-HIRES
P8b. FAIL → H representation investigation; no TESSA
```

No rush-driven gate skipping is allowed.
