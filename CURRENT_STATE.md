# RealSaS-OPT — Current State

**Date:** 2026-10-19  
**Production authority:** `main` until reviewed merge  
**Active research branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Review vehicle:** draft PR #122 — `IRIS: TESSA V2 structuralization + geometry-explicit evidence redesign`  
**Active research lane:** `SURFACE_TESSA`  
**Parked lane:** `RELATION_V10` — not cancelled, no optimizer authorization  
**Scientific status:** **NO GLOBAL IRIS FIT8 PASS / NO TESSA PASS / SURF-A0 RAW-FIELD BASELINE FALSIFIED / SURF-C0 NATIVE V8 CONTROL MEASURED / SURF-A1 FROZEN-H SOURCE + PREREG IMPLEMENTED, A100 RUNTIME NOT YET EXECUTED**  
**Optimizer authorization:** **SURF-A1 step 1 is permitted only after its exact A100 zero-step admission receipt passes. TESSA V2 and Relation V10 remain blocked.**

## 1. Authority law

1. `main` remains merged product-code authority.
2. Completed V8 is immutable model/history authority: source `5083c9915533ec8ec6530c7d229c635887abfea7`, step `12000`, checkpoint `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`, binding `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`.
3. Completed V9 is immutable focused-relation history; it is not a global IRIS V9 authority.
4. PR #122 is review/research source only. Green source CI is not scientific green.
5. IRIS owns learned scene/evidence; TESSA may propose explicit structured meshes; **Compiler is the only authority that may mint canonical M**.
6. Teacher topology/rig/skin may supervise training truth but may not enter inference conditioning.
7. Fit8 is a shared-system qualification cohort, not an unseen-generalization claim.

Machine-readable continuation authority: `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json`.

## 2. Product invariant and long-term objective

The target remains one generic animation-ready carrier:

```text
8-view observations
        ↓
IRIS visual/geometric/mechanical continuous carrier H/evidence
        ↓
qualified high-fidelity direct surface
        ↓
OPTIONAL TESSA structuralization / learned retopology
        ↓
Compiler qualification
        ↓
ONE canonical animation-ready M
        ↓
AXIS / MIRA / presentation-runtime consumers
```

There is no second deformable visual mesh. TESSA survives only if it improves useful topology/structural economy without sacrificing source fidelity or downstream mechanics. GSA is not frozen as permanent architecture; its future role must be derived from what IRIS/TESSA/Compiler can prove generically.

Research synthesis remains deliberate:

- StdGEN++-class lesson: preserve geometry/raster supervision through representation transitions and resolve thin/high-resolution structure explicitly;
- MeshAnythingV2-class lesson: a high-fidelity extracted surface is not automatically the best animation/production topology;
- RealSaS rule: neither external method owns authority; all claims must survive our typed evidence, Compiler and downstream courts.

## 3. Fit8 role

Fit8 is not valuable only because it contains eight different subjects. It forces the **same architecture, same contracts, same Compiler boundaries and same execution machinery** to work across heterogeneous assets without subject-specific parameters or semantic shortcuts.

Current Fit8 prepared manifest remains exact V4 authority with eight subjects and shared sampling policy. The program uses Fit8 to remove one-subject assumptions before future large unseen campaigns; no current Fit8 result claims 10k-unseen generalization.

## 4. V9 relation result and parked V10

V9 focused relation completed 1600 steps at source `7029ee3f48787aa9f7e27eecef89155eb9368a96`.

Earned scoped result:

- projected alpha-boundary BA ~`0.7599`;
- projected appearance-boundary BA ~`0.8209`;
- front ~`0.5588`;
- pairwise visibility rows ~`0.55–0.57` and not valid global-visibility proof;
- occlusion support was positive-only.

Relation V10 remains parked until all-hit/ray-stack truth, authored-motion conditioning disposition, Truth V3 validity/support, initialization, split and readiness gates are independently sealed.

## 5. SURF-A0-RAWFIELD — falsified

The exact V8 raw scalar field was adapted to the historical distance-like `T512 → sparse T1024 → indexed MT` decoder with no extraction delta, no learned FlexiCubes cube parameters, no TESSA and zero optimizer steps.

A100 preflight passed; Snake then failed closed at apparatus source `d9701923b2c19bd1c9b9771fe071bb92264a041d`:

```text
INDEXED_MT_BUILD_AND_NORMALS
SparseSurfaceTruncationError
SPARSE_SHELL_TRUNCATES_ZERO_SURFACE:62735467 boundary crossing tetrahedra
```

Measured before termination included `6,525,575` refined coarse cells, `313,227,600` candidate fine tetrahedra, `95,514,148` crossing tetrahedra and `125,076,400` exact faces.

Decision: **raw V8 field + historical distance-like sparse selector is falsified as the direct A-arm baseline.** Do not widen the shell, disable fail-closed truncation, flip signs or reinterpret the failed candidate as success.

## 6. SURF-C0-NATIVE-R24 — completed diagnostic

The exact surface path that actually received V8 G2 supervision was then materialized for all eight subjects:

```text
exact V8 scene planes
        ↓
field + learned extraction_delta + H
        ↓
learned 21-D cube parameters
        ↓
pinned FlexiCubes @ R24
```

Runtime authority:

- apparatus source `ccb71011f41b041bc5c137544bc1b97ca394ea9c`;
- Drive evidence root `RealSaS_IRIS_TESSA_V2_FIT8_20261019/ccb71011f41b/V5_NATIVE_V8_CONTROL`;
- batch receipt SHA256 `a83c0c466d5251c8cc1bc13475866c766c4796e7b0b08c7b96d031c0662667de`;
- control court SHA256 `656f674e12cf3a4ffe79fd16dd25f8ffe4949f736a8e39c9dcd4987df19f3617`;
- optimizer steps `0`; TESSA `false`; high-resolution capability claim `false`.

Global measurement:

- mean subject mask IoU `0.5605675`;
- minimum subject mask IoU `0.4053549`;
- minimum view IoU `0.3425416`;
- maximum mask gap `392.4207 px`;
- mean symmetric Chamfer `0.0612046` IRIS units.

Fragmentation is also material on compact-teacher subjects: Snake `3 → 85` components, T-Rex `1 → 59`, Spider `15 → 88`, GreenBlob `4 → 92`, Shark `1 → 67`; Bat is close at `63 → 65`.

There was no preregistered PASS/FAIL threshold for this diagnostic, so it remains measurement-only. Engineering interpretation is nevertheless clear enough to set the next experiment:

> **native V8 surface quality debt is present, and high-resolution transition debt remains separately open.**

This does not by itself falsify sealed H.

## 7. SURF-A1-FROZEN-H — active experiment

Exact hypothesis:

> sealed V8 H contains enough geometric information for a new resolution-aware, raster-supervised extraction decoder to rescue surface quality without rewriting the V8 carrier.

New authority files:

- `canonical/SURF_A1_FROZEN_H_INPUT_BINDING_V1.json`;
- `canonical/SURF_A1_FROZEN_H_PREREGISTRATION_V1.json`;
- `canonical/COLAB_RESEARCH_WORKER_UX_V1.json`.

Source boundary:

- `models/iris/vnext/surface_rescue_v1.py` — extraction-only residual decoder;
- `models/iris/vnext/surface_rescue_raster_v1.py` — explicit RealSaS camera → PyTorch3D differentiable raster apparatus;
- `models/iris/vnext/surface_rescue_training_v1.py` — fixed R24/R32/R48 schedule and objective;
- `tools/iris_vnext/build_fit8_surface_raster_truth_v1.py` — exact Stage01 mask + authored-rest depth/normal sidecar;
- `tools/iris_vnext/preflight_surf_a1_a100_v1.py` — zero-step admission, no optimizer;
- `tools/iris_vnext/train_fit8_surf_a1_frozen_h_v1.py` — resumable shared 2400-step decoder fit;
- `tools/iris_vnext/audit_fit8_surf_a1_representation_v1.py` — independent final R48 court.

Hard rules:

- V8 `scene_core/H` is frozen and hash-checked;
- no appearance/relation/mechanics/topology-head training;
- no teacher topology/rig/skin in inference forward;
- no subject-specific trainable parameters;
- inherited V8 deformation is converted to **cell-relative** semantics rather than naively reusing absolute `0.03` at finer grids;
- zero-step R24 must reproduce native V8 faces exactly and vertices within `2e-6` IRIS units;
- PyTorch3D hard coverage must match independent first-party teacher-geometry center raster with precision/recall ≥ `0.98` on all 64 views before step 1;
- source masks remain a separate silhouette target, not raster-apparatus truth;
- no reduced-resolution/capacity fallback after observing failure.

Training schedule if and only if preflight passes:

```text
R24: 400 steps
R32: 600 steps
R48: 1400 steps
TOTAL: 2400 shared-decoder steps
```

The final A1 court is preregistered before runtime. A1 PASS only earns permission to design `SURF-A2-HIRES`; it does **not** authorize TESSA.

## 8. TESSA V2 state

TESSA V2 source architecture remains implemented but training is blocked.

Reference-class architecture remains 1024 width / 16 heads / 24 layers / 4096 FFN with 512 scene + 512 surface/mechanics memory tokens, local 4096 causal context plus causal global-history summaries, strict free-running grammar and independent decoded-mesh courts.

`canonical/TESSA_V2_FIT8_PREREGISTRATION_V1.json` now explicitly requires a **qualified `SURF-A2-HIRES` direct A-arm**. Historical raw T512/T1024, native R24 control, or SURF-A1 may not be substituted as TESSA training surface authority.

## 9. Notebook / execution UX contract

Research notebooks are compute workers, not scientific authority.

Required operator experience:

```text
open notebook
→ Run all
→ approve Google Drive exactly once
→ autonomous hash-bound execution
→ meaningful progress logs
→ resumable Drive checkpoint/receipt writes
```

Forbidden: second Google OAuth UI, `auth.authenticate_user()`, interactive rclone/gcloud auth, hidden low-resolution escape or manual cell sequencing. Source must be exact-SHA pinned, precompiled and source-tested before A100 optimizer admission.

## 10. Immediate continuation state machine

```text
SOURCE CONTRACT + CI GREEN
        ↓
BUILD/VERIFY SURF-A1 RASTER TRUTH
        ↓
A100 ZERO-STEP PREFLIGHT
  - exact source/input hashes
  - A100 resource gate
  - native R24 parity 8/8
  - raster parity 64/64
  - frozen scene_core hash
        ↓ PASS ONLY
SURF-A1 2400-STEP SHARED DECODER FIT
        ↓
INDEPENDENT R48 REPRESENTATION COURT
        ↓
PASS → DESIGN SURF-A2-HIRES DIRECT A-ARM
FAIL → DEDICATED H REPRESENTATION INVESTIGATION
        ↓
NO TESSA UNTIL QUALIFIED SURF-A2
```

## 11. Required read order

1. `AGENTS.md`
2. `CURRENT_STATE.md`
3. `CURRENT_PLAN.md`
4. `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json`
5. `canonical/SURF_A1_FROZEN_H_PREREGISTRATION_V1.json`
6. `docs/research/IRIS_TESSA_V2_DECISION_LEDGER_20261019.md`
7. `docs/research/IRIS_TESSA_V2_STRUCTURALIZATION_PLAN_20261019.md`
8. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
9. `docs/research/IRIS_V9_FOCUSED_RELATION_RESULT_20261019.json`
