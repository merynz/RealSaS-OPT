# RealSaS-OPT — Current Plan

**Date:** 2026-10-20  
**Status:** **H-GEO-P1 CLOSED SCIENTIFIC FAIL; H-GEO-P2-SCENE-FIELD ACTIVE**  
**Production authority:** `main`  
**Research review branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Source-gate vehicle:** draft PR #126  
**Active lane:** `SURFACE_TESSA`  
**Active experiment:** `H-GEO-P2-SCENE-FIELD`  
**Parked lane:** `RELATION_V10`  
**Optimizer policy:** only P2 may optimize after exact source/input/A100/gradient-connectivity admission. No SURF-A2 or TESSA optimizer step is authorized.

## Objective

Build the IRIS→TESSA boundary honestly:

```text
8 views
  ↓
IRIS continuous evidence / metric geometry
  ↓
qualified high-fidelity direct surface
  ↓
OPTIONAL TESSA structuralization
  ↓
Compiler
  ↓
ONE canonical animation-ready M
```

IRIS must establish correct/high-fidelity shape before TESSA is evaluated. TESSA may improve topology/structure; it may not be used as a second reconstructor to hide IRIS geometry debt.

## Evidence already closed

- `SURF-A0-RAWFIELD`: falsified historical raw-field transition.
- `SURF-C0-NATIVE-R24`: measurement-only native V8 surface debt.
- `SURF-A1-FROZEN-H`: scientific FAIL after substantial extraction rescue.
- `H-REP-P0`: scientific FAIL absolute, but strong local geometry information demonstrated in sealed V8 H.
- `H-GEO-P1`: scientific FAIL absolute; exact-triangle field-only training materially beat V8 on distance, near-surface classification and normals, but endpoint improvement and all absolute qualification gates remained insufficient.

P1 therefore showed that geometry supervision works, but because upstream `scene_core.scene` was frozen it did not justify architecture expansion by itself.

## Active workstream — H-GEO-P2-SCENE-FIELD

P2 is an **existing-architecture control**, not the explicit geometry branch.

Fixed controls:

- initialize from exact immutable V8, not P1;
- reuse exact P1 triangle-dataset bytes, no rebuild or resampling;
- no architecture expansion;
- train only existing `scene_core.scene + scene_core.field` plus the training-only metric head;
- freeze view bank, cube extraction, mechanics, appearance, relations and structure;
- no subject-specific trainable parameters;
- no TESSA;
- teacher mesh is target-only.

Runtime admission must prove both scene and field receive finite non-zero gradients before optimizer step 1. Live scene planes are recomputed inside each training step so upstream geometry gradients cannot be accidentally detached.

Training is preregistered for 4800 steps with separate lower scene LR and field LR, absolute-step seeded resume semantics and frozen-module hash non-regression.

## Independent court

Fresh equal-capacity probes compare:

```text
P2_H_XYZ
P1_H_XYZ
V8_H_XYZ
XYZ_ONLY
P2_SHUFFLED_H_XYZ
```

Absolute gates stay unchanged from P1:

- mean distance MAE `<=0.025` IRIS;
- worst endpoint p95 `<=0.075` IRIS;
- worst normal alignment `>=0.70`;
- worst near balanced accuracy `>=0.80`.

P2 must also beat P1 by preregistered margins and retain local-alignment plus bounded V8 representation drift. Thresholds may not change after results.

## Decision boundary

```text
P2 PASS
→ existing IRIS architecture has sufficient metric-geometry capacity
→ design a new high-resolution direct-surface experiment
→ DO NOT train TESSA yet

P2 FAIL
→ field-only + scene+field controls both miss qualification
→ preregister an explicit geometry-carrier branch experiment
→ DO NOT train TESSA
```

Neither branch directly authorizes SURF-A2.

## Execution order

1. Seal H-GEO-P1 evidence in repo. **DONE**
2. P2 input binding + preregistration. **DONE**
3. P2 runtime/preflight/trainer/court source. **DONE**
4. Dedicated + broad source CI. **ACTIVE**
5. Freeze exact final green source SHA. **NEXT**
6. Fast-forward active research branch to that exact SHA.
7. Generate V8-style A100 Run-All notebook pinned to that SHA.
8. Run P2 and inspect only the independent court.
9. Advance to direct-surface design or explicit-geometry-branch preregistration strictly from that result.

## Notebook contract

Physical phase cells, exactly one Drive mount, no Drive-tree discovery, no second OAuth, exact direct paths/hashes, source precompile + CLI smoke, exact A100 gate, deterministic resumability, visible logs, fail closed, independent court.
