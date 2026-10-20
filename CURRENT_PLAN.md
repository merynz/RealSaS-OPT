# RealSaS-OPT — Current Plan

**Date:** 2026-10-20  
**Status:** **H-GEO-P2 CLOSED SCIENTIFIC FAIL; H-GEO-P3-EXPLICIT-HIRES-CARRIER ACTIVE**  
**Production authority:** `main`  
**Research review branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Source-gate vehicle:** draft PR #128  
**Active lane:** `SURFACE_TESSA`  
**Active experiment:** `H-GEO-P3-EXPLICIT-HIRES-CARRIER`  
**Parked lane:** `RELATION_V10`  
**Optimizer policy:** only P3 may optimize after exact source/input/A100/gradient-connectivity/shared-V8-preservation admission. No SURF-A2 or TESSA optimizer step is authorized.

## Objective

Build the IRIS→TESSA boundary honestly:

```text
8 views
  ↓
shared IRIS scene/evidence
  + dedicated metric geometry evidence if required
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
- `H-GEO-P1`: exact-triangle field-only retraining materially improved V8 but failed locked metric quality.
- `H-GEO-P2`: exact-triangle scene+field retraining completed 4800 steps. It passed absolute distance, normal and near gates plus every vs-P1 improvement gate, but failed endpoint p95 and shared-H retention. Authority: `docs/research/IRIS_H_GEO_P2_RESULT_20261020.json`.

P2 therefore establishes two things simultaneously: existing scene+field has substantial metric-geometry capacity, and forcing the remaining high-frequency endpoint precision into shared H creates measurable retention tension. The preregistered next action is an explicit geometry-carrier branch study.

## Active workstream — H-GEO-P3-EXPLICIT-HIRES-CARRIER

P3 is an **architecture study**, not a product-model promotion.

Fixed controls:

- exact immutable V8 remains the shared model;
- shared V8 is frozen byte-exact and absent from the optimizer;
- the dedicated geometry branch initializes by copying V8 scene and field weights, never from P2;
- P2 checkpoint is comparison-only in the final court;
- exact P1 authored-triangle dataset bytes are reused, with no rebuild/resampling;
- geometry branch gets a zero-residual 64→128 triplane refiner and a zero-initialized continuous XYZ Fourier residual;
- no subject-specific trainable parameters;
- no TESSA;
- teacher mesh is target-only.

At step 0 the new high-resolution path must be an exact bilinear upsample plus zero residual, and the XYZ residual must be exactly zero. Preflight must prove finite non-zero gradients reach geometry scene, geometry field, refiner and position projection while shared V8 receives none.

Training budget is 4800 steps, matching P2. Scene LR remains lower than field/new-module LR. Resume is absolute-step seeded and hash-bound.

## Independent court

Fresh equal-capacity probes compare:

```text
P3_GEO_XYZ
P2_H_XYZ
V8_H_XYZ
XYZ_ONLY
P3_SHUFFLED_GEO_XYZ
```

Absolute gates remain the same locked quality bar:

- mean distance MAE `<=0.025` IRIS;
- worst endpoint p95 `<=0.075` IRIS;
- worst normal alignment `>=0.70`;
- worst near balanced accuracy `>=0.80`.

P3 must also reduce endpoint p95 by at least 10% versus fresh-court P2, hold P2-level distance/normal/near within preregistered non-regression margins, beat XYZ/shuffled local controls, and prove exact shared-V8 state + anchor-H preservation.

## Decision boundary

```text
P3 PASS
→ explicit geometry carrier earns metric-geometry qualification
→ design a new high-resolution direct-surface experiment from H_geo
→ DO NOT train TESSA yet

P3 FAIL
→ audit endpoint-frequency / scale ceiling in explicit carrier
→ do not relax thresholds
→ do not use TESSA as rescue
```

Neither outcome directly authorizes SURF-A2.

## Execution order

1. Seal H-GEO-P2 evidence in repo. **DONE**
2. P3 input binding + preregistration. **DONE**
3. P3 architecture/runtime/preflight/trainer/court source. **DONE**
4. Dedicated + broad source CI. **ACTIVE — PR #128**
5. Freeze exact final green P3 source SHA. **NEXT**
6. Fast-forward active research branch to that exact SHA.
7. Generate V8-style A100 Run-All notebook pinned to that SHA.
8. Run P3 and inspect only the independent court.
9. Advance to direct-surface design or endpoint-frequency audit strictly from that result.

## Notebook contract

Physical phase cells, exactly one Drive mount, no Drive-tree discovery, no second OAuth, exact direct paths/hashes, source precompile + CLI smoke, exact A100 gate, deterministic resumability, visible logs, fail closed, independent court.
