# RealSaS-OPT — Current Plan

**Date:** 2026-10-20  
**Status:** **H-GEO-P3 CLOSED SCIENTIFIC FAIL; H-GEO-P3-EFREQ SEALED UNRESOLVED; H-GEO-P3-SCALE-XFER ZERO-OPTIMIZER STAGE LOCALIZATION ACTIVE**  
**Production authority:** `main`  
**Research review branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Source-gate vehicle:** draft PR #130  
**Active lane:** `SURFACE_TESSA`  
**Active experiment:** `H-GEO-P3-SCALE-XFER`  
**Parked lane:** `RELATION_V10`  
**Optimizer policy:** **no optimizer run is authorized.** No P4, SURF-A2 or TESSA optimizer step is authorized until SCALE-XFER localizes the first evidence-loss/ambiguity stage and a new experiment is separately preregistered.

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

IRIS must establish correct/high-fidelity shape before TESSA is evaluated. TESSA may improve topology/structure; it may not hide IRIS geometry debt.

## Closed evidence

- `SURF-A0-RAWFIELD`: falsified.
- `SURF-C0-NATIVE-R24`: native V8 surface debt measured.
- `SURF-A1-FROZEN-H`: scientific FAIL after substantial extraction rescue.
- `H-REP-P0`: absolute FAIL with strong local geometry information in sealed V8 H.
- `H-GEO-P1`: field-only exact-triangle retraining materially improved V8 but failed locked metric quality.
- `H-GEO-P2`: scene+field retraining passed distance/normal/near and all vs-P1 gates, but endpoint p95 and shared-H retention failed.
- `H-GEO-P3`: dedicated geometry carrier passed distance/normal/near and exact shared-V8 preservation, but endpoint p95 failed and worsened versus P2.
- `H-GEO-P3-EFREQ`: zero-optimizer audit completed `UNRESOLVED`. No locked single dominance condition passed; nevertheless P3 was smoother than P2 at fine scale, endpoint tails had smaller projected footprints, and non-adjacent surface competition increased.

EFREQ authority: `docs/research/IRIS_H_GEO_P3_EFREQ_RESULT_20261020.json`.

## Active workstream — H-GEO-P3-SCALE-XFER

This experiment executes **zero optimizer/probe steps** and writes no checkpoint.

### Fixed query court

Reuse `H_GEO_P3_EFREQ_QUERY_TABLE.csv` byte-for-byte. Tail/control membership, distance matching and endpoint ranking may not be recomputed.

### Factor A — actual frozen view-token separability

For each exact query:

- take exact nearest authored-surface point;
- find closest non-adjacent competing authored-surface point;
- project both through the exact eight normalized cameras;
- bilinearly sample the existing `37×37×384` frozen DINO token grid with `align_corners=False` semantics;
- measure same-token-cell fraction, distinct-token view count, token cosine distance and projected separation in token-cell units.

This tests spatial evidence resolution directly rather than relying only on triangle-footprint proxies. It is still not a claim of true occlusion visibility.

### Factor B — exact multi-surface target structure

From authored teacher triangles measure:

- nearest-vs-competing distance margin;
- competing surface normal angle up to sign;
- whether another non-adjacent surface lies within `1/256`, `1/128` or `1/64` IRIS of the exact nearest distance.

### Factor C — stage-by-stage carrier transfer

Apply deterministic ± perturbations at:

`1/1024, 1/512, 1/256, 1/128, 1/64, 1/32` IRIS.

Measure per-subject channel-whitened RMS response at:

1. P2 H baseline;
2. P3 sampled 64² base planes;
3. P3 sampled bilinear 128² planes;
4. P3 sampled learned-refined 128² planes;
5. P3 geometry-field H;
6. P3 continuous-position residual;
7. P3 final H_geo.

The purpose is to find the **first** stage where a distinction present upstream becomes inadequate downstream.

## Locked decision boundary

SCALE-XFER outputs exactly one of:

```text
TARGET_MULTISURFACE_LIMITED
VIEW_INPUT_RESOLUTION_LIMITED
CARRIER_TRANSFER_LIMITED
MIXED
UNRESOLVED
```

That classification selects a **future preregistration family only**:

- target → multi-surface / multi-hypothesis geometry target P4;
- view input → multiscale / higher-spatial-resolution view-evidence P4;
- carrier → evidence-preserving geometry-carrier P4;
- mixed → minimal joint P4 containing only identified failing stages;
- unresolved → expand zero-optimizer localization.

No result directly authorizes SURF-A2 or TESSA.

## Execution order

1. Seal H-GEO-P3 evidence. **DONE**
2. Run EFREQ zero-optimizer audit. **DONE — UNRESOLVED**
3. Seal EFREQ audit + exact query CSV hashes. **DONE**
4. SCALE-XFER exact input binding + preregistration. **DONE**
5. SCALE-XFER pure helpers + runtime audit source. **DONE**
6. Dedicated + broad source CI. **ACTIVE — PR #130**
7. Freeze exact final green SCALE-XFER source SHA. **NEXT**
8. Fast-forward active research branch to that exact SHA.
9. Generate V8-style A100 Run-All SCALE-XFER notebook pinned to that SHA.
10. Run audit: optimizer/probe optimizer must remain exactly zero.
11. Read stage-localized classification + query table.
12. Preregister exactly one minimal P4 family; training remains separately gated.

## Notebook contract

Physical phase cells, exactly one Drive mount, no Drive-tree discovery, no second OAuth, exact direct paths/hashes, source precompile + CLI smoke, exact A100 gate, visible logs, fail closed, zero optimizer/probe steps, no checkpoint writes.
