# RealSaS-OPT — Current Plan

**Date:** 2026-10-20  
**Status:** **H-GEO-P3 CLOSED SCIENTIFIC FAIL; H-GEO-P3-EFREQ ZERO-OPTIMIZER DIAGNOSTIC ACTIVE**  
**Production authority:** `main`  
**Research review branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Source-gate vehicle:** draft PR #129  
**Active lane:** `SURFACE_TESSA`  
**Active experiment:** `H-GEO-P3-EFREQ`  
**Parked lane:** `RELATION_V10`  
**Optimizer policy:** **no optimizer run is authorized.** No P4, SURF-A2 or TESSA optimizer step is authorized until the zero-optimizer endpoint audit selects the next preregistration family.

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

## Closed evidence

- `SURF-A0-RAWFIELD`: falsified historical raw-field transition.
- `SURF-C0-NATIVE-R24`: measurement-only native V8 surface debt.
- `SURF-A1-FROZEN-H`: scientific FAIL after substantial extraction rescue.
- `H-REP-P0`: absolute FAIL, strong local geometry information in sealed V8 H.
- `H-GEO-P1`: field-only exact-triangle retraining materially improved V8 but failed locked metric quality.
- `H-GEO-P2`: scene+field retraining passed absolute distance/normal/near and all vs-P1 improvement gates, but failed endpoint p95 and shared-H retention.
- `H-GEO-P3`: dedicated 128² geometry carrier passed absolute distance/normal/near, local alignment and exact shared-V8 preservation, but failed endpoint p95 and endpoint improvement versus P2. Authority: `docs/research/IRIS_H_GEO_P3_RESULT_20261020.json`.

P3 therefore removes the shared-H retention confound but does **not** solve endpoint-tail precision. The same four difficult morphologies remain in the endpoint-fail set, while scalar distance, near-surface evidence and normals are strong. The next question is diagnostic, not another architecture guess.

## Active workstream — H-GEO-P3-EFREQ

This experiment executes **zero optimizer steps** and writes no checkpoint.

### Factor A — target ambiguity / discontinuity

On exact authored teacher triangles:

- non-adjacent surface gap at the closest point;
- nearest-face/non-adjacent-face flips under deterministic `±epsilon` perturbations;
- closest-point and offset-vector directional change.

This tests whether the nearest-point vector target itself changes discontinuously around thin sheets, competing surfaces or medial regions.

### Factor B — frozen view-evidence resolution

Reconstruct the normalized orthographic cameras from the exact prepared 13-D camera features and measure for each nearest triangle:

- in-frame view count;
- informative view count with `|normal·forward| >= 0.25`;
- maximum/median projected triangle area in source pixels;
- maximum projected triangle edge length.

These are geometric coverage/footprint proxies only. They do not claim true occlusion visibility.

### Factor C — carrier spatial bandwidth

At `epsilon = 1/256, 1/128, 1/64` IRIS:

- exact target-vector variation;
- P2-H whitened finite-difference response;
- P3-H_geo whitened finite-difference response.

P3 internals are also audited:

- 128² refiner residual RMS relative to bilinear base;
- refiner residual high-band FFT power fraction;
- learned continuous-position residual RMS.

### Query court

The P3 checkpoint training aux head is used **only** to rank each subject's endpoint tail. It has no qualification authority.

- exact P1 holdout query bytes;
- distance `<=0.12` IRIS;
- top 10% P3-aux endpoint-error tail, max 384/subject;
- deterministic distance-stratified median-error controls;
- exact teacher mesh normalization replay required;
- no model or probe training.

## Locked decision boundary

The audit outputs exactly one of:

```text
TARGET_AMBIGUITY_DOMINANT
VIEW_EVIDENCE_DOMINANT
CARRIER_BANDWIDTH_DOMINANT
MIXED
UNRESOLVED
```

That classification selects a **future preregistration family only**:

- target ambiguity → vector-field / multi-hypothesis target study;
- view evidence → true multi-scale view-evidence geometry study;
- carrier bandwidth → true high-frequency geometry-carrier study;
- mixed → preregister a factorial P4 ablation;
- unresolved → expand zero-optimizer diagnostics.

No result directly authorizes SURF-A2 or TESSA.

## Execution order

1. Seal H-GEO-P3 evidence in repo. **DONE**
2. EFREQ exact input binding + preregistration. **DONE**
3. EFREQ pure analysis helpers + audit source. **DONE**
4. Dedicated + broad source CI. **ACTIVE — PR #129**
5. Freeze exact final green EFREQ source SHA. **NEXT**
6. Fast-forward active research branch to that exact SHA.
7. Generate V8-style A100 Run-All EFREQ notebook pinned to that SHA.
8. Run audit: optimizer steps must remain exactly zero.
9. Read `H_GEO_P3_EFREQ_AUDIT.json` and query table.
10. Preregister one next geometry study family from the locked classification.

## Notebook contract

Physical phase cells, exactly one Drive mount, no Drive-tree discovery, no second OAuth, exact direct paths/hashes, source precompile + CLI smoke, exact A100 gate, visible logs, fail closed, zero optimizer steps, no checkpoint writes.
