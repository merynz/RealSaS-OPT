# RealSaS-OPT — Current State

**Date:** 2026-10-20  
**Production authority:** `main` until reviewed merge  
**Active research branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Source-gate vehicle:** draft PR #129 — `IRIS: H-GEO-P3 endpoint-frequency diagnostic`  
**Active research lane:** `SURFACE_TESSA`  
**Active experiment:** `H-GEO-P3-EFREQ`  
**Runtime source:** pending final green freeze  
**Parked lane:** `RELATION_V10` — not cancelled  
**Scientific status:** **NO GLOBAL IRIS FIT8 PASS / NO TESSA PASS / SURF-A0 FALSIFIED / SURF-C0 MEASURED / SURF-A1 FAIL / H-REP-P0 FAIL WITH LOCAL SIGNAL / H-GEO-P1 FAIL WITH MATERIAL RESCUE / H-GEO-P2 FAIL WITH 3-OF-4 ABSOLUTE GEOMETRY GATES PASS + SHARED-H RETENTION FAIL / H-GEO-P3 FAIL WITH 3-OF-4 ABSOLUTE GEOMETRY GATES PASS + EXACT SHARED-V8 PRESERVATION / EFREQ ZERO-OPTIMIZER AUDIT ACTIVE**  
**Optimizer policy:** **zero optimizer steps are authorized in H-GEO-P3-EFREQ.** No P4, SURF-A2 or TESSA optimizer step is authorized.

Machine-readable continuation authority: `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json`.
Platform execution lanes: `docs/platform/EXECUTION_LANES.md` (`PLATFORM_EXECUTION_LANES`).

## 1. Product invariant

```text
8-view observations
        ↓
IRIS shared visual / semantic / mechanical evidence
        +
qualified explicit metric geometry evidence
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

There is no independent deformable visual mesh. TESSA is not a second reconstructor and may not repair an unqualified upstream surface. Compiler alone mints canonical M.

## 2. Immutable V8 authority

- source `5083c9915533ec8ec6530c7d229c635887abfea7`;
- step `12000`;
- checkpoint SHA256 `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding SHA256 `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`.

Later experiments do not rewrite V8 history.

## 3. Closed evidence chain

### SURF-A0 / SURF-C0 / SURF-A1
Historical raw-field transition is falsified; native V8 extraction is measured diagnostic debt; SURF-A1 improved extraction substantially but failed locked high-fidelity surface gates. None authorizes TESSA.

### H-REP-P0
Extractor-free probes proved real local geometry information in sealed V8 H, but absolute metric geometry gates were not earned.

### H-GEO-P1
Exact source `4a4236db120ffec32dc33d278341e21f58d93529`; 3600 steps. Field-only exact-triangle retraining materially improved V8 but failed locked metric quality.

### H-GEO-P2
Exact execution source `3a572989ab379975ed7cda5f0dd8b6f33de246b3`; 4800 steps. Scene+field retraining passed absolute distance, normal and near gates plus all four vs-P1 improvement gates, but failed endpoint p95 and shared-H retention.

### H-GEO-P3 — sealed scientific FAIL
Authority: `docs/research/IRIS_H_GEO_P3_RESULT_20261020.json`.

- exact execution source `e3c48678f3f563bb60624278e206f4ba8f71c272`;
- 4800 carrier steps completed;
- checkpoint SHA256 `95af736e1e9163cad9d568ea9c67770fe2fbf3391bbdb72d8c03c472b5b5d74b`;
- P2 checkpoint comparison-only, never initialization;
- exact P1 triangle dataset reused;
- shared V8 state and anchor H preserved exactly;
- teacher truth absent from inference; TESSA not invoked.

Independent court:

| Metric | P3 | Locked gate | Result |
|---|---:|---:|---|
| mean distance MAE | `0.01678` | `<=0.025` | PASS |
| worst endpoint p95 | `0.08786` | `<=0.075` | FAIL |
| worst normal abs-dot | `0.76790` | `>=0.70` | PASS |
| worst near BA | `0.91524` | `>=0.80` | PASS |

P3 also passed distance/normal/near non-regression versus fresh-court P2, local-alignment controls and exact shared-preservation gates, but endpoint worsened versus P2 (`0.08483 → 0.08786`). Endpoint-fail subjects remain Snake, Horse, Spider and GreenBlob.

This falsifies the preregistered claim that V8-weight-copy + 64→128 residual refinement + low-frequency continuous XYZ residual is sufficient to close endpoint tail precision. It does **not** falsify the dedicated geometry-carrier concept.

## 4. Active experiment — H-GEO-P3-EFREQ

Question:

> Why do exact closest-surface endpoint tail errors persist after distance/normal/near qualification and exact shared-V8 preservation?

Authority:

- `canonical/H_GEO_P3_EFREQ_INPUT_BINDING_V1.json`;
- `canonical/H_GEO_P3_EFREQ_PREREGISTRATION_V1.json`;
- `docs/research/IRIS_H_GEO_P3_EFREQ_DECISION_LEDGER_20261020.md`;
- `tools/iris_vnext/h_geo_p3_efreq_v1.py`;
- `tools/iris_vnext/audit_h_geo_p3_efreq_v1.py`.

This is a **zero-optimizer diagnostic**. It compares endpoint-tail queries to deterministic distance-matched controls using:

- non-adjacent surface gap and closest-face instability from exact authored teacher triangles;
- perturbation-scale target-vector instability at `1/256`, `1/128`, `1/64` IRIS;
- P2-H vs P3-H_geo whitened finite-difference response;
- 128² refiner residual RMS + high-band power;
- learned continuous-position residual magnitude;
- orthographic in-frame/informative-view counts and projected nearest-triangle footprint proxies.

The P3 training aux is allowed only to rank tail queries. It cannot requalify P3 and no fresh probe is trained.

Locked output classification:

- `TARGET_AMBIGUITY_DOMINANT`;
- `VIEW_EVIDENCE_DOMINANT`;
- `CARRIER_BANDWIDTH_DOMINANT`;
- `MIXED`;
- `UNRESOLVED`.

No classification itself authorizes a new optimizer run.

## 5. SURF-A2 / TESSA

`SURF-A2-HIRES` remains blocked on the endpoint-frequency diagnosis and a subsequent preregistered geometry decision.

`TESSA-V2-B1` remains blocked until a genuinely qualified high-resolution direct A-arm exists. TESSA may structuralize correct geometry; it may not repair unqualified IRIS geometry.

## 6. Notebook worker law

```text
open notebook
→ Run all
→ approve Google Drive exactly once
→ exact-hash autonomous execution
→ separated visible phases
→ zero optimizer steps
→ diagnostic result + query table
```

No full-Drive discovery, second OAuth, hidden fallback, source drift or threshold relaxation is allowed.

## 7. Immediate continuation

```text
seal P3 result                                      [DONE]
        ↓
EFREQ binding + preregistration                     [DONE]
        ↓
zero-optimizer audit source + dedicated CI          [ACTIVE — PR #129]
        ↓
freeze exact final green EFREQ SHA                  [NEXT]
        ↓
fast-forward active research branch
        ↓
A100 Run-All EFREQ notebook
        ↓
inspect classification + query table
        ↓
preregister exactly one next geometry study family
```
