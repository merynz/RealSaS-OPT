# RealSaS-OPT — Current State

**Date:** 2026-10-20  
**Production authority:** `main` until reviewed merge  
**Active research branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Source-gate vehicle:** draft PR #128 — `IRIS: H-GEO-P3 explicit high-resolution geometry carrier`  
**Active research lane:** `SURFACE_TESSA`  
**Active experiment:** `H-GEO-P3-EXPLICIT-HIRES-CARRIER`  
**Runtime source:** pending final green freeze  
**Parked lane:** `RELATION_V10` — not cancelled  
**Scientific status:** **NO GLOBAL IRIS FIT8 PASS / NO TESSA PASS / SURF-A0 FALSIFIED / SURF-C0 MEASURED / SURF-A1 FAIL / H-REP-P0 FAIL WITH LOCAL SIGNAL / H-GEO-P1 FAIL WITH MATERIAL RESCUE / H-GEO-P2 FAIL WITH 3-OF-4 ABSOLUTE GEOMETRY GATES PASS + SHARED-H RETENTION FAIL / H-GEO-P3 SOURCE PREPARATION**  
**Optimizer policy:** no SURF-A2 or TESSA optimizer step. H-GEO-P3 may optimize only after exact source/input/A100/gradient-connectivity/shared-V8-preservation admission passes.

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

### H-GEO-P1 — field-only exact-triangle control

Exact source `4a4236db120ffec32dc33d278341e21f58d93529`; 3600 steps. P1 materially beat V8 but failed all four absolute geometry gates and the required endpoint improvement gate. It did not prove whole-architecture insufficiency because upstream `scene_core.scene` remained frozen.

### H-GEO-P2 — scene + field existing-architecture control

Exact execution source `3a572989ab379975ed7cda5f0dd8b6f33de246b3`; authority `docs/research/IRIS_H_GEO_P2_RESULT_20261020.json`.

- A100 preflight PASS; non-zero finite gradients reached both scene and field at optimizer step 0;
- exact V8 initialization, never P1;
- exact P1 triangle dataset bytes reused;
- 4800 carrier steps completed;
- checkpoint SHA256 `0bdcd24af776e322bfbb1a004d6034167770b7950ecccb9b04d6d689597854ee`;
- teacher truth absent from inference; TESSA not invoked.

Independent court:

| Metric | P2 | Locked gate | Result |
|---|---:|---:|---|
| mean distance MAE | `0.01743` | `<=0.025` | PASS |
| worst endpoint p95 | `0.08587` | `<=0.075` | FAIL |
| worst normal abs-dot | `0.74521` | `>=0.70` | PASS |
| worst near BA | `0.90323` | `>=0.80` | PASS |

P2 also passed all four preregistered improvement gates versus P1 and all local-alignment controls. Shared-H retention failed: minimum subject H cosine vs frozen V8 was `0.56570` (GreenBlob), although raw-field retention passed.

Therefore existing IRIS scene+field has strong metric-geometry capacity, but forcing ultra-fine geometry into shared H leaves a locked endpoint-precision deficit and representation-retention tension. P2's preregistered FAIL boundary authorizes a new explicit-geometry-carrier preregistration only.

## 4. Active experiment — H-GEO-P3-EXPLICIT-HIRES-CARRIER

Question:

> Can a dedicated geometry carrier, conditioned on the same frozen V8 view evidence, close endpoint precision while leaving shared V8 H exactly unchanged?

Authority:

- `canonical/H_GEO_P3_INPUT_BINDING_V1.json`;
- `canonical/H_GEO_P3_PREREGISTRATION_V1.json`;
- `docs/research/IRIS_H_GEO_P3_DECISION_LEDGER_20261020.md`;
- `models/iris/vnext/explicit_geometry_carrier_v1.py`;
- `tools/iris_vnext/h_geo_p3_runtime_v1.py`;
- `tools/iris_vnext/preflight_h_geo_p3_a100_v1.py`;
- `tools/iris_vnext/train_fit8_h_geo_p3_seeded_v1.py`;
- `tools/iris_vnext/audit_fit8_h_geo_p3_seeded_v1.py`.

Architecture under test:

```text
frozen V8 view_bank
   ├→ frozen V8 scene + field → shared H (must remain exact)
   └→ dedicated geometry_scene (V8 weight copy)
        → 64² planes
        → zero-residual 2× geometry refinement → 128² planes
        → dedicated geometry_field (V8 weight copy)
        + zero-init continuous XYZ residual
        → H_geo 384-D
```

P2 checkpoint is court comparison only and forbidden for initialization. P3 reuses the exact P1 triangle-dataset bytes and matches P2's 4800-step training budget.

### Locked P3 court

Fresh equal-capacity probes compare `P3_GEO_XYZ`, `P2_H_XYZ`, `V8_H_XYZ`, `XYZ_ONLY`, and `P3_SHUFFLED_GEO_XYZ`.

Absolute quality remains unchanged: distance `<=0.025`, endpoint p95 `<=0.075`, worst normal `>=0.70`, worst near BA `>=0.80`. P3 must additionally reduce endpoint p95 by at least 10% versus fresh-court P2 while keeping P2-level distance/normal/near within locked non-regression margins. Shared V8 state hashes and anchor H must remain exact.

PASS permits design of a new direct-surface experiment from `H_geo`; it does not itself authorize SURF-A2 or TESSA. FAIL sends the program to a focused explicit-carrier endpoint-frequency audit, not threshold relaxation or TESSA rescue.

## 5. SURF-A2 / TESSA

`SURF-A2-HIRES` remains blocked on P3 and the resulting direct-surface decision.

`TESSA-V2-B1` remains blocked until a genuinely qualified high-resolution direct A-arm exists. TESSA may structuralize correct geometry; it may not repair unqualified IRIS geometry.

## 6. Notebook worker law

```text
open notebook
→ Run all
→ approve Google Drive exactly once
→ exact-hash autonomous execution
→ separated visible phases
→ resumable Drive checkpoint/receipts
→ independent court
```

No full-Drive discovery, second OAuth, hidden fallback, source drift or threshold relaxation is allowed.

## 7. Immediate continuation

```text
seal P2 result                                  [DONE]
        ↓
P3 binding + preregistration                    [DONE]
        ↓
P3 source / dedicated CI                        [ACTIVE — PR #128]
        ↓
freeze exact final green P3 SHA                 [NEXT]
        ↓
fast-forward active research branch
        ↓
V8-style A100 Run-All P3 notebook
        ↓
independent P3 vs P2 vs V8 fresh-probe court
```
