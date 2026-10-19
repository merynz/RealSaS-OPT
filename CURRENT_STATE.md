# RealSaS-OPT — Current State

**Date:** 2026-10-20  
**Production authority:** `main` until reviewed merge  
**Active research branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Source-gate vehicle:** draft PR #126 — `IRIS: H-GEO-P2 existing-architecture scene+field control`  
**Active research lane:** `SURFACE_TESSA`  
**Active experiment:** `H-GEO-P2-SCENE-FIELD`  
**Runtime source:** pending final green freeze  
**Parked lane:** `RELATION_V10` — not cancelled  
**Scientific status:** **NO GLOBAL IRIS FIT8 PASS / NO TESSA PASS / SURF-A0 FALSIFIED / SURF-C0 MEASURED / SURF-A1 SCIENTIFIC FAIL / H-REP-P0 SCIENTIFIC FAIL / H-GEO-P1 SCIENTIFIC FAIL WITH MATERIAL V8 IMPROVEMENT / H-GEO-P2 SOURCE PREPARATION**  
**Optimizer policy:** no SURF-A2 or TESSA optimizer step. H-GEO-P2 may optimize only after exact source/input/A100/gradient-connectivity admission passes.

Machine-readable continuation authority: `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json`.
Platform execution lanes: `docs/platform/EXECUTION_LANES.md` (`PLATFORM_EXECUTION_LANES`).

## 1. Product invariant

```text
8-view observations
        ↓
IRIS continuous visual / geometric / mechanical evidence H
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

## 3. Surface / representation evidence chain

### SURF-A0-RAWFIELD — falsified

The historical raw-field `T512 → sparse T1024` direct-surface transition is not a valid direct arm. Preserve as falsification only.

### SURF-C0-NATIVE-R24 — measured diagnostic

Native V8 extraction measured mean subject mask IoU `0.56057`, min subject IoU `0.40535`, min view IoU `0.34254`, max mask gap `392.42 px`, mean symmetric Chamfer `0.06120` IRIS. No preregistered PASS was claimed.

### SURF-A1-FROZEN-H — scientific FAIL

Exact source `afc18debbc4d5858a4a632884b7cfd2141dcc35a`; all apparatus gates passed; 2400 steps completed. Final independent court improved C0 strongly but missed locked mean IoU, Chamfer, mask-gap, p95 and normal gates. This did **not** prove H lacks geometry. SURF-A2 and TESSA stayed blocked.

### H-REP-P0 — scientific FAIL, local geometry signal demonstrated

Frozen source `479e8e53cf0d7df234a3e64b824bb704b861bf19`. Extractor-free fresh probes showed real local H signal: H distance error was roughly one third of XYZ/shuffled-H controls, but absolute metric geometry gates were not earned. This authorized a separately preregistered carrier-training study, not TESSA.

### H-GEO-P1 — completed scientific FAIL with substantial rescue

Exact execution source `4a4236db120ffec32dc33d278341e21f58d93529`.

Authority: `docs/research/IRIS_H_GEO_P1_RESULT_20261020.json`.

- exact authored-rest-triangle truth; no point-cloud proxy primary target;
- maximum normalization replay error `3.033220208337184e-08` IRIS;
- A100 preflight PASS with optimizer step `0`;
- only existing `scene_core.field` body + training-only metric head trainable;
- 3600 carrier steps completed;
- checkpoint SHA256 `2f73a783391a16f09d12e50f240c0fbaaae40f0e044d1546bf12590eba0c5e5d`;
- teacher truth absent from inference; TESSA not invoked.

Independent fresh-probe court:

| Metric | P1 H | frozen V8 H | Locked absolute gate |
|---|---:|---:|---:|
| mean distance MAE | `0.03320` | `0.05030` | `<=0.025` FAIL |
| worst endpoint p95 | `0.10370` | `0.11347` | `<=0.075` FAIL |
| worst normal abs-dot | `0.66920` | `0.60072` | `>=0.70` FAIL |
| worst near BA | `0.74737` | `0.69025` | `>=0.80` FAIL |

P1 passed all local-alignment and retention gates and beat V8 on distance, near and normal. It failed the locked absolute court and the required endpoint reduction vs V8. Therefore field-only retraining is insufficient, but P1 **does not establish whole-architecture insufficiency** because `scene_core.scene` remained frozen.

## 4. Active experiment — H-GEO-P2-SCENE-FIELD

P2 is the missing existing-architecture control before a dedicated geometry branch can be scientifically justified.

Question:

> Can the exact existing V8 architecture reach metric geometry when both `scene_core.scene` and `scene_core.field` are jointly trained with the exact same authored-triangle truth used by P1?

Authority:

- `canonical/H_GEO_P2_INPUT_BINDING_V1.json`;
- `canonical/H_GEO_P2_PREREGISTRATION_V1.json`;
- `docs/research/IRIS_H_GEO_P2_DECISION_LEDGER_20261020.md`;
- `tools/iris_vnext/h_geo_p2_runtime_v1.py`;
- `tools/iris_vnext/preflight_h_geo_p2_a100_v1.py`;
- `tools/iris_vnext/train_fit8_h_geo_p2_v1.py`;
- `tools/iris_vnext/audit_fit8_h_geo_p2_v1.py`.

### Controlled variable

P1 and P2 use the **same exact P1 triangle dataset bytes**. P2 initializes from exact V8, never from P1. The only intended representational change is the trainable domain:

```text
P1: scene_core.field
P2: scene_core.scene + scene_core.field
```

No architecture expansion, no subject-specific parameters, no TESSA, no Compiler repair.

### Admission

Before optimizer step 1, P2 must pass exact clean source, A100 >=39 GiB, exact V8/input/dataset hashes, exact trainable census, and a backward-only real-query gate proving finite non-zero gradients reach both `scene_core.scene` and `scene_core.field`. Live scene planes must be recomputed inside every optimizer step; detached cached planes are forbidden for the training forward.

### Training / court

P2 preregisters 4800 carrier steps. Independent fresh equal-capacity probes compare `P2_H_XYZ`, `P1_H_XYZ`, `V8_H_XYZ`, `XYZ_ONLY`, and `P2_SHUFFLED_H_XYZ` on identical P1 held-out triangle queries.

Absolute gates remain P1's locked quality bar: distance `<=0.025`, endpoint p95 `<=0.075`, worst normal `>=0.70`, worst near BA `>=0.80`. P2 must also materially beat P1 and preserve local-alignment/retention gates.

PASS means existing IRIS architecture has sufficient metric capacity and permits design of a new high-resolution direct-surface experiment. FAIL means an explicit geometry-carrier branch is justified for a **new preregistration**. Neither result directly authorizes SURF-A2 or TESSA.

## 5. SURF-A2 / TESSA

`SURF-A2-HIRES` remains blocked on the H-GEO-P2 result and the resulting direct-surface decision.

`TESSA-V2-B1` remains blocked until a genuinely qualified high-resolution direct A-arm exists. Historical raw T512/T1024, native R24, SURF-A1, H-REP and H-GEO carrier results are not substitutes.

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
H-GEO-P1 exact-triangle field-only run            [DONE — SCIENTIFIC FAIL]
        ↓
seal P1 result + preregister P2 scene+field       [DONE]
        ↓
P2 source contract / CI                            [ACTIVE]
        ↓
freeze exact final green P2 SHA                    [NEXT]
        ↓
V8-style A100 Run-All P2 notebook                  [THEN]
        ↓
independent P2 vs P1 vs V8 fresh-probe court
```
