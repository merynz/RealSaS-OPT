# RealSaS-OPT — Current State

**Date:** 2026-10-19  
**Production authority:** `main` until reviewed merge  
**Active research branch / PR head:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Source staging branch:** `research/iris-vnext-hybrid-tessa-v2-20261019-hrep-p0`  
**Review vehicle:** draft PR #122 — `IRIS: TESSA V2 structuralization + geometry-explicit evidence redesign`  
**Active research lane:** `SURFACE_TESSA`  
**Active experiment:** `H-REP-P0` — sealed-V8 representation diagnostic  
**Parked lane:** `RELATION_V10` — not cancelled, no optimizer authorization  
**Scientific status:** **NO GLOBAL IRIS FIT8 PASS / NO TESSA PASS / SURF-A0 FALSIFIED / SURF-C0 MEASURED / SURF-A1 COMPLETED SCIENTIFIC FAIL / H-REP-P0 SOURCE+PREREG STAGED, NOT YET RUN**  
**Optimizer authorization:** **No SURF-A2 or TESSA optimizer step. H-REP-P0 probe optimization is permitted only after its exact source/input/A100 admission is green. Carrier retraining is not authorized.**

Machine-readable continuation authority: `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json`.

## 1. Authority law

1. `main` remains merged product-code authority.
2. Completed V8 is immutable model/history authority: source `5083c9915533ec8ec6530c7d229c635887abfea7`, step `12000`, checkpoint `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`, binding `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`.
3. Completed V9 is immutable focused-relation history; it is not global IRIS authority.
4. PR #122 / research source is not scientific evidence by itself. Green CI is necessary source integrity, not scientific PASS.
5. IRIS owns learned scene/evidence; TESSA may propose explicit structured meshes; **Compiler alone mints canonical M**.
6. Teacher topology/rig/skin may supervise training truth but never enter inference conditioning.
7. Fit8 is a shared-system qualification cohort, not an unseen-generalization claim.
8. Failed preregistered courts are never repaired by changing thresholds after results.

## 2. Long-term product invariant

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

There is no second deformable visual mesh. TESSA survives only if it adds structural/topological value without sacrificing source fidelity or downstream mechanics. GSA is not permanent by fiat; its eventual role follows the qualified carrier contract.

## 3. Historical anchors

### IRIS-V8 — immutable

- source `5083c9915533ec8ec6530c7d229c635887abfea7`;
- step `12000`;
- checkpoint `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`;
- shared `scene_core` sealed after M1.

### IRIS-V9-R — completed scoped relation experiment

Projected boundary evidence was learned; full ordering/visibility was not. Relation V10 remains parked until its separate Truth-V3/all-hit/ray-stack readiness gates are earned.

### SURF-A0-RAWFIELD — falsified

Exact V8 raw field + historical distance-like `T512 → sparse T1024` selector failed closed on Snake at `SPARSE_SHELL_TRUNCATES_ZERO_SURFACE`. Preserve as falsification evidence; do not widen the shell, flip field sign, disable truncation or reinterpret the ~125M-face candidate as success.

### SURF-C0-NATIVE-R24 — completed diagnostic

Exact V8 trained surface path at R24 measured:

- mean subject mask IoU `0.5605675`;
- minimum subject mask IoU `0.4053549`;
- minimum view IoU `0.3425416`;
- maximum mask gap `392.4207 px`;
- mean symmetric Chamfer `0.0612046` IRIS.

This control had no preregistered PASS/FAIL threshold, so it remains measurement-only. It established native extraction-quality debt without falsifying sealed H.

## 4. SURF-A1-FROZEN-H — completed scientific FAIL

### 4.1 Question

Could exact sealed V8 H support a new resolution-aware, raster-supervised extraction decoder without rewriting the carrier?

### 4.2 Exact runtime authority

Execution source: `afc18debbc4d5858a4a632884b7cfd2141dcc35a`.

Preflight evidence:

- Drive file id `12rEV36p_qTdzIkbmlszrTNouuj862aOr`;
- SHA256 `0fd889134164ee0cb80bf05d3409967ce5a37ff2f9f2c38b41a7218c2392ce75`;
- status `PASS_SURF_A1_A100_PREFLIGHT__TRAINING_AUTHORIZED`;
- A100-SXM4-40GB, Torch `2.11.0+cu130`, CUDA `13.0`;
- pinned FlexiCubes `4cc7d6c...`, PyTorch3D `978cd992...`;
- hard-raster parity: `64/64`, minimum precision `1.0`, minimum recall `1.0`;
- zero-step native R24 parity: faces exact `8/8`, vertex differences ~`6e-8–1.8e-7` IRIS.

Training evidence:

- Drive file id `1fI6CXDFgmvTTZARo3aAjQm0ubfsVORVm`;
- SHA256 `9d5ac2f01b5e84d8a9c91e93008dfd2a50a7a8908b0626950e4d11a43d918123`;
- optimizer steps `2400` (`R24:400`, `R32:600`, `R48:1400`);
- final checkpoint SHA256 `431b29030722fa69b51d037d2a6b39dc1d4857c9d0cd9d2a5aad8c13ba37d04a`;
- sealed V8 scene-core hash unchanged;
- teacher truth absent from inference forward;
- TESSA not invoked.

Independent court evidence:

- Drive file id `1eptd5tfwI74FHvaUkD--Rias0DOgMLCa`;
- SHA256 `a8c4070a126edaddd602853c62b3e2b58335ef2fb62750c52f088ef8e15e631e`;
- status `SCIENTIFIC_FAIL_SURF_A1_FROZEN_H_REPRESENTATION_RESCUE`.

Global R48 court:

| Metric | SURF-C0 native | SURF-A1 | A1 prereg gate | Result |
|---|---:|---:|---:|---|
| mean subject mask IoU | `0.56057` | `0.77953` | `>=0.80` | FAIL |
| min subject mask IoU | `0.40535` | `0.67901` | `>=0.65` | PASS |
| min view mask IoU | `0.34254` | `0.65213` | `>=0.55` | PASS |
| mean symmetric Chamfer | `0.06120` | `0.04305` | `<=0.04` | FAIL |
| max mask gap | `392.42` | `258.66` | `<=128` | FAIL |
| worst teacher→candidate p95 | — | `0.11117` | `<=0.075` | FAIL |
| min subject normal abs-dot mean | — | `0.57822` | `>=0.70` | FAIL |
| non-manifold edges | — | `0` | `0` | PASS |
| zero-area faces | — | `0` | `0` | PASS |

Interpretation:

> SURF-A1 substantially rescued native C0 but did not earn its preregistered qualification. This is a real scientific fail because all apparatus/admission gates passed. It does **not** prove that sealed H contains no useful geometry; it proves that the tested sealed-H→shared-extraction path is insufficient across Fit8.

Do not relax A1 thresholds. Do not promote SURF-A1 to SURF-A2. Do not train TESSA.

## 5. H-REP-P0 — active representation investigation

The next information-maximizing question is deliberately extractor-free:

> Is local surface geometry actually decodable from sealed V8 `H(x)` or the sealed raw field, independent of FlexiCubes / SURF-A1 mesh extraction?

Authority:

- `canonical/H_REP_P0_INPUT_BINDING_V1.json`;
- `canonical/H_REP_P0_PREREGISTRATION_V1.json`;
- `models/iris/vnext/h_rep_probe_v1.py`;
- `tools/iris_vnext/build_fit8_h_rep_probe_dataset_v1.py`;
- `tools/iris_vnext/h_rep_probe_runtime_v1.py`;
- `tools/iris_vnext/train_fit8_h_rep_probe_v1.py`;
- `tools/iris_vnext/audit_fit8_h_rep_probe_v1.py`.

### 5.1 Four equal-capacity arms

All arms use the same `387-D` input envelope and exactly the same parameter count:

1. `H_XYZ` — sealed local H + XYZ;
2. `RAW_FIELD_XYZ` — raw V8 field in one evidence channel + zero padding + XYZ;
3. `XYZ_ONLY` — all evidence channels zero + XYZ negative control;
4. `SHUFFLED_H_XYZ` — fixed within-subject shuffled H + XYZ local-alignment control.

No subject ID, no subject-specific trainable parameters, no SURF-A1 decoder, no FlexiCubes, no TESSA, no mesh extraction.

### 5.2 Truth scope

Probe targets reuse source-defensible geometry already sealed in V4 prepared shards:

- `teacher_surface_points_iris`;
- `mechanical_points`;
- `mechanical_normals`.

Distance/offset targets are nearest-point **point-cloud proxies**, explicitly not exact SDF truth. Normal targets are validity-gated nearest mechanical normals. Teacher truth is target-only and never a probe input.

### 5.3 Decision boundary

H-REP-P0 PASS requires absolute held-out quality plus improvement over both `XYZ_ONLY` and `SHUFFLED_H_XYZ`; raw-field has its own absolute + XYZ-control gates. Thresholds are preregistered before runtime.

```text
H or raw field passes
    ↓
local geometry is decodable without mesh extraction
    ↓
design a different extraction/transition path
DO NOT retrain H yet
DO NOT train TESSA

neither passes
    ↓
local geometry decodability not demonstrated
    ↓
preregister a carrier-representation retraining experiment
DO NOT train TESSA
```

Neither result directly authorizes `SURF-A2-HIRES`.

## 6. SURF-A2 / TESSA state

`SURF-A2-HIRES` is blocked on H-REP-P0 and the resulting direct-surface architecture decision.

`TESSA-V2-B1` remains blocked until a genuinely qualified high-resolution direct A-arm exists. Historical raw T512/T1024, native R24 control and SURF-A1 are forbidden substitutes for that upstream authority.

## 7. Notebook / execution UX contract

Research notebooks are compute workers, not authority. Required operator UX:

```text
open notebook
→ Run all
→ approve Google Drive exactly once
→ autonomous hash-bound execution
→ meaningful phase logs
→ resumable Drive checkpoint/receipt writes
```

For all new notebooks:

- physical cells must separate major phases like the successful V8 worker;
- exactly one `google.colab.drive.mount('/content/drive', force_remount=False)`;
- no `auth.authenticate_user()`, second OAuth, rclone/gcloud interactive auth;
- **no Drive-tree discovery** (`os.walk`, `rglob`, full-MyDrive scan) for known scientific inputs;
- exact mounted paths + IDs + byte/hash authority are resolved before notebook generation;
- source SHA, inputs, prereg, model and checkpoints fail closed;
- no hidden reduced-capacity/resolution fallback.

## 8. Immediate continuation state machine

```text
SURF-A1 SCIENTIFIC FAIL SEALED
        ↓
H-REP-P0 SOURCE + PREREG
        ↓
SOURCE CONTRACT / CI GREEN
        ↓
FREEZE EXACT GREEN SHA
        ↓
V8-STYLE MULTI-CELL A100 RUN-ALL WORKER
        ↓
BUILD DETERMINISTIC TRAIN/HOLDOUT POINT-PROXY DATASET
        ↓
CACHE/VERIFY EXACT SEALED V8 H + RAW FIELD
        ↓
TRAIN 4 EQUAL-CAPACITY PROBES (1200 steps each)
        ↓
INDEPENDENT HELD-OUT COURT
        ↓
PASS → extraction-transition redesign; H remains sealed
FAIL → preregister carrier-representation retraining
        ↓
NO TESSA UNTIL QUALIFIED SURF-A2
```

## 9. Required read order

1. `AGENTS.md`
2. `CURRENT_STATE.md`
3. `CURRENT_PLAN.md`
4. `canonical/IRIS_RESEARCH_EXPERIMENT_LEDGER_V1.json`
5. `canonical/H_REP_P0_INPUT_BINDING_V1.json`
6. `canonical/H_REP_P0_PREREGISTRATION_V1.json`
7. `canonical/SURF_A1_FROZEN_H_PREREGISTRATION_V1.json`
8. `docs/research/IRIS_TESSA_V2_DECISION_LEDGER_20261019.md`
9. `docs/research/IRIS_TESSA_V2_STRUCTURALIZATION_PLAN_20261019.md`
10. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
