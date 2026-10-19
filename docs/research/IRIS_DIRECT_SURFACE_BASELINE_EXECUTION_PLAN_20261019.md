# IRIS Direct-Surface Baseline Execution Plan — 2026-10-19

Status: **SOURCE IMPLEMENTED / ARTIFACT EXECUTION PENDING / NO TRAINING AUTHORIZATION**

Branch: `research/iris-vnext-hybrid-tessa-v2-20261019`  
Review vehicle: PR #122  
Purpose: establish the TESSA-free high-resolution surface baseline that both tests the StdGEN++-style surface hypothesis and supplies the fair A arm for TESSA V2.

## 1. Scientific question

Before TESSA receives any optimizer step, answer:

> How far does the exact completed V8 IRIS continuous field get when decoded with the already selected high-resolution sparse surface path, without TESSA or any learned retopology layer?

This is not a topology-head experiment. It is a model-only surface materialization followed by independent fidelity and Compiler courts.

## 2. Exact authority

### Model

- historical model: `IRISVNextV1`;
- V8 source: `5083c9915533ec8ec6530c7d229c635887abfea7`;
- global step: `12000`;
- checkpoint SHA256: `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding SHA256: `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`;
- `scene_core`: historical post-M1 sealed authority.

The direct baseline does **not** migrate V8 into `IRISVNextEvidenceModelV3`. The sparse adapter explicitly supports the historical `model.scene_core.query()` API so the surface is produced from exact V8 scene-core bytes.

### Prepared Fit8 observations

Reuse the already sealed V9 preparation at source:

`7029ee3f48787aa9f7e27eecef89155eb9368a96`

Exact local/Drive binding is machine-readable in:

`canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json`

Bound preparation manifest:

- schema: `RealSaS.IRISVNextFit8Manifest.v4`;
- SHA256: `6561bf2087f4fac3699c673b5c93fb3d8dbc2d1ff90a5065d976cae70688b4d8`;
- subject order: `Snake, Bat, T-Rex, Horse, Spider, GreenBlob, Shark, George`.

No new teacher export or cohort rebuild is required for this baseline.

## 3. Surface decoder

Use only:

`RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT.v5`

with the exact default `SparseRegularTetraPolicyV5`:

- coarse cells: **512**;
- sparse fine cells: **1024**;
- no deformation;
- normalized coordinate identity `[-1,1] -> [-1,1]`;
- raw learned field sign unchanged;
- no field uncertainty in extraction authority;
- no `extraction_delta` mixed into this baseline;
- implicit normals from the selected sparse decoder.

The scientific CLI exposes **no base/fine resolution switch**. Reduced 8/16 or similar grids exist only in source unit tests and never count as baseline evidence.

## 4. Numeric policy

Surface baseline numeric policy:

`FP32_NO_AUTOCAST`

Rationale: V9 BF16 relation inference is not authority for high-resolution zero-surface location. The direct geometry baseline should not add a new precision confound before the A/B court.

## 5. Source components

Implemented:

- `models/iris/vnext/sparse_surface_v3.py`
  - exact V8 `scene_core.query()` compatibility;
  - V3 `query_local()` compatibility for later controlled experiments;
  - identity coordinate/scalar transform;
  - T512->T1024 decoder bridge;
  - hash-bound per-subject candidate receipt;
  - explicit `compiler_qualified=false` / `scientific_pass_claimed=false`.
- `canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json`
  - exact read-only V9 prepared bytes;
  - exact V8 checkpoint bytes;
  - full-resolution policy;
  - training/remote-write prohibition.
- `tools/iris_vnext/hydrate_fit8_direct_surface_inputs_v1.py`
  - Drive file-ID hydration only;
  - byte SHA verification;
  - prepared manifest/receipt cross-check;
  - no remote write.
- `tools/iris_vnext/preflight_fit8_direct_surface_a100_v1.py`
  - engineering-only exact-V8 scene/query benchmark;
  - no reduced mesh;
  - no quality verdict;
  - >=39 GiB CUDA-memory admission envelope.
- `tools/iris_vnext/materialize_fit8_direct_surface_v1.py`
  - exact checkout binding;
  - exact V8 checkpoint verification;
  - exact prepared-shard verification;
  - FP32 scene encode;
  - full T512->sparse T1024 decode for all eight subjects;
  - batch receipt with zero training and Compiler pending.

## 6. Execution state machine

```text
SOURCE CONTRACT GREEN
        ↓
READ-ONLY INPUT HYDRATION
        ↓
BYTE/HASH/LINEAGE RECEIPT PASS
        ↓
A100-CLASS ENGINEERING PREFLIGHT
        ↓
NO REDUCED MESH / NO QUALITY CLAIM
        ↓
FULL FIT8 T512→T1024 MATERIALIZATION
        ↓
8/8 MODEL-ONLY SURFACE RECEIPTS
        ↓
SOURCE-FIDELITY COURT
        ↓
DIRECT SURFACE → COMPILER COURT
        ↓
A-ARM BASELINE SEALED
        ↓
ONLY THEN TESSA V2 TRAINING/DECODE B-ARM
```

## 7. Materialization outputs

Per subject:

- `DIRECT_IRIS_SURFACE.stage12.npz`;
- `DIRECT_IRIS_SURFACE_RECEIPT.json`.

Batch:

- `FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json`.

Required batch status after materialization:

`MATERIALIZED_MODEL_SURFACES__COMPILER_COURT_PENDING`

This status is not a geometry PASS.

## 8. Courts after materialization

### 8.1 Source-fidelity court

Evaluate on the same admitted source cameras and source-defensible teacher geometry where available:

- mask/silhouette fidelity;
- depth fidelity;
- normal fidelity;
- surface distance distribution including p95/max;
- thin-structure continuity / open-chain diagnostics where independently observable;
- component and degeneracy diagnostics.

Do not invent hidden-surface color/semantic truth.

### 8.2 Compiler/product court

The candidate must still pass downstream qualification before it can become canonical M evidence:

- static Compiler mesh qualification;
- support binding;
- multiview/G5 qualification;
- authored-motion/G3 qualification;
- no forbidden topology/degenerate/non-manifold regressions.

The Compiler court must not repair a failed learned surface into a scientific model success.

## 9. TESSA A/B consequence

After the direct arm is sealed:

```text
A = exact V8 H/field -> T512/T1024 surface -> Compiler
B = same admissible IRIS evidence/surface -> TESSA V2 -> Compiler
```

TESSA is promotable only if B preserves source fidelity and mechanical/Compiler qualification while adding meaningful structural economy/topology value. If A already qualifies and B adds no meaningful value, policy prefers A.

## 10. Explicit non-claims

Until real artifacts exist:

- no direct-surface Fit8 PASS;
- no TESSA PASS;
- no 32k-face generation claim;
- no V10 relation authorization;
- no product canonical-M promotion;
- no optimizer step from this plan.

## 11. Immediate next action

1. source gate green on the exact final branch head;
2. hydrate exact bound inputs read-only;
3. execute A100 engineering preflight;
4. if resource PASS, run full eight-subject T512->T1024 materializer;
5. inspect receipts before designing/locking the source-fidelity thresholds that depend on available authority;
6. seal direct A-arm before TESSA V2 optimization.
