# RealSaS-OPT — Current State

**Date:** 2026-10-19  
**Production authority:** `main` until reviewed merge  
**Post-V8 base:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Active architecture/evidence branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Review vehicle:** draft PR #122 — `IRIS: TESSA V2 structuralization + geometry-explicit evidence redesign`  
**Scientific status:** **NO GLOBAL IRIS FIT8 PASS / NO TESSA PASS / V9 HAS SCOPED RELATION PASS ONLY**  
**Training authorization:** **NO NEW OPTIMIZER STEP AUTHORIZED**

## 1. Authority separation

1. `main` remains merged product-code authority.
2. Completed V8 remains immutable architecture/measurement history bound to source `5083c9915533ec8ec6530c7d229c635887abfea7` and its exact checkpoint lineage.
3. Completed V9 remains immutable focused-relation experiment history; later redesign does not rewrite V9 truth, model or result.
4. PR #122 is review/research source only.
5. Green CI proves source contracts, not scientific quality.
6. Model evidence, Compiler qualification and product canonical state remain separate authorities.

## 2. Preserved product invariants

- one canonical mechanical carrier `M` after deterministic Compiler qualification/selection;
- no second deformable visual mesh;
- IRIS emits learned scene/evidence, not final product mesh authority;
- TESSA may emit mesh proposals only;
- Compiler remains the authority that may mint canonical M;
- unknown/unsupported truth must abstain rather than become fabricated negative or semantic intent.

## 3. V8 baseline — immutable

Exact completed V8 authority:

- source: `5083c9915533ec8ec6530c7d229c635887abfea7`;
- global step: `12000`;
- checkpoint SHA256: `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding SHA256: `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`;
- `scene_core` sealed after M1.

Earned V8 interpretation:

- post-M1 geometry/extraction retention was materially improved by sealing shared H;
- downstream appearance/relation/topology objectives remained learnable from sealed H;
- this does **not** establish a global Fit8 scientific PASS;
- topology loss reduction is not an independent topology qualification court.

Default remains: do not reopen H merely because one specialist path is weak.

## 4. V9 focused relation — completed scoped result

V9 focused relation `ATTEMPT_001` completed 1600 optimizer steps at source `7029ee3f48787aa9f7e27eecef89155eb9368a96`.

Scoped result:

- projected alpha-boundary held-out balanced accuracy ~`0.7599`;
- projected appearance-boundary held-out balanced accuracy ~`0.8209`;
- front held-out balanced accuracy ~`0.5588`;
- visibility A ~`0.5659`;
- visibility B ~`0.5534`;
- occlusion support was positive-only and is not a binary discrimination court.

Interpretation:

- clean camera-conditioned projected-boundary truth is genuinely learnable from the sealed V8 representation/path;
- V9 does not prove ordering solved;
- V9 visibility targets duplicated pair orientation (`visA=front`, `visB=1-front`) and do not establish complete-stack visibility;
- weak V9 ordering does not by itself prove raw H lacks order information because V9 retained the historical frozen 384→64 relation embedding.

## 5. Relation redesign state

PR #122 contains `GraphicsRelationHeadV3` / evidence V3 source changes:

- full/raw H is available to the relation path rather than forcing the historical 64-D bottleneck;
- explicit `XYZ_A/XYZ_B`, delta geometry, camera-space depth, projected UV and same-ray residual are conditioning inputs;
- front/order remains pairwise;
- visibility is removed from the pair abstraction and moved to complete-ray-stack evidence;
- new pair API has no free unsupervised neural confidence;
- appearance/mechanical V3 remove orphan confidence/descriptor/uncertainty outputs from authority;
- historical uncalibrated scene uncertainty remains frozen/hidden from V3 evidence.

No V10 relation optimizer step is authorized. Relation Truth V3 still requires measured all-hit/authored-motion support, exact materialization, preregistration and readiness.

## 6. Contact persistence state

The historical teacher type can represent contact persistence, but the active Fit8 mechanical preparation did not provide `contact_face_mask`.

Therefore:

```text
no source-defensible contact mask
        ↓
contact_persistence_valid = false
        ↓
ABSTAIN / zero gradient
```

Stored zero must not be interpreted as negative contact truth.

## 7. TESSA V2 architectural decision

TESSA is no longer treated as a mandatory small IRIS structure head.

Current preferred split:

```text
8-view observations
        ↓
IRIS continuous scene carrier H / field
        ↓
high-fidelity surface witness
        ↓
OPTIONAL TESSA V2 structuralization / learned retopology
        ↓
mesh proposal
        ↓
Compiler qualification
        ↓
canonical M
```

TESSA V2 source defaults:

- width `1024`;
- `16` attention heads;
- `24` decoder layers;
- FFN `4096`;
- `1024` coordinate bins;
- `512` scene + `512` surface/mechanics memory tokens;
- exact local causal context `4096`;
- causal global structural-history summaries;
- AMT-compatible adjacency-aware grammar retained;
- strict free-running grammar/generator;
- independent decoded-mesh topology court;
- token CE + coordinate metric + structural-token objective.

These are research defaults, not mobile runtime promises and not demonstrated 32k free-running capability.

## 8. Why TESSA remains optional

StdGEN++-class and MeshAnything-class ideas solve different problems:

- IRIS/StdGEN++ side: high-fidelity continuous/sparse surface reconstruction, geometric/raster supervision, thin-structure continuity;
- TESSA/MeshAnything side: explicit structural connectivity / production-style retopology prior.

Therefore TESSA survives only if an A/B court proves value:

```text
A = same IRIS evidence → direct high-res surface → Compiler
B = same IRIS evidence → TESSA V2 → Compiler
```

If A qualifies and B adds no meaningful topology/face-economy value, policy prefers A.

## 9. Direct high-resolution surface baseline — source implemented

The selected historical sparse decoder has been connected to vNext/V8 field authority:

`RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT.v5`

Scientific baseline policy is fixed:

- T512 coarse grid;
- sparse T1024 refinement;
- no hidden deformation;
- normalized coordinate identity;
- raw learned field sign identity;
- no `extraction_delta` or uncalibrated uncertainty mixed into the A-arm baseline;
- FP32 / no autocast.

Reduced grids exist only in unit tests. The scientific CLI provides no grid-resolution override.

Important compatibility correction already implemented:

- historical V8 `IRISVNextV1` uses `model.scene_core.query()`;
- post-redesign V3 can use `model.query_local()`;
- the sparse adapter supports both but the direct A arm explicitly uses exact historical V8, not a V3 migration.

## 10. Exact direct-surface input binding

Machine-readable authority:

`canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json`

Prepared observations are read-only reuse of the sealed V9 V4 cohort:

- preparation source `7029ee3f48787aa9f7e27eecef89155eb9368a96`;
- manifest schema `RealSaS.IRISVNextFit8Manifest.v4`;
- manifest SHA256 `6561bf2087f4fac3699c673b5c93fb3d8dbc2d1ff90a5065d976cae70688b4d8`;
- 8 bound shard file IDs + SHA256 values;
- exact V8 checkpoint file ID + SHA256.

Read-only hydration source:

`tools/iris_vnext/hydrate_fit8_direct_surface_inputs_v1.py`

It verifies bytes, manifest rows and preparation receipt. It has no remote write path and authorizes no training.

## 11. Direct-surface execution source

Implemented:

- `models/iris/vnext/sparse_surface_v3.py` — exact field→T512/T1024 adapter and candidate receipt;
- `tools/iris_vnext/hydrate_fit8_direct_surface_inputs_v1.py` — exact read-only input hydration;
- `tools/iris_vnext/preflight_fit8_direct_surface_a100_v1.py` — A100-class engineering-only probe;
- `tools/iris_vnext/materialize_fit8_direct_surface_v1.py` — 8-subject full-resolution materializer;
- unit/source contracts that forbid reduced scientific resolution and prove exact-V8 query API compatibility.

A100 preflight semantics:

- exact V8 + one real prepared scene;
- exact production field-query chunk;
- no reduced-resolution mesh;
- no quality verdict;
- minimum 39 GiB CUDA-memory admission envelope;
- resource/throughput result only.

## 12. Direct-surface scientific state

**No full direct-surface artifact has been executed yet on this branch.**

Therefore currently absent:

- 8/8 full T512→T1024 surface receipts;
- source-fidelity mask/depth/normal/surface-distance court;
- direct-surface Compiler qualification;
- G5/multiview direct-surface court;
- G3/authored-motion direct-surface court;
- sealed A-arm baseline.

Source-contract green must never be reported as those results.

## 13. Current execution threshold

Exact next state machine:

```text
FINAL SOURCE GATE ON CURRENT PR HEAD
        ↓
READ-ONLY INPUT HYDRATION
        ↓
BYTE/HASH/LINEAGE RECEIPT PASS
        ↓
A100-CLASS ENGINEERING PREFLIGHT
        ↓
FULL FIT8 T512→T1024 MATERIALIZATION
        ↓
8/8 MODEL-ONLY SURFACE RECEIPTS
        ↓
SOURCE-FIDELITY + COMPILER/G5/G3 COURTS
        ↓
DIRECT A-ARM SEALED
        ↓
TESSA V2 TRAINING READINESS
```

TESSA optimizer steps are premature before the A arm exists.

## 14. Parallel post-V8 relation workstream

The surface/TESSA work does not cancel the relation evidence program.

Relation V10 remains blocked on:

- exact all-hit relation evidence support;
- authored-motion conditioning disposition;
- Truth V3 authority/validity;
- support/class-balance/abstention courts;
- V9-state vs neutral initialization court;
- frozen-domain hot-path fix;
- preregistration/materialization/readiness.

Do not mix relation V10 authorization with TESSA/surface authorization.

## 15. Required read order

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `docs/research/IRIS_TESSA_V2_STRUCTURALIZATION_PLAN_20261019.md`
4. `docs/research/IRIS_DIRECT_SURFACE_BASELINE_EXECUTION_PLAN_20261019.md`
5. `docs/research/IRIS_TESSA_V2_DECISION_LEDGER_20261019.md`
6. `docs/research/IRIS_V9_FOCUSED_RELATION_RESULT_20261019.json`
7. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
8. `docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`
9. `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`
10. `docs/research/IRIS_VNEXT_DECISION_LEDGER_20261018.md`
11. `AGENTS.md`
12. `SYSTEM_INDEX.md`
