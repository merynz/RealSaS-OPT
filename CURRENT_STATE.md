# RealSaS-OPT — Current State

**Date:** 2026-10-19  
**Production authority:** `main` until reviewed merge  
**Post-V8 base:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Active architecture/evidence branch:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Review vehicle:** draft PR #122 — `IRIS: TESSA V2 structuralization + geometry-explicit evidence redesign`  
**Scientific status:** **NO GLOBAL IRIS FIT8 PASS / NO TESSA PASS / RAW-FIELD SPARSE A-ARM FALSIFIED ON SNAKE / NATIVE V8 SURFACE CONTROL PENDING**  
**Training authorization:** **NO NEW OPTIMIZER STEP AUTHORIZED**

## 1. Authority separation

1. `main` remains merged product-code authority.
2. Completed V8 remains immutable architecture/measurement history bound to source `5083c9915533ec8ec6530c7d229c635887abfea7` and its exact checkpoint lineage.
3. Completed V9 remains immutable focused-relation experiment history; later redesign does not rewrite V9 truth, model or result.
4. PR #122 is review/research source only.
5. Green CI proves source contracts, not scientific quality.
6. Model evidence, Compiler qualification and product canonical state remain separate authorities.
7. The 2026-10-19 Snake sparse failure is a real scientific falsification of one post-V8 extraction adaptation, not a V8 checkpoint mutation.

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

Default remains: do not reopen H merely because one specialist/extraction transition is weak.

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
- V9 visibility targets duplicated pair orientation and do not establish complete-stack visibility;
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
- strict free-running grammar/generator;
- independent decoded-mesh topology court;
- token CE + coordinate metric + structural-token objective.

These are research defaults, not mobile runtime promises and not demonstrated 32k free-running capability.

## 8. Why TESSA remains optional

IRIS/StdGEN++-class surface reconstruction and TESSA/MeshAnything-class structuralization solve different problems.

TESSA survives only if an A/B court proves value:

```text
A = same IRIS evidence → qualified direct surface witness → Compiler
B = same IRIS evidence/witness → TESSA V2 → Compiler
```

If A qualifies and B adds no meaningful topology/face-economy value, policy prefers A.

## 9. Raw-field sparse T512→T1024 A-arm — FALSIFIED

Historical sparse decoder candidate:

`RealSaS.ZeroSurfaceDecoder.SparseRegularT512T1024IndexedMT.v5`

The adapter deliberately used exact completed V8, raw `scene_core.query()["field"]`, identity coordinates/sign, FP32/no-autocast, no extraction delta and no TESSA.

A100 engineering preflight passed. Full scientific materialization then failed on the first subject, Snake, at execution source:

`d9701923b2c19bd1c9b9771fe071bb92264a041d`

Failure:

```text
stage: INDEXED_MT_BUILD_AND_NORMALS
exception: SparseSurfaceTruncationError
message: SPARSE_SHELL_TRUNCATES_ZERO_SURFACE:62735467 boundary crossing tetrahedra
```

Preserved diagnostics:

- coarse field min/max: `-0.3515409529 / 0.5678277612`;
- coarse zero bracket: `true`;
- T512 query points: `135,005,697`;
- refined coarse cells: `6,525,575` (`~4.8619%` of T512 cells);
- candidate fine tetrahedra: `313,227,600`;
- crossing tetrahedra: `95,514,148`;
- conservative boundary-crossing tetrahedra: `62,735,467`;
- exact faces counted before fail-closed truncation: `125,076,400`;
- fine field min/max: `-0.1913980544 / 0.2545053363`;
- fine zero bracket: `true`.

Interpretation:

- this is **not** a CUDA/resource failure;
- do not widen the refinement band after observing this result;
- do not disable `fail_on_boundary_crossing`;
- do not force the ~125M-face raw zero-field candidate through as a success;
- the conservative boundary counter may overcount literal shell intersections because it flags crossings in boundary-parent coarse cells, but the overall raw-field complexity is independently large;
- the result falsifies the **raw V8 scalar field + historical distance-like sparse selector** as the current direct A-arm baseline.

It does **not** yet falsify sealed H or V8's native learned FlexiCubes witness.

## 10. Why the previous A-arm definition drifted from V8-trained surface semantics

V8 G2 did not supervise the raw zero-field surface alone. The trained extraction path is:

```text
R24 regular grid
        ↓
V8 query_evidence
        ↓
field + H + learned extraction_delta
        ↓
learned 21-D cube parameters from cube_extraction(H corners)
        ↓
pinned FlexiCubes
        ↓
extracted surface witness
        ↓
teacher-surface Chamfer + FlexiCubes deviation/L1
```

Therefore the failed raw-field T512/T1024 experiment is clean falsification evidence, but it is not the native V8 surface witness promised by the architectural A/B plan.

## 11. Native V8 FlexiCubes control — source implemented, runtime pending

New source:

- `tools/iris_vnext/audit_v8_raw_field_sparse_compatibility_v1.py` — preserved T512 raw-field compatibility forensics; no model re-query and no failure override;
- `tools/iris_vnext/materialize_fit8_v8_native_flexicubes_v1.py` — exact V8 native R24 trained extraction witness;
- `tools/iris_vnext/audit_fit8_v8_native_flexicubes_control_v1.py` — independent sealed-V4 geometry + exact Stage01 eight-view mask measurement;
- `tests/iris/test_v8_surface_control_v1.py` — distance-like vs oscillatory field diagnostics and native cube-head path source tests.

Native control semantics:

- exact V8 checkpoint/scene planes;
- exact training resolution `R24`;
- exact learned field;
- exact learned `extraction_delta`;
- exact learned 21-D FlexiCubes cube parameters;
- pinned upstream FlexiCubes commit;
- FP32/no-autocast;
- no teacher input in inference forward;
- no TESSA;
- zero optimizer steps;
- measurement only;
- `high_resolution_capability_claimed = false`.

## 12. High-resolution transition is now an open design debt

No naive R24→T1024 reuse of V8 `extraction_delta` is authorized.

`maximum_extraction_delta = 0.03` is an absolute normalized displacement. Relative to the grid it is sub-cell scale at R24 but many fine cells at T1024. The learned cube parameters and deformation semantics therefore need an explicit resolution-aware transition rather than blind supersampling.

The next design decision depends on the native R24 control:

```text
native R24 control good
        → sealed H/native V8 surface likely retained
        → redesign high-resolution transition/field regularity/extraction semantics

native R24 control bad
        → reopen geometry/extraction quality workstream
        → still do not reopen H without a dedicated carrier falsification court
```

## 13. Exact input authority remains unchanged

`canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json` still binds the exact Stage01, sealed V4 prepared cohort and immutable V8 checkpoint bytes.

Its historical T512/T1024 purpose now records a **falsified experiment**; it must not be silently reinterpreted as a successful A-arm baseline.

Prepared cohort:

- source `7029ee3f48787aa9f7e27eecef89155eb9368a96`;
- manifest schema `RealSaS.IRISVNextFit8Manifest.v4`;
- manifest SHA256 `6561bf2087f4fac3699c673b5c93fb3d8dbc2d1ff90a5065d976cae70688b4d8`.

Stage01 bundle SHA256:

`9d9c9cbf63c979326cd2f39f5435006b81dddb201028a34ebb4cf10e1b434936`

## 14. Current execution threshold

Exact next state machine:

```text
GREEN SOURCE HEAD WITH NATIVE CONTROL TOOLS
        ↓
RUN PRESERVED SNAKE T512 RAW-FIELD COMPATIBILITY FORENSIC
        ↓
MATERIALIZE EXACT V8 NATIVE R24 FLEXICUBES WITNESSES FOR FIT8
        ↓
RUN SEALED-V4 GEOMETRY + EXACT STAGE01 64-VIEW MASK CONTROL
        ↓
DECIDE: NATIVE SURFACE DEBT vs HIGH-RES TRANSITION DEBT
        ↓
DEFINE / PREREGISTER CORRECTED HIGH-RES DIRECT A-ARM
        ↓
ONLY AFTER A-ARM READINESS → TESSA V2 TRAINING READINESS
```

TESSA optimizer steps remain premature.

## 15. Parallel post-V8 relation workstream

Relation V10 remains separately blocked on:

- exact all-hit relation evidence support;
- authored-motion conditioning disposition;
- Truth V3 authority/validity;
- support/class-balance/abstention courts;
- V9-state vs neutral initialization court;
- frozen-domain hot-path fix;
- preregistration/materialization/readiness.

Do not mix relation V10 authorization with TESSA/surface authorization.

## 16. Required read order

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `docs/research/IRIS_TESSA_V2_STRUCTURALIZATION_PLAN_20261019.md`
4. `docs/research/IRIS_TESSA_V2_DECISION_LEDGER_20261019.md`
5. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
6. `docs/research/IRIS_V9_FOCUSED_RELATION_RESULT_20261019.json`