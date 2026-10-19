# RealSaS-OPT — Current Plan

**Date:** 2026-10-19  
**Status:** **POST-V8 QUALITY PROGRAM — DIRECT-SURFACE A-ARM EXECUTION + RELATION V10 EVIDENCE QUALIFICATION**  
**Active source:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Review vehicle:** draft PR #122  
**Production authority:** `main` until reviewed merge  
**New optimizer policy:** **NO RELATION V10 OR TESSA V2 OPTIMIZER STEP** until each workstream independently reaches its readiness boundary.

## 0. Non-negotiable rules

### One canonical carrier

```text
K x candidate M*
        ↓
static admission + candidate mechanics
        ↓
deterministic selection
        ↓
ONE canonical M/G/W_M
        ↓
Graphics + Motion + Runtime
```

A second deformable visual mesh remains forbidden.

### V8 carrier seal

```text
Geometry + Mechanics → shared H → M1 seal → specialist learning
```

Do not reopen `scene_core` merely because a specialist path is weak. Reopening H requires a separate representation-falsification court.

### Truth / conditioning

- unknown is first-class;
- unidentifiable presentation/semantic truth abstains;
- view-conditioned truth requires matching conditioning;
- deterministic geometry already available to the system should be explicit conditioning rather than something a latent head must rediscover;
- authored-motion evidence is not automatically training truth;
- Compiler may qualify evidence but may not rescue model incapacity and convert it into scientific success.

### Source vs science

- source compile/test green = implementation evidence only;
- loss decreasing = optimization evidence only;
- teacher-forced topology CE = not free-running topology quality;
- product qualification = not automatically a model-learning PASS.

## 1. Historical anchors

### V8

Immutable completed baseline:

- source `5083c9915533ec8ec6530c7d229c635887abfea7`;
- step `12000`;
- checkpoint SHA256 `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding SHA256 `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`.

Keep the V8 carrier seal unless a proper raw-H probe falsifies it.

### V9

Focused relation `ATTEMPT_001` completed 1600 steps at source `7029ee3f48787aa9f7e27eecef89155eb9368a96`.

Scoped earned result:

- projected alpha boundary BA ~0.7599;
- projected appearance boundary BA ~0.8209;
- front ~0.5588;
- visibility A ~0.5659;
- visibility B ~0.5534;
- occlusion support positive-only and not binary-court eligible.

V9 is retained as a clean-boundary experiment, not final relation architecture proof.

## 2. Program split after the architecture audit

Two parallel research workstreams now exist:

```text
WORKSTREAM R — RELATION / VISIBILITY
  truth + conditioning + representation probes

WORKSTREAM S — SURFACE / STRUCTURALIZATION
  direct high-res IRIS surface A-arm
  then optional TESSA structuralization B-arm
```

Neither workstream authorizes the other.

## 3. Workstream R — Relation Truth V3 / V10 remains blocked

### 3.1 What changed architecturally

Relation V3 source now follows:

```text
raw H_A/H_B
+ XYZ_A/XYZ_B
+ ΔXYZ / distance
+ camera-space depth_A/depth_B/Δdepth
+ projected UV_A/UV_B/ΔUV
+ same-ray residual
+ camera embedding
        ↓
pairwise front/order head
```

Visibility is a separate complete-ray-stack problem.

The pair path no longer emits free neural confidence or pairwise global visibility.

### 3.2 Why V9 did not falsify H

V9 trained only a view branch on top of the historical frozen relation embedding. Therefore weak front performance establishes:

> historical relation embedding + V9 view path is weak for ordering

not:

> raw sealed H contains no ordering information.

If needed before V10, use the frozen-carrier probe ladder:

```text
A: historical 64-D relation embedding + camera
B: raw H pair + fresh head + camera
C: raw H + explicit geometry + camera
D: C + complete ray-stack context
```

Only A≈B≈C≈D justifies suspecting H itself.

### 3.3 Truth/support prerequisites

Before V10 step 1:

1. exact all-hit relation evidence artifact;
2. complete stack support per subject/view;
3. true global visibility labels decoupled from pairwise front;
4. authored-motion pair/order flip statistics;
5. conditioning disposition for motion-derived truth;
6. no fabricated occlusion negatives;
7. same/cut/attachment abstention unless new independent authority exists;
8. Truth V3 manifest with explicit validity/provenance;
9. support/class-balance/split leakage court;
10. hot-path frozen hash fix;
11. V9-final vs neutral initialization court;
12. preregistered absolute held-out thresholds;
13. materialization/round-trip/readiness seal.

### 3.4 V10 model capacity policy

Do not increase relation capacity merely because V9 front was weak. First resolve:

```text
truth → conditioning → representation bottleneck → pair-vs-stack abstraction
```

Only then use capacity/adapters if the clean-target court demonstrates need.

## 4. Workstream S — direct IRIS high-resolution surface first

This is the current immediate execution workstream.

Scientific question:

> Before TESSA learns anything, how good is the exact completed V8 continuous field when decoded through the selected high-resolution sparse surface path?

### 4.1 Exact direct A-arm

```text
sealed V9 prepared 8-view observations
        ↓
exact completed V8 IRISVNextV1
        ↓
sealed V8 scene_core / raw scalar field
        ↓
selected T512 coarse → sparse T1024 zero-surface decoder
        ↓
model-only surface candidate
        ↓
independent source-fidelity + Compiler/G5/G3 courts
```

No TESSA is involved in A.

### 4.2 Input authority

Machine-readable binding:

`canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json`

Prepared cohort:

- source `7029ee3f48787aa9f7e27eecef89155eb9368a96`;
- V4 manifest SHA256 `6561bf2087f4fac3699c673b5c93fb3d8dbc2d1ff90a5065d976cae70688b4d8`;
- exact 8 shard Drive IDs + SHA256s;
- exact preparation receipt.

V8 checkpoint:

- exact file ID bound;
- exact SHA256 `0be15df...`;
- step 12000 and binding `735b37f...` required;
- sealed scene_core required.

### 4.3 Full-resolution policy

Only `SparseRegularTetraPolicyV5()` is scientific baseline authority:

- base cells = `512`;
- fine cells = `1024`;
- no deformation;
- raw field sign unchanged;
- normalized coordinate mapping unchanged;
- no uncalibrated uncertainty;
- no extraction delta in the baseline;
- FP32/no-autocast.

The execution CLI intentionally has no base/fine resolution flags.

### 4.4 Exact source path

- `models/iris/vnext/sparse_surface_v3.py`
- `tools/iris_vnext/hydrate_fit8_direct_surface_inputs_v1.py`
- `tools/iris_vnext/preflight_fit8_direct_surface_a100_v1.py`
- `tools/iris_vnext/materialize_fit8_direct_surface_v1.py`
- `docs/research/IRIS_DIRECT_SURFACE_BASELINE_EXECUTION_PLAN_20261019.md`

### 4.5 A100 engineering preflight

Preflight is not a reduced scientific experiment.

It may:

- hydrate exact bytes;
- load exact V8;
- encode one real prepared subject;
- benchmark the real 32k field-query chunk;
- measure VRAM/host RAM/disk/throughput;
- estimate engineering runtime.

It may not:

- generate a reduced mesh;
- claim geometry quality;
- change resolution;
- run an optimizer.

Current admission envelope requires >=39 GiB CUDA memory plus host/disk constraints. This admits A100-40GB class or better rather than silently using the 1660Ti self-hosted runner for the full scientific materialization.

## 5. Direct surface source-fidelity court

After all 8 full surfaces exist, evaluate from independent source-defensible authority:

- silhouette/mask fidelity;
- depth fidelity;
- normal fidelity;
- surface distance including p95/max;
- finite/domain validity;
- degenerate faces;
- connected components;
- boundary/open-chain diagnostics;
- thin-structure continuity where the observation can actually identify it.

Do not invent hidden RGB/semantic truth.

The threshold document must be locked before using the same measurements to decide TESSA promotion.

## 6. Direct surface Compiler court

A model-only Stage12-like candidate is not canonical M.

Required A-arm qualification:

1. static Compiler qualification;
2. source/support binding;
3. G5/multiview qualification;
4. G3/authored-motion qualification;
5. no unacceptable component/non-manifold/degenerate/open-chain failure;
6. exact receipt/hash lineage.

Compiler repair must be reported separately from raw model-surface quality.

## 7. TESSA V2 role and capacity

Only after A is sealed do we fit/evaluate B.

TESSA V2 is an optional second learned structuralization layer:

```text
same IRIS evidence / high-res witness
        ↓
TESSA V2
        ↓
structured mesh proposal
        ↓
Compiler
```

Reference-class first serious fit:

- d_model 1024;
- 16 heads;
- 24 decoder layers;
- FFN 4096;
- 1024 coordinate bins;
- 512 scene memory tokens;
- 512 surface/mechanics memory tokens;
- local exact context 4096;
- causal global structural-history summaries.

Do not start from a reduced diagnostic model and infer incapacity from that result.

## 8. TESSA objective / free-running court

Training objective:

```text
L_tessa =
    λ_token   * next-token CE
  + λ_coord   * metric coordinate loss
  + λ_special * structural-token loss
```

A falling loss is not a topology PASS.

Independent free-running court must inspect:

- grammar/EOS completion;
- component count;
- boundary loops/open chains;
- non-manifold edges;
- degenerate faces;
- source-surface p95/max;
- mask/depth/normal preservation;
- support binding;
- static/G5/G3 Compiler results;
- face-count/structural economy;
- cohort worst subject.

## 9. TESSA A/B promotion policy

```text
A = exact V8 → full high-res direct surface → Compiler
B = same admissible IRIS evidence → TESSA V2 → Compiler
```

TESSA is promotable only if B:

1. preserves static Compiler qualification;
2. preserves G5;
3. preserves G3;
4. preserves support binding;
5. remains inside preregistered source-surface p95/max tolerance;
6. preserves mask/depth/normal fidelity;
7. does not introduce unacceptable component/non-manifold/degenerate/open-chain regression;
8. provides meaningful structural economy/topology value.

If A passes and B provides no meaningful benefit, prefer A.

## 10. Other authority cleanup preserved

### Confidence / uncertainty

No inference scalar may influence product authority without supervision/calibration/deterministic derivation or explicit diagnostic-only typing.

### Contact persistence

Current Fit8 preparation has no authoritative `contact_face_mask`; contact persistence remains invalid/abstained.

### Geometry supervision after baseline

If A reveals genuine field/extraction quality debt, the next geometry training design may adopt source-defensible:

- mask/silhouette;
- depth;
- normal;
- surface distance;
- FlexiCubes deviation;
- continuity/hole regularization.

Do not increase spatial resolution further until the independent court shows the selected T512/T1024 path is the limiting factor rather than learned-field quality.

## 11. Current execution order

### Immediate — Workstream S

```text
S0. exact final PR #122 source gate green
S1. hydrate canonical direct-surface binding read-only
S2. verify hydration receipt and exact bytes
S3. run A100-class engineering preflight
S4. if resource PASS, execute full 8-subject T512→T1024 materializer
S5. inspect 8/8 receipts; no quality conclusion yet
S6. lock and run source-fidelity court
S7. run direct-surface Compiler + G5 + G3 courts
S8. seal direct A-arm baseline
S9. build TESSA V2 exact teacher/readiness corpus
S10. run reference-capacity TESSA V2 fit
S11. free-run B and execute preregistered A/B court
```

### Parallel — Workstream R

```text
R0. finish/inspect exact richer relation evidence materialization
R1. lock authored-motion conditioning disposition
R2. define Truth V3 from measured authority
R3. support/split/abstention courts
R4. raw-H/geometry/ray-stack probe if required
R5. qualify initialization
R6. fix frozen-hash hot path
R7. preregister V10
R8. materialize/readiness
R9. only then optimizer step 1
```

## 12. Current hard stops

Do **not**:

- run TESSA training before A-arm baseline exists;
- run V10 relation training before Relation Truth V3 readiness;
- use 1660Ti source runner as full high-resolution scientific surface lane;
- reduce T512/T1024 to obtain a quicker scientific answer;
- call source CI a surface/TESSA PASS;
- fabricate occlusion negatives/contact negatives/hidden appearance truth;
- reopen H because V9 ordering was weak;
- let Compiler repair hide raw model failure.

## 13. Next artifact threshold

The next meaningful scientific artifact is not another architecture commit. It is:

`FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json`

with all eight exact-V8 subjects materialized at full T512→sparse T1024 and status:

`MATERIALIZED_MODEL_SURFACES__COMPILER_COURT_PENDING`

Only after that artifact exists should the program move to the direct-surface quality/Compiler court and then TESSA training readiness.
