# RealSaS-OPT — Current Plan

**Date:** 2026-10-19  
**Status:** **POST-V8 QUALITY PROGRAM — NATIVE V8 SURFACE CONTROL BEFORE HIGH-RES A-ARM REDESIGN**  
**Active source:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Review vehicle:** draft PR #122  
**Production authority:** `main` until reviewed merge  
**Optimizer policy:** **NO RELATION V10 OR TESSA V2 OPTIMIZER STEP** until each workstream independently reaches its readiness boundary.

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

Do not reopen `scene_core` merely because a specialist or extraction transition is weak. Reopening H requires a separate representation-falsification court.

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
- product qualification = not automatically a model-learning PASS;
- a fail-closed scientific extraction result remains a failure; do not relax the observed court after seeing it.

## 1. Historical anchors

### V8

Immutable completed baseline:

- source `5083c9915533ec8ec6530c7d229c635887abfea7`;
- step `12000`;
- checkpoint SHA256 `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding SHA256 `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`.

Keep the V8 carrier seal unless a proper raw-H court falsifies it.

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

## 2. Program split

Two parallel post-V8 workstreams remain:

```text
WORKSTREAM R — RELATION / VISIBILITY
  truth + conditioning + representation probes

WORKSTREAM S — SURFACE / STRUCTURALIZATION
  native V8 surface control
  → corrected high-resolution direct A-arm
  → optional TESSA B-arm
```

Neither workstream authorizes the other.

## 3. Workstream R — Relation Truth V3 / V10 remains blocked

Relation V3 source uses:

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

Visibility is a separate complete-ray-stack problem. Pair relation emits no free neural confidence or pairwise global visibility.

Before V10 step 1 still require:

1. exact all-hit relation evidence artifact;
2. complete stack support per subject/view;
3. true global visibility labels decoupled from pairwise front;
4. authored-motion pair/order flip statistics;
5. conditioning disposition for motion-derived truth;
6. no fabricated occlusion negatives;
7. same/cut/attachment abstention unless independent authority exists;
8. Truth V3 manifest with explicit validity/provenance;
9. support/class-balance/split leakage court;
10. hot-path frozen hash fix;
11. V9-final vs neutral initialization court;
12. preregistered held-out thresholds;
13. materialization/round-trip/readiness seal.

Do not increase relation capacity merely because V9 front was weak. Resolve truth → conditioning → representation bottleneck → pair-vs-stack abstraction first.

## 4. Workstream S — raw-field sparse A-arm result

### 4.1 What was tested

The first post-V8 direct A-arm implementation deliberately isolated the exact completed V8 raw scalar field:

```text
sealed V9 prepared 8-view observations
        ↓
exact completed V8 IRISVNextV1
        ↓
sealed scene_core / raw scalar field only
        ↓
T512 coarse → sparse T1024 historical zero-surface decoder
        ↓
model-only candidate
```

Fixed policy:

- `SparseRegularTetraPolicyV5()`;
- base cells `512`;
- fine cells `1024`;
- raw field sign identity;
- no extraction delta;
- no learned FlexiCubes cube parameters;
- FP32/no-autocast;
- no TESSA;
- no reduced-resolution scientific escape.

This made the experiment clean, but runtime evidence now shows that this was not a valid baseline for the **V8-trained extraction semantics**.

### 4.2 Exact runtime falsification

A100 engineering preflight passed. Full materialization at apparatus source
`d9701923b2c19bd1c9b9771fe071bb92264a041d` then failed on the first subject, Snake:

```text
failed stage = INDEXED_MT_BUILD_AND_NORMALS
exception    = SparseSurfaceTruncationError
message      = SPARSE_SHELL_TRUNCATES_ZERO_SURFACE:62735467 boundary crossing tetrahedra
```

Measured before fail-closed termination:

- coarse field range `[-0.3515409529, 0.5678277612]`;
- T512 zero bracket = true;
- T512 query points = `135,005,697`;
- refined coarse cells = `6,525,575` (`~4.8619%`);
- candidate fine tetrahedra = `313,227,600`;
- crossing tetrahedra = `95,514,148`;
- conservative boundary-crossing tetrahedra = `62,735,467`;
- exact faces counted = `125,076,400`;
- fine field range `[-0.1913980544, 0.2545053363]`;
- fine zero bracket = true.

Decision:

> **RAW V8 FIELD + HISTORICAL DISTANCE-LIKE SPARSE SELECTOR IS FALSIFIED AS THE CURRENT DIRECT A-ARM BASELINE.**

Do not:

- widen `refine_band` after observing the result;
- set `fail_on_boundary_crossing=False`;
- flip/repair field signs;
- accept the enormous raw candidate as a scientific success;
- interpret the failure as proof that sealed H is bad.

The boundary-crossing count is conservative because boundary-parent cells label all crossing tetrahedra inside them; the count is not a literal exact clipping census. That caveat does not erase the independently huge crossing/face complexity.

## 5. Why the failed A-arm did not test V8's trained surface witness

V8 G2 surface supervision used:

```text
native R24 grid
        ↓
query_evidence
        ├─ field
        ├─ H
        └─ learned extraction_delta
        ↓
learned 21-D cube parameters from cube_extraction(H corners)
        ↓
pinned FlexiCubes
        ↓
extracted witness
        ↓
teacher-surface Chamfer + FlexiCubes deviation/L1
```

The raw zero-field alone was only one input to this learned extraction system.

Therefore the next question is not “how far should we widen the T512 shell?” It is:

> Does exact V8 still produce a useful surface through the path that actually received V8 extraction supervision?

## 6. Mandatory native V8 control

New source control:

- `tools/iris_vnext/audit_v8_raw_field_sparse_compatibility_v1.py`;
- `tools/iris_vnext/materialize_fit8_v8_native_flexicubes_v1.py`;
- `tools/iris_vnext/audit_fit8_v8_native_flexicubes_control_v1.py`;
- `tests/iris/test_v8_surface_control_v1.py`.

The native materializer must use:

- exact V8 checkpoint and binding;
- exact sealed V4 prepared observations;
- exact V8 `extraction_training_resolution = 24`;
- model field;
- learned extraction delta;
- learned 21-D cube parameters;
- pinned upstream FlexiCubes source;
- FP32/no-autocast;
- zero optimizer steps;
- no teacher topology/input in inference forward;
- no TESSA.

Status is measurement-only:

`MATERIALIZED_NATIVE_V8_FLEXICUBES_WITNESSES__MEASUREMENT_ONLY`

It must explicitly retain:

`high_resolution_capability_claimed = false`.

## 7. Native control court

For all eight Fit8 subjects measure independently:

### Geometry

- symmetric surface distance / Chamfer;
- candidate→teacher and teacher→candidate p50/p95/p99/max;
- mechanical surface distance;
- normal alignment against sealed V4 mechanical normals;
- finite/domain validity;
- degenerates;
- component count;
- boundary/non-manifold diagnostics.

### Exact source masks

Using the already admitted Stage01 cameras/masks and existing conservative triangle/unit-pixel-square raster semantics:

- 8 views × 8 subjects = 64 masks;
- IoU;
- precision;
- recall;
- symmetric maximum mask gap.

There is still no preregistered scientific PASS threshold for this control. It is a diagnosis separating native V8 surface quality from high-resolution transition failure.

## 8. Raw-field compatibility forensic

The preserved Snake `.work/Snake/v5_coarse_field.f32` remains useful evidence.

`audit_v8_raw_field_sparse_compatibility_v1.py` measures without re-querying the model:

- coarse sign-crossing cell support;
- band-only refinement support;
- boundary-parent fraction;
- sign-crossing boundary-parent fraction;
- domain-boundary sign support;
- grid-edge sign-flip rate;
- finite-difference field-change / spatial-step indicators.

These are **distance-like/Lipschitz compatibility indicators**, not a formal Eikonal proof.

The audit cannot override the failed extraction.

## 9. High-resolution transition redesign boundary

Do not naively execute V8 native FlexiCubes at T1024.

V8 `maximum_extraction_delta = 0.03` is an absolute normalized displacement. Nominal grid step is:

- R24: `2/24 ≈ 0.08333` → max delta ≈ `0.36` cell;
- T1024: `2/1024 ≈ 0.001953` → same max delta ≈ `15.36` cells.

So blindly reusing the native extraction-delta head at T1024 changes its geometric meaning by more than an order of magnitude.

After native R24 measurement:

```text
IF native R24 witness is good:
    treat debt as high-resolution transition / field regularity / extraction semantics
    design resolution-aware high-res surface path
    preserve sealed H unless separately falsified

IF native R24 witness is poor:
    reopen geometry/extraction quality workstream
    audit G2/native witness quality first
    still require a separate court before reopening shared H
```

Possible future interventions are eligible only after this split is measured:

- resolution-normalized extraction deformation;
- surface-conditioned sparse refinement rather than raw-field-magnitude shell alone;
- field gradient/Eikonal-like regularity only if source/ablation justifies it;
- mask/depth/normal/surface supervision through the chosen high-res transition;
- continuity/hole regularization under source-defensible selectors.

No intervention is authorized merely because the raw sparse adapter failed.

## 10. Exact input authority

Machine-readable byte binding remains:

`canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json`

Prepared cohort:

- source `7029ee3f48787aa9f7e27eecef89155eb9368a96`;
- V4 manifest SHA256 `6561bf2087f4fac3699c673b5c93fb3d8dbc2d1ff90a5065d976cae70688b4d8`;
- exact preparation receipt and 8 shard IDs/hashes.

Stage01 bundle SHA256:

`9d9c9cbf63c979326cd2f39f5435006b81dddb201028a34ebb4cf10e1b434936`

V8:

- exact checkpoint SHA256 `0be15df...`;
- step 12000;
- binding `735b37f...`;
- sealed scene_core required.

The binding's T512/T1024 purpose now describes a falsified experiment and must not be silently treated as a successful scientific A-arm.

## 11. TESSA V2 remains blocked

TESSA V2 stays optional and untrained.

Reference-class first serious fit remains:

- d_model 1024;
- 16 heads;
- 24 decoder layers;
- FFN 4096;
- 1024 coordinate bins;
- 512 scene memory tokens;
- 512 surface/mechanics memory tokens;
- local exact context 4096;
- causal global structural-history summaries.

But no TESSA optimizer step until a corrected direct A-arm reaches readiness.

The raw-field sparse failure is not permission to skip A and move straight to B.

## 12. Future A/B promotion policy

Once a corrected high-resolution A-arm exists:

```text
A = exact V8/admitted IRIS evidence → corrected high-res surface witness → Compiler
B = same admissible IRIS evidence/witness → TESSA V2 → Compiler
```

TESSA is promotable only if B:

1. preserves static Compiler qualification;
2. preserves G5;
3. preserves G3;
4. preserves support binding;
5. stays inside preregistered source-surface p95/max tolerance;
6. preserves mask/depth/normal fidelity;
7. avoids unacceptable component/non-manifold/degenerate/open-chain regression;
8. provides meaningful structural economy/topology value.

If A passes and B provides no meaningful benefit, prefer A.

## 13. Current execution order

### Immediate — Workstream S

```text
S0. keep current PR #122 source gates green
S1. run preserved Snake raw-field sparse compatibility forensic
S2. materialize exact V8 native R24 FlexiCubes witnesses for all Fit8 subjects
S3. run sealed-V4 geometry + exact Stage01 64-view mask control
S4. record native-surface-vs-high-res-transition diagnosis
S5. define corrected resolution-aware high-resolution A-arm
S6. preregister its source-fidelity and failure courts before evaluation
S7. materialize corrected A for all 8 subjects
S8. run source-fidelity + Compiler + G5 + G3 courts
S9. seal qualified direct A-arm if earned
S10. only then build/qualify TESSA V2 fit readiness
S11. reference-capacity TESSA fit, strict free-run B, preregistered A/B court
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

## 14. Current hard stops

Do **not**:

- run TESSA training before corrected A-arm readiness;
- run V10 relation training before Relation Truth V3 readiness;
- use a reduced grid to turn the failed raw sparse path into a scientific answer;
- widen sparse thresholds after seeing Snake;
- disable sparse truncation fail-closed policy to force output;
- naively apply native R24 extraction delta at T1024;
- call native R24 measurement a high-resolution PASS;
- call source CI a surface/TESSA PASS;
- fabricate occlusion negatives/contact negatives/hidden appearance truth;
- reopen H because an extraction transition failed;
- let Compiler repair hide raw model failure.

## 15. Next artifact threshold

The next meaningful scientific artifacts are now, in order:

1. `V8_RAW_FIELD_SPARSE_COMPATIBILITY.json` from the preserved failed Snake T512 field;
2. `FIT8_V8_NATIVE_FLEXICUBES_BATCH_RECEIPT.json` with 8/8 exact native V8 R24 witnesses;
3. `FIT8_V8_NATIVE_FLEXICUBES_CONTROL_COURT.json` containing sealed-V4 geometry and exact Stage01 64-view mask measurements.

Only those measurements authorize the next high-resolution transition design. They do not authorize TESSA training by themselves.