# RealSaS-OPT — Current Plan

**Date:** 2026-10-20  
**Status:** **POST-V8 SURFACE PROGRAM — SURF-A1 CLOSED FAIL; H-REP-P0 SOURCE GREEN; A100 RUN NEXT**  
**Active PR source:** `research/iris-vnext-hybrid-tessa-v2-20261019`  
**Review vehicle:** draft PR #122  
**Production authority:** `main` until reviewed merge  
**Active lane:** `SURFACE_TESSA`  
**Active experiment:** `H-REP-P0`  
**Frozen H-REP-P0 execution source:** `479e8e53cf0d7df234a3e64b824bb704b861bf19`  
**Runtime worker:** `NB-r1` — generated, not yet run  
**Parked lane:** `RELATION_V10`  
**Optimizer policy:** **Only H-REP-P0 low-capacity probes may optimize after the notebook's exact runtime/input/A100 admission passes. V8 H stays sealed. No SURF-A2, TESSA, Relation V10 or carrier-retraining optimizer step is authorized.**

## 0. Program objective

Build one generic visual + geometric + mechanical carrier that is faithful enough to source observations and structurally useful enough for animation:

```text
source observations
   ↓
IRIS continuous learned evidence / H
   ↓
qualified high-fidelity surface
   ↓
optional TESSA structuralization if it proves value
   ↓
Compiler qualification
   ↓
ONE canonical animation-ready M
   ↓
AXIS / MIRA / graphics-motion-runtime consumers
```

The intended synthesis remains:

- **StdGEN++-class lesson:** preserve geometry/raster supervision through representation transitions; address thin/high-resolution structure explicitly;
- **MeshAnythingV2-class lesson:** high-fidelity extracted surface and production topology are different problems;
- **RealSaS law:** external methods are references, not authority; Compiler alone mints canonical M and TESSA stays optional until an A/B court earns it.

## 1. Non-negotiable rules

- one canonical deformable `M`; no independent deformable visual mesh;
- exact V8 `scene_core/H` stays sealed during H-REP-P0;
- teacher topology/rig/skin never enters inference/probe conditioning;
- failed courts are not relaxed after results;
- Compiler may qualify but may not hide model incapacity;
- source CI is not scientific evidence;
- Fit8 uses one shared architecture/contracts and no subject-specific trainable parameters;
- Fit8 does not claim unseen generalization;
- TESSA cannot rescue an unqualified direct surface;
- carrier retraining requires its own future preregistration and is not authorized by A1 failure alone.

## 2. Evidence chain that led here

### SURF-A0 — raw-field sparse path falsified

Exact V8 raw field did not satisfy the historical distance-like `T512→sparse-T1024` materialization policy. Preserve this as a clean falsification of that transition only.

### SURF-C0 — native V8 R24 measured

Native V8 extraction retained useful but insufficient surface fidelity:

- mean subject mask IoU `0.56057`;
- min subject IoU `0.40535`;
- min view IoU `0.34254`;
- max mask gap `392.42 px`;
- mean symmetric Chamfer `0.06120` IRIS.

This diagnostic had no preregistered pass threshold.

### SURF-A1 — frozen-H rescue completed and failed preregistered court

All admission gates passed before step 1:

- exact source `afc18debbc4d5858a4a632884b7cfd2141dcc35a`;
- exact A100 / pinned renderer/extractor;
- R24 native parity faces exact `8/8`;
- all 64 teacher-geometry raster views precision/recall `1.0`;
- sealed scene-core hash unchanged.

The shared extraction-only decoder completed all `2400` steps and improved C0 substantially, but final R48 court remained below preregistration:

- mean IoU `0.77953 < 0.80`;
- mean symmetric Chamfer `0.04305 > 0.04`;
- max gap `258.66 > 128 px`;
- worst teacher→candidate p95 `0.11117 > 0.075`;
- min normal abs-dot `0.57822 < 0.70`.

Meanwhile min subject IoU `0.67901`, min view IoU `0.65213`, non-manifold edges `0`, and zero-area faces `0` passed.

Decision:

> A1 did not earn SURF-A2. Because it improved native C0 strongly, do not infer “H contains no geometry.” First isolate local representation decodability without mesh extraction.

## 3. Active experiment — H-REP-P0

### 3.1 Exact question

Can exact sealed V8 local evidence encode surface geometry strongly enough that a low-capacity point probe can decode it **without** FlexiCubes, a surface decoder, TESSA or Compiler repair?

This experiment separates two hypotheses:

```text
H-REP-P0 PASS
→ local geometry is demonstrably present in H or raw field
→ A1 failure is still compatible with extraction/transition debt
→ redesign extraction transition; do not retrain H yet

H-REP-P0 FAIL
→ local geometry decodability is not demonstrated even before extraction
→ open a separately preregistered carrier-representation retraining study
```

Neither branch authorizes TESSA or SURF-A2 directly.

## 4. Frozen source authority

The exact scientific execution source is frozen at:

`479e8e53cf0d7df234a3e64b824bb704b861bf19`

PR #124 closed by fast-forward to that exact SHA after all required source gates were green:

- `iris-h-rep-p0-source-gate` PASS;
- `model-mainline-source-gate` PASS;
- `iris-tessa-v2-structuralization-source-gate` PASS;
- `iris-v2-source-contract` PASS;
- `architecture-freeze-source-gate` PASS.

Post-freeze documentation commits do not redefine this scientific source.

The frozen runtime additionally enforces:

- exact clean source checkout;
- exact A100-class GPU;
- at least `39 GiB` CUDA memory;
- exact V8/prepared/binding/prereg authority chain;
- no optimizer step before those gates;
- exact per-global-step deterministic minibatch schedule for resume equivalence.

Sampler authority: `RealSaS.HRepP0StepSeededSampler.v1`.

## 5. H-REP-P0 controls

Four probes are trained with **identical architecture and exact equal parameter count**. Every arm receives the same `384 evidence + 3 XYZ = 387` dimensional envelope.

1. `H_XYZ`
   - channels `0:384`: exact sealed local H;
   - channels `384:387`: query XYZ.
2. `RAW_FIELD_XYZ`
   - channel `0`: exact sealed V8 raw field;
   - channels `1:384`: zero;
   - XYZ appended.
3. `XYZ_ONLY`
   - all 384 evidence channels zero;
   - XYZ appended.
4. `SHUFFLED_H_XYZ`
   - within-subject fixed permutation of H, destroying local H↔XYZ alignment while preserving subject-level H statistics;
   - XYZ appended.

No subject ID or semantic label enters any arm.

Probe architecture:

- width `256`;
- `2` residual blocks;
- point-cloud distance head;
- offset-to-reference head;
- normal-up-to-sign head;
- near-surface binary head;
- identical parameter count for all four arms.

## 6. H-REP-P0 truth

Use only exact sealed V4 geometry already admitted for independent surface courts:

- `teacher_surface_points_iris`;
- `mechanical_points`;
- `mechanical_normals`.

Build deterministic train and holdout query sets independently for all eight subjects. Targets:

- nearest-reference distance;
- vector offset to nearest reference point;
- nearest mechanical normal with validity radius;
- near-surface class.

Important claim boundary:

> nearest-point distance here is a **point-cloud geometry proxy**, not an exact signed distance field and not occupancy authority.

Teacher data supplies targets only. It never enters the probe input or V8 forward.

## 7. Fixed training protocol

Per arm:

- one shared model across all eight subjects;
- AdamW;
- `1200` optimizer steps;
- batch `2048`;
- peak LR `2e-4`;
- `60`-step warmup;
- cosine to `2e-5`;
- gradient clip `1.0`;
- checkpoint every `200` steps;
- deterministic subject round-robin;
- deterministic batch indices derived from absolute global step + subject index;
- resumable exact hash-bound checkpoint.

All arms see the same subject schedule and exact batch indices at each global step. Resume cannot restart or shift the minibatch RNG stream.

V8 model remains entirely frozen and its full scene-core hash is checked before/after feature caching.

## 8. Independent held-out court

Preregistered before runtime.

### H_XYZ absolute gates

- mean distance MAE `<= 0.025` IRIS;
- worst-subject endpoint p95 `<= 0.075` IRIS;
- minimum-subject normal abs-dot mean `>= 0.70`;
- minimum-subject near-surface balanced accuracy `>= 0.80`.

### H_XYZ local-information controls

- distance MAE reduction vs `XYZ_ONLY` `>=20%`;
- distance MAE reduction vs `SHUFFLED_H_XYZ` `>=15%`;
- mean near-BA gain vs `XYZ_ONLY` `>=0.08`;
- mean near-BA gain vs `SHUFFLED_H_XYZ` `>=0.05`.

### RAW_FIELD_XYZ gates

Same absolute geometry targets plus:

- distance MAE reduction vs XYZ `>=20%`;
- near-BA gain vs XYZ `>=0.08`.

The carrier-local-geometry result is positive if **H_XYZ passes both its absolute and local-alignment gates OR RAW_FIELD_XYZ passes its absolute and XYZ-control gates**.

Thresholds may not change after results.

## 9. Notebook contract / next execution

Runtime worker `NB-r1` is generated against frozen SHA `479e8e53...` and is the next program boundary. It is not scientific evidence until executed.

Operator UX:

```text
Open → Run all → one Drive approval → autonomous execution
```

Physical phases:

1. one Drive mount + immutable constants;
2. exact detached source checkout + clean-tree gate;
3. dependency/precompile + exact A100 admission;
4. direct-path exact input hydration with zero Drive discovery;
5. deterministic point-proxy dataset build/restore;
6. zero-step scientific admission with frozen V8 scene-core hash and equal-capacity check;
7. four-arm exact-resume training;
8. independent held-out court;
9. typed completion receipt.

Forbidden:

- second OAuth / `auth.authenticate_user()`;
- interactive rclone/gcloud;
- full Drive scan, `os.walk`, `rglob` discovery for known inputs;
- hidden fallback;
- manual cell sequencing;
- FlexiCubes/PyTorch3D/TESSA/Compiler rescue inside H-REP-P0.

## 10. SURF-A2 and TESSA remain blocked

`SURF-A2-HIRES` cannot start merely because H-REP-P0 passes. A PASS first requires an extraction-transition design/preregistration that can produce a new direct A-arm.

`TESSA-V2-B1` remains blocked until that later direct A-arm itself qualifies.

## 11. Relation V10 remains parked

Do not context-switch back to relation training while H-REP-P0 is unresolved. Preserve Relation V10 at its Truth-V3/all-hit/ray-stack readiness boundary.

## 12. Immediate execution order

```text
P0. seal SURF-A1 result in state/ledger                       [DONE]
P1. implement H-REP-P0 source + prereg + controls             [DONE]
P2. source contract / CI green                                [DONE]
P3. freeze exact green SHA 479e8e53...                        [DONE]
P4. generate V8-style multi-cell single-consent zero-scan A100 notebook NB-r1 [DONE]
P5. execute NB-r1 and build deterministic train/holdout point-proxy dataset [NEXT]
P6. cache/verify frozen V8 H + raw field
P7. train four exact equal-capacity arms, 1200 steps each
P8. independent held-out court
P9a. PASS → extraction-transition redesign; keep H sealed
P9b. FAIL → preregister carrier-representation retraining
```

No TESSA or SURF-A2 shortcut exists around this state machine.
