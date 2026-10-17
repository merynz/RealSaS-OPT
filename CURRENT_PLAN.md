# RealSaS-OPT — Current Plan

**Date:** 2026-10-16
**Status:** USER-AUTHORIZED IRIS VNEXT / SINGLE-CARRIER / GRAPHICS IMPLEMENTATION
**Review line:** `research/iris-vnext-single-carrier-graphics` / draft PR #116
**Production authority:** `main` until this review line passes and is merged

## 0. Non-negotiable architecture rule

RealSaS must never reintroduce independent mechanical and visual carriers.

```text
K x learned candidate M*
        ↓
Compiler static admission
        ↓
candidate AXIS + MIRA
        ↓
bounded mechanical selection probes
        ↓
Compiler survivor selection
        ↓
ONE canonical M/G/W_M
        ↓
Graphics + Motion + Runtime
```

Candidates are evidence only. No candidate has product authority. There is exactly one canonical mechanics bundle after selection.

Graphics may add 2D presentation support derived from canonical M, but it may not create or deform a second 3D mesh.

---

## 1. IRIS vNext

Keep frozen DINOv2-S as the controlled image-token front end and replace the demo-sized trainable scene core with one stronger shared scene representation.

Baseline:

```text
scene width          512
camera embedding     128
scene layers           6
attention heads         8
FF multiplier           4
triplane          3x96x64x64
field width            384
field residual blocks    6
local H(x)             384
```

The shared core fans out into evidence heads rather than separate scene-understanding networks:

```text
Geometry
  H(x), field, uncertainty,
  bounded extraction displacement,
  21-D Flexi-style cube extraction freedom

MechanicalDemand
  stretch major/minor,
  principal tangent direction,
  bend demand,
  rigidity probability,
  uncertainty

Graphics Appearance
  RGBA candidate/confidence,
  appearance descriptor,
  alpha/appearance boundaries

Graphics Relations
  same-drawable affinity,
  cut probability,
  fragment embedding,
  visibility/front/occlusion,
  attachment affinity,
  boundary evidence,
  uncertainty

Structure
  shared-scene + SurfaceWitness memory
  -> TESSA topology grammar decoder
```

No named semantic taxonomy is permitted.

---

## 2. Training-time mechanical truth

Authored source mechanics are teacher truth only.

```text
authored M/G/W/native motion
        ↓
per-edge strain
per-face deformation/Jacobian
area change
condition ratio
principal direction
bend demand
rigidity
optional persistent contact
        ↓
MechanicalDemandTruth
```

Inference remains image/camera only. Teacher G/W/motion may never enter the inference forward call.

This is not a feedback loop from post-hoc pass/fail. It is direct supervised teaching of the expected mechanical life of a surface.

---

## 3. Structure / TESSA role

TESSA remains useful as an explicit topology language, but it is no longer a second scene-understanding model.

Baseline:

```text
topology width          512
heads                      8
decoder layers              8
memory tokens             256
memory refinement layers    2
```

Structure memory sees:

```text
shared triplane tokens
+
SurfaceWitness {
  XYZ,
  normal,
  H(x),
  uncertainty,
  MechanicalDemandEvidence
}
```

The grammar remains explicit. Candidate generation must produce a bounded diverse set, initially `K=4`.

---

## 4. Geometry / extraction quality

Adopt the useful StdGEN++/FlexiCubes lessons without copying their semantic ontology:

- coarse-to-fine active-surface evaluation;
- learnable bounded extraction freedom;
- connectivity supervision at source;
- high-resolution extraction where topology is actually active;
- model failure rather than large Compiler repair.

Teacher connectivity uses actual authored surface/component truth where valid. Do not invent inside/outside labels for open/non-watertight teachers.

---

## 5. Canonical mechanics timing

The old rule `Stage19 static canonical M -> AXIS/MIRA -> dynamic qualification` is not the vNext authority model.

The vNext mechanics graph is:

```text
MeshCandidateSet
        ↓
static admission only
        ↓
candidate AXIS G
        ↓
candidate MIRA W_M
        ↓
topology-selection mechanical probes
        ↓
SOLE canonical M/G/W_M
```

A separate production dynamic-motion proof remains downstream of the selected canonical puppet. Candidate-selection probes and production motion proof are different courts and must stay different.

Compiler may select among model candidates; it may not design a replacement mesh.

---

## 6. Graphics

`CAA` is legacy compatibility vocabulary only. The active domain name is **Graphics**.

Observed source pixels remain source-wins. Learned appearance can support unobserved/ambiguous areas but cannot silently override observed art.

The model does not output persistent fragment IDs. It emits relation evidence. Compiler canonicalizes identities and solves the global composition problem:

```text
role-free learned relation evidence
+ source RGBA / alpha
+ canonical-M physical depth
+ canonical M/G/W motion
+ persistent contact
+ temporal evidence
        ↓
Compiler global solve
        ↓
GraphicsFragments
per-view visibility
per-view partial order
attachment anchors
fragment motion fields
bounded silhouette support
```

Graphics must bind to the sole canonical M.

---

## 7. Runtime consumption

A hash reference is not behavioral consumption.

vNext RSS/native runtime must materialize and consume:

- canonical face -> Graphics fragment ownership;
- visible fragments per view;
- explicit partial-order edges;
- attachment anchors;
- bounded 2D support parameters.

Native raster ordering rule:

1. if Compiler provides explicit precedence between the two fragments, that precedence wins;
2. otherwise physical camera depth wins;
3. exact depth ties use deterministic fallback;
4. triangle-array order is never presentation authority.

This allows relation evidence to solve real draw-order/occlusion cases without turning the entire character into a fixed painter-order stack.

---

## 8. Paperman-style 2D support

Allow a bounded 2D support sidecar derived from canonical mechanics:

```text
posed canonical M/G/W
        ↓
fragment projection
        ↓
FragmentMotionField
+ bounded SilhouetteSupportBand
        ↓
final Graphics composition
```

This support is not another mesh. It cannot change 3D mechanical geometry or own independent deformation.

Use it to distinguish:

```text
true geometric hole      -> IRIS/topology problem
small projected art crack -> Graphics support problem
```

---

## 9. Compiler rule

> Compiler may enforce truth; Compiler may not compensate for model incapacity.

Compiler owns validation, hashes/IDs/addressing, global relation solving, candidate selection, exact contact/barycentric solves, bounded numerical sanitation and fail-closed qualification.

Large topology repair, large hole filling, hundreds of collapses/relaxations, or semantic skin transfer hiding carrier drift are failures of learned output, not Compiler success.

Record per-subject mutation amount, repair count, invented information, abstention and failure during Fit8 as the behavioral genericity court.

---

## 10. Trainer

Retire the old eight-times-serial model update.

New scientific update:

```text
8 exact subjects resident/pinned
        ↓
B=8 source scene encode
        ↓
batched query banks / teacher surface banks
        ↓
one scalar objective
        ↓
one backward
        ↓
one AdamW optimizer step
```

Use BF16 tensor cores, SDPA/Flash paths and fused optimizer where available. Training does not require bitwise deterministic CUDA kernels. Scientific reproducibility comes from exact source/policy/model/manifest binding, immutable checkpoints, and fixed FP32 evaluation courts.

Use one optimizer lineage with controlled parameter groups and curriculum, not optimizer resets/LR shocks.

Initial curriculum:

```text
G0 engineering throughput preflight
G1 geometry / field
G2 extraction + connectivity + MechanicalDemand
A1 Graphics appearance evidence
R1 Graphics relation evidence
T1 structure / animation-aware topology
J1 controlled joint alignment only if previous courts do not regress
```

---

## 11. Fit8

The legacy run that reached roughly step 5400 is preserved as a control artifact only. vNext invalidates its model/optimizer continuation.

Fit8 remains the same eight subjects and one shared architecture/parameter set. Knight remains a separate regression witness.

Before the fresh vNext run:

1. all vNext CI and legacy-regression gates PASS;
2. native Graphics content consumption PASS;
3. exact vNext eight-subject shards are produced and hashed;
4. MechanicalDemand teacher truth exists for all eight;
5. role-free Graphics teacher truth exists where source evidence supports it;
6. A100 G0 proves B=8 execution, memory headroom and finite numerics;
7. manifest/policy/model/source binding is sealed.

Then start fresh IRIS vNext Fit8. Do not weaken any binding guard to reuse the old checkpoint.

---

## 12. Immediate implementation order

```text
A. model + teacher contracts                         [implemented on PR #116]
B. single-carrier candidate/canonical mechanics     [implemented core]
C. role-free Graphics compiler                      [implemented core]
D. Graphics RSS/native consumption                  [in integration/gate]
E. vNext DAG invariants                             [implemented]
F. exact B=8 Fit8 trainer                           [implemented core]
G. teacher/shard materialization for real Fit8      [next]
H. A100 G0 preflight                                [next]
I. fresh IRIS vNext Fit8                            [after G0 PASS]
```

No merge to `main` until the review branch gates are green and the state/plan files match the code. Do not reactivate PR #113/#114 or the old subject-serial trainer.
