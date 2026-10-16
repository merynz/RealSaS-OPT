# RealSaS-OPT — Current State

**Date:** 2026-10-16
**Canonical production line:** `main` until reviewed vNext is merged
**Active review line:** `research/iris-vnext-single-carrier-graphics` / draft PR #116
**Active program:** IRIS vNext + post-selection single-carrier mechanics + behaviorally consumed Graphics, then fresh Fit8

## Executive state

The earlier narrow `IRIS scene_planes -> TESSA decoder-only` bridge was a useful intermediate result, but it is no longer the complete target. The user has authorized the broader vNext architecture after reviewing the remaining Knight demo failures and the risks of another visual/mechanical authority split.

The central invariant is now:

> Candidate meshes are proposal evidence only. After candidate selection, RealSaS has exactly one canonical `M/G/W_M`. Graphics, Motion and Runtime bind only to that carrier. A second deformable visual mesh is forbidden.

The legacy shared IRIS Fit8 run was resumed past the earlier ~4400 checkpoint and reached approximately **5400 global steps**. It is preserved as a control artifact. It is not continued as the vNext scientific checkpoint because both the model architecture and trainer semantics are changing.

## Why the architecture changed

Current main already has strong post-fit dynamic courts. Stage35 consumes final `W_M` and performs carrier-native deformation/topology compatibility evidence; Stage41 separately proves compiled production motion. The missing idea was not another post-hoc dynamic test. It was to teach the topology-producing model, during training, what mechanical demand a surface is expected to experience.

Authored teacher M/G/W/native motion may therefore produce training-only `MechanicalDemandTruth` such as stretch, principal deformation direction, bend demand, area change, condition ratio, rigidity and optional persistent-contact evidence. None of these teacher mechanics are inference inputs.

Research anchors:

- StdGEN++: high-resolution sparse extraction, learnable extraction freedom, connectivity supervision at source;
- Animation-Aware Quadrangulation / animated FlexiCubes experiments: topology should be designed with expected deformation in mind;
- Paperman: final 2D drawing support need not be identical to the 3D surface, but it must remain owned by the correct 3D mechanical carrier.

## IRIS vNext model

The active implementation is under `models/iris/vnext/`.

```text
8 x RGBA + cameras
        ↓
frozen DINOv2-S
        ↓
shared scene core Z
   ┌──────────────┼──────────────┐
   ↓              ↓              ↓
Geometry       Graphics        Structure
H(x)           evidence        memory
field          appearance      mech demand
uncertainty    relations       TESSA grammar
Flexi-style extraction freedom  
   └──────────────┬──────────────┘
                  ↓
             K x candidate M*
```

Baseline dimensions are scene width 512, six scene layers, eight heads, 96-channel 64x64 triplanes, 384-D local evidence field, and a 512-wide / 8-head / 8-layer TESSA structure decoder with 256 memory tokens.

`H(x)` is a learned substrate, not product authority. The network may emit geometry, extraction, mechanical-demand, appearance and relation evidence, but Compiler remains the product authority boundary.

No named body-part semantic taxonomy is used.

## Mechanical-demand supervision

`models/iris/vnext/teacher_truth_v1.py` derives training-only mechanical labels from authored animation. The inference model predicts:

```text
stretch_major
stretch_minor
tangent_direction_2theta
bend_demand
rigidity_probability
uncertainty
```

Teacher G/W/motion bytes are explicitly forbidden from the inference forward path.

## Candidate meshes and canonicalization

The previous timing `static M -> canonical M -> AXIS/MIRA -> dynamic test` is retired in vNext.

The new logical mechanics path is:

```text
IRIS vNext / Structure
        ↓
MeshCandidateSet {M*0..K}
        ↓
Compiler static admission only
        ↓
candidate AXIS G
        ↓
candidate MIRA W_M
        ↓
bounded topology-selection mechanical probes
        ↓
Compiler deterministic survivor selection
        ↓
SOLE canonical M/G/W_M
```

`compiler/realsas_compiler_core/mechanical_candidate_v1.py` implements the authority boundary. A candidate that claims canonical authority before selection fails closed.

The candidate-selection probe is not Stage41 production motion proof. The two courts remain distinct: one selects a viable carrier; the later one validates actual compiled motion on the selected canonical puppet.

## Graphics replaces CAA as active product vocabulary

`CAA` remains only as historical compatibility terminology for existing artifacts/codecs. The active vNext product domain is **Graphics**.

The learned relation interface is role-free:

```text
same_drawable_affinity
cut_probability
fragment_embedding
visibility_probability
front_probability
occlusion_probability
rigidity_probability
attachment_affinity
alpha_boundary_probability
appearance_boundary_probability
uncertainty
```

No `arm`, `leg`, `cape`, `sword`, `hair`, etc. labels exist in the production contract.

Compiler combines learned evidence with source RGBA/alpha, physical depth, canonical M/G/W motion and persistent contact to produce:

```text
GraphicsFragment ownership on canonical M
per-view visibility
per-view partial order
attachment anchors
fragment motion fields
bounded 2D silhouette support
```

Observed source pixels remain source-wins.

## Runtime consumption

A key current-main defect is now explicit: the historical PresentationGraph can be binding-checked without its contents being behaviorally consumed by the RSS/native render path. vNext forbids this state.

The new implementation includes:

- `compiler/realsas_compiler_core/graphics_v1.py` — role-free global Graphics compile on canonical M;
- `compiler/realsas_compiler_core/graphics_runtime_v1.py` — exact face-to-fragment ownership, visibility, per-view order and attachments;
- `compiler/realsas_compiler_core/runtime_graphics_binary_v1.py` — `graphics.bin` RSS payload and manifest contract;
- `runtime/realsas_cpp/src/graphics_v1.hpp` — native parser and relation-aware layer comparator.

Graphics precedence overrides pseudo-3D depth only for fragment pairs with explicit compiled precedence. Unrelated pairs retain physical depth and deterministic tie-breaking. Triangle array index is forbidden as presentation authority.

The Paperman-style support band is 2D-only and bounded; it never creates a second mesh or changes canonical mechanical geometry.

## Trainer

The old subject-serial shared trainer is retired for vNext.

`models/iris/vnext/cohort_v1.py` implements the new Fit8 contract:

```text
all 8 exact subjects
        ↓
one resident batched cohort
        ↓
B=8 scene encode
        ↓
batched geometry / teacher-surface queries
        ↓
one loss
        ↓
one backward
        ↓
one optimizer step
```

The checkpoint binding includes exact source files, manifest, policy, model config and optimizer semantics. Resume fails closed on binding or payload drift. CUDA training is BF16/tensor-core oriented; fixed FP32 evaluation courts provide scientific comparability rather than forcing bitwise training kernels.

## Current implementation evidence

The dedicated `iris-vnext-single-carrier-v1` CI gate has already passed the first model/teacher/single-carrier/Graphics contract set and legacy TESSA regressions. The gate is being expanded to cover the new B=8 trainer, vNext DAG and native Graphics parser/ordering consumer.

The review branch intentionally does not claim Fit8 scientific PASS.

## vNext execution graph

`canonical/VNEXT_EXECUTION_GRAPH_V1.json` is the machine-readable intended DAG. It contains exactly one canonical mechanics stage:

```text
05 candidate set
06 static admission
07 candidate AXIS
08 candidate MIRA
09 candidate mechanical probes
10 SOLE canonical M/G/W
11 source-wins Graphics appearance compile
12 Graphics relation compile
13-15 production motion + proof
16 bounded 2D Graphics motion sidecars
17-19 runtime package + native Graphics consumption
20 closure
```

Appearance/Graphics compile is downstream of canonical selection; candidate Graphics cannot become a parallel product carrier.

## Compiler law

> Compiler may enforce truth; Compiler may not compensate for model incapacity.

Allowed: validation, IDs/hashes/addressing, very small numerical sanitation, global relation/constraint solving, candidate selection, exact contact/barycentric solve, fail closed.

Large topology repair, large hole closure, semantic skin transfer hiding carrier drift, or an independent visual mesh are model/architecture failures and cannot be silently repaired into a PASS.

## Fit8 re-entry gate

The next scientific IRIS Fit8 run is fresh vNext training, not a continuation of the ~5400 legacy optimizer state.

Before starting it, all of the following must be true:

1. vNext model/teacher/single-carrier/Graphics/native-runtime contracts PASS;
2. exact eight-subject vNext cohort shards exist and are hash-bound;
3. training-only MechanicalDemand truth exists for all eight admitted subjects;
4. the B=8 A100 engineering preflight proves real batched execution, memory use and stable numerics;
5. source/model/policy/manifest binding is sealed;
6. no subject-specific architecture or Compiler branch exists.

At that point the program returns to IRIS Fit8.

## Deferred, not deleted

One-image synthesis, named semantic taxonomies and broad semantic-owner census work are not on the active path. Historical PRs #113/#114 remain deferred and non-authoritative. Their useful research provenance is preserved.

## Required read order

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `canonical/VNEXT_SINGLE_CARRIER_CONTRACT_V1.json`
4. `canonical/VNEXT_EXECUTION_GRAPH_V1.json`
5. `docs/research/IRIS_VNEXT_SINGLE_CARRIER_GRAPHICS_V1.md`
6. `AGENTS.md`
7. `SYSTEM_INDEX.md`
8. `docs/corpus/FIT8_ADMISSION.md`
9. historical Knight/V2 records only as regression/provenance

When the user says `continue / devam et`, continue closing this vNext review line toward the fresh IRIS Fit8 re-entry gate. Do not reactivate the old subject-serial trainer or an independent visual-mesh architecture.
