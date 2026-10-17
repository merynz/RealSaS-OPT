# RealSaS-OPT — Current State

**Date:** 2026-10-16  
**Canonical production line:** `main` until reviewed vNext is merged  
**Active review line:** `research/iris-vnext-single-carrier-graphics` / draft PR #116  
**Active program:** close IRIS vNext engineering, then fresh scientific Fit8

## Executive state

The user-authorized vNext architecture is built around one non-negotiable invariant:

> Candidate meshes are proposal evidence only. After bounded candidate evaluation, RealSaS mints exactly one canonical `M/G/W_M`. Graphics, Motion and Runtime bind only to that selected carrier. A second deformable visual mesh is forbidden.

The earlier legacy shared-IRIS Fit8 run resumed from committed step 4400, committed through **5400**, executed through **5563**, and was interrupted with 5400 as the exact resume point. That run is preserved as a control artifact only. vNext does **not** resume its optimizer state.

## Current-main continuity while vNext is reviewed

This branch does not reinterpret historical/current-main PASS records. Until vNext is reviewed and merged:

- `main` remains production authority;
- `canonical/V2_IMPLEMENTATION_READINESS.json` remains the current-main readiness record;
- existing SurfaceAddressing / canonical-carrier / Knight regression seals remain valid historical/current-main evidence;
- vNext files are review-line evidence only.

`PLATFORM_EXECUTION_LANES` remains governed by the existing current-main platform authority map. The vNext review does not silently rename, delete or replace those execution-lane contracts.

## Why vNext exists

Current main already has post-fit dynamic courts. Stage35 tests final `W_M` with carrier-native deformation/topology evidence; Stage41 separately proves compiled production motion. The missing capability is **training-time mechanical understanding**, not another post-hoc gate.

Authored teacher M/G/W/native animation may therefore derive training-only `MechanicalDemandTruth` such as stretch, principal deformation direction, bend demand, area change, conditioning, rigidity and optional contact persistence. Teacher rig/skin/motion is never an inference input.

## IRIS vNext model

Active model namespace: `models/iris/vnext/`.

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
FlexiCubes extraction
   └──────────────┬──────────────┘
                  ↓
             K x candidate M*
```

Baseline capacity remains 512-wide shared scene reasoning, 6 scene layers, 8 heads, 96-channel 64x64 triplanes, 384-D local `H(x)`, and a 512-wide / 8-head / 8-layer TESSA structure decoder with 256 memory tokens.

No named body-part taxonomy (`arm`, `cape`, `sword`, etc.) exists in the production contract.

## FlexiCubes extraction contract

The 21 extraction values are no longer treated as arbitrary supervised targets. Their exact semantics are pinned to NVIDIA FlexiCubes:

```text
12 beta edge weights
8  alpha corner weights
1  gamma quad-split weight
```

Source binding:

- repository: `https://github.com/nv-tlabs/FlexiCubes.git`
- commit: `4cc7d6c3d0cee83c011ce36721b81adff0dd7db6`
- license: Apache-2.0

`models/iris/vnext/flexicubes_v1.py` verifies exact commit/remote/clean source before use. The 21 raw values receive gradients through the pinned differentiable extractor; no `cube_extraction_targets[21]` teacher labels are admitted.

AMP is allowed in the shared scene trunk, but the FlexiCubes extraction court is explicitly FP32.

## Model-produced SurfaceWitness

Teacher geometry is not allowed to become hidden structure-memory input.

The active topology conditioning path is:

```text
model field + H(x)
      ↓
pinned FlexiCubes
      ↓
model-produced surface
      ↓
SurfaceWitness {xyz, normal, H(x), uncertainty, MechanicalDemandEvidence}
      ↓
TESSA grammar decoder
```

IRIS field/extraction coordinates are `[-1,1]^3`. TESSA's established grammar uses `[-0.5,0.5]^3`; the conversion is explicit (`xyz_tessa = 0.5 * xyz_iris`) and StructureMemory fails closed on coordinate-domain drift.

Authored teacher geometry supplies geometry/topology targets, not the inference witness.

## Mechanical-demand supervision

`models/iris/vnext/teacher_truth_v1.py` derives training-only motion supervision. The model predicts:

```text
stretch_major
stretch_minor
tangent_direction_2theta
bend_demand
rigidity_probability
uncertainty
```

Teacher G/W/native-motion bytes remain offline teacher evidence only.

## Candidate mechanics and canonicalization

The old timing

```text
static M -> canonical M -> AXIS/MIRA -> dynamic test
```

is retired for vNext.

The target path is:

```text
K candidate M*
 -> static admission
 -> candidate AXIS G
 -> candidate MIRA W_M
 -> bounded candidate mechanical probes
 -> deterministic survivor selection
 -> SOLE canonical M/G/W_M
```

Candidate-selection probes remain distinct from the later production-motion proof. Compiler may select among proposals; it may not invent a replacement mesh.

## Graphics replaces CAA as active product vocabulary

`CAA` remains historical compatibility terminology only. Active vNext vocabulary is **Graphics**.

The model exposes role-free evidence such as:

```text
same_drawable_affinity
cut_probability
fragment_embedding
visibility_probability
front_probability
occlusion_probability
attachment_affinity
alpha_boundary_probability
appearance_boundary_probability
confidence / uncertainty
```

Compiler combines this with source-wins RGBA/alpha, physical depth, canonical M/G/W motion and contact evidence to solve canonical face ownership, visibility, partial order and attachment anchors.

## Runtime Graphics consumption

Current-main PresentationGraph content can be binding-checked without behaviorally controlling the raster. vNext forbids hash-only Graphics binding.

The review line now includes:

- `compiler/realsas_compiler_core/graphics_v1.py`
- `compiler/realsas_compiler_core/graphics_runtime_v1.py`
- `compiler/realsas_compiler_core/runtime_graphics_binary_v1.py`
- `runtime/realsas_cpp/src/graphics_v1.hpp`
- `runtime/realsas_cpp/src/runtime_v2_graphics_reference.cpp`

The Graphics-native player requires `graphics.bin`; the historical player rejects Graphics-authorized packages. They therefore do not claim authority over the same package.

At an overlap pixel, explicit compiled fragment precedence forms a DAG. Topological order is mandatory for those edges; among unrelated currently admissible fragments, physical depth remains the tie-break. This avoids both hidden triangle-order authority and comparator cycles.

A real RSS/native E2E uses the existing Knight single-M material fixture and requires changing only Graphics visibility content to change native RGBA behavior.

## 2D presentation support

Canonical M remains the only 3D carrier. Paperman-style fragment motion / bounded silhouette support may exist only as 2D Graphics sidecars driven by canonical M/G/W.

```text
posed canonical M/G/W
 -> fragment projection
 -> FragmentMotionField
 -> bounded SilhouetteSupportBand
 -> Graphics composition
```

A geometric hole remains a Geometry failure. A tiny projected support crack may be a Graphics support failure. The sidecar cannot hide missing 3D geometry or mint another mesh.

## Active trainer

The old subject-serial trainer is retired.

`models/iris/vnext/cohort_v1.py` is the migration prototype.  
**Active scientific trainer:** `models/iris/vnext/cohort_v2.py` / `IRISVNextFit8TrainerV2`.

Its curriculum is:

```text
G1 geometry
G2 FlexiCubes surface extraction
M1 mechanical demand
A1 appearance evidence
R1 role-free relations
T1 mechanical-aware topology
J1 joint fine-tune
```

One optimizer update uses one shared B=8 scene encode, no subject-serial model-forward loop, one backward and one optimizer step. Ragged FlexiCubes meshes are extracted per subject only after the common batched field/H query because topology output is inherently ragged.

TESSA teacher sequences are not truncated to one common asset length. Shards contain padded truncated-window banks, valid-window counts and global position offsets. Each subject selects one valid window deterministically; the decoder still receives one B=8 forward with per-subject position offsets.

CUDA trunk training uses BF16/tensor-core paths where supported; extraction is FP32. Exact inputs, policies, model source, pinned FlexiCubes identity and checkpoint payloads are hash-bound.

## Teacher materialization

Two new offline tools define the vNext truth boundary:

- `tools/iris_vnext/blender_authored_teacher_export_v1.py`
  - exact source-hash import;
  - authored rest topology;
  - anonymous mesh-object face ownership;
  - native animation deformation sequence;
  - source-world coordinates only, no canonical authority.
- `tools/iris_vnext/build_fit8_vnext_shards_v1.py`
  - reuses exact frozen DINO/camera/field evidence from legacy Fit8 shards;
  - binds authored teacher evidence to the exact legacy IRIS normalization;
  - derives MechanicalDemand truth;
  - derives source-wins appearance and role-free Graphics relation truth from observation pixels + authored geometry rays;
  - builds authored TESSA topology windows;
  - writes hash-bound vNext shards and cohort manifest.

Direct 21-D FlexiCubes labels and teacher SurfaceWitness inputs are explicitly forbidden by the active trainer.

## Compiler law

> Compiler may enforce truth; Compiler may not compensate for model incapacity.

Allowed: exact validation, IDs/hashes/addressing, bounded numerical sanitation, global relation/constraint solving, candidate selection, exact contact/barycentric solves, fail closed.

Forbidden: large topology/hole repair, semantic skin transfer hiding carrier drift, independent visual mesh, subject-specific rescue branches.

## Fit8 re-entry gate

The next scientific Fit8 run is a fresh vNext run. It does not continue the legacy 5400 optimizer state.

Before starting it:

1. vNext model/teacher/single-carrier/Graphics/native-runtime contracts PASS;
2. native runtime source extension is sealed append-only;
3. exact eight authored teacher exports are materialized and source-hash bound;
4. exact eight vNext shards + cohort manifest are materialized and hash-bound;
5. B=8 A100 engineering preflight passes model, FlexiCubes, TESSA windows, memory and finite-numeric courts;
6. source/model/policy/FlexiCubes/manifest binding is sealed;
7. no subject-specific architecture or Compiler branch exists.

At that point the program is back at **IRIS Fit8** and waits for the user before starting scientific training.

## Deferred, not deleted

One-image synthesis, named semantic taxonomies and broad semantic-owner census work remain deferred. Historical PRs #113/#114 stay non-authoritative research provenance.

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

When the user says `continue / devam et`, continue closing this review line toward the fresh IRIS Fit8 re-entry gate. Do not reactivate the legacy subject-serial trainer or an independent visual-mesh architecture.
