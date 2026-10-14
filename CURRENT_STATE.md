# RealSaS-OPT — Current State

**Date:** 2026-10-14
**Canonical code line:** `main` only
**Active architecture program:** Unified IRIS reconstruction redesign
**Current plan authority:** `CURRENT_PLAN.md`

## Immediate state

The previously started shared IRIS Fit8 A100 training continuation has been **stopped by user decision** before completion because the current IRIS/TESSA representation is no longer accepted as the intended final architecture.

The latest sealed stopped-run checkpoint was approximately global step **4400**. Preserve that run, checkpoint, result and trace as an **IRIS-vCurrent Fit8 control/baseline**. It did not mint Fit8 scientific PASS, unseen/generalization evidence or product authority. Do **not** resume it merely to reach the old 19,200-step schedule.

The active program is now a fresh redesign of the geometry front end:

```text
one image
   ↓
Unified IRIS
   ├─ internally synthesizes canonical 8-view evidence
   ├─ shared high-capacity scene representation
   ├─ geometry / SDF / uncertainty head
   ├─ RGB/RGBA appearance head
   ├─ generic semantic/ownership head
   ├─ bounded extraction Δxyz + configuration head
   └─ integrated topology head (TESSA role folded into IRIS)
   ↓
Compiler qualification / canonical M
   ↓
AXIS G
   ↓
MIRA W_M on exact M
   ↓
Compiler dynamic / appearance / presentation proof
   ↓
Runtime
```

Read `CURRENT_PLAN.md` for the normative design program, open questions, dataset plan, DAG proposal and implementation order.

## User-authorized architecture decisions

The following are active decisions, not pending proposals:

1. **Final customer input is one PNG / one image.** IRIS may internally construct eight canonical 45-degree views.
2. **The new IRIS starts at StdGEN-class representation capacity.** Do not spend a scientific run on a deliberately undersized model whose only purpose is later capacity escalation.
3. **IRIS and TESSA are being partially unified.** The useful explicit topology-generation role survives as an IRIS topology head/module; the separate GSA→17-D→TESSA scene-understanding bottleneck is not the target architecture.
4. **RGB/appearance and semantic/ownership heads are required.** They share the scene representation with geometry.
5. **A bounded surface-extraction deformation/configuration head is required** if implemented safely; this is extraction-carrier deformation, not MIRA skeletal skinning.
6. **Compiler boundaries are kept only where they are genuinely better:** immutable authority, provenance, independent courts, qualification, bounded generic repair, canonical addressing, mechanics/presentation proof and runtime sealing.
7. **The deterministic contained tetra path remains valuable as an independent geometry witness/court**, not as a mandatory lossy neural bottleneck.
8. **AXIS and MIRA remain separate downstream mechanics models for this intervention.** IRIS does not output a finished animated puppet.
9. **The same eight diverse Fit8 subject identities remain the first architecture cohort.** Difficult subjects are not replaced to make the redesign easier.

## Why the old geometry chain is being replaced

The current shipped/research path is scientifically modular but throws away rich scene information between models:

```text
IRIS image tokens / triplane
   ↓
SDF
   ↓
tetra zero surface
   ↓
GSA / RiggingSurface
   ↓
17-D TESSA conditioning
   ↓
second learned surface encoder
   ↓
TESSA topology / XYZ proposal
```

The new target keeps one shared scene representation alive through geometry, appearance, semantic ownership and topology prediction. Compiler stages may derive compact evidence for qualification/mechanics, but they should not force the neural system to discard scene knowledge and relearn it in a second model.

## Current implementation versus target

The repository implementation still contains the existing public model split:

```text
IRIS → TESSA → AXIS → MIRA → Compiler → Runtime
```

Current `SYSTEM_INDEX.md` describes that shipped implementation and remains useful for code ownership/navigation. It must not be mistaken for the newly approved target architecture before implementation lands.

The target under `CURRENT_PLAN.md` is:

```text
Unified IRIS (including topology/TESSA role)
        ↓
Compiler canonical M
        ↓
AXIS
        ↓
MIRA
        ↓
Compiler / Runtime
```

Do not silently relabel the current code as if this target already existed.

## Research-state consequence

This intervention changes together:

- input cardinality and canonical-view synthesis;
- IRIS model capacity;
- shared scene representation;
- geometry extraction;
- RGB/appearance outputs;
- semantic/ownership outputs;
- topology/TESSA boundary;
- compiler data flow;
- teacher dataset;
- released DAG.

Therefore the next scientific execution requires a **new EngineRelease and fresh/explicit child Attempt**. It is not a resume of the old Fit8 IRIS run.

Any existing platform campaign/Attempt rows from the old Fit8 path may still be technically OPEN/RUNNING until operator cleanup is performed. Treat them as **superseded scientific state**, not permission to continue old learned work. Close/exit them through the Go control plane before starting the new attempt; do not edit old ledgers to manufacture a terminal result.

Immutable raw source identities may be reused where exact dependency identity still holds. Observation/model artifacts whose contracts change must be regenerated under the new release.

## New Fit8 meaning

Fit8 remains the heterogeneous architecture cohort:

- Snake
- Bat
- T-Rex
- Horse
- Spider
- GreenBlob
- Shark
- George

The next experiment is no longer "finish IRIS-vCurrent 19.2k". It is:

> **Fit8 Unified IRIS Reconstruction + Canonical M Architecture Acceptance.**

The model must remain shared across the cohort; compiler fixes must remain subject-free. Knight remains a separate fixed regression witness.

Fit8 still establishes architecture/system closure, not learned unseen generalization. LOFO/unseen come after the redesigned architecture closes and is frozen.

## New dataset program

The existing Fit8 authored 3D sources are retained and expanded into a deterministic teacher corpus rather than replaced.

For each asset, the new teacher pipeline should extract source-authored:

```text
object / mesh identity
material slots
geometry / topology
armature / bone hierarchy
skin weights
parent / constraint relations
rigid accessories
available motion
```

and render canonical teacher evidence:

```text
RGB
alpha
depth
normal
instance/material IDs
mechanical-owner evidence
amodal component masks
occlusion graph
attachment graph
```

Primary semantics are category-agnostic instance/ownership/boundary/attachment/deformation-mode evidence. Human-only `body / cloth / hair` taxonomy is not the generic foundation.

The same extractor should later scale to the larger audited authored corpus after Fit8 validation.

## Canonical M target

The intervention is judged by the final qualified `M`, not by a pretty static neural render.

Desired `M` properties:

- strong static geometry and source/canonical-view silhouette fidelity;
- valid/manifold animation-friendly topology;
- no unexplained holes/gaps or pathological slivers;
- coherent owner/material addressing;
- exact provenance back to shared IRIS evidence;
- stable motion stretch/area/Jacobian/condition behavior;
- low projected flip/self-intersection rates;
- suitable basis for AXIS and exact-carrier MIRA;
- 2D-oriented projection quality across eight directions;
- later mobile/LOD reduction without changing the mechanical basis silently.

The deterministic tetra surface remains an independent witness of the implicit field. A learned FlexiCubes-like explicit surface may improve reconstruction/training, while the integrated topology head remains responsible for an animation-oriented mesh proposal until dynamic evidence proves a simpler replacement.

## Compiler policy during redesign

Keep compiler work that is demonstrably valuable:

- source/generated provenance;
- independent geometry courts;
- static mesh qualification;
- bounded generic repair;
- canonical M promotion;
- SurfaceAddressing;
- owner/fragment resolution;
- source-wins appearance qualification;
- AXIS/MIRA binding;
- dynamic motion proof;
- contact/grip/order/presentation proof;
- runtime package/native parity.

Retire or redesign work that exists only because separate models forgot each other's information, including the GSA→17-D→second-scene-encoder bottleneck and support reconstruction that can be replaced by direct shared-field bindings.

Compiler repair is not permission for a weak model. If large repairs become routine, fix the model/head.

## AXIS / MIRA boundary

Current mechanics principle remains carrier-native:

```text
canonical M
  → AXIS G
  → MIRA W_M on exact M
  → Compiler exact (M,G,W_M) proof
```

The AXIS input adapter should be redesigned to consume M-anchored rich surface evidence rather than depending on a pre-M lossy surface summary if that can be done cleanly. Exact adapter design remains open in `CURRENT_PLAN.md`.

MIRA continues to predict on exact final M. No semantic skin transfer to another topology becomes canonical.

## Current Knight baseline

The approved Knight presentation/mechanics regression remains useful and unchanged in scope:

- one canonical M for geometry/topology/motion;
- source-wins RGB + existing CAA on M-face/barycentric addresses;
- exact sealed regression artifacts remain historical/product-near witnesses;
- grip/weapon relation and semantic overlap/draw-order remain known visual concerns;
- Knight is not part of the Fit8 cohort and must not block it.

Do not reinterpret the old Knight witness as proof of the new Unified IRIS architecture.

## Platform / execution rules

Canonical source remains on `main`.
Research state belongs to Go `Attempt`; production state to `ProductRevision`.
Do not create a long-lived research branch as architecture authority.

Before new execution:

1. finalize/exit superseded old Fit8 sessions through the Go control plane;
2. land the new schemas/model/compiler/DAG implementation on main;
3. mint a new EngineRelease;
4. start a fresh/explicit child campaign/Attempt from source identity;
5. run the new one-image Unified IRIS path;
6. qualify canonical M, then AXIS/MIRA/mechanics/presentation.

External Colab/A100 remains compute only; its checkpoint/result does not mint scientific qualification by itself.

## Immediate priority

Do **not** spend compute resuming IRIS-vCurrent.

The current work order is:

```text
1. teacher-corpus schemas + deterministic extractor/normalizer
2. Unified IRIS architecture/spec + code layout
3. new model/compiler artifact schemas
4. new dependency DAG / EngineRelease design
5. full-capacity implementation
6. CPU/GPU preflight and exact one-image-forward guards
7. new Fit8 Unified attempt
8. canonical M qualification
9. AXIS/MIRA + dynamic proof
10. LOFO/unseen only after architecture freeze
```

## Required read order for agents

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `AGENTS.md`
4. `docs/platform/EXECUTION_LANES.md`
5. `docs/platform/AGENT_HANDOFF.md`
6. `SYSTEM_INDEX.md` for current implementation ownership
7. `canonical/CANONICAL_HANDOFF_20261007.md` for sealed mechanics provenance only
8. older recovery/scientific material only when needed for provenance

When the user says only `continue / devam et`, continue the Unified IRIS redesign program in `CURRENT_PLAN.md`. **Do not resume the stopped old Fit8 A100 training.**
