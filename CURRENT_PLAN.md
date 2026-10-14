# RealSaS-OPT — Current Plan

**Date:** 2026-10-14
**Status:** USER-AUTHORIZED ARCHITECTURE INTERVENTION — DESIGN / IMPLEMENTATION PROGRAM
**Canonical code line:** `main` only
**Research state:** a fresh EngineRelease + fresh/child Attempt is required before new scientific execution
**Supersedes as active priority:** the stopped IRIS-vCurrent Fit8 shared-training continuation

This document is the current architecture plan agents must use after reading `CURRENT_STATE.md`.
It is intentionally a planning authority, not a claim that the described model, DAG or datasets already exist or pass.
Open questions are marked **OPEN**. Decisions already made by the user are marked **LOCKED**.

---

## 0. Executive decision

The current IRIS/TESSA geometry path is being redesigned before continuing Fit8.
The goal is not a smaller or cheaper approximation of the old architecture. The goal is a materially stronger and simpler representation that preserves the useful RealSaS compiler/proof work while eliminating information-destroying model boundaries.

The target product boundary is:

```text
one user PNG
    ↓
IRIS
    ├─ internally constructs an uncertainty-aware canonical 8-view scene
    ├─ builds one shared high-capacity spatial representation
    ├─ predicts geometry, appearance, semantic/ownership evidence
    ├─ predicts learned extraction deformation/configuration evidence
    └─ predicts an animation-oriented explicit mesh/topology proposal
    ↓
Compiler
    ├─ independently qualifies source/geometry/topology/addressing
    ├─ performs only generic bounded repairs that are demonstrably useful
    └─ seals canonical mechanical M
    ↓
AXIS
    ↓
canonical G
    ↓
MIRA on exact M
    ↓
canonical W_M
    ↓
Compiler mechanical / appearance / presentation / dynamic proof
    ↓
Runtime package
```

**LOCKED:** IRIS does **not** output a finished animated puppet. Its product-facing geometric endpoint is an evidence-rich mesh proposal which the Compiler may promote to canonical `M`. AXIS and MIRA remain downstream mechanics models for now.

**LOCKED:** final customer input is one image / one PNG. Internally generating and using eight canonical views is allowed and preferred.

**LOCKED:** the new model starts at StdGEN-class representation capacity. Do not run a small-capacity ladder whose only purpose is to rediscover that a larger published representation works better.

**LOCKED:** this is a fresh architecture intervention. The stopped Fit8 IRIS-vCurrent run is retained only as a control/baseline artifact; it is not resumed as the final IRIS.

---

## 1. Why the architecture is being reopened

The old path is scientifically modular but contains a representation bottleneck that is no longer justified as a product architecture:

```text
8 RGB
  → IRIS image tokens / shared triplane
  → continuous SDF
  → deterministic tetra zero surface
  → dense/compact GSA RiggingSurface
  → 17-dimensional per-node TESSA conditioning
  → a second learned surface encoder
  → autoregressive topology / XYZ proposal
  → Compiler support binding / static repair
  → Stage19 M
```

The current TESSA conditioning adapter reduces the upstream scene to normalized XYZ, normal/normal-validity, eight support-view bits, graph degree and persistence. It does not carry the original image tokens, rich triplane feature, appearance, semantic ownership or dense uncertainty field. TESSA then spends a substantial second transformer reconstructing a latent surface model from that compressed substrate.

That separation was valuable for scientific isolation, but it is not automatically the best final neural architecture.

The redesign therefore follows this rule:

> **Do not throw rich learned scene information away merely to recreate it in a second model.**

The compiler remains where it is better than a neural head: immutable authority, source provenance, exact qualification, independent courts, bounded generic repair, addressing, mechanical proof and fail-closed promotion. It must not be assigned work merely to preserve historical stage count.

---

## 2. What is preserved from current RealSaS

The intervention does **not** discard the strongest RealSaS work.

Preserve unless a later implementation review proves a strictly better replacement:

1. **Canonical source / camera / normalization contracts.**
2. **Immutable artifact identity and provenance.**
3. **Compiler-owned product authority.** A learned output is a proposal/evidence object until promoted.
4. **Independent source-fidelity measurement.** Learned views and learned geometry do not grade themselves.
5. **Deterministic contained tetra witness** as an independent implicit-field court.
6. **Final stable SurfaceAddressing on canonical M.**
7. **Stage19-style exact mechanical carrier identity.** AXIS and MIRA must consume the same final carrier basis.
8. **AXIS hard causal rig authority + compiler validation** unless a later mechanics intervention explicitly replaces it.
9. **MIRA carrier-native `W_M` on exact M.** No silent skin transfer to another basis.
10. **Dynamic mechanical proof:** stretch, area, condition, self-intersection, contact and motion consequences remain first-class.
11. **Source-wins appearance principle.** Learned appearance is completion/evidence, not permission to overwrite an observed source pixel silently.
12. **Presentation/ownership/draw-order remain explicit product concerns** rather than being inferred from camera Z alone.
13. **Go Attempt / EngineRelease / ProductRevision separation.** Old Attempts are never rewritten to pretend the new architecture existed earlier.

Current shipped mechanics boundary remains useful context:

```text
M → AXIS G → MIRA W_M → Compiler exact (M,G,W_M) proof
```

The redesign changes how `M` and its visual evidence are produced. It does not collapse AXIS/MIRA into IRIS in this intervention.

---

## 3. Reference architecture: use StdGEN-class capacity directly

The new IRIS scene representation should begin at the capacity class demonstrated by StdGEN S-LRM rather than the smaller IRIS-vCurrent capacity.

Published/released StdGEN implementation uses approximately:

```text
image encoder:       DINO ViT-B/16, 768 feature dim
scene transformer:   1024 hidden, 16 layers, 16 heads
query triplane:       32x32 low-res → 64x64 high-res
triplane channels:   80 per plane
field decoders:      SDF, RGB, semantic, Δxyz deformation
cell decoder:        21 FlexiCubes configuration weights
surface grid:        128 x 128 x 128 class
```

**LOCKED CAPACITY FLOOR for the first full IRIS-vNext training attempt:**

```text
encoder class             >= DINO ViT-B/16 / 768-D class
shared scene transformer  >= 1024-D, 16-layer, 16-head class
triplane                   >= 3 x 64 x 64 x 80
geometry field             continuous SDF + uncertainty
appearance field           RGBA / material evidence
semantic field             generic ownership/instance evidence
extraction field           bounded Δxyz + cell/configuration weights
topology head              explicit animation-oriented mesh proposal
```

This is a capacity floor, not a requirement to copy every upstream implementation choice verbatim.

### Licensing constraint

StdGEN's repository has a top-level Apache-2.0 license, but individual files include upstream/local license headers, including NVIDIA-origin renderer code with a restrictive header. **Do not copy file bodies blindly.** Reuse architectural ideas and interfaces; either reimplement cleanly from the paper/public algorithm or perform a file-level license audit before importing code.

The RealSaS implementation should own its training code and weights.

---

## 4. Unified IRIS-vNext model

### 4.1 One shared scene, multiple heads

The central representation becomes one shared spatial scene latent `Z`:

```text
single source image
       │
       ▼
canonical-view synthesizer
       │
       ├─ exact source-observed view
       └─ 7 generated canonical hypotheses + uncertainty/provenance
       │
       ▼
image token encoder
       │
       ▼
shared 3D scene transformer / triplane Z
       │
       ├─ G: geometry / SDF / uncertainty
       ├─ A: appearance / RGBA / material
       ├─ S: semantic / ownership / boundary / affinity
       ├─ F: bounded extraction Δxyz + cell configuration
       └─ T: topology / explicit mechanical mesh proposal
```

Heads share scene evidence; they do **not** automatically share product authority.

### 4.2 Geometry head `G`

Outputs at query point `x`:

```text
sdf(x)
geometry_uncertainty(x)
optional normal / gradient-derived evidence
```

Keep the strong signed-field supervision accumulated in IRIS-vCurrent. The new head must be at least as good at continuous field queries; adding other heads is not permission to weaken signed geometry.

### 4.3 Appearance head `A`

Output should be view-aware rather than a naïve globally constant texture field because RealSaS targets 2D artwork:

```text
RGBA(x, target_view, optional_owner_descriptor)
appearance_confidence
source_support_state
```

Observed source appearance remains stronger authority than learned completion.

The head should support:

- exact/near-exact visible source reconstruction,
- hidden/unseen material hypotheses,
- cross-view appearance consistency,
- canonical 8-direction output,
- future one-image product use without requiring artist-authored eight-view sheets.

### 4.4 Semantic / ownership head `S`

Do **not** hard-code StdGEN's human-only ontology (`hair / cloth / body`) as the primary representation.

Primary generic outputs:

```text
instance / owner descriptor (recommended 64-D class)
same-owner affinity
owner-boundary probability
material / source-region descriptor
amodal-support probability
deformation-mode evidence
attachment evidence
pairwise occlusion / overlap evidence
uncertainty / abstention
```

Secondary optional normalized roles may include:

```text
CORE_BODY
DEFORMING_APPENDAGE
APPAREL
HAIR_FUR
RIGID_ATTACHMENT
DEFORMING_ATTACHMENT
HELD_OBJECT
FACE_DETAIL
OTHER_COMPONENT
```

Anatomical labels such as arm/wing/tail are secondary metadata, not the core generic ontology.

Compiler converts descriptors/relations into subject-local canonical owner/fragment identities; the network does not predict globally meaningful owner IDs.

### 4.5 Extraction deformation/configuration head `F`

This is **not MIRA** and not skeletal skinning.

It predicts bounded deformation/configuration of the implicit-surface extraction carrier, analogous in purpose to StdGEN's:

```text
Δxyz per extraction-grid vertex
cell / FlexiCubes-style configuration weights
```

Use bounded parameterization to prevent unstable flips.

Its purposes are:

- improve local surface placement,
- give mesh-level depth/normal/silhouette losses direct leverage,
- represent details a fixed extraction lattice may miss,
- produce an explicit learned surface `F_surface` for comparison against the deterministic tetra witness.

It does **not** become animation deformation authority.

### 4.6 Topology head `T` — integrate TESSA instead of serializing through GSA

TESSA's useful idea is retained: an explicit learned topology/XYZ proposal suitable for animation.

The separate TESSA model boundary is retired.

The current TESSA decoder can serve as a starting point, but its surface encoder must not begin from the 17-D GSA summary. The topology decoder should cross-attend directly to shared IRIS scene/surface memory:

```text
shared triplane / scene tokens
+ field samples
+ geometry uncertainty
+ semantic/owner descriptors
+ visibility/source evidence
        ↓
mesh-token decoder
        ↓
vertices + faces + component/owner evidence
```

Recommended first implementation:

- preserve TESSA's explicit topology sequence / adjacency serialization ideas;
- preserve coordinate quantization only if it remains useful at full capacity;
- replace `TESSASurfaceEncoderV1(17-D GSA)` with shared scene memory adapters;
- allow per-vertex continuous residual refinement after token coordinate prediction;
- emit direct field/query bindings for each proposed vertex so Compiler need not recreate support by nearest-neighbor graph search where exact shared-field addressing is available.

**LOCKED:** TESSA becomes an IRIS topology head/module, not a separately trained scene-understanding model.

---

## 5. Single image in, canonical eight views inside IRIS

### 5.1 Product contract

**LOCKED:** final product input cardinality is one image.

IRIS internally creates eight canonical 45-degree views:

```text
0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
```

These are hypotheses/evidence, not eight equal-trust source observations.

Each view carries provenance:

```text
SOURCE_OBSERVED
GENERATED_CANONICAL_VIEW
```

and confidence/uncertainty.

Rule:

```text
exact observed source > generated view hypothesis
```

### 5.2 No mirror shortcut

Generate all eight directions independently under one shared identity/scene condition. Do not manufacture half the set by mirroring; asymmetric weapons, clothing, faces and accessories are product-relevant.

### 5.3 Canonical-view model boundary

The view synthesizer is logically a submodule of IRIS.

**OPEN:** whether the first implementation exposes only raster outputs to the shared scene encoder or also passes a private generator latent directly into scene fusion.

Preference:

- keep eight generated views inspectable and serializable;
- preserve exact source image separately;
- allow a latent side-channel only if dependency/provenance is explicit and it materially improves consistency.

### 5.4 Consistency target

The model is not rewarded for producing eight individually pretty but mutually inconsistent images.

Training/evaluation must include multi-view identity, silhouette, depth/normal and geometric consistency. Generated views should be evaluated by the downstream recovered scene/mesh, not merely perceptual scores.

---

## 6. Three geometry objects, three different jobs

The redesigned path deliberately distinguishes three geometric objects:

```text
E = deterministic implicit-field witness
F = learned differentiable extraction surface
M* = topology-head mechanical mesh proposal
```

### 6.1 `E` — deterministic tetra witness

Preserve the contained deterministic tetra decoder (`T512 → sparse T1024` family) as an independent court over the signed field.

Its role changes:

- **not** a neural bottleneck between IRIS and topology generation;
- **not** necessarily the product mechanical mesh;
- independent verification of zero-surface geometry;
- exterior/sign census;
- silhouette / source-fidelity court;
- comparison target for the learned extraction surface and topology-head proposal.

This preserves a strong RealSaS idea that StdGEN does not provide by itself: a learned model should not be its own only geometry judge.

### 6.2 `F` — learned extraction surface

Produced from SDF + bounded grid deformation + cell/configuration weights.

Used for:

- differentiable mesh-level RGB/mask/depth/normal losses;
- high-quality explicit reconstruction;
- semantic layer extraction experiments;
- static comparison against `E` and source views.

**OPEN:** whether `F` can eventually replace the dedicated topology head for some subject classes. Do not assume this in vNext; static reconstruction topology is not yet proven to be the best animation carrier.

### 6.3 `M*` — animation-oriented topology proposal

Generated by integrated topology head `T`.

Must be optimized/qualified for both 2D projection quality and dynamic deformation, not only static Chamfer/render fidelity.

Required properties before promotion:

- manifold / valid oriented topology;
- no degenerate/sliver pathologies beyond frozen quality policy;
- high source silhouette fidelity in required views;
- no unacceptable projected flips/self-intersections;
- stable motion Jacobian/condition/stretch/area behavior;
- owner/material boundaries addressable;
- suitable density for runtime and later LOD reduction;
- exact field/source provenance.

Compiler may promote `M*` to canonical `M` only after independent qualification.

---

## 7. GSA / RiggingSurface after the redesign

GSA is not deleted. Its role changes from **neural information bottleneck** to **derived compiler/mechanics evidence**.

Old:

```text
IRIS → tetra → GSA → 17-D TESSA input → TESSA
```

New:

```text
                    Unified IRIS scene
                    /               \
         topology-head M*          signed field
                    \               /
                     Compiler qualification
                              ↓
                         canonical M
                              ↓
             M-anchored surface evidence / GSA
                              ↓
                            AXIS
```

The derived surface should carry richer sidecar descriptors where useful:

- final M vertex/face address,
- shared-scene feature reference or projected compact embedding,
- geometry uncertainty,
- source/view support,
- semantic/owner descriptor,
- material/appearance support,
- local structural graph information.

Do not collapse these back into a 17-D vector unless a downstream model explicitly needs only that subset.

---

## 8. Compiler boundary: keep only work that is genuinely better there

### 8.1 Compiler responsibilities that remain

1. **Evidence provenance and immutable binding.**
2. **Source-observed vs generated-view authority distinction.**
3. **Independent deterministic tetra witness and source-fidelity courts.**
4. **Static mesh qualification.**
5. **Generic bounded static repair when repair is demonstrably safer than regeneration.**
6. **Canonical promotion from proposal to exact `M`.**
7. **SurfaceAddressing and stable face/barycentric domains.**
8. **Subject-local owner/fragment resolution from learned evidence.**
9. **Source-wins material compilation and explicit completion provenance.**
10. **AXIS rig validation/materialization.**
11. **MIRA exact-carrier skin qualification.**
12. **Dynamic mechanical proof and repair attribution.**
13. **Contact/grip/order/presentation relation qualification.**
14. **Runtime package sealing and native-consumer parity.**

### 8.2 Compiler work to retire or redesign

Retire when the unified representation makes it unnecessary:

- GSA → 17-D → second neural surface encoder as the only TESSA path;
- a separate TESSA model checkpoint whose scene understanding must be rebuilt from GSA;
- nearest-XYZ / KD-tree material-support reconstruction when a vertex already has an exact shared-field/query binding;
- repeated projection/backprojection identity recovery between independently learned geometry and appearance domains;
- visual/mechanical rematching that exists only because the neural producers did not share scene identity;
- any stage whose only job is to reconstruct information an upstream shared latent already knew and could serialize explicitly.

### 8.3 Repair rule

Compiler repair is not a dumping ground for weak learned geometry.

A repair operator remains only when all are true:

- generic / subject-free;
- bounded and deterministic;
- provenance-preserving;
- improves a frozen quality court;
- does not silently invalidate appearance/semantic/topology addresses;
- all downstream consequences are requalified.

If large repair is routinely necessary, fix the model/head instead.

---

## 9. Canonical M target

The redesign is successful only if it produces a better canonical mechanical carrier, not merely a nicer reconstruction render.

Canonical `M` should be simultaneously:

### Static geometry quality

- close to source/teacher geometry where admitted;
- strong 8-direction 2D silhouette fidelity;
- complete visible surface coverage;
- high normal/depth fidelity where truth exists;
- no non-manifold edges;
- no degenerate/sliver faces outside frozen policy;
- no unexplained holes/gaps;
- coherent semantic/material boundaries.

### Topology quality

- animation-friendly connectivity;
- no arbitrary triangulation changes after skinning without a new mechanical equivalence proof;
- explicit provenance from topology head through any compiler repair;
- deterministic addressing after qualification;
- suitable basis for MIRA `W_M`.

### 2D-specific quality

- projected behavior is first-class, not a byproduct of 3D quality;
- canonical eight directions are measured;
- silhouette, owner boundaries and overlap-critical regions receive enough topology;
- later product LOD should target roughly mobile-friendly complexity (historically ~800–900 triangles where achievable), but reduction must not be forced before deformation/presentation equivalence is demonstrated.

### Dynamic quality

Before canonical product promotion, test motion probes for:

- edge stretch;
- triangle area ratio;
- Jacobian / condition number;
- projected flips;
- self-intersections;
- contact residual;
- attachment/grip consequence;
- owner-boundary continuity;
- appearance gaps during motion.

A static PASS cannot substitute for dynamic carrier quality.

---

## 10. AXIS handoff

AXIS remains a separate mechanics model in this intervention.

Current AXIS consumes a surface representation and predicts hard causal tree + ordered discrete XYZ; Compiler validates/materializes it.

The redesigned handoff should occur **after canonical M exists** whenever possible:

```text
canonical M
+ M-anchored surface evidence
+ shared-scene geometry/semantic descriptors
        ↓
AXIS
        ↓
G proposal
        ↓
Compiler validation/materialization
```

### Required redesign question

**OPEN:** exact AXIS-vNext input adapter.

Preferred direction:

- preserve AXIS causal tree mechanics;
- condition on final M-anchored evidence rather than a pre-M lossy GSA cloud;
- optionally expose compact shared-scene/semantic descriptors if they improve joint/attachment inference;
- never give AXIS teacher rig labels at inference;
- keep image-only/single-image product input invariant upstream.

Do not merge AXIS into unified IRIS in this architecture intervention. First obtain a high-quality canonical M and clean M-anchored evidence boundary.

---

## 11. MIRA handoff

MIRA stays carrier-native:

```text
exact canonical M + qualified G → MIRA → W_M on exact M
```

This is a strong current invariant and should not be weakened.

New semantic/ownership evidence may be offered as an **optional explicit conditioning feature** only if it is shown to improve generic skin prediction without creating a hidden dependency on authored labels.

No same-index skin transfer from another topology.

---

## 12. Dataset program

The architecture change requires a richer teacher corpus, but the existing authored Fit8 subjects remain useful.

### 12.1 Fit8 Unified Teacher V1

Use the same eight selected Fit8 source identities:

- Snake
- Bat
- T-Rex
- Horse
- Spider
- GreenBlob
- Shark
- George

Do not replace difficult subjects to make the new model easier.

### 12.2 Raw authored truth extraction

For each source asset, ingest deterministically:

```text
object hierarchy
mesh/object IDs
vertices/faces
material slots
armature
bone hierarchy
skin weights
object parenting / constraints
rigid accessories
source animations where available
```

Raw authored structure is primary truth. Do not infer it from rendered pixels when it exists in the file.

### 12.3 Canonical render teacher bundle

For each canonical 45-degree view render/store:

```text
rgb.png
alpha.png
depth.exr
normal.exr
instance_id
material_id
mechanical_owner / dominant-weight evidence
camera.json
```

Add component-isolation/amodal renders:

```text
amodal/<component>.png
```

and structured files:

```text
component_manifest.json
occlusion_graph.json
attachment_graph.json
```

### 12.4 Semantic truth

Keep two layers:

**Layer A — raw authored identity:** object, mesh, material, bone weights, parent relation.

**Layer B — deterministic normalized RealSaS roles:** generic role/deformation class/attachment class.

The normalizer must be versioned and subject-independent. Name parsing may be evidence, not sole authority.

### 12.5 Occlusion truth

At each source/canonical ray, retain all relevant ordered hits where feasible:

```text
(component_A, depth_A)
(component_B, depth_B)
...
```

Derive pairwise relations such as:

```text
FRONT_OF(A,B,view)
BEHIND(A,B,view)
NO_OVERLAP(A,B,view)
UNKNOWN
```

This is more useful for RealSaS presentation than a single global `front/back` class.

### 12.6 Attachment truth

From authored structure derive separate evidence for:

- rigid bone-parented object;
- skinned/deforming attachment;
- held object;
- independent/root object;
- attachment target/owner bone where explicit.

Do not conflate "can grip" with "is in contact in this pose".

### 12.7 Appearance truth tiers

Define provenance tiers:

```text
TIER 0: exact observed user/source RGB
TIER 1: authored 3D material render under controlled canonical renderer
TIER 2: learned/generated completion hypothesis
```

Compiler/source-wins rules distinguish them.

The existing authored 3D assets can supply enough appearance supervision to train the first unified geometry/appearance model. A separate artist-layered 2D corpus may still be needed for final Hollow-Knight/Spine-like hidden-line/style behavior; mark that as a later dataset extension rather than blocking the first unified reconstruction attempt.

### 12.8 Single-view training pairs

Training sample structure:

```text
INPUT: one selected source/canonical RGB view
TEACHER-ONLY: true canonical V0..V7 RGB/alpha/depth/normal/semantic/geometry
TARGET: unified scene + explicit surfaces/topology
```

Teacher-only views and authored 3D truth are forbidden forward inputs in the one-image product arm.

Train/test splits must be by character/source project, never by frames/views of the same asset.

### 12.9 Scale-up corpus

After deterministic extractor/normalizer validation on Fit8, run the same pipeline over the larger audited authored corpus (historically ~139 candidates) subject to license/source-quality admission.

Do not redesign the ontology per subject.

---

## 13. Training objective families

This section defines the intended full training system, not an ablation ladder.

### Canonical-view synthesis losses

- canonical RGB reconstruction;
- alpha/silhouette;
- cross-view identity/perceptual consistency;
- depth/normal consistency where teacher exists;
- uncertainty calibration for unseen/backside content.

### Shared field losses

- existing strong signed F* / SDF supervision;
- exterior sign census;
- near-zero surface accuracy;
- uncertainty calibration.

### Explicit learned surface losses

- mask / silhouette;
- depth;
- normal;
- source/canonical RGB;
- semantic/instance projection;
- FlexiCubes-style surface/deviation/configuration regularization;
- anti-flip bounded extraction-deformation regularization.

### Appearance losses

- source-visible RGB/RGBA;
- view-conditioned canonical appearance;
- material/owner consistency;
- hidden completion only where admitted teacher truth exists.

### Semantic/ownership losses

- role classification where available;
- instance descriptor contrastive/affinity loss;
- same-owner / different-owner pairs;
- boundary loss;
- amodal-support loss;
- pairwise occlusion relation loss;
- attachment/deformation-mode classification where source truth exists;
- abstention/unknown calibration.

### Topology-head losses

- teacher mesh token / adjacency sequence;
- XYZ coordinate target;
- topology validity;
- field-consistency between proposed vertices and shared SDF;
- source projection fidelity;
- owner/material boundary alignment;
- animation-oriented regularizers derived from canonical motion probes where admitted.

### Dynamic carrier objective

**OPEN / DESIGN REQUIRED:** how much dynamic probe loss belongs inside topology-head training versus remaining a post-training Compiler court.

Preference: include differentiable/bounded proxy terms during training, but retain the independent full dynamic Compiler court as promotion authority.

---

## 14. New DAG / stage structure

Exact numeric ordinals are **not locked yet**. Names/dependencies are more important than preserving historical numbers.

A new EngineRelease should represent the new graph. Do not mutate an old released DAG in place.

Recommended semantic nodes:

```text
SOURCE_ASSET_ADMITTED
    ↓
RAW_AUTHORED_TRUTH_EXTRACTED
    ↓
UNIFIED_TEACHER_BUNDLE_SEALED
    ↓
SINGLE_VIEW_INPUT_QUALIFIED
    ↓
CANONICAL_FRAME_CAMERA_QUALIFIED
    ↓
IRIS_UNIFIED_MODEL_PREREGISTERED
    ↓
IRIS_UNIFIED_CHECKPOINT_BOUND
    ↓
IRIS_CANONICAL8_EVIDENCE
    ↓
IRIS_SHARED_SCENE_EVIDENCE
    ├───────────────┬────────────────┬────────────────┐
    ↓               ↓                ↓                ↓
SIGNED_FIELD    APPEARANCE       SEMANTIC       LEARNED_SURFACE
EVIDENCE        EVIDENCE         EVIDENCE       EVIDENCE
    │               │                │                │
    └──────┬────────┴───────┬────────┘                │
           ↓                │                         │
DETERMINISTIC_TETRA_WITNESS │                         │
           │                │                         │
           └────────┬───────┴──────────────┬──────────┘
                    ↓                      ↓
          TOPOLOGY_HEAD_PROPOSAL      STATIC CROSS-COURTS
                    │                      │
                    └──────────┬───────────┘
                               ↓
                   CANONICAL_M_QUALIFIED
                               ↓
                 M_ANCHORED_SURFACE_EVIDENCE
                               ↓
                         AXIS_RIG_PROPOSAL
                               ↓
                         RIG_QUALIFIED G
                               ↓
                         MIRA_SKIN W_M
                               ↓
                     EXACT_MGW_QUALIFIED
                      /                    \
                     ↓                      ↓
          APPEARANCE_MATERIAL_COMPILE   MOTION_PROOF
                     \                      /
                      ↓                    ↓
                   PRESENTATION_OWNERSHIP_ORDER
                               ↓
                         RUNTIME_PACKAGE
                               ↓
                      NATIVE_DYNAMIC_PROOF
```

### Nodes expected to disappear or fundamentally change

- separate IRIS-only zero-surface → GSA → TESSA model training boundary;
- standalone TESSA checkpoint as a separate scene model;
- TESSA material support reconstruction that exists only to recover field identity lost at the model boundary;
- any old appearance/presentation node that duplicates owner/material identity now emitted by unified IRIS without adding independent qualification value.

### Nodes expected to survive conceptually

- observation/camera/normalization;
- deterministic geometry witness;
- final mesh qualification/addressing;
- AXIS;
- MIRA;
- exact M/G/W dynamic proof;
- appearance qualification;
- presentation/order/contact/grip qualification;
- runtime/native parity.

---

## 15. Attempt and migration policy

### 15.1 Stop old scientific continuation

The shared IRIS-vCurrent Fit8 A100 run was intentionally stopped at the latest sealed ~4400-step checkpoint after the user rejected the representation as the final IRIS architecture.

Retain its checkpoint/result/trace as:

```text
IRIS_VCURRENT_FIT8_CONTROL_BASELINE
```

Do not resume it merely to reach the historical 19,200 schedule.

No Fit8 scientific PASS was minted by the stopped run.

### 15.2 Fresh research state

Because the following change together:

- input cardinality / view synthesis;
- model capacity;
- shared representation;
- geometry extraction;
- semantic/appearance outputs;
- TESSA integration;
- Compiler boundaries;
- dataset truth;
- released DAG;

a **new EngineRelease and fresh/explicit child Attempt** are required.

Start from Stage01/source identity in platform terms. Reuse immutable raw source bytes only where dependency identity proves they are still valid. Do not force reuse of observation/model artifacts whose contracts have changed.

### 15.3 Fit8 remains the architecture cohort

The same eight subjects remain the first architecture/genericity cohort.

The new experiment is no longer "continue the old IRIS fit". It is:

> **Fit8 Unified IRIS Reconstruction + Canonical M Architecture Acceptance.**

Knight remains a separate regression witness.

---

## 16. Qualification strategy

Architecture choice is not an ablation ladder, but the new full model still needs independent courts.

### IRIS internal-output courts

- one input really is the only forward image authority;
- generated eight-view provenance correct;
- canonical-view consistency;
- signed field exterior/near-zero correctness;
- deterministic tetra witness;
- learned-surface source fidelity;
- semantic/owner accuracy and calibration;
- topology proposal validity.

### Canonical M courts

- deterministic static mesh quality;
- source/canonical projection fidelity;
- no unacceptable holes/gaps;
- owner/material address completeness;
- comparison against independent implicit witness;
- dynamic deformation probes;
- runtime density/performance envelope.

### System courts

- AXIS consumes only qualified final M-anchored evidence;
- MIRA predicts on exact same M;
- dynamic `(M,G,W_M)` proof closes;
- appearance source-wins behavior preserved;
- draw order / overlap / attachment relations actually consumed;
- native package parity.

P999-like stress metrics may remain diagnostic/scientific courts, but product acceptance must distinguish "strict stress failure" from actual visible/mechanical product failure. Never silently relax a frozen metric after seeing the test subject; version the court if its product correlation is intentionally changed.

---

## 17. Open design questions

These are intentionally unresolved; agents must not silently decide them without recording evidence.

1. **Canonical-view synthesizer architecture.** Use a CharacterGen/StdGEN-derived multiview generator family, another public model, or a clean-room RealSaS model? Need quality/license/training-code audit.
2. **Raster-only vs latent side-channel** from view generator into shared scene transformer.
3. **Exact DINO family.** StdGEN-proven ViT-B/16 is the capacity reference; evaluate whether DINOv2-B/14 is a clearly superior compatible replacement before locking weights.
4. **Topology head form.** Preserve autoregressive TESSA sequence exactly, replace it with parallel mesh decoding, or hybrid? The requirement is explicit animation-oriented topology from shared scene memory.
5. **Coordinate discretization.** Current TESSA uses 1024 bins; determine whether continuous residuals should be mandatory.
6. **Learned extraction surface vs mechanical M.** Could a FlexiCubes-style surface ever become the final animation carrier directly? Do not assume yes until dynamic proof.
7. **Dynamic loss inside training.** Determine which motion-probe terms are differentiable/useful without teaching a narrow motion family.
8. **Semantic descriptor dimension and normalized ontology.** Primary representation should remain category-agnostic.
9. **AXIS-vNext input adapter.** Define exact M-anchored rich surface evidence without turning semantic labels into inference truth.
10. **Appearance hidden truth.** Determine how far authored 3D material renders can supervise final 2D artist-style completion before a layered 2D corpus becomes necessary.
11. **Triangle budget.** Define canonical research M vs product/mobile LOD policy; do not trade away deformation quality prematurely.
12. **Compiler repair budget.** Freeze explicit limits after measuring new model output; large routine repairs indicate model failure.
13. **Stage numbering.** Final ordinals only after dependency graph is accepted; do not preserve numbers at the cost of bad dependencies.

---

## 18. Implementation program — not a capacity ablation ladder

The model target above is fixed. Engineering still needs an ordered build because multiple modules depend on each other.

### Work package A — authority and DAG reset

- mark old Fit8 IRIS run stopped/control in current state;
- create new research intervention identity;
- draft new EngineRelease DAG with semantic node names above;
- define new schemas before executing training;
- ensure one-image forward authority is machine-checkable.

### Work package B — teacher corpus generator

- extend controlled Blender/source ingest;
- export raw authored hierarchy/mesh/material/rig/weights;
- render canonical RGB/alpha/depth/normal;
- export instance/material/mechanical-owner maps;
- generate amodal component isolation;
- generate occlusion/attachment graphs;
- version semantic normalizer;
- validate on all eight Fit8 assets.

### Work package C — unified IRIS model

Build the full-capacity shared scene model from the start:

- canonical 1→8 view synthesizer;
- DINO-B-class encoder;
- 1024x16 shared transformer;
- 64²x80+ triplane;
- SDF/uncertainty head;
- RGBA head;
- semantic/ownership head;
- bounded extraction deformation/configuration head;
- integrated TESSA/topology head.

No small-model science run is required before implementing the target capacity.

### Work package D — geometry/compiler simplification

- retain deterministic tetra witness as independent court;
- remove GSA-as-TESSA-bottleneck;
- create direct shared-field bindings for topology-head vertices;
- redesign Stage18/19 equivalent around unified proposal + independent courts;
- preserve/rewrite only compiler repairs that remain genuinely beneficial;
- emit canonical M + SurfaceAddressing + rich M-anchored surface evidence.

### Work package E — AXIS/MIRA bridge

- define AXIS M-anchored input adapter;
- preserve carrier-native MIRA;
- rerun exact mechanics proof on new M;
- keep model boundaries explicit until evidence supports a later merge.

### Work package F — appearance/presentation consumption

- teach CAA/material compiler to consume IRIS appearance evidence with provenance;
- teach owner/presentation compiler to consume semantic descriptors/affinities;
- preserve source-wins;
- prove draw-order/attachment/contact consumers actually use the new evidence.

### Work package G — Fit8 unified attempt

Only after schemas/model/truth/DAG preflight:

- start new platform campaign/Attempts;
- train one shared unified IRIS across the eight subjects under the new one-image forward contract;
- qualify each subject independently;
- run AXIS/MIRA and dynamic proof;
- fix only generic owner-routed failures;
- freeze architecture after cohort closure, then proceed to LOFO/unseen.

---

## 19. Retired assumptions

The following assumptions are no longer active-plan constraints:

- "IRIS must only output SDF/geometry evidence."
- "TESSA must be a separately trained model."
- "TESSA should reconstruct scene understanding from a 17-D GSA summary."
- "Eight images must be supplied by the user."
- "Appearance should be learned/compiled independently from the scene geometry representation."
- "Semantic evidence is inherently human-specific."
- "Compiler should keep a stage merely because an older architecture had it."
- "A long preregistered old training run should continue after the representation itself is rejected."

Still active:

- IRIS is image-only from the customer's point of view;
- teacher 3D/semantic/rig data never becomes inference input truth;
- Compiler owns canonical promotion and proof;
- AXIS/MIRA remain mechanics owners downstream for this intervention;
- final runtime consumes sealed artifacts and performs no generative correction.

---

## 20. Definition of success for this intervention

This architecture intervention is successful when a fresh one-image Fit8 attempt can produce, with one shared model family and no subject-specific rescue:

1. uncertainty-aware internal canonical eight-view evidence;
2. a coherent shared geometry/appearance/semantic scene representation;
3. a high-quality explicit reconstruction surface;
4. an animation-oriented topology proposal from the same scene memory;
5. a Compiler-qualified canonical `M` with strong static 2D/3D geometry and topology;
6. an M-anchored surface evidence package suitable for AXIS;
7. AXIS `G` + MIRA `W_M` on the exact same M;
8. dynamic mechanical closure on heterogeneous motion families;
9. source-preserving appearance and ownership/order evidence that materially reduces current visual gaps/overlap failures;
10. a cleaner DAG than the old IRIS→tetra→GSA→TESSA re-encoding chain.

The architecture must be judged by the final qualified canonical M and downstream animation/presentation result, not by an isolated neural benchmark.

---

## 21. Agent continuation rule

When the user says only `continue / devam et`, do **not** resume the stopped 19.2k IRIS-vCurrent run.

Continue from this program in order:

1. source/teacher-corpus schema and extractor design,
2. unified IRIS architecture/spec and file layout,
3. new released DAG / Attempt boundary,
4. full-capacity implementation,
5. preflight,
6. new Fit8 Unified attempt.

Read before implementation:

- `CURRENT_STATE.md`
- `CURRENT_PLAN.md`
- `AGENTS.md`
- `SYSTEM_INDEX.md`
- current IRIS/TESSA source
- StdGEN paper + public S-LRM implementation, with file-level license audit
- `docs/platform/EXECUTION_LANES.md`
- `docs/platform/AGENT_HANDOFF.md`

Historical Fit8 and Knight artifacts remain evidence/regression material, not architecture authority for the new unified model.
