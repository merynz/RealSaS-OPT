# IRIS-vNext hierarchical decision tree

**Date:** 2026-10-14  
**Status:** active architecture review; unresolved leaves must not be silently frozen  
**Scope:** FIT8 full-truth -> Unified IRIS -> F/T -> Compiler/CAO

This document enforces one design rule: every high-level architectural choice must branch into explicit sub-decisions, measurable trade-offs, evidence requirements and freeze conditions. A reduction is valid only if deliberate; an expansion is valid only if its downstream value is stated and tested.

## 0. Global invariants

1. **Generic articulation.** No human/anime-only anatomical assumption may become a core contract.
2. **One surface identity.** Geometry, semantic/owner identity and appearance addressing must remain co-addressable on the same continuous scene/surface domain.
3. **Three independent courts on one carrier.** Static geometry quality, dynamic mechanical quality and visible appearance coherence are separate qualifications over the same canonical M.
4. **Compiler authority.** Learned outputs remain evidence/proposals until independent promotion.
5. **FIT8 before scale.** FIT8 is the architecture/schema closure cohort. Do not scale the teacher extractor to the full corpus before truth contracts close.
6. **Product input stays one image.** The first reconstruction architecture court may use eight authored canonical views to avoid conflating reconstruction closure with the learned 1->8 prior.
7. **References are translated by reason.** Every mechanism inherited from InstantMesh/StdGEN/StdGEN++ or legacy IRIS/TESSA must be marked KEEP / TRANSLATE / REJECT with rationale.

---

## 1. Lineage: what is substrate versus real innovation

### 1.1 InstantMesh substrate

InstantMesh already provides the main reconstruction skeleton later inherited by StdGEN:

- DINO ViT-B/16-class image encoder;
- 1024-D, 16-layer, 16-head image-to-triplane transformer;
- learned 32x32 plane queries upsampled to 64x64;
- 80 channels per plane;
- SDF, RGB, extraction deformation and FlexiCubes configuration prediction;
- bounded extraction deformation;
- differentiable mesh rendering with RGB/mask/depth/normal supervision.

Therefore these are not treated as StdGEN semantic innovations merely because StdGEN also contains them.

### 1.2 StdGEN innovation layer

StdGEN materially adds:

- a semantic decoder on the same triplane field;
- semantic-equivalent NeRF rendering;
- semantic-equivalent SDF extraction;
- multi-layer outside-to-inside semantic/color training;
- semantically decomposed explicit meshes.

### 1.3 StdGEN++ innovation layer

StdGEN++ materially adds or strengthens:

- specialized LoRA reconstruction branches to relieve global/detail conflict;
- high-resolution 256x256x384 target extraction through sparse coarse-to-fine active queries;
- structure-aware hole regularization for thin semantic regions;
- explicit texture decomposition into editable appearance layers;
- selective multi-layer refinement rather than assuming post-process refinement always helps;
- a stronger dataset with specialized facial/texture truth.

### 1.4 RealSaS extension layer

RealSaS has additional requirements not solved by the above lineage:

- generic articulation rather than human/anime-only characters;
- animation-ready low-density canonical carrier topology;
- authored motion consequence supervision;
- 3D mechanics -> 2D appearance/presentation bridge;
- compiler-owned canonical promotion and fail-closed proof;
- dynamic draw-order/overlap/attachment authority;
- source-exact appearance priority through CAO.

---

## 2. Image encoder

### 2.1 Backbone family

Candidates:

- sealed DINOv2-S/14, 384-D — historical control only;
- DINO v1 ViT-B/16, 768-D — StdGEN/InstantMesh-class available baseline;
- DINOv2-B/14-class — candidate if license/deployment terms are acceptable;
- DINOv3-B/16 — target/reference only until dependency licensing is explicitly accepted.

Do not make DINOv3 a required dependency merely because it is newer.

### 2.2 Trainability

Branches:

- frozen encoder;
- LoRA/adapters;
- final blocks trainable;
- full fine-tune.

Freeze criteria:

- owner/material boundary quality;
- SDF/source fidelity;
- appearance detail retention;
- convergence under fixed scene-transformer capacity;
- GPU memory and training stability.

### 2.3 Feature depth

Branches:

- last-layer tokens only;
- multi-level token fusion;
- native-resolution detail stem only if a measured residual failure remains after multi-level fusion.

CAO preserves exact observed-source appearance independently, therefore IRIS does not need to memorize exact visible pixels merely to compensate for a lossy image backbone.

---

## 3. Image-to-scene transformer and triplane

### 3.1 Capacity floor

Reference starting class:

```text
image features:        768-D class
scene transformer:     1024-D
layers:                16
heads:                 16
query triplane:         32x32 -> 64x64
plane channels:         >=80
```

Legacy IRIS-v4 is materially smaller: 384-D image tokens -> 256-D scene width, 4 scene layers, 8 heads, 64 plane channels. Treat v4 as a control, not a capacity template.

### 3.2 Triplane base resolution

Candidates are not one global number:

- 32 -> 64 reference;
- 48 -> 96;
- 64 -> 128;
- low global plane + sparse region-local refinement.

Freeze metrics:

- normalized/world-space feature pitch;
- zero-surface error;
- 8-view projected error;
- owner/material boundary F1;
- thin-structure recall;
- active-query recall downstream;
- memory and throughput.

**Do not conflate triplane resolution with extraction-grid resolution.** StdGEN++ obtains a large quality gain by increasing extraction resolution without replacing the basic triplane concept.

### 3.3 Channel width

Candidates:

- 80 baseline;
- wider shared field;
- fixed shared width + region-specialized adapters;
- factored subspaces only if measured gradient conflict requires them.

Freeze only after G/A/S/F gradient-interference diagnostics.

---

## 4. Training representation: NeRF -> SDF -> explicit F

### 4.1 Sigma/NeRF role

Decision: density/sigma is a **training scaffold**, not product geometry authority.

Use it for:

- early RGB/mask/semantic volume supervision;
- hidden/interior semantic and appearance learning under decomposition masks;
- stabilizing a shared field before explicit-surface extraction.

Do not expose sigma as canonical runtime geometry.

### 4.2 Semantic-equivalent NeRF

StdGEN/StdGEN++ multiply effective opacity by semantic membership during ray integration. RealSaS must generalize this without global owner IDs.

Required split:

- primary semantic representation = relational owner/material descriptors and affinities;
- extraction/decomposition representation = subject-local normalized region-membership probabilities resolved from teacher truth.

The second representation may participate in semantic-equivalent NeRF/SDF; the primary owner embedding must not be forced into an arbitrary global class simplex.

### 4.3 Stage curriculum

Provisional curriculum:

```text
U1: shared triplane + sigma/RGB/semantic surface learning
U2: hidden/interior RGB + semantic decomposition learning
U3: SDF + semantic-equivalent SDF + high-resolution F extraction
U4: F-conditioned animation-carrier topology T
U5: canonical M + CAO static/dynamic visible coherence
```

Every stage must specify which parameters are trainable and which previous capabilities are regression-gated.

---

## 5. Semantic representation

### 5.1 Primary generic semantics

Do not use hair/cloth/body as the core ontology.

Primary axes:

- owner descriptor / same-owner affinity;
- owner boundary;
- material/source-region descriptor / same-material affinity;
- deformation mode;
- attachment relation;
- amodal support;
- overlap/occlusion relation;
- uncertainty/abstention.

Optional coarse roles are secondary metadata only.

### 5.2 Variable cardinality

Do not hard-code an owner/material slot count before FIT8 corpus census.

Teacher extraction must first measure:

- source object count;
- connected semantic component count;
- material-region count;
- mechanically distinct ownership count;
- decomposition fragment count under valid view/pose changes.

Only then freeze any slot/query capacity.

### 5.3 Generic hidden-layer curriculum

StdGEN's outside->inside ordering is human-specific. RealSaS must build a subject-local **Component/Visibility Graph** from authored truth.

Nodes may represent owner/material/components. Edges may encode:

- attachment/parent relation;
- containment;
- persistent front/behind relation where valid;
- observed occlusion relation;
- material/owner adjacency.

Training decomposition subsets must be valid graph-derived peel/isolation states, not arbitrary human layer names.

---

## 6. F: semantic-aware high-fidelity extraction

### 6.1 What F is

F is the best static explicit reconstruction surface. It is not automatically the final animation carrier.

### 6.2 Grid shape and resolution

StdGEN baseline: approximately 100x100x150 dense extraction.  
StdGEN++ target: 256x256x384 with coarse-to-fine sparse evaluation.

RealSaS must choose **sampling pitch**, not blindly copy dimensions.

Inputs to the decision:

- normalized bounding-box aspect ratio;
- smallest teacher-visible structure thickness;
- curvature distribution;
- owner/material boundary frequency;
- subject morphology;
- target active-domain recall.

Candidate architecture may use anisotropic grids if the canonical normalization makes one axis systematically longer.

### 6.3 Coarse grid

Candidates must be measured, e.g. 48^3 / 64^3 / 96^3 / 128^3 class depending on normalization and active-domain recall.

Freeze criterion: zero missed teacher surface regions after dilation at a predefined confidence margin, especially thin Bat wings, Spider legs, Shark fins and Mech panels.

### 6.4 Fine grid

Choose from target world-space/normalized pitch; dimensions are derived, not guessed.

Metrics:

- Chamfer/surface distance;
- strict F-score;
- topology/hole count;
- thin-structure continuity;
- semantic-boundary fidelity;
- active query count and peak memory.

### 6.5 Active-domain proposal

Minimum:

```text
geometry surface proposal
UNION semantic/owner boundary proposal
UNION high uncertainty
UNION thin-structure risk
UNION contact/overlap-critical region
```

Geometry occupancy alone is not sufficient for RealSaS because appearance/ownership boundaries matter downstream.

### 6.6 Extraction deformation

Keep bounded deformation. Sub-decisions:

- bound relative to voxel pitch;
- shared versus axis-specific bound;
- whether boundary/thin regions receive a different limit;
- anti-flip/deviation regularization strength.

### 6.7 Hole/continuity loss

StdGEN++ demonstrates that post-hoc refinement cannot reliably repair bad source topology. Therefore continuity must be learned at extraction time.

RealSaS risk classes are generic:

- wings/fins;
- tendrils/tails;
- capes/clothing;
- armor/panel strips;
- any owner/material subset whose teacher surface is continuous but equivalent-SDF extraction fractures it.

The FIT8 teacher must label/measure these failures rather than assuming one semantic class is always hole-prone.

---

## 7. T: animation-carrier retopology

### 7.1 Problem definition

T is not a second full reconstruction model.

Input:

```text
high-fidelity F surface
+ shared scene memory
+ G/S descriptors
+ authored mechanical teacher evidence
```

Output:

```text
M* animation/runtime topology proposal
+ exact binding back to continuous/F surface
+ owner/material boundary bindings
+ confidence/provenance
```

### 7.2 What to preserve from TESSA

Potentially valuable:

- global topology reasoning;
- component/chart/face grammar;
- bounded causal decoding;
- explicit topology proposal contract;
- differentiable mechanical consequence metrics.

Reject as vNext bottleneck:

- 17-D GSA scene input;
- independent relearning of the scene from compressed geometry;
- unconstrained XYZ invention disconnected from F/shared field.

### 7.3 Decoder family is OPEN

Compare at least:

- autoregressive TESSA-style face grammar conditioned on F/shared memory;
- mesh/set encoder + learned representative vertices + connectivity decoder;
- chart/patch proposal + stitching;
- graph/edge-collapse style learned retopology.

Do not choose "autoregressive because TESSA already exists" without a court.

### 7.4 Vertex placement contract

Strong preference:

```text
surface/F binding + bounded residual
```

rather than free XYZ generation.

This keeps T from changing character shape merely to simplify topology and automatically preserves A/S addressability.

### 7.5 Mechanical objective

Legacy TESSA already contains differentiable per-triangle metrics:

- Jacobian;
- singular values;
- area ratio;
- condition number;
- edge stretch;
- orientation.

Important finding: this objective exists but is not currently consumed by canonical TESSA training. vNext T should make dynamic mechanical consequence a real training signal rather than a dormant utility.

### 7.6 Density/LOD target

Do not hard-code 800-900 triangles as the training target until fidelity/mechanics curves are measured.

Required curve:

```text
triangle budget -> static projection error
                -> dynamic stretch/condition/area
                -> owner/material boundary error
                -> mobile/runtime cost
```

The final runtime target can then be selected from an evidence-backed Pareto frontier.

---

## 8. CAO: Canonical Appearance Authority

### 8.1 Role

CAO owns final appearance and presentation materialization on canonical M in both static and dynamic states.

It is not just texture filling.

### 8.2 Generic appearance decomposition

Fragments must be subject-local and may be split/merged by Compiler qualification.

A fragment carries at least:

- canonical M address range;
- owner descriptor;
- material descriptor;
- source support/provenance;
- mechanical owner/attachment relation where available;
- amodal support;
- opacity/appearance evidence;
- view/pose pairwise overlap evidence.

### 8.3 Authority ordering

Observed source remains stronger than learned completion.

Generated future canonical views are hypotheses and can never silently become SOURCE_EXACT.

### 8.4 Visible-surface coherence

Required invariant:

> Every geometrically visible canonical-M sample has a qualified appearance/presentation explanation or an explicit transparent/occluded/intentionally-masked state.

This closes the historical failure where canonical M passed mechanics/static courts yet runtime render contained holes, peppering or fragmented alpha regions.

### 8.5 Dynamic draw order

Raw camera Z is evidence, not complete 2D presentation authority.

CAO/Compiler should consume:

- posed geometry;
- owner/material decomposition;
- pairwise overlap relations;
- attachment/contact evidence;
- learned S-head relation evidence;
- source/art-style rules where qualified.

---

## 9. FIT8 teacher schema as decision instrument

Teacher truth is not merely supervision; it decides architecture leaves.

FIT8 extractor must provide enough information to answer:

- required semantic slot cardinality;
- minimum extraction sampling pitch;
- thin-structure risk distribution;
- authored component/visibility graph complexity;
- dynamic pose deformation metrics;
- attachment/owner relation complexity;
- static/dynamic overlap relation counts;
- material/owner disagreement frequency;
- whether appearance needs view-dependence beyond source-exact CAO support.

No full-corpus scale-up before these measurements are available and schema validity is proven across all eight morphologies.

---

## 10. Freeze protocol

Each unresolved leaf receives one of:

- **LOCKED_BY_REFERENCE_AND_MATCHED_REQUIREMENT** — published mechanism directly matches our requirement and no RealSaS-specific conflict exists;
- **LOCKED_BY_FIT8_MEASUREMENT** — FIT8 evidence chooses among candidates;
- **OPEN_ABLATION_REQUIRED** — alternatives must be trained/evaluated under matched budget;
- **DEFERRED_PRODUCT_PHASE** — e.g. learned 1->8 canonical view synthesis;
- **REJECTED_BY_CONTRACT** — violates genericity, provenance, canonical-surface identity or Compiler authority.

The architecture is not frozen until every product-critical leaf has one explicit state.