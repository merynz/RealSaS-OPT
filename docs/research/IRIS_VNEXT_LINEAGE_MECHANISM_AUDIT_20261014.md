# IRIS-vNext lineage and mechanism audit

**Date:** 2026-10-14  
**Status:** active research authority; facts and open design implications, not a final architecture freeze  
**Scope:** InstantMesh -> StdGEN -> StdGEN++ -> legacy IRIS/TESSA -> IRIS-vNext

This document exists to prevent accidental rediscovery, accidental deletion, and false novelty attribution. Every mechanism is tracked by origin, actual training/inference role, and RealSaS relevance.

## 0. Audit rules

1. Distinguish **substrate** from **paper novelty**.
2. Distinguish **implemented** from **merely described**.
3. Distinguish **code exists** from **training path actually consumes it**.
4. Distinguish **static reconstruction quality** from **animation-carrier quality**.
5. Distinguish **semantic category decomposition** from **generic owner/material/attachment relations**.
6. Do not make a RealSaS design choice only because an upstream paper reports it; identify the requirement it solves.
7. External code is a behavioral/reference source. Licensing is audited separately before code reuse.

---

## 1. InstantMesh is the reconstruction substrate

Reference repository revision inspected: `TencentARC/InstantMesh@08822c52fdc399b93ea00e4fa9e596344ed52ccc`.

### 1.1 Image encoder and camera conditioning

InstantMesh uses a DINO v1 ViT-B/16-class image encoder (`facebook/dino-vitb16`, 768-D class). Camera parameters are not only appended after image feature extraction. Its wrapper maps the camera vector into the ViT hidden dimension and injects it through the custom ViT AdaLN path before returning image tokens.

Implication for RealSaS:

```text
legacy IRIS:
image-only DINO patch features
    + camera embedding later in CameraAwareViewTokenBank

InstantMesh lineage:
image + camera-conditioned ViT tokens
    -> image-to-triplane transformer
```

This is a real architectural difference, not a capacity-only difference.

**OPEN decision:** compare:

- late camera conditioning only;
- encoder-level camera conditioning only;
- both, with explicit anti-redundancy/ablation.

Evaluation must include geometry and cross-view owner/material identity, not only SDF error.

### 1.2 Image-to-triplane transformer

The established substrate is approximately:

```text
image features:        768
query latent width:    1024
transformer layers:    16
attention heads:       16
query planes:          3 x 32 x 32
upsampled planes:      3 x 64 x 64 x 80
```

Each transformer block contains cross-attention from triplane queries to image tokens, self-attention among triplane tokens, then an MLP. A learned 32x32 query lattice is upsampled with a stride-2 transposed convolution.

This is the same broad architecture later inherited by StdGEN.

### 1.3 NeRF stage is not disposable history

InstantMesh's NeRF trainer learns the scene through volume rendering before the explicit mesh regime.

Observed training targets:

```text
RGB MSE
+ 2.0 * LPIPS
+ mask MSE
```

It also perturbs input camera parameters during training and uses random resize/crop for render supervision.

The important architectural lesson is not "use NeRF at runtime". It is:

> a volumetric rendering court teaches a spatial scene representation before sign-sensitive explicit extraction is introduced.

### 1.4 NeRF -> SDF is a parameter-transfer transition

The mesh trainer does not merely start a conceptually new stage from scratch.

It explicitly initializes SDF/RGB MLPs from the trained NeRF decoder:

- shared hidden density/color MLP weights are copied;
- density final-layer sign is inverted into the SDF convention;
- an isosurface threshold offset is folded into the SDF bias;
- RGB final-layer weights are copied;
- extraction deformation and FlexiCubes configuration heads are newly introduced.

Therefore the real lineage is:

```text
volumetric scene learning
     |
     | parameter transfer
     v
signed explicit-surface learning
```

**IRIS-vNext design branch:**

- one shared field trunk with sigma/SDF mode-specific heads;
- explicit NeRF->SDF parameter transfer like InstantMesh;
- staged joint heads with frozen/regression-gated transitions.

Do not choose between these before measuring feature retention and optimization stability on FIT8.

### 1.5 InstantMesh mesh-stage supervision

The mesh stage adds geometric rendering losses and extraction regularizers:

```text
RGB MSE
+ LPIPS
+ mask
+ depth L1
+ normal alignment
+ SDF sign-edge regularization
+ FlexiCubes surface regularization
+ FlexiCubes weight regularization
```

This is important because StdGEN's later semantic mesh stage is built on a mature differentiable explicit-surface court rather than inventing one from scratch.

### 1.6 Extraction is differentiable, but not animation-aware

InstantMesh predicts:

- SDF;
- bounded extraction-grid deformation;
- FlexiCubes cell/configuration weights;
- RGB.

The deformation is bounded relative to grid resolution to reduce triangle flips.

This is a strong static reconstruction mechanism. It contains no animation-carrier objective analogous to RealSaS stretch/condition/area/Jacobian requirements.

---

## 2. StdGEN: what is actually new over InstantMesh

Reference inspected: `hyz317/StdGEN@f1b0c822689dc2539c774bb73090a67c66e2e766` and released `main` sources.

### 2.1 Reconstruction backbone is largely inherited

StdGEN preserves the main InstantMesh skeleton:

- DINO ViT-B/16-class encoder;
- 1024/16-layer/16-head image-to-triplane transformer;
- 32->64 triplane family;
- 80 triplane channels;
- NeRF then SDF/FlexiCubes training lineage;
- bounded extraction deformation;
- differentiable mesh rendering.

Do not attribute these pieces to the semantic innovation itself.

### 2.2 Semantic decoder is colocated with geometry/appearance

StdGEN adds a semantic decoder over the same sampled triplane feature used by RGB/density or SDF/RGB/deformation heads.

This is a major principle for RealSaS:

> semantics are a spatial field aligned with geometry and appearance, not a late screen-space segmentation product.

### 2.3 Semantic-equivalent NeRF

During volume rendering, semantic membership participates in effective opacity/transmittance. Removing a semantic region changes ray integration itself rather than simply masking the final 2D image.

This allows internal/occluded color and semantics to receive training signal when outer categories are removed.

RealSaS translation cannot use a fixed global hair/cloth/body simplex as primary semantics. It needs:

```text
primary: relational owner/material descriptors
secondary extraction/decomposition simplex: subject-local normalized region membership
```

Only the second should be used directly in semantic-equivalent NeRF/SDF formulas.

### 2.4 Semantic-equivalent SDF

StdGEN couples semantic probabilities to the SDF before FlexiCubes extraction. The important property is that the original SDF zero/exterior constraint remains a hard geometric constraint while semantic margins choose which subset of the inside field remains extractable.

RealSaS should preserve this coupling for compiler-resolved categorical subsets while keeping relational owner embeddings separate.

### 2.5 Multi-layer curriculum is the real semantic teaching innovation

The released implementation supports semantic levels such as:

```text
full character
body + clothing
base body
```

and layer-specific extraction such as cloth-only/hair-only.

The human-specific ordering is not portable, but the curriculum principle is.

RealSaS replacement:

- authored Component/Visibility Graph;
- graph-valid peel/isolation states;
- hidden RGB + semantics jointly supervised;
- no global human anatomy assumption.

### 2.6 StdGEN extraction resolution

Released configuration uses an anisotropic grid around `100 x 100 x 150` for full character extraction.

Important: this is extraction-grid resolution, not triplane resolution.

---

## 3. StdGEN++: actual bottlenecks and actual changes

Reference: arXiv `2601.07660`.

### 3.1 Dual branch addresses representation competition, not generic semantics

StdGEN++ identifies a conflict between global body reconstruction and fine facial detail. It introduces two LoRA-specialized branches:

- full-body branch;
- cropped/rescaled facial branch.

LoRA is inserted in self-attention/cross-attention projection layers; reported rank is 128.

The transferable lesson is **selective specialization under measured gradient/detail conflict**.

The non-transferable part is **human face as a universal privileged region**.

RealSaS candidate router signals:

- owner/material boundary density;
- geometry/semantic uncertainty;
- thin-structure risk;
- projected high-frequency detail;
- contact/overlap importance;
- high CAO appearance decomposition complexity.

Do not instantiate a second branch until FIT8 demonstrates that a shared branch actually bottlenecks these regions.

### 3.2 High-resolution extraction is the biggest measured geometry gain

StdGEN++ keeps the triplane paradigm but scales explicit extraction from StdGEN's roughly `100 x 100 x 150` to `256 x 256 x 384`.

It avoids dense evaluation by:

1. evaluating a coarse SDF grid;
2. marking negative-SDF occupancy;
3. dilating with a 3x3x3 neighborhood;
4. nearest-neighbor upsampling that proposal to the fine grid;
5. evaluating expensive SDF/deformation/semantic predictions only in the active domain.

Ablation on hair F1 at a strict 0.5% threshold:

```text
StdGEN baseline                 0.6416
+ coarse-to-fine high-res       0.6995
+ specialized facial branch     0.7245
```

This proves two separate effects:

- resolution/active-domain scaling provides a large gain;
- specialized representation adds further gain.

RealSaS must not conflate them.

### 3.3 Fullbody and local extraction volumes are explicitly different

Reported full-body extraction:

```text
256 x 256 x 384
physical span ~0.7 x 0.7 x 1.05 of normalized character cube
```

Reported facial branch:

```text
180 x 180 x 180
span ~0.25 x 0.25 x 0.25
```

Thus specialization is not only attention weights: it changes spatial normalization and effective sampling pitch.

RealSaS generic specialization, if adopted, must specify:

- local coordinate frame;
- crop/region scale;
- sampling pitch;
- merge/stitch policy back to the global field;
- authority when global and local predictions disagree.

### 3.4 Hole loss targets a sign-field topology failure

StdGEN++ adds a hole-filling loss over sign-changing edges in semantic-equivalent SDFs, especially for thin semantic structures. It nudges unstable positive values toward neighboring interior support while preserving a usable zero crossing.

The transferable lesson is not "hair needs a hole loss".

RealSaS should identify risk regions from truth:

- wing membranes;
- fins;
- long thin appendages;
- cloth/capes;
- panel strips;
- semantic boundaries whose teacher support is continuous.

### 3.5 Test-time refinement is explicitly demoted

StdGEN++ reports that post-hoc optimization cannot reliably repair a bad initial topology. Refinement can smooth the boundary of a hole without reconstructing missing topology and can even degrade thin structures.

They therefore use refinement selectively, mainly where smooth body surfaces benefit.

RealSaS consequence:

> Compiler repair must remain bounded qualification/repair, not a substitute for correct learned surface support.

Historical `canonical M PASS but visible holes remain` is exactly the class of failure that should be prevented upstream or rejected by CAO coherence, not cosmetically repaired downstream.

### 3.6 Texture decomposition is generative, not segmentation-only

StdGEN++ treats appearance decomposition as a multi-frame generative inpainting problem. Separate semantic texture layers exchange information through spatial/temporal attention so hidden backgrounds (e.g. skin behind iris/lashes) are reconstructed rather than merely alpha-masked.

This is a major lesson for CAO:

- decomposition needs hidden appearance completion;
- region separation and completion are coupled;
- a monolithic texture atlas is insufficient for editable layered presentation.

The human facial layer taxonomy is not generic and is rejected as a core RealSaS ontology.

---

## 4. Legacy IRIS: what is genuinely strong and what is undersized

### 4.1 Shared triplane idea already exists

`models/iris/v4/scene_first_signed_v4.py` already has:

```text
CameraAwareViewTokenBankV4
    -> SharedTriplaneSceneFusionV4
    -> ContinuousSignedSurfaceFieldV4
```

It is not conceptually alien to InstantMesh/StdGEN.

### 4.2 Capacity is materially smaller

Legacy V4 default class:

```text
image token dim:      384
scene width:          256
scene layers:         4
heads:                8
query plane:          32 -> 64
plane channels:       64
field hidden:         192
```

Compared with the established InstantMesh/StdGEN class, the largest difference is scene-transformer capacity, not merely plane resolution.

### 4.3 Camera conditioning is later and explicit

IRIS adds:

- projected image token;
- explicit camera embedding;
- explicit token-XY embedding;

before flattening all views into scene memory.

This may be beneficial for exact RealSaS camera contracts, but it has not been compared against camera-conditioned ViT tokens under the new G/A/S objectives.

### 4.4 Legacy IRIS optimization is not an InstantMesh-like render curriculum

The current shared FIT8 trainer is based on direct signed-field supervision (`F*`) plus sign/silhouette objectives. It queries 3D points and regresses the source-derived signed field.

This is a RealSaS strength because authored 3D FIT8 truth can provide far more direct geometric supervision than the 2D-only regime emphasized by StdGEN++.

However, it means IRIS does **not** currently receive the volumetric RGB/semantic curriculum that teaches hidden appearance/semantics in StdGEN.

IRIS-vNext should exploit both advantages:

```text
authored 3D geometry truth
+ differentiable RGB/semantic rendering
+ hidden component supervision
```

rather than replacing strong 3D truth with a weaker 2D-only imitation.

---

## 5. Legacy TESSA: separate useful ideas from dormant ideas

### 5.1 Current input bottleneck is real

`models/tessa/v1/conditioning_v1.py` reduces the upstream surface to 17 values per sample:

```text
XYZ                       3
normal + normal-valid     4
view-support bits         8
graph degree              1
persistence               1
----------------------------
                          17
```

This discards shared scene appearance, semantic ownership and rich image/triplane information.

### 5.2 TESSA decoder has independent value

Useful ideas include:

- one global topology sequence;
- true connected components rather than arbitrary training chart splits;
- adjacent-face serialization;
- bounded local causal attention;
- component/chart/face grammar;
- explicit Compiler qualification after learned proposal.

These are not invalidated merely because the input representation is poor.

### 5.3 Free quantized XYZ should be re-examined

Current TESSA quantizes each coordinate into 1024 bins in the canonical frame and can emit geometry as part of topology generation.

In IRIS-vNext, high-fidelity F already supplies geometry. Therefore a more constrained carrier model is preferable:

```text
F/shared-surface binding
+ bounded residual
+ learned connectivity/topology
```

rather than unconstrained coordinate reinvention.

### 5.4 Mechanical objective exists but is dormant

`models/tessa/v1/mechanical_objective_v1.py` provides differentiable:

- per-triangle deformation Jacobian;
- singular values;
- area ratio;
- condition number;
- maximum edge stretch ratio;
- orientation comparison.

Repository audit finds this objective exported and tested, but not consumed by the canonical TESSA training path.

This distinction is important:

> RealSaS has already designed useful dynamic-carrier losses, but has not yet actually trained TESSA with them.

IRIS-vNext T should treat these or improved equivalents as first-class training signals.

### 5.5 T becomes retopology, not reconstruction

Because F provides a high-fidelity static surface, T's target problem becomes:

```text
high-fidelity surface
+ semantic/material boundaries
+ shared scene features
+ authored motion consequences
    ->
low(er)-density animation/runtime carrier
```

This is a different problem from legacy TESSA's "infer topology + coordinates from compressed GSA evidence".

The decoder family therefore remains OPEN.

---

## 6. Cross-lineage decision branches now required

### 6.1 Camera conditioning

Compare:

```text
A. pure-image backbone -> late camera token conditioning
B. camera-conditioned image encoder -> scene transformer
C. both
```

Metrics:

- cross-view correspondence;
- source projection error;
- owner/material affinity consistency;
- asymmetric left/right identity;
- appearance consistency;
- training stability.

### 6.2 NeRF->SDF transition mechanism

Compare:

```text
A. InstantMesh-style parameter transfer
B. shared trunk + separate sigma/SDF heads
C. shared trunk + one sign/density convertible head
```

Must regression-test RGB/semantic hidden-layer competence after SDF activation.

### 6.3 Scene feature resolution versus extraction resolution

Never use one number for both.

Independent decisions:

```text
triplane base resolution
triplane channel width
coarse extraction grid
fine extraction sampling pitch
local specialized extraction scale
```

### 6.4 Specialization trigger

Do not create a generic second branch because StdGEN++ has a facial branch.

First measure whether global/shared capacity fails systematically at:

- thin structures;
- high curvature;
- semantic/material boundaries;
- high-frequency projected appearance;
- contact-critical regions.

Only then choose adapter/router capacity.

### 6.5 T decoder family

Compare under the same F surface and mechanical objective:

- TESSA-style autoregressive face grammar;
- learned representative-vertex set + connectivity decoder;
- chart/patch proposal + stitching;
- learned collapse/split/flip retopology over F;
- hybrid only if each half has a separately justified role.

### 6.6 Appearance decomposition

CAO must distinguish:

```text
visible source-exact appearance
learned hidden appearance
owner decomposition
material decomposition
pose/view overlap ordering
intentional transparency
```

A single RGBA field alone is insufficient for the final 2D presentation authority.

---

## 7. Current non-negotiable conclusions

These are stronger than implementation preferences and can guide subsequent branches:

1. **Use FIT8 full truth before scale-up.** The eight morphologies are the schema/architecture closure cohort.
2. **Do not solve learned 1->8 in the first architecture attempt.** Use eight authored views while keeping the input contract replaceable.
3. **Keep one co-addressable continuous scene.** Geometry, appearance and semantics must not require post-hoc identity rematching.
4. **NeRF is training scaffolding, not product geometry authority.**
5. **F is high-fidelity static extraction; T is animation-carrier retopology.**
6. **Dynamic mechanical quality must actually enter T training.** A dormant utility is not a capability.
7. **High-resolution extraction must solve topology at the source.** Compiler repair cannot be the primary hole-repair method.
8. **CAO is not texture fill.** It is final static/dynamic appearance and presentation authority on canonical M.
9. **Generic articulation forbids hard-coded human semantic ontology.**
10. **Every reduction/expansion must be traceable to a measured problem.**

No scientific PASS is claimed by this audit.