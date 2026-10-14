# IRIS-vNext open-surface / arbitrary-topology representation audit

**Date:** 2026-10-14  
**Status:** active architecture research — no representation freeze, no scientific PASS  
**Trigger:** exact-byte FIT8 census found genuinely non-watertight/open components in Bat and George, while other subjects contain closed volumetric components.

This audit exists because a generic-articulation model cannot assume that every authored surface is the boundary of a closed volume. Any geometry representation used by IRIS-vNext must therefore be evaluated against both closed and open/two-sided surface support.

---

## 1. Problem statement from FIT8, not from theory alone

The FIT8 census demonstrates all of the following in one eight-subject closure cohort:

```text
closed manifold continuous surfaces
multiple disconnected closed components
open surface components
mixed closed + open components inside one asset
smooth skinning
rigid bone-owned components
material-color appearance
UV-textured appearance
double-sided/unlit presentation
```

Examples:

- Horse: one position-welded closed manifold surface.
- Snake/T-Rex/Spider/Shark: source FBX components are closed edge-manifold under the diagnostic source-polygon audit.
- Bat: position-weld diagnostic contains open/non-manifold components; source material is unlit and double-sided.
- George: one source mesh/material but diagnostic geometry resolves dozens of components, including both closed and open components.

Therefore:

> `one global metric SDF with a universally valid inside/outside classification` is not a safe corpus invariant.

This does **not** mean that signed fields should be discarded. It means their authority must be validity-gated.

---

## 2. Candidate A — global UDF

References examined:

- NeuralUDF (CVPR 2023)
- NeUDF (CVPR 2023)
- DUDF (CVPR 2024)
- DualMesh-UDF (ICCV 2023)
- 2S-UDF (CVPR 2024)

### Strengths

1. UDF naturally represents open and closed surfaces without requiring a global inside/outside partition.
2. NeuralUDF/NeUDF demonstrate multi-view volume rendering with arbitrary topology.
3. DUDF improves differentiability near the zero level set through a hyperbolically scaled formulation.
4. 2S-UDF shows that a staged density/weight curriculum can improve UDF rendering stability and occlusion awareness.

### Costs for RealSaS

1. A true UDF has no sign change across the surface, so ordinary Marching Cubes/FlexiCubes zero-crossing logic no longer applies directly.
2. Neural UDF gradients/distances are most problematic near the exact region where mesh extraction needs them. DualMesh-UDF therefore introduces a dedicated adaptive-octree/QEF/filtering extractor.
3. Moving F from FlexiCubes to UDF-specific extraction would discard a large, well-tested part of the InstantMesh -> StdGEN -> StdGEN++ substrate.
4. StdGEN semantic-equivalent SDF has no direct one-line equivalent on an unsigned field; semantic subset extraction would need a new formulation.
5. Orientation/two-sided normal ambiguity remains a separate concern.

### Current judgment

**KEEP AS REFERENCE / FALLBACK, NOT PRIMARY FIRST CHOICE.**

UDF is the cleanest mathematical answer to arbitrary topology, but it introduces a second hard problem — stable high-quality differentiable meshing — precisely where RealSaS already has a strong extraction lineage. Replacing SDF/FlexiCubes globally would be a large architectural expansion and is not yet justified by FIT8.

---

## 3. Candidate B — SDF / level-set + validity (NeAT family)

Reference examined:

- NeAT (CVPR 2023), paper and released implementation.

### Mechanism

NeAT predicts from the same point:

```text
signed scalar f(x)
validity V(x)
positional feature
```

Validity participates in volume-rendering alpha, so a zero-level crossing that does not correspond to an actual surface can be suppressed. At export, released code masks low-validity field samples before Marching Cubes. NeAT also modifies rendering to support both sides of open surfaces instead of relying on a closed-surface outside->inside event.

### Why it is interesting for RealSaS

It separates two questions that legacy IRIS implicitly conflated:

```text
where is the level set?
where does a real authored surface actually exist?
```

For a closed component, `f(x)` may be supervised as true metric SDF and validity is high around the entire intended zero-set.

For an open component, `f(x)` can be treated as a local/oriented level-set helper while validity/support terminates the surface at its real open boundary.

### Risks

1. A learned signed scalar around an open surface is not automatically a globally meaningful metric SDF.
2. NeAT's released exporter uses validity-masked Marching Cubes, not the differentiable FlexiCubes path used by StdGEN.
3. Pointwise validity alone can produce brittle/jagged boundaries if not regularized structurally.
4. Validity semantics must not be confused with owner/material semantic probabilities.

### Current judgment

**STRONG CONCEPTUAL INPUT.**

The existence/validity factorization is likely useful even if the exact NeAT extraction mechanism is not adopted.

---

## 4. Candidate C — sparse active-cell FlexiCubes (SparseFlex family)

References examined:

- SparseFlex / TripoSF (ICCV 2025), paper, supplement, released representation/decoder code.

### Mechanism

SparseFlex retains the FlexiCubes-style local representation:

```text
per grid corner:
  signed scalar
  deformation

per active voxel:
  interpolation/configuration weights
```

but it does **not** instantiate a dense voxel domain. A sparse set of active voxels defines where the isosurface extractor is allowed to exist.

The released decoder contains learned pruning heads after sparse subdivision. At inference, predicted low-occupancy children are physically removed from the sparse tensor before mesh extraction.

Paper training includes:

```text
L_render = depth + normal + mask + SSIM/LPIPS-on-normal
L_prune  = BCE(ground-truth voxel occupancy, predicted occupancy)
L_flex   = FlexiCubes regularization
```

Ground-truth voxel occupancy is derived from the input surface point cloud rather than from watertight SDF preprocessing.

### Why it is especially relevant

SparseFlex unifies two needs that StdGEN++ treats separately:

1. high extraction resolution without dense-grid memory explosion;
2. open boundaries via sparse support/pruning.

It therefore suggests a more generic replacement for StdGEN++'s rule:

```text
coarse negative-SDF occupancy -> dilate -> fine SDF queries
```

because `negative SDF` is not a valid universal activation rule for an open sheet.

A RealSaS-specific rule could instead be:

```text
coarse learned/teacher surface-support cells
        UNION
geometry-near-surface proposal
        UNION
owner/material boundary proposal
        UNION
high-uncertainty proposal
        UNION
contact/overlap-critical proposal
            -> fine sparse extraction domain
```

### Major RealSaS advantage over SparseFlex's original setting

SparseFlex's VAE starts from surface point clouds and uses rendering supervision. FIT8 has the exact authored mesh, materials, normals, cameras, skin and animation.

RealSaS can therefore generate stronger teacher evidence:

```text
exact surface-cell intersection / support
source triangle binding per active cell
closed/open/non-manifold component class
local source normals/orientation validity
metric SDF where qualified
unsigned distance everywhere
render depth/normal/mask
owner/material region support
```

### Risks / open questions

1. SparseFlex's released model is a point-cloud VAE/shape representation, not an image-to-triplane LRM. We can adopt the representation/extraction principle without adopting its whole network.
2. Paper describes frustum-aware sectional voxel training, but the public VAE release does not appear to include the full training path; treat that part as paper-confirmed, not code-confirmed.
3. The sparse scalar field inside active voxels still needs a stable local orientation/sign convention for open surfaces.
4. Sparse voxel pruning must be conservative: one false-negative cell can literally cut a hole in the surface.
5. Semantic-equivalent extraction must define whether semantics change scalar values, active-cell support, or both.

### Current judgment

**PRIMARY CANDIDATE FOR F REPRESENTATION RESEARCH.**

Sparse active-cell support appears more compatible with the existing StdGEN/StdGEN++ extraction lineage than a global UDF replacement, while directly addressing a generic-articulation failure mode exposed by FIT8.

---

## 5. The likely abstraction: distance/orientation and surface existence are separate axes

Do not freeze the final equations yet, but the current evidence strongly supports separating:

```text
GEOMETRIC PROXIMITY / LEVEL SET
from
SURFACE EXISTENCE / SUPPORT
```

Candidate shared field evidence:

```text
u(x)  = unsigned distance / proximity evidence
f(x)  = signed or locally-oriented level-set evidence where valid
q(x)  = sign/orientation validity
s(x)  = surface-support / existence evidence
```

This allows different teacher authority by component:

### Qualified closed orientable component

```text
u(x): metric UDF truth
f(x): metric SDF truth
q(x): high/valid
s(x): actual near-surface support
```

### Qualified open orientable sheet

```text
u(x): metric UDF truth
f(x): local/oriented level-set truth only where orientation validity is established
q(x): locality/orientation-validity mask
s(x): exact authored open-surface support / active-cell truth
```

### Non-manifold / ambiguous component

```text
u(x): still broadly valid
f(x): invalid or only partially valid
q(x): explicit abstention
s(x): surface-support truth remains available
```

The model/Compiler must never reinterpret an invalid `f(x)` value as metric signed-distance authority.

---

## 6. Consequence for G

Legacy:

```text
G -> sdf + uncertainty
```

Possible vNext expansion under evaluation:

```text
G ->
  surface proximity / unsigned distance
  signed-or-oriented level-set evidence
  sign/orientation validity
  geometry uncertainty
  surface-existence evidence
```

This is an intentional expansion motivated by exact FIT8 topology, not feature accumulation for its own sake.

`metric SDF` may remain a privileged supervision court on components where it is mathematically valid.

---

## 7. Consequence for NeRF scaffold

The early volumetric court must not reintroduce closed-surface assumptions.

Candidate branches:

### Closed-qualified samples

Use established InstantMesh/StdGEN density or SDF-derived rendering supervision plus exact authored RGB/semantic targets.

### Open/two-sided samples

Compare:

- NeAT-style validity-weighted two-sided rendering;
- surface-support-conditioned density;
- SparseFlex-style explicit mesh rendering after sufficient F warm-up.

Do not choose a single closed-surface density conversion merely for implementation uniformity.

---

## 8. Consequence for semantic-equivalent extraction

StdGEN's semantic-equivalent SDF remains valuable **inside a valid extraction domain**.

For generic RealSaS the operation may need two independent masks:

```text
surface support / active cells
AND
semantic region membership
```

Conceptually:

```text
F_region = Extract(
    scalar_field modified by semantic membership,
    active_cells constrained by real surface-support evidence
)
```

This prevents semantic equations from manufacturing geometry in cells where no physical/authored surface exists.

Open question: semantic subset removal may also change which sparse cells are active; this should be supervised from decomposition-state truth rather than inferred from category names.

---

## 9. Consequence for F extraction resolution

Resolution decisions now have **two independent budgets**:

```text
support-grid / sparse-cell resolution
within-cell scalar/deformation/config quality
```

Candidate evaluation ladder should be based on world-space sampling pitch, not arbitrary cube labels.

For each FIT8 subject measure at minimum:

- thinnest mechanically/presentation-relevant surface feature;
- open-boundary curvature/radius;
- owner/material boundary frequency;
- active-cell recall under coarse support prediction;
- false-negative support holes;
- false-positive support shell/bridges;
- memory/query count;
- depth/normal/silhouette error;
- extracted component/boundary topology fidelity.

Only after these measurements should values analogous to 64/96/128/256/etc. be frozen.

---

## 10. Consequence for T

T must preserve open/closed surface class rather than "repairing" every open carrier into a closed manifold.

Animation-carrier qualification therefore needs topology-policy by component class:

```text
closed deforming surface
open two-sided deforming sheet
rigid bone-owned open/closed part
explicit attachment
```

Dynamic metrics such as Jacobian/stretch/area remain meaningful on triangle surfaces regardless of whether the component encloses volume.

A blanket `M must be globally closed manifold` policy is therefore incompatible with generic authored articulation unless the product contract explicitly chooses a proven conversion.

---

## 11. Consequence for CAO

Open/two-sided surfaces make presentation semantics more important, not less.

CAO needs explicit evidence for:

```text
front-side appearance
back-side appearance
source material double-sidedness
view-conditioned side ownership
surface normal/orientation confidence
intentional transparency
open boundary support
pose/view overlap relation
```

A two-sided Bat wing is not an appearance hole merely because no volumetric backside exists. Conversely a missing appearance sample inside an authored open sheet's support is still an unexplained visible failure.

Thus visible-coherence is measured against **qualified surface support**, not against a presumed closed solid.

---

## 12. Current ranking and freeze gate

Current research ranking:

```text
1. sparse surface-support / active-cell extraction + local signed field
   (SparseFlex-inspired, adapted to triplane LRM)

2. pointwise SDF/level-set + validity
   (NeAT-inspired; strong conceptual fallback or auxiliary evidence)

3. global UDF + dedicated UDF mesher
   (most general field, largest architecture departure)
```

This ranking is not a final architecture freeze.

Before selecting candidate 1, FIT8 must answer:

1. Are Bat/George open components visibly/mechanically significant or mostly export debris?
2. Can source normals/orientation establish stable local sign on important open components?
3. What support-grid resolution gives zero false-negative holes on authored truth?
4. Can sparse support be learned from 8-view scene features without bridges/missing cells?
5. Does semantic-equivalent extraction remain stable when support terminates at open boundaries?
6. Does F preserve source boundary loops and two-sided normals sufficiently for T/CAO?

No scientific or product PASS is claimed.