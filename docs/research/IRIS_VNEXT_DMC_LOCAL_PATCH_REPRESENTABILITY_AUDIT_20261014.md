# IRIS-vNext DMC local-patch representability audit

**Date:** 2026-10-14  
**Status:** active architecture research — no final F freeze, no scientific/product PASS  
**Depends on:** `IRIS_VNEXT_F_SUPPORT_GRID_COLLISION_AUDIT_20261014.md`, `IRIS_VNEXT_F_LOCAL_SHEET_MULTIPLICITY_20261014.md`

This audit asks a narrower question than `SDF vs UDF` or `sparse vs dense`:

> Given a sparse high-resolution cell domain, is the standard FlexiCubes / DMC corner-scalar representation expressive enough for the local authored surface multiplicity found in FIT8?

The answer is more nuanced than the earlier working assumption. FlexiCubes is already a Dual Marching Cubes family extractor and can emit multiple dual vertices from one cube. Therefore `multiple surface patches in one cell` is **not by itself** proof that FlexiCubes is incapable.

The sharper representability question is whether the authored surface requires **multiple distinct crossings on the same grid edge**, because a standard corner-scalar field provides one interpolated zero crossing per edge.

---

## 1. Correction: FlexiCubes is not simply one vertex / one sheet per cell

The released FlexiCubes machinery uses a DMC lookup structure and can associate more than one dual vertex group with a cube. The lookup capacity reaches multiple local contour components in ambiguous cubes.

Therefore the earlier simplified argument

```text
multiple components in one cell
=> FlexiCubes necessarily collapses them
```

is rejected.

The actual local data path is closer to:

```text
corner scalar/sign configuration
        |
identify sign-changing cube edges
        |
one interpolated zero crossing per edge
        |
DMC lookup groups crossings into one or more dual vertices
        |
FlexiCubes alpha/beta/gamma + deformation refine geometry/triangulation
```

This distinction matters. The DMC dual-vertex stage has more topological capacity than a one-vertex-per-cell SurfaceNets interpretation.

---

## 2. The remaining hard bound: one scalar crossing per grid edge

A scalar sampled at edge endpoints provides one linear zero-crossing estimate on that edge.

If authored truth intersects the same grid edge at two or more genuinely distinct parameters:

```text
edge e
  t = 0.19 -> sheet A
  t = 0.73 -> sheet B
```

then one endpoint-sign pair plus one interpolation cannot encode both crossings simultaneously without additional local state.

Increasing DMC dual-vertex count does not automatically recover crossings that were never represented at the edge stage.

Therefore the teacher court measures exact segment-triangle intersections on support-cell edges, deduplicates intersections by edge parameter, and records source component identity for each crossing.

---

## 3. N=128 exact edge-crossing census

The current deterministic prototype was run on collision cells for Snake, Spider, Bat and George at N=128.

| Subject | Collision cells | Cells with multi-crossing edge | Cells with multi-crossing from distinct components | Distinct-component multi-crossing edges | Max crossings on one cell edge |
|---|---:|---:|---:|---:|---:|
| Snake | 152 | 118 | **118** | 152 | 2 |
| Spider | 370 | 314 | **304** | 480 | 3 |
| Bat | 1,243 | 1,041 | **1,040** | 2,256 | 3 |
| George | 2,548 | 2,030 | **1,942** | 4,076 | 3 |

At N=64, George reaches four distinct crossings on one cell edge in the prototype.

Coincident multi-component crossings at essentially the same edge parameter are rare. The dominant failure class is not duplicate identity on one geometric crossing; it is **multiple separate crossing positions** along the same edge.

This falsifies the claim that all local multiplicity can be resolved only by DMC's existing multi-dual-vertex grouping.

---

## 4. Resolution sweep: multi-crossing does not disappear

Selected persistent pairs were re-evaluated at N=128, 256 and 512.

Representative distinct-multi-crossing cell fractions:

```text
Snake pair (0,1)
  N128 ~77.6%
  N256 ~77.8%
  N512 ~80.1%

Spider exact/contact pair (1,2)
  N128 ~50.0%
  N256 ~71.4%
  N512 ~98.1%

Bat exact/contact pair (2,10)
  N128 ~79.2%
  N256 ~79.5%
  N512 ~77.7%

Bat near pair (2,9)
  N128 ~89.1%
  N256 ~91.9%
  N512 ~89.3%

George exact Neck/Chest pair (26,29)
  N128 ~24.0%
  N256 ~32.7%
  N512 ~42.0%

George near Shoulder/Neck pair (22,26)
  N128 100.0%
  N256 ~95.7%
  N512 ~96.3%
```

These values are research diagnostics, not scientific benchmark claims. The important qualitative result is monotonicity failure: resolution increase does **not** systematically collapse multi-crossing to zero.

The surfaces are genuinely close/contacting/layered over extended regions.

Therefore:

> `increase N until one crossing per edge becomes true` is not a viable generic primary strategy.

---

## 5. Local patch identity is finer than global component identity

A second exact-cell audit was run across **all active cells**, not only multi-component collision cells. Local triangle adjacency inside each cell is used as a lower-bound local surface-patch count.

At N=128:

| Subject | Active cells | Multi-patch cells | Same-global-component multi-patch cells | Max local patch count |
|---|---:|---:|---:|---:|
| Snake | 22,648 | 194 | 42 | 2 |
| Bat | 20,232 | 1,259 | 16 | 3 |
| T-Rex | 9,616 | 154 | **154** | **4** |
| Horse | 17,531 | 42 | **42** | 2 |
| Spider | 13,530 | 438 | 68 | 3 |
| GreenBlob | 56,486 | 353 | 3 | 2 |
| Shark | 7,808 | 170 | **170** | 2 |
| George | 18,613 | 2,664 | 116 | **4** |

This is one of the most important FIT8 findings so far.

T-Rex, Horse and Shark demonstrate that even a single global connected component can fold or approach itself so that multiple disconnected local patches occupy one cell.

Therefore none of these is a sufficient local-sheet identity:

```text
global connected component
material ID
mechanical owner / dominant bone
source object ID
```

A **local patch / sheet relation** is its own geometric factor.

---

## 6. Interesting capacity coincidence — but not a proof of sufficiency

At N=128 the measured lower-bound maximum local patch count in FIT8 is four.

The existing DMC/FlexiCubes lookup family can represent multiple dual contour groups in one cube, with capacity in this range.

This is encouraging: the good DMC dual-vertex machinery may not need to be discarded.

However the edge-crossing audit shows the actual bottleneck can occur **before** dual-vertex grouping:

```text
local patch multiplicity capacity: potentially adequate
edge crossing multiplicity: not represented by one scalar interpolation
```

So the current preferred research direction is not `replace DMC entirely`.

It is:

> preserve sparse DMC/FlexiCubes topology/dual-vertex machinery where useful, but generalize the local crossing representation or condition it by local sheet identity.

---

## 7. Updated F candidate families

### F-A — standard sparse FlexiCubes / single unconditioned scalar field

Keep as a required baseline.

Strengths:

- simplest;
- inherits mature InstantMesh -> StdGEN -> SparseFlex lineage;
- differentiable and efficient;
- already supports multiple DMC dual vertices per cube.

Measured weakness:

- cannot directly encode multiple distinct zero crossings on one grid edge from one corner-scalar pair.

Current state: **strongly weakened as the generic primary representation, but not removed from matched-budget court.**

### F-B — sparse support + local-sheet-conditioned scalar/FlexiCubes fields

Concept:

```text
active cell c
   |
local sheet queries k=1..K
   |
for each k:
  local scalar/orientation/crossing evidence
  FlexiCubes-compatible geometry/config
   |
Compiler-qualified stitching / merge
```

Advantages:

- preserves most FlexiCubes machinery;
- one edge may carry different crossing estimates for different local sheet hypotheses;
- ordinary K=1 cells remain cheap;
- extra capacity is paid only in local multiplicity regions.

Risks:

- local-sheet permutation/cardinality matching;
- cross-cell sheet identity continuation;
- accidental duplicate sheets;
- topology stitching authority.

### F-C — explicit multi-crossing Hermite/DMC representation

Instead of deriving one crossing from corner scalars, predict or supervise a variable local set of crossing primitives.

Candidate local evidence:

```text
active cell
+ patch tokens
+ per-patch edge crossing parameter(s)
+ crossing normal / orientation validity
+ patch confidence
+ patch affinity to neighbor-cell patch tokens
+ optional FlexiCubes alpha/beta/gamma or dual-vertex residual
```

This has a clear classical geometry-processing lineage in Dual Contouring / Manifold Dual Contouring / Dual Marching Cubes: local contour components and Hermite edge evidence are first-class rather than hidden consequences of a single scalar field.

Advantages:

- directly matches the measured failure mode;
- teacher truth can supervise crossings exactly from authored triangles;
- open surfaces do not require a fabricated global sign;
- topology and geometry can be supervised separately.

Costs:

- larger architectural change;
- variable-cardinality local matching;
- needs robust manifold/open-boundary assembly;
- NeRF/SDF training scaffold no longer maps one-to-one onto final F parameters.

### F-D — global UDF / dedicated arbitrary-topology mesher

Retain as a fallback/reference family.

It solves the global sign assumption more directly but changes the extraction substrate most aggressively and does not by itself solve local patch identity, semantics, dynamic carrier requirements or CAO binding.

---

## 8. TopoMesh 2026 changes the training design, not the generic truth contract

TopoMesh is a high-value new reference because it explicitly identifies a topology/geometry optimization conflict in FlexiCubes-like decoders.

Paper-confirmed mechanism:

```text
standard FlexiCubes scalar s
    sign(s)      -> topology
    magnitude(s) -> geometry

TopoMesh decouples:
    topology = occupancy/sign o + triangulation gamma
    geometry = magnitude u + alpha/beta/deformation
```

It then uses:

- direct topology supervision;
- direct vertex supervision;
- direct normal/orientation supervision;
- Teacher Forcing with ground-truth topology for geometry learning;
- GT-guided voxel pruning to avoid early false-negative holes;
- progressive 32^3 -> 64^3 -> 128^3 training.

These ideas map very well to RealSaS F.

### What should be translated

RealSaS should strongly consider:

```text
TOPOLOGY EVIDENCE
separate from
GEOMETRY EVIDENCE
```

and train geometry under qualified teacher topology/patch structure before asking the network to jointly predict both.

For F-B/F-C this could become:

```text
Topo:
  active-cell support
  local patch cardinality / incidence
  patch-edge incidence / neighbor continuation
  triangulation/config class

Geom:
  crossing positions
  local normals/orientation
  vertex/dual-vertex residuals
  bounded extraction deformation
```

Teacher Forcing is especially attractive because FIT8 teacher construction owns exact source triangles and can derive local patch/crossing truth directly.

### What must not be copied blindly

TopoMesh obtains direct DMC correspondence by converting arbitrary meshes into a **watertight DMC-compliant** remeshed target. Its remeshing pipeline includes voxelization, flood-fill, dilation/distance processing and volume-enclosing output.

That is incompatible with a core RealSaS invariant established by FIT8:

```text
open/two-sided authored surfaces are first-class truth
```

Bat and George contain visibly significant open support. Closing every authored sheet into a volume would mutate topology and may destroy:

- two-sided wing/panel semantics;
- source appearance side ownership;
- local contact/layer identity;
- mechanical/open-boundary truth.

Therefore:

```text
KEEP:
  topology/geometry decoupling
  direct mesh-level supervision
  teacher forcing
  GT-guided sparse pruning
  progressive resolution
  explicit correspondence principle

TRANSLATE:
  DMC topological unification -> authored local patch/crossing correspondence

REJECT AS UNIVERSAL TRUTH PREPROCESS:
  force every source into one watertight volume
```

TopoMesh can still be an excellent court for **closed-qualified components** or for a separate derived reconstruction target, but it must not overwrite raw/qualified open-surface truth.

---

## 9. PoNQ / QEM family: useful alternative pressure

PoNQ demonstrates a different idea: represent local surface patches by points, normals and quadric error metrics rather than a regular scalar grid. It can handle open boundaries/sharp features and derives a global mesh from local patch geometry.

Its relevance to RealSaS is mainly conceptual pressure on F-C/T:

- local patch geometry need not be represented only as a scalar zero set;
- QEM is naturally aligned with retopology/low-error representative vertices;
- explicit local normals and patch geometry can preserve sharp features.

However PoNQ is not an obvious replacement for the current F substrate because:

- it discards much of the mature triplane/FlexiCubes path;
- its global topology guarantees/volume-boundary assumptions need careful reconciliation with arbitrary open articulated surfaces;
- it does not solve RealSaS dynamic carrier or CAO identity by itself.

Keep it as a representation/T-retopology reference rather than current F default.

---

## 10. Direct teacher supervision becomes a major RealSaS advantage

TopoMesh reports a large gap between rendering-only and direct mesh-level supervision in its controlled ablation. RealSaS has stronger teacher access than typical image-to-3D systems:

```text
source triangles
source normals
source polygon/face addresses
qualified geometric equivalence
local patch connectivity
exact triangle-cell support
exact edge crossing parameters
owner/material/mechanical relations
native animation
```

Therefore F should not throw this away and train only through:

```text
RGB + depth + normal + silhouette
```

The first F court should include direct local structural supervision wherever authority exists.

Candidate teacher channels opened by this audit:

```text
extraction_cell_support(c,h)
local_patch_count_lower_bound(c,h)
local_patch_source_triangle_sets(c,h)
local_patch_neighbor_affinity
edge_crossing_set(e,h)
edge_crossing_source_address
edge_crossing_normal
edge_crossing_orientation_validity
patch_owner/material/mechanical descriptors
```

Exact authority classes remain to be designed before adding these to the canonical schema.

---

## 11. Relation to T

F and T should not collapse into one problem.

F local-patch structure aims to represent the **high-fidelity authored/static surface** without losing close/folded/open sheets.

T still solves:

```text
high-fidelity F
+ shared scene
+ authored dynamic consequence
-> lower-density animation/runtime carrier M*
```

However F-C-style explicit patch/crossing correspondence could give T much cleaner input than an anonymous triangle soup:

- local sheet continuation;
- sharp/boundary features;
- open-boundary identity;
- owner/material boundary bindings;
- contact-critical regions.

This is one reason to preserve source/teacher addressability through F rather than emitting only a final triangle list.

---

## 12. Current decision state

### Resolved by measurement

1. `local sheet == global component` is false.
2. `local sheet == owner` is false.
3. `local sheet == material` is false.
4. resolution-only separation is not a generic solution.
5. standard one-crossing-per-edge scalar encoding is a real representational risk on FIT8 contact/layer zones.
6. sparse high-resolution support remains strongly justified.
7. FlexiCubes/DMC dual-vertex machinery has useful capacity and should not be discarded merely because cells contain multiple patches.

### Open

1. F-B sheet-conditioned scalar fields vs F-C explicit Hermite/multi-crossing representation.
2. exact local patch matching/permutation method.
3. cross-cell patch continuation representation.
4. whether topology/geometry decoupling should use TopoMesh-like occupancy+magnitude variables or a more direct patch-incidence/crossing parameterization.
5. exact sparse support resolution ladder.
6. Compiler stitching/merge court.
7. direct structural-loss weighting versus render/field losses.

---

## 13. Required next matched-budget court

Before freezing F, construct teacher-local patch/crossing targets on FIT8 and compare at minimum:

```text
F-A  sparse standard FlexiCubes scalar baseline
F-B  sparse local-sheet-conditioned FlexiCubes
F-C  sparse explicit patch/crossing Hermite-DMC
```

Use identical shared scene features and comparable parameter/query budgets.

Measure:

```text
static source-surface distance / F-score
open-boundary recall
false bridge count
missing-sheet count
edge-crossing recall
local patch cardinality error
owner/material boundary fidelity
8-view silhouette/depth/normal error
CAO visible-coherence support
runtime/query memory
```

Then, only for candidates that survive static F, measure whether their exact bindings make downstream T dynamic qualification easier.

No final F representation is frozen by this audit.
