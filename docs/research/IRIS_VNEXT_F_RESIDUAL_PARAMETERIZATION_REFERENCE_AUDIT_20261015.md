# IRIS-vNext F residual-parameterization reference audit

**Date:** 2026-10-15  
**Status:** architecture research — no final F parameterization freeze, no model/product PASS  
**Depends on:** FIT8 local-patch/crossing teacher, face-strand topology audit, residual-budget census, topology-authority audit

This audit asks a narrower question than `which mesher should IRIS use?`:

> Given that most FIT8 support is compatible with a sparse DMC/FlexiCubes-style base but a small measured residual domain contains multiple edge crossings, face-interior loops, open endpoints and branch/non-manifold relations, how should that residual be parameterized without throwing away the mature base representation?

The comparison deliberately separates **reference mechanism** from **truth ontology**. A method may provide a useful mechanism while relying on assumptions that RealSaS cannot adopt globally.

---

## 1. FlexiCubes / Dual Marching Cubes — keep the base, identify the exact boundary

Released FlexiCubes is built on Dual Marching Cubes (DMC), not one-vertex SurfaceNets/DC. Its lookup tables allow more than one dual vertex inside a cube for DMC configurations that require multiple contour components.

Useful substrate:

```text
corner scalar / occupancy configuration
  -> sign-changing grid-edge crossings
  -> DMC edge groups
  -> one or more dual vertices per cube
  -> flexible alpha / beta / gamma / deformation geometry
```

This matches FIT8 evidence well for the overwhelming scalar-compatible domain.

However the standard scalar path still computes one interpolated zero crossing from the two scalar values on one grid edge. FIT8 exact teacher evidence contains multiple distinct authored crossings on the same edge, including same-local-patch crossings.

Therefore the boundary is now explicit:

```text
KEEP:
  sparse support
  DMC multi-dual-vertex grouping
  FlexiCubes geometry/configuration degrees of freedom

DO NOT ASSUME:
  one unconditioned scalar crossing per edge is universal authored truth
```

---

## 2. TopoMesh — topology/geometry decoupling is high value; universal remesh ontology is not

TopoMesh identifies an optimization conflict in FlexiCubes-style decoding: one signed scalar controls both topology through its sign and geometry through its magnitude. It decouples these roles conceptually into topology occupancy and geometry magnitude, directly supervises topology/vertices/normals, uses teacher forcing, GT-guided sparse pruning and progressive resolution.

These mechanisms map strongly to RealSaS F:

```text
TOPOLOGY STATE
  active support
  base DMC configuration
  residual activation / residual topology class
  local incidence

GEOMETRY STATE
  crossing positions
  normals / orientation validity
  dual-vertex / local-vertex positions
  bounded deformation
```

RealSaS should strongly preserve this separation in the first F overfit court.

But TopoMesh obtains direct correspondence by converting input geometry into a watertight DMC-compliant target. FIT8 has visibly significant open/two-sided authored surfaces and non-manifold-qualified cases.

Therefore:

```text
KEEP:
  topology/geometry decoupling
  direct structural supervision
  teacher forcing
  GT-guided sparse pruning
  progressive resolution

REJECT AS UNIVERSAL TRUTH PREPROCESS:
  force authored geometry into one watertight DMC volume
```

---

## 3. Neural Marching Cubes — strongest reference for compact learned local templates

Neural Marching Cubes (NMC) is relevant for a different reason. Instead of restricting a cube to classical MC templates, it learns local topology and geometry under a richer compact per-cube representation using nearby context.

The released implementation documents a significantly richer per-cube output than NDC: approximately

```text
5 boolean topology values
51 floating geometry values
```

and a lookup/table-driven local tessellation family. The paper describes a reduced finite family of local topological cases including an explicit tunnel decision and face/interior vertices.

This demonstrates an important design possibility for F-R:

> explicit residual topology does not automatically require an unconstrained variable graph decoder.

A compact residual-template family plus continuous geometry parameters may be learnable and easier to qualify.

But NMC is still rooted in a signed-distance / occupancy cube model and topology applicable to that implicit-surface family. RealSaS teacher truth contains cases outside that ontology:

```text
open authored sheets
face-interior open endpoints
multiple distinct crossings on one edge
source/owner/material sheet identity
non-manifold-qualified authored relations
closed face loops not evidenced by the four face corners
```

Therefore NMC is a **parameterization reference**, not a universal RealSaS topology table.

---

## 4. Neural Dual Contouring / Unsigned NDC — useful open-surface lesson, insufficient multiplicity

Neural Dual Contouring predicts local dual vertices and connectivity rather than deriving everything from handcrafted QEF/gradient logic.

Its unsigned variant is particularly relevant because it demonstrates that open surfaces can be reconstructed without imposing a global inside/outside sign field. This supports the RealSaS decision to keep open/two-sided truth first-class and to separate support from global sign authority.

However NDC's compactness comes from a strong structural limit:

```text
one vertex per grid cell
one face relation per intersected grid edge family
```

The NMC repository itself notes that this simplification cannot reconstruct some cube cases and can introduce non-manifold edges.

FIT8 already falsifies one-vertex/one-crossing universality in collision/layer zones.

Therefore:

```text
KEEP / TRANSLATE:
  sign-agnostic open-surface support
  learned edge/face existence
  local geometry prediction

DO NOT USE AS FULL F:
  one vertex per cell / one simple local sheet assumption
```

---

## 5. Manifold Dual Contouring / DMC lineage — preserve contour-component identity

Classical Manifold Dual Contouring and DMC variants address the failure of one contour vertex per cell by identifying multiple local contour components and assigning separate vertices/groups.

This aligns directly with FIT8 evidence that:

```text
local sheet != global component
local patch != owner
local patch != material
```

and supports the emerging local incidence representation:

```text
cell
  -> local patch / contour component
      -> edge crossing set
      -> face strand set
      -> neighboring face-strand incidence
```

But these methods still contour a sampled implicit field. They cannot recover authored local topology that the sampling representation never encoded.

Thus they are assembly/topology references, not proof that corner scalar truth is sufficient.

---

## 6. MC33 / topologically-correct MC — consistency with an interpolant is not authorship fidelity

MC33/TMC-style methods are valuable because they distinguish **topological consistency/correctness with respect to a chosen interpolant** from naive lookup-table extraction.

They can resolve face/interior ambiguity of a trilinear scalar interpolant and may add interior vertices for topological correctness.

But RealSaS has a stricter teacher:

```text
AUTHORED SURFACE TRUTH
```

not merely

```text
TOPOLOGY OF THE TRILINEAR INTERPOLANT OF EIGHT CORNER SAMPLES
```

A face-interior authored closed loop with no corner evidence can be absent from that interpolant entirely. Being topologically correct to the interpolant would still be wrong relative to the authored asset.

Therefore MC33/TMC are useful base-consistency courts, not final truth authority.

---

## 7. Residual topology should be factorized by failure class

The residual should not be one undifferentiated `exception token`.

Current FIT8 teacher evidence supports at least the following orthogonal local primitives.

### 7.1 Edge residual

```text
EdgeResidual
  grid edge identity
  local patch / sheet identity
  ordered CrossingSet
      t / xyz
      normal + validity
      source address / provenance
```

Activated when one local patch requires more crossings on one edge than the base representation can express.

### 7.2 Face residual

```text
FaceResidual
  shared face identity
  StrandSet
      BOUNDARY_TO_BOUNDARY_ARC
      CLOSED_LOOP
      OPEN_INTERIOR_ENDPOINT
      BRANCH_OR_NONMANIFOLD
  patch -> strand incidence
  strand geometry / source binding
```

### 7.3 Cell residual

A cell-level topology state is required only when edge/face incidence is insufficient to determine interior assembly, e.g. tunnel/interior connectivity ambiguity.

Candidate form:

```text
CellResidual
  local contour-component count / tokens
  edge/face-strand -> component incidence
  optional compact interior topology template
  confidence / uncertainty / abstention
```

Do not assume one global template vocabulary can cover every authored relation. Template IDs may cover common manifold cases while explicit incidence remains the fallback authority.

---

## 8. Strongest current parameterization hypothesis

The current strongest research hypothesis is not `F-B` or `F-C` alone.

It is a specific staged representation:

```text
SPARSE SUPPORT
      |
      v
TOPOLOGY BASE
  DMC/FlexiCubes-compatible
  topology/geometry decoupled
      |
      +--> ordinary cell/face/edge -> base path
      |
      +--> residual-risk entity
              |
              +--> compact learned residual template where qualified
              |
              +--> explicit variable incidence/crossing fallback
      |
      v
COMPILER QUALIFICATION / ASSEMBLY
```

This combines three useful lessons without importing their incompatible assumptions:

```text
FlexiCubes/DMC:
  mature sparse differentiable base

TopoMesh:
  direct supervision + topology/geometry decoupling + teacher forcing

NMC / Manifold-DC lineage:
  richer local topology and compact local templates / contour-component identity
```

UNDC contributes the sign-agnostic/open-surface lesson.

---

## 9. Why a template-only residual is not frozen

A finite template bank would be attractive for speed and matching stability.

But the current FIT8 cohort already contains:

- closed-manifold smooth surfaces;
- many disconnected components;
- open sheets;
- non-manifold-qualified incidence;
- same-component local folds;
- multiple edge crossings;
- face loops/endpoints/branches.

A template bank trained only on these eight subjects could appear closed while failing future generic articulation.

Therefore the safe design is:

```text
common topology cases -> compact template proposal
unknown/unrepresentable case -> explicit incidence proposal or abstain
Compiler -> qualify
```

No `OTHER` template is allowed to silently approximate topology.

---

## 10. Training consequence

The first learned F experiment should separate three questions.

### Residual activation

```text
Does this edge/face/cell require capacity beyond base DMC/FlexiCubes?
```

Optimize heavily for false-negative avoidance because a missed residual can create false bridges or missing sheets.

### Residual topology

Under teacher forcing:

```text
given exact activation + exact local incidence class
predict topology/template/incidence
```

### Residual geometry

Under exact teacher topology:

```text
predict crossing / strand / local vertex geometry
```

Only after those close separately should topology + geometry + activation be jointly predicted.

This is the direct RealSaS translation of TopoMesh's stabilization lesson, generalized beyond one watertight DMC target.

---

## 11. Relation to T and CAO

F residual topology remains a high-fidelity static surface representation concern.

T still solves the lower-density animation carrier problem and must not simply inherit F's dense cell complex as runtime topology.

However relevant lineage should survive:

```text
F patch / strand / boundary / source address
    -> T retopology constraints / feature importance
    -> canonical M binding
    -> CAO presentation-fragment split/merge/order evidence
```

This is the desired 3D-mechanics / 2D-presentation bridge without introducing a second visible-mesh authority.

---

## 12. Decision state

### Strongly supported

1. sparse DMC/FlexiCubes remains the best base substrate currently examined;
2. topology and geometry prediction/training should be decoupled;
3. direct authored local-structure supervision is a major RealSaS advantage;
4. open-surface support cannot depend on universal signed/watertight truth;
5. residual topology should be sparse and factorized by edge/face/cell failure class;
6. compact learned local templates are worth testing before choosing a fully unconstrained residual graph decoder;
7. explicit incidence/abstention must remain available when a compact template is not qualified.

### Still open

1. exact compact residual template vocabulary;
2. whether common face residuals should use NMC-like explicit face/interior vertices or Hermite strand primitives;
3. cell interior/tunnel representation;
4. residual activation predictor architecture and threshold court;
5. topology-template permutation/matching loss;
6. cross-cell shared-face ownership of residual state;
7. differentiable assembly implementation;
8. matched compute/quality comparison against full F-C.

---

## 13. Next court

Do not train full IRIS yet.

Implement a teacher-driven **residual topology coding census**:

1. canonicalize every residual edge/face/cell event;
2. derive isomorphism-invariant local topology signatures;
3. count how much FIT8 residual support is covered by a small shared template vocabulary;
4. hold out morphology/subject when deriving template vocabulary to detect memorization;
5. measure explicit-fallback rate;
6. only then choose between:
   - mostly-template F-R;
   - template + explicit incidence fallback;
   - full explicit F-C.

The desired representation is the smallest generic local topology language that exactly covers authored truth and fails closed outside its proven domain.
