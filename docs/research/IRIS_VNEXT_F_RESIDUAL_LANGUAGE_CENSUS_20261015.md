# IRIS-vNext F residual-language census

**Date:** 2026-10-15  
**Status:** architecture-decision evidence — no final residual vocabulary freeze, no model/product PASS  
**Grid:** N=128 current qualified FIT8 teacher/prototype courts

This audit asks whether the explicit topology-residual domain should be represented primarily as a finite whole-cell/whole-face template vocabulary or as a small compositional primitive language with variable incidence.

The motivating references point in both directions:

- Neural Marching Cubes shows that a compact learned local topology template family can work well;
- FIT8 authored truth contains open sheets, multiple edge crossings, face-interior loops/endpoints and branch/non-manifold relations that should not be silently forced into one closed implicit template ontology.

The first genericity check therefore measures **primitive closure under leave-one-subject-out (LOFO)** rather than memorizing complete event configurations.

---

## 1. Edge residual primitive census

The current F-B edge base allows:

```text
one crossing per local patch / grid edge
```

Residual patch-edge multiplicity at N=128 across FIT8 is:

```text
2 crossings: 5,148 events  (~98.36% of edge-residual patch-edge events)
3 crossings:    86 events  (~1.64%)
```

Every one of the eight subjects contains at least one `2`-crossing and one `3`-crossing residual event in the current court.

Therefore, for every 7->1 LOFO split:

```text
held-out observed edge-multiplicity primitive-type coverage = 100%
```

This does **not** authorize a product cap of three crossings. The teacher language should remain variable-cardinality.

It does show that FIT8 does not currently require eight morphology-specific edge residual classes.

The natural generic primitive is:

```text
EdgeCrossingSet(local_patch, grid_edge)
    cardinality = variable
    ordered crossing parameters
    normals / orientation validity
    source / semantic bindings
```

not:

```text
EDGE_CASE_2
EDGE_CASE_3
```

as frozen ontology.

---

## 2. Face topology primitive census

After excluding numerical/coplanar teacher canonicalization from learned topology capacity, the current N=128 face-residual reason counts are:

| Primitive / predicate | Count across subject face courts | Share of reason occurrences |
|---|---:|---:|
| `OPEN_INTERIOR_ENDPOINT` | 1,137 | ~72.47% |
| `BRANCH_OR_NONMANIFOLD` | 190 | ~12.11% |
| `CLOSED_LOOP` | 156 | ~9.94% |
| `PATCH_GT2_STRANDS` | 86 | ~5.48% |

Counts are reason occurrences, not disjoint face counts; one face may activate more than one reason.

Observed subject support:

```text
OPEN_INTERIOR_ENDPOINT
  Bat, George, Horse, Shark, Snake, Spider, T-Rex

CLOSED_LOOP
  Bat, George, Horse, Shark, Snake, Spider, T-Rex

BRANCH_OR_NONMANIFOLD
  Bat, Snake, T-Rex

PATCH_GT2_STRANDS
  all eight subjects, including GreenBlob
```

Thus for every 7->1 LOFO split, every topology-residual primitive type present in the held-out subject is also present in at least one of the seven remaining subjects:

```text
held-out observed face primitive-type coverage = 100%
```

Again, this is **primitive closure**, not proof that complete face/cell configurations are shared.

---

## 3. Why primitive closure matters

FIT8 was deliberately selected to span strongly different morphology:

```text
serpentine
flying
large dinosaur
quadruped
arthropod
amorphous
shark/aquatic
mechanical multi-part
```

Yet the local residual topology events reduce to a small set of recurring structural atoms.

This is evidence in favor of a generic **compositional topology language** rather than anatomy-specific classes.

The language can describe a new subject by composing familiar local primitives even when the exact whole-cell configuration has never appeared before.

That is better aligned with RealSaS generic articulation than learning names such as:

```text
wing case
cape case
robot panel case
spider-leg case
```

---

## 4. NMC lesson translated more carefully

Neural Marching Cubes demonstrates that local topology can be represented compactly with a finite learned case/template family plus continuous geometry parameters.

The valuable lesson is:

```text
local topology can be discrete / structured
while
local geometry remains continuous
```

The part that should **not** be copied literally is the assumption that one finite whole-cube case table derived from an implicit signed/occupancy surface is the natural generic RealSaS truth ontology.

Current FIT8 evidence instead suggests:

```text
COMPOSITIONAL PRIMITIVES
    edge crossing sets
    face contour strands
    open endpoints
    closed loops
    branch incidence
    patch <-> strand incidence

PLUS
    optional compact templates for common compositions

PLUS
    explicit incidence fallback / abstention
```

Templates become a compression/decoder convenience, not truth authority.

---

## 5. Proposed hierarchy of authority

The emerging F residual language should distinguish four levels.

### Level 0 — base scalar/DMC domain

No explicit residual required.

### Level 1 — primitive residual observations

```text
EdgeCrossingSet
FaceStrand
PatchFaceIncidence
```

These are representation-neutral teacher facts.

### Level 2 — compositional topology graph

```text
local patch nodes
edge-crossing nodes
face-strand nodes
incidence / continuation edges
```

This is the generic topology-residual authority domain.

### Level 3 — compact template code

A frequent subgraph may be assigned a learned/canonical template ID for efficient prediction and decoding.

But:

```text
TEMPLATE ID
!=
TRUTH
```

The Compiler qualifies the decoded incidence against Level 1/2 evidence.

Unknown topology must remain representable by explicit incidence or fail/abstain.

---

## 6. Why this is safer than a whole-template-only decoder

A whole-template vocabulary creates two genericity risks:

1. combinatorial explosion from composing otherwise familiar primitives;
2. an `OTHER` class that silently approximates unseen topology.

A primitive graph avoids both:

```text
new composition of known primitives
    -> still expressible

truly unseen primitive / unsupported incidence
    -> uncertainty / explicit fallback / Compiler rejection
```

This matches the project's fail-closed authority philosophy.

---

## 7. Geometry remains separate

Primitive topology identity should not include triangle count, source polygon tessellation or exact segment sampling density.

Those would fragment one topology class into many signatures merely because of different triangulation.

For example:

```text
CLOSED_LOOP
```

is topology.

Its:

```text
control points
Hermite samples
crossing positions
normals
curvature/detail
```

are geometry parameters.

This is another direct application of the TopoMesh topology/geometry decoupling lesson, but on the richer RealSaS local topology domain.

---

## 8. Current strongest F-R residual hypothesis

```text
F-R

sparse support
  -> topology/geometry-decoupled DMC/FlexiCubes base
  -> residual activation
       |
       +-- predict compositional primitive graph
       |      edge crossing sets
       |      face strands
       |      patch/strand incidence
       |      neighbor continuation
       |
       +-- optionally classify common graph to compact template ID
       |
       +-- explicit graph fallback / abstention for non-template case
  -> Compiler qualification / assembly
```

This is more generic than a fixed NMC-like whole-cube table while retaining the efficiency lesson of compact local templates.

---

## 9. Important limitations of this census

The result does **not** prove that the full topology-graph vocabulary is closed.

Primitive type closure can be 100% while held-out subjects still contain unseen:

- combinations of primitive types;
- graph degree patterns;
- cell-interior tunnel incidence;
- cross-face continuation patterns;
- higher crossing cardinalities at other resolutions or future assets;
- non-orientable or more pathological authored topology.

Therefore no fixed vocabulary is frozen from FIT8.

The correct conclusion is only:

> a compositional primitive language is currently better supported than a morphology-specific or whole-template-only ontology.

---

## 10. Next court

The next census should move one level above primitive types without jumping directly to raw geometry:

1. derive an **incidence-aware local topology graph signature** per residual cell;
2. canonicalize it under cube rotation/reflection where semantically safe;
3. keep open/closed/orientation/source-side state explicit rather than erasing it;
4. measure global top-K graph-template coverage;
5. measure 7->1 LOFO graph-template coverage;
6. compare:
   - whole-graph template prediction;
   - primitive-graph autoregressive/set prediction;
   - template proposal + explicit graph fallback.

Only this court can decide whether common residual compositions deserve a finite template decoder or whether explicit compositional incidence should be the primary learned representation.
