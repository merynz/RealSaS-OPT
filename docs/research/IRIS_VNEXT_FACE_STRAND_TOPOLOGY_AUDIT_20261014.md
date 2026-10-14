# IRIS-vNext shared-face strand topology audit

**Date:** 2026-10-14  
**Status:** active architecture research — no final F freeze, no scientific/product PASS  
**Depends on:** local-patch/crossing teacher, F representability court, face-qualified continuation court

This audit asks what must actually cross an extraction-cell boundary when authored geometry is preserved exactly enough for RealSaS.

The previous local-patch continuation court showed that cell-local patch IDs are not stable cross-cell identities. The face-qualified continuation court then showed that source-lineage continuation is structurally complete on FIT8 at N=128 once only patches that physically reach the shared face are evaluated.

The remaining question is finer:

> Is one patch continuation relation per neighboring cell pair enough, or can one local patch intersect the shared face in multiple independent contour strands?

The answer is yes: multiple shared-face strands are common enough to be an explicit representation concern.

---

## 1. Face-qualified continuation result at N=128

Exact source-triangle lineage plus exact patch-to-cell-face support was evaluated for the eight FIT8 subjects.

All shared-face patch endpoints that actually reach the shared face had a source-lineage continuation in the neighboring active cell under the current diagnostic teacher topology.

Branching remains sparse:

| Subject | 1-to-1 continuation edge fraction | branching face-neighbor pair fraction | max patch degree |
|---|---:|---:|---:|
| Snake | 99.790% | 0.106% | 2 |
| Bat | 99.837% | 0.083% | 3 |
| T-Rex | 98.586% | 0.696% | 3 |
| Horse | 99.693% | 0.154% | 2 |
| Spider | 99.039% | 0.492% | 2 |
| GreenBlob | 99.991% | 0.004% | 2 |
| Shark | 98.672% | 0.675% | 2 |
| George | 98.605% | 0.766% | 2 |

This supports a sparse/local continuation design rather than a global persistent patch-ID ontology.

However patch-level branching is not the full topology story.

---

## 2. Shared-face contour strands at N=128

For every neighboring active cell pair, authored source triangles were intersected with the exact shared cell face.

Triangle-plane segments were connected only when:

1. their source triangles are adjacent in qualified source/teacher topology; and
2. their face-intersection endpoints meet within the explicit geometry tolerance.

Unrelated sheets that happen to cross geometrically are not merged.

Measured multi-strand shared-face frequency:

| Subject | multi-strand face fraction | max strands on one face |
|---|---:|---:|
| Snake | 1.387% | 3 |
| Bat | 5.135% | 4 |
| T-Rex | 2.284% | 4 |
| Horse | 1.049% | 3 |
| Spider | 3.324% | 4 |
| GreenBlob | 0.600% | 3 |
| Shark | 2.852% | 3 |
| George | 11.091% | 4 |

Thus:

```text
local patch
!=
one shared-face contour strand
```

A single local patch may expose multiple disconnected contour strands on one neighboring-cell interface.

---

## 3. Strand topology classes

The exact shared-face intersections were classified by the graph of authored triangle-plane segments.

At N=128 the following classes occur:

### Snake

```text
BOUNDARY_TO_BOUNDARY_ARC  45,922
OPEN_INTERIOR_ENDPOINT       118
CLOSED_LOOP                   26
BRANCH_OR_NONMANIFOLD          1
```

### Bat

```text
BOUNDARY_TO_BOUNDARY_ARC  42,387
OPEN_INTERIOR_ENDPOINT       530
BRANCH_OR_NONMANIFOLD       130
CLOSED_LOOP                    6
```

### T-Rex

```text
BOUNDARY_TO_BOUNDARY_ARC  19,603
CLOSED_LOOP                   80
BRANCH_OR_NONMANIFOLD        60
OPEN_INTERIOR_ENDPOINT         8
```

### Horse

```text
BOUNDARY_TO_BOUNDARY_ARC  35,331
OPEN_INTERIOR_ENDPOINT       117
CLOSED_LOOP                   17
```

### Spider

```text
BOUNDARY_TO_BOUNDARY_ARC  28,162
OPEN_INTERIOR_ENDPOINT         4
CLOSED_LOOP                   12
```

### GreenBlob

```text
BOUNDARY_TO_BOUNDARY_ARC 113,888
```

### Shark

```text
BOUNDARY_TO_BOUNDARY_ARC  16,043
OPEN_INTERIOR_ENDPOINT       107
CLOSED_LOOP                   22
```

### George

```text
BOUNDARY_TO_BOUNDARY_ARC  42,660
OPEN_INTERIOR_ENDPOINT       480
CLOSED_LOOP                    2
```

The exact counts are diagnostic, not benchmark claims. The structural result is what matters: ordinary boundary arcs are dominant, but interior loops, open endpoints and branch/non-manifold face topology are real teacher cases.

---

## 4. Why this is stricter than classic Marching-Cubes ambiguity

Classic Marching-Cubes face ambiguity asks how a small number of edge crossings on one quadrilateral face should be connected. The Asymptotic Decider and later topologically consistent MC variants resolve which valid connectivity should be chosen from the scalar field.

RealSaS FIT8 exposes cases outside the comfortable form of that problem:

```text
multiple authored contour strands on one shared face
closed contour entirely inside the face
open-surface contour terminating inside the face
branch/non-manifold authored relation
multiple distinct crossings on one grid edge
```

A closed loop entirely inside one shared face can exist while the four face corners carry no sign change. A standard corner-sampled bilinear face field cannot recover information it never sampled.

Therefore:

> face disambiguation is necessary for a scalar-field core, but it is not sufficient for generic authored surface truth.

---

## 5. Resolution helps but does not eliminate the residual

Selected N=256 results:

### T-Rex

```text
BOUNDARY_TO_BOUNDARY_ARC 78,540
CLOSED_LOOP                 126
BRANCH_OR_NONMANIFOLD       139
OPEN_INTERIOR_ENDPOINT        8
```

### Bat

```text
BOUNDARY_TO_BOUNDARY_ARC 171,122
OPEN_INTERIOR_ENDPOINT      1,054
BRANCH_OR_NONMANIFOLD         254
CLOSED_LOOP                    18
```

### George

```text
BOUNDARY_TO_BOUNDARY_ARC 171,145
OPEN_INTERIOR_ENDPOINT        886
CLOSED_LOOP                     2
```

### Spider

```text
BOUNDARY_TO_BOUNDARY_ARC 113,262
OPEN_INTERIOR_ENDPOINT          4
CLOSED_LOOP                      4
```

For T-Rex, the closed-loop fraction falls as resolution increases:

```text
N128 ~0.405%
N256 ~0.160%
N512 ~0.085%
```

Higher sparse resolution is therefore valuable.

But the residual does not vanish, and finer grids can reveal additional local structure. `increase N until all local topology becomes scalar-trivial` is not accepted as a generic proof strategy.

---

## 6. Updated assembly abstraction

The previous abstraction:

```text
cell local patch token
   -> neighboring cell local patch token
```

is too coarse as the primary interface.

The more accurate local incidence structure is:

```text
CELL
  |
  +-- local patch / sheet
         |
         +-- edge crossing set
         |
         +-- face strand set
                 |
                 +-- neighboring cell face strand
```

Cross-cell continuation authority should therefore be based on shared-face strand/crossing incidence, not equality of patch-local IDs.

Patch tokens remain useful inside a cell, but their continuation is mediated by explicit local boundary evidence.

---

## 7. Strongest current F research candidate

Do not call this merely a hybrid. The proposal has a specific base domain and a specific residual domain.

### Base domain

```text
sparse active cells
+ DMC/FlexiCubes-compatible scalar/topology/geometry variables
+ TopoMesh-style topology/geometry decoupling under direct teacher supervision
```

### Explicit topology residual domain

Only where teacher/predicted evidence proves the base local representation is insufficient:

```text
EdgeCrossingSet
    multiple distinct crossings on one grid edge

FaceStrandSet
    multiple contour strands
    interior closed loop
    open interior endpoint
    branch/non-manifold relation

PatchIncidence
    local sheet -> edge/face topology
```

The residual is sparse and local. It is not a second global mesh representation.

The Compiler remains responsible for qualification, stitching, identity preservation and promotion to the high-fidelity F surface.

---

## 8. Relation to F-B and F-C

The earlier candidates should now be refined.

### F-B

Local-sheet-conditioned scalar/FlexiCubes remains valuable because it closes most multi-sheet edge-crossing loss, especially in layered subjects.

But sheet conditioning cannot represent every same-sheet multi-crossing or interior face loop.

### F-C

A fully explicit Hermite/multi-crossing representation remains the upper-capacity reference.

However current FIT8 evidence suggests paying F-C complexity everywhere may be unnecessary.

### New precise court

Compare:

```text
F-A  standard sparse FlexiCubes
F-B  local-sheet-conditioned FlexiCubes
F-R  sparse DMC/FlexiCubes core + explicit topology residual
F-C  fully explicit patch/crossing representation
```

`F-R` is a research label only, not a frozen architecture name.

The key question is whether F-R reaches F-C authored-topology fidelity while retaining most of the mature DMC/FlexiCubes path and substantially lower variable-cardinality cost.

---

## 9. CAO consequence

Shared-face strand identity should survive through F/T/Compiler bindings where it is presentation-relevant.

This does **not** mean:

```text
F strand == CAO fragment
```

It means CAO can consume exact lineage instead of rediscovering close/layered sheet identity from pixels after the fact.

The emerging bridge is:

```text
F local geometric strand/sheet identity
       -> canonical M surface lineage
       -> CAO fragment split/merge/order qualification
```

This directly supports the RealSaS goal of one mechanical geometry authority with faithful 2D presentation.

---

## 10. Current resolved/open state

### Resolved by evidence

1. patch-local ID is not a cross-cell identity contract;
2. exact source lineage provides complete face-qualified continuation in the current N=128 FIT8 teacher court;
3. local patch may expose multiple contour strands on one face;
4. closed-loop and open-endpoint face topology exist in FIT8;
5. classic scalar face disambiguation alone is not a complete generic truth representation;
6. higher sparse resolution helps but is not a proof that topology residuals disappear;
7. explicit local boundary topology is a better stitching authority than global component/owner/material equality.

### Still open

1. exact `F-R` parameterization;
2. learned predictor for topology-residual activation;
3. false-negative policy for residual activation;
4. neighbor face-strand affinity representation;
5. whether topology residual should predict exact Hermite samples or compact topological templates plus geometry residuals;
6. interaction with TopoMesh-style teacher forcing;
7. compute/memory crossover versus full F-C;
8. dynamic stability of strand identity under authored motion;
9. how much strand lineage T must preserve after retopology;
10. CAO fragment seeding from F/S/all-hit evidence.

---

## 11. Next court

Build a controlled `F-R` oracle representability court:

```text
base F-B capacity
+
activate explicit topology residual only on teacher-proven exception cells/faces/edges
```

Measure against F-C:

- exact authored edge-crossing recall;
- shared-face strand recall;
- closed-loop/open-endpoint recall;
- cross-cell continuity;
- false bridge / missing sheet count;
- residual activation fraction;
- variable-cardinality payload;
- estimated decoder/query cost.

Only if F-R closes the same authored-topology failures as F-C should it advance to learned overfit experiments.
