# IRIS-vNext F residual-budget census

**Date:** 2026-10-14  
**Status:** architecture-decision evidence — no final F freeze, no scientific/product PASS  
**Depends on:** FIT8 local-patch/crossing oracle, shared-face strand audit, topology-residual court

This document measures how much of the sparse extraction domain actually requires capacity beyond a local-sheet-conditioned DMC/FlexiCubes core.

The purpose is not to collapse all failure classes into one percentage. Three domains remain separate:

```text
EDGE MODEL-CAPACITY RESIDUAL
FACE TOPOLOGY RESIDUAL
TEACHER CANONICALIZATION / DEGENERACY
```

Only the first two justify additional learned/extraction capacity.

---

## 1. N=128 edge residual budget

The edge court begins from F-B:

```text
one crossing per local patch / grid edge
```

A cell requires edge residual capacity when authored truth contains more than one canonicalized crossing for the same local patch on the same grid edge.

| Subject | active cells | cells beyond F-B | cell fraction | residual patch-edge slot fraction | extra crossing fraction |
|---|---:|---:|---:|---:|---:|
| Snake | 22,648 | 476 | 2.102% | 0.709% | 0.711% |
| Bat | 20,232 | 378 | 1.868% | 0.507% | 0.523% |
| T-Rex | 9,616 | 428 | 4.451% | 1.442% | 1.487% |
| Horse | 17,531 | 496 | 2.829% | 0.751% | 0.757% |
| Spider | 13,530 | 686 | 5.070% | 1.408% | 1.402% |
| GreenBlob | 56,486 | 386 | 0.683% | 0.179% | 0.180% |
| Shark | 7,808 | 276 | 3.535% | 1.219% | 1.229% |
| George | 18,613 | 971 | 5.217% | 1.769% | 1.749% |

The distinction between cell fraction and slot fraction matters.

A few percent of cells may contain at least one difficult edge, while the actual local crossing payload requiring explicit extra capacity is usually around or below ~1-2% of patch-edge slots.

Thus a variable-cardinality edge representation applied to **every** sparse cell would pay substantially more complexity than the teacher truth currently demands.

---

## 2. N=128 face topology residual budget

The face court assumes the F-B scalar core may represent ordinary boundary-to-boundary face contours.

Explicit topology residual is activated for teacher cases such as:

```text
CLOSED_LOOP
OPEN_INTERIOR_ENDPOINT
BRANCH_OR_NONMANIFOLD
STRAND_MULTI_PATCH
PATCH_GT2_STRANDS
```

Point/coplanar degeneracy is intentionally excluded from the learned topology-capacity count and tracked as canonicalization work.

Measured topology-residual face fractions:

| Subject | topology-residual shared-face fraction |
|---|---:|
| Snake | 0.196% |
| Bat | 1.599% |
| T-Rex | 0.716% |
| Horse | 0.225% |
| Spider | 0.095% |
| GreenBlob | 0.001% |
| Shark | 0.477% |
| George | 1.207% |

These values are small but not zero.

They are also morphology-dependent rather than an open-surface-only corner case:

- Bat/George stress open/layered/part-heavy topology;
- T-Rex/Spider/Shark show residual topology on closed articulated surfaces;
- GreenBlob is almost entirely scalar-trivial at this pitch.

---

## 3. Canonicalization is a separate budget

If point/coplanar face degeneracies are included, the apparent residual fraction increases for some subjects:

```text
Snake      1.677%
Bat        2.218%
T-Rex      0.716%
Horse      1.451%
Spider     0.095%
GreenBlob  0.001%
Shark      2.884%
George     2.109%
```

This does **not** mean those extra cases require a larger neural topology head.

The crossing-gap audit already showed that numerical duplicate intersections can depend on triangulation/import precision and must be canonicalized using an explicit spatial/provenance policy.

Therefore:

```text
TEACHER NUMERICAL / COPLANAR DEGENERACY
!=
MODEL REPRESENTATION FAILURE
```

This separation is now executable in `fit8_f_topology_residual_court_v1.py` and `fit8_f_residual_budget_court_v1.py`.

---

## 4. Cross-cell continuation cost is also sparse

At N=128, after qualifying only local patches that physically reach the shared cell face:

- every evaluated shared-face patch endpoint had a source-lineage continuation in the neighboring active cell;
- continuation edges are overwhelmingly one-to-one;
- branching face-neighbor pairs remain below 0.8% for all eight subjects;
- maximum observed local patch continuation degree is 2-3.

This argues against a global persistent patch-ID ontology.

The correct representation is local incidence:

```text
local patch
  -> edge crossing set
  -> face strand set
  -> neighbor face strand incidence
```

with variable topology only where authored evidence requires it.

---

## 5. The resulting F-R hypothesis is now precise

`F-R` remains a research label, not a product name.

### Base representation

```text
sparse active support
+ local-sheet-conditioned scalar / DMC-FlexiCubes core
+ direct topology/geometry teacher supervision
+ progressive sparse resolution
```

### Residual representation

Activated only on teacher/predicted exception entities:

```text
EDGE RESIDUAL
  extra crossing parameter(s)
  crossing normal/orientation
  local patch/source binding

FACE RESIDUAL
  explicit contour strand(s)
  interior loop/open endpoint support
  branch/non-manifold incidence
  patch -> strand incidence

CONTINUATION
  strand -> neighboring face strand relation
```

The residual is not a second mesh.

It is explicit local topology state attached to the same sparse extraction complex.

---

## 6. Why this is more attractive than making all of F explicit

A fully explicit F-C patch/crossing decoder remains the upper-capacity reference and must stay in the matched court.

But current FIT8 teacher measurements imply:

```text
ordinary DMC/FlexiCubes-compatible domain = overwhelming majority
explicit residual domain                 = small minority
```

Therefore the central question becomes:

> Can F-R match F-C authored-topology fidelity while paying variable-cardinality complexity only on the measured residual support?

If yes, F-R is structurally simpler and retains more of the mature InstantMesh -> StdGEN -> FlexiCubes lineage.

If no, full F-C remains justified.

---

## 7. Required false-negative policy

Sparse residual activation creates a new failure mode:

```text
teacher needs residual
but predictor emits ordinary base cell/face
```

That failure can directly create:

- missing sheet;
- false bridge;
- closed-loop loss;
- open-boundary loss;
- wrong local ownership;
- later CAO hole/order failure.

Therefore residual activation must be trained/evaluated fail-closed.

A first learned experiment should emphasize **residual recall**, not merely class-balanced accuracy.

Compiler policy should be able to reject/abstain on high topology uncertainty rather than silently falling back to a single scalar topology.

---

## 8. Next oracle court

Before training the full image-to-scene IRIS trunk, compare from exact teacher spatial evidence:

```text
F-A  standard sparse FlexiCubes
F-B  sheet-conditioned FlexiCubes
F-R  F-B + sparse explicit topology residual
F-C  full explicit patch/crossing representation
```

Required metrics:

- authored edge-crossing recall;
- shared-face strand recall;
- closed-loop/open-endpoint recall;
- cross-cell continuity;
- false bridge count;
- missing sheet count;
- topology/component fidelity;
- residual activation recall/precision;
- variable-cardinality payload;
- memory/query estimate;
- differentiable optimization stability.

Only then should F-R or F-C be wired behind the expensive shared IRIS scene model.
