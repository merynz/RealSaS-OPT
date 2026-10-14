# IRIS-vNext topology authority and triangulation audit

**Date:** 2026-10-15  
**Status:** teacher/topology authority research — no final geometry-authority freeze, no model PASS

This audit was triggered when the first Euler/genus court exposed a false T-Rex result caused by naive FBX n-gon fan triangulation.

The central lesson is:

> A topology invariant is only as trustworthy as the vertex-equivalence and polygon-triangulation authority that produced the triangle carrier.

Therefore the topology court is now downstream of two explicit qualification layers:

```text
render/source identity
  -> geometric/mechanical vertex equivalence
  -> topology-qualified polygon triangulation
  -> surface topology invariants
```

Appearance/render identity is preserved separately and is never erased by the geometric quotient.

---

## 1. T-Rex falsifies silent fan triangulation

The raw FBX authored polygon connectivity for T-Rex is closed and edge-manifold in the source census.

Naive fan triangulation nevertheless produced:

```text
6 non-manifold triangle edges
Euler characteristic = 9
```

for the one connected surface.

Forensic inspection found six problematic derived diagonal keys.

### Derived internal diagonal aliases an authored boundary elsewhere

```text
(336, 660)
(355, 356)
(653, 660)
(682, 684)
```

### Derived internal diagonal reused independently by multiple source polygons

```text
(336, 660)
(653, 887)
(669, 760)
```

A single key may appear in both lists.

These derived diagonals collapse independent polygon-local incidence into one ordinary indexed-mesh edge identity and manufacture non-manifold topology.

Thus:

```text
NAIVE_NGON_FAN_IS_NOT_TEACHER_AUTHORITY
```

---

## 2. Topology-qualified triangulation contract

For each source polygon, a candidate triangulation must at minimum prove:

```text
triangle count = n - 2
triangle vertices belong to source polygon
source polygon boundary edge appears exactly once inside that polygon triangulation
polygon-local internal diagonal appears exactly twice inside that polygon triangulation
```

Across source polygons:

```text
derived internal diagonal
    must not alias any authored source boundary edge

derived internal diagonal vertex-pair identity
    must not be independently reused by another source polygon
    unless a separate equivalence proof explicitly authorizes it
```

These checks are executable in:

`tools/corpus/fit8_polygon_triangulation_qualification_v1.py`

They are combinatorial only. Geometry/coverage/normal/depth and source-renderer parity remain separate courts.

---

## 3. FIT8 fan result

The legacy diagnostic fan triangulation gives:

```text
Snake   PASS topology court
T-Rex   FAIL topology court
Spider  PASS topology court
Shark   PASS topology court
```

T-Rex alone is enough to reject silent fan triangulation as a universal corpus policy.

---

## 4. Constrained ear oracle closes FBX topology

A deterministic diagnostic constrained-ear prototype was evaluated with two forbidden-diagonal rules:

```text
1. do not choose an authored boundary edge as a derived internal diagonal;
2. do not reuse an already-consumed polygon-local internal diagonal identity.
```

All four FIT8 FBX subjects produced a topologically valid triangulation:

| Subject | triangle count | boundary edges | non-manifold edges | component topology |
|---|---:|---:|---:|---|
| Snake | 1,618 | 0 | 0 | 3 components, each χ=2, g=0 |
| T-Rex | 1,820 | 0 | 0 | 1 component, χ=2, g=0 |
| Spider | 2,712 | 0 | 0 | 15 components, each χ=2, g=0 |
| Shark | 644 | 0 | 0 | 1 component, χ=2, g=0 |

This proves that the source polygon topology is compatible with a clean triangle carrier.

It does **not** prove that this ear triangulation is the unique authored geometric surface.

---

## 5. Non-planar n-gons prevent topology qualification from becoming geometry authority

FIT8 FBX source polygons are frequently not exactly planar.

Diagnostic maximum distance to the per-polygon PCA plane, normalized by subject bbox diagonal:

```text
Snake   p95 ~0.105%, worst ~0.426%
T-Rex   p95 ~0.106%, worst ~0.278%
Spider  p95 ~0.068%, worst ~0.400%
Shark   p95 ~0.118%, worst ~0.315%
```

Therefore two topology-valid triangulations of one authored n-gon can define different piecewise-linear 3D interiors.

The architecture must distinguish:

```text
TOPOLOGY_QUALIFIED_TRIANGULATION
from
GEOMETRY_AUTHORSHIP_QUALIFIED_TRIANGULATION
```

The second court should include source importer/renderer behavior where available plus normal/depth/material parity and authored-motion mechanical consequence.

The Compiler must preserve triangulation provenance.

---

## 6. glTF render identity is not geometric/mechanical identity

For Bat, Horse, GreenBlob and George, exact same-position render vertices were grouped and compared using their mechanical deformation signature.

No coincident position group contained a mechanical-signature conflict.

| Subject | duplicate same-position groups | mechanical-conflict groups |
|---|---:|---:|
| Bat | 271 | 0 |
| Horse | 1,093 | 0 |
| GreenBlob | 107 | 0 |
| George | 2,744 | 0 |

The same groups were evaluated under native authored motion:

| Subject | sampled motion contexts | equivalence groups with separation >1e-7 | worst separation |
|---|---:|---:|---:|
| Bat | 20 | 0 | 0.0 |
| Horse | 65 | 0 | 0.0 |
| GreenBlob | 45 | 0 | 0.0 |
| George | 100 | 0 | 0.0 |

This supports a mechanically qualified quotient for this cohort.

---

## 7. Render seams remain first-class evidence

Mechanical equivalence does not imply render equivalence.

Every duplicate group in the four glTF subjects carries at least one render-attribute difference in the current audit.

Examples of seam-bearing groups:

```text
Bat       normal and/or UV differences
Horse     normal differences; some material differences
GreenBlob normal and/or UV differences
George    normal and/or UV differences
```

Thus the correct lineage is:

```text
RAW RENDER VERTEX / CORNER IDENTITY
        |
        +--> normal / UV / material / source appearance -> A / CAO
        |
        +--> qualified many-to-one geometric/mechanical equivalence
                           |
                           -> G / F / T / mechanics
```

A quotient relation is created. Render identities are not deleted.

Executable court:

`tools/corpus/fit8_render_vertex_equivalence_v1.py`

---

## 8. Qualified FIT8 topology baseline

Using:

- topology-qualified constrained source-polygon triangulation for the four FBX subjects;
- exact-position + mechanical-signature equivalence, independently motion-witnessed, for the four glTF subjects;

current topology invariants are:

### Closed manifold sphere-topology components

```text
Snake      3 components, all χ=2 / g=0
T-Rex      1 component, χ=2 / g=0
Spider    15 components, all χ=2 / g=0
Shark      1 component, χ=2 / g=0
Horse      1 component, χ=2 / g=0
GreenBlob  2 components, both χ=2 / g=0
```

### Bat

```text
12 components
86 boundary edges
18 non-manifold edges
```

Ten components currently satisfy the preconditions for a genus value and are g=0. Components touching non-manifold incidence abstain from genus authority.

### George

```text
61 components
216 boundary edges
0 non-manifold edges
```

The qualified quotient contains a mixture of closed and open components. Four components have qualified g=1 under the current combinatorial court; the remainder with valid genus are g=0.

This is another reason component count alone is insufficient for F qualification.

---

## 9. Topology invariant court

`tools/corpus/fit8_surface_topology_invariants_v1.py` computes on an already-qualified triangle carrier:

```text
connected triangle components
V / E / F
euler characteristic
boundary edge count
boundary graph components
boundary-loop count only when the boundary graph is loop-qualified
non-manifold edge incidence
orientability only when manifold preconditions permit it
genus only when orientability + boundary-loop preconditions permit it
```

It deliberately performs no hidden welding, triangulation or repair.

---

## 10. Consequence for F-R / F-C court

The next candidate-surface court must compare more than component partition.

At minimum per qualified source component:

```text
component correspondence
Euler characteristic
boundary incidence / boundary-loop structure
non-manifold incidence
orientability where valid
genus where valid
```

But even exact topology invariants are not sufficient for final qualification.

A candidate may preserve χ/genus while still moving a boundary, changing geometry or corrupting appearance ownership. Therefore later courts also require source-address binding, geometric error, normals/depth/silhouette and CAO visible-coherence.

---

## 11. Current open triangulation leaf

The constrained-ear result resolves **existence of a topology-clean carrier**, not final geometric triangulation authority.

Open evidence required before freezing FBX teacher triangulation:

1. compare deterministic constrained triangulation against source application/importer triangulation where recoverable;
2. compare canonical-view depth/normal/silhouette/material output;
3. measure dynamic mechanical consequence across alternative valid triangulations on non-planar polygons;
4. preserve raw source polygon + chosen triangulation + exact binding/provenance;
5. reject any triangulation that changes source topology or causes unacceptable presentation/mechanical drift.

No model training should silently consume naive fan topology as truth.
