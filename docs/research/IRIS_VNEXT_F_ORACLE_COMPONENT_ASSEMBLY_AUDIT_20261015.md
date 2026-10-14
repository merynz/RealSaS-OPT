# IRIS-vNext F oracle component-assembly audit

**Date:** 2026-10-15  
**Status:** representation-evidence court — component partition only; NOT full topology/mesh PASS  
**Grid:** N=128 teacher-local patch domain

This audit asks one deliberately narrow question:

> If exact teacher-local patch instances are connected only through qualified source-lineage continuation across neighboring cells, does their global transitive closure recover the exact source connected-component partition?

It does **not** yet ask whether genus, boundary loops, non-manifold incidence, local triangulation, geometry, normals or presentation are reconstructed exactly.

---

## 1. Court

Each active-cell local patch becomes a node:

```text
PatchInstance = (cell_xyz, patch_local_id, source_triangle_ids)
```

Cross-cell edges come from the already-qualified teacher continuation relation.

The resulting patch graph is unioned globally and compared against the exact source/qualified triangle-component partition.

Failures are separated into:

```text
FALSE MERGE
  one representation component contains triangles from >1 source component

SPLIT
  one source component maps to >1 representation component

MISSING SOURCE TRIANGLE
  source triangle appears in no patch node
```

No object/material/owner equality participates in assembly authority.

---

## 2. FIT8 N=128 result

| Subject | patch nodes | continuation edges | source components | assembled components | false merges | split source components | missing triangles |
|---|---:|---:|---:|---:|---:|---:|---:|
| Snake | 22,842 | 46,083 | 3 | 3 | 0 | 0 | 0 |
| Bat | 21,544 | 42,965 | 12 | 12 | 0 | 0 | 0 |
| T-Rex | 9,788 | 19,675 | 1 | 1 | 0 | 0 | 0 |
| Horse | 17,573 | 35,364 | 1 | 1 | 0 | 0 | 0 |
| Spider | 13,988 | 28,026 | 15 | 15 | 0 | 0 | 0 |
| GreenBlob | 56,839 | 113,824 | 2 | 2 | 0 | 0 | 0 |
| Shark | 7,978 | 16,093 | 1 | 1 | 0 | 0 | 0 |
| George | 21,481 | 42,736 | 61 | 61 | 0 | 0 | 0 |

For all eight subjects:

```text
max representation pieces per source component = 1
```

Therefore the N=128 oracle local-patch/continuation substrate closes the **connected-component partition** exactly for the current FIT8 teacher geometry.

---

## 3. What this resolves

The result supports the following representation principle:

> A global persistent patch-ID ontology is not required merely to preserve source connected components.

Cell-local patch identity plus qualified local continuation is sufficient for component-level assembly in this court.

This strengthens the sparse F-R direction because local variable-cardinality state does not automatically imply globally unstable identity.

---

## 4. What this does NOT resolve

Exact connected-component count is a weak topological invariant.

For example, these could all pass this court while still being wrong:

```text
handle removed or invented
boundary loop closed incorrectly
hole punched into a sheet
false local bridge inside one component
wrong non-manifold edge incidence
wrong triangle adjacency cycle
correct component count but wrong Euler characteristic
correct topology but wrong geometry
```

Therefore:

```text
EXACT_COMPONENT_PARTITION
!=
EXACT_TOPOLOGY
!=
EXACT_GEOMETRY
```

No F-R architecture freeze follows from this court alone.

---

## 5. Next topology courts

The next escalation is intentionally ordered from combinatorial to geometric:

### Court A — source triangle-adjacency incidence

Measure whether local patch + continuation evidence preserves every qualified source triangle-adjacency relation and introduces no false adjacency relation.

### Court B — primal surface topology invariants

Per qualified source component compare:

```text
V / E / F incidence
boundary-edge count and boundary-loop structure
non-manifold edge incidence
Euler characteristic
orientability where qualified
genus only where the mathematical preconditions make it valid
```

### Court C — actual oracle F-R surface assembly

Only after the local incidence contract closes, materialize a teacher-only F-R surface and compare:

```text
false bridges
missing sheets
boundary drift
component topology
surface geometry
normal/depth/silhouette
```

The target is to prove that F-R's sparse explicit residual retains the same authored topology fidelity as full F-C, not merely that their patch graphs have the same number of components.
