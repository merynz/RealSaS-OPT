# IRIS-vNext cell-supported patch authority refinement

**Date:** 2026-10-15  
**Status:** teacher/representation refinement evidence; no final F freeze, no scientific/product PASS  
**Supersedes:** the interpretation that `local_patch_lower_bound` is sufficiently exact for F-B capacity measurement.  
**Does not delete:** prior residual-budget evidence; it refines the local-sheet authority used by that court.

---

## 1. Trigger

`ExtractionCellTeacherV1.local_patch_lower_bound` intentionally uses global qualified triangle adjacency restricted to triangles intersecting one extraction cell.

That abstraction is a lower bound on local sheet multiplicity, not exact clipped-local connectivity.

A failure mode is possible:

```text
triangle A intersects cell
triangle B intersects cell
A and B share an authored/qualified edge globally
shared edge lies outside the cell
```

The old lower-bound groups A and B into one local patch even though the two surface pieces are not connected inside the cell.

That can falsely create:

- same-patch multi-crossing;
- `PATCH_GT2_STRANDS`;
- inflated residual-cell counts;
- inflated whole-boundary-graph complexity.

---

## 2. New authority layer

Executable qualifier:

`tools/corpus/fit8_cell_supported_patch_qualification_v1.py`

The qualifier consumes explicit topology-qualified shared-edge witnesses:

```text
TriangleSharedEdgeWitness
  triangle_a
  triangle_b
  shared edge xyz endpoints
  provenance
```

and admits adjacency only when the shared edge has **positive-length support inside the current extraction cell**.

A point touch is preserved as evidence but does not connect two local sheets.

The witness edge must be an actual edge of both source triangles. The witness set must declare itself exhaustive. The qualifier may only **split** a coarse lower-bound patch; it may never merge two lower-bound patches.

Therefore the hierarchy is now:

```text
source / qualified triangle topology
        -> local_patch_lower_bound       # cheap/coarse diagnostic
        -> cell_supported_patch          # local sheet qualification
        -> edge/face residual courts
```

No component/material/owner equality participates in patch connectivity authority.

---

## 3. N=128 structural measurement

A reproduced FIT8 N=128 diagnostic court compared the coarse global-adjacency partition with positive-length cell-supported shared-edge connectivity.

| Subject | active cells in this rerun | cells whose patch partition splits | fraction | extra qualified local patches | max additional patches in one cell |
|---|---:|---:|---:|---:|---:|
| Snake | 22,654 | 219 | 0.967% | 233 | 3 |
| Bat | 20,232 | 102 | 0.504% | 111 | 2 |
| T-Rex | 9,610 | 108 | 1.124% | 112 | 2 |
| Horse | 17,531 | 187 | 1.067% | 211 | 2 |
| Spider | 15,426 | 174 | 1.128% | 193 | 6 |
| GreenBlob | 58,650 | 100 | 0.171% | 101 | 2 |
| Shark | 7,810 | 132 | 1.690% | 152 | 3 |
| George | 18,613 | 289 | 1.553% | 315 | 2 |

### Carrier-parity caution

This rerun reproduces the previous active-cell carrier exactly for Bat, Horse and George, and within a few cells for Snake/T-Rex/Shark. The current local reproduction does **not** reproduce the previous Spider and GreenBlob active-cell totals exactly; those two rows are diagnostic until importer/triangulation carrier parity is re-bound.

The structural conclusion does not depend on those two rows: over-merge is already directly measured on six independently varied subjects.

---

## 4. F-B edge residual was materially overestimated

Only cells whose patch partition changed were re-evaluated for:

```text
F-B = one crossing / qualified local patch / grid edge
```

Delta inside those changed cells:

| Subject | changed cells | old changed-cells beyond F-B | new changed-cells beyond F-B | delta beyond-F-B cells | old unmatched crossings | new unmatched crossings | delta unmatched |
|---|---:|---:|---:|---:|---:|---:|---:|
| Snake | 219 | 148 | 12 | -136 | 245 | 13 | -232 |
| Bat | 102 | 84 | 14 | -70 | 130 | 26 | -104 |
| T-Rex | 108 | 100 | 16 | -84 | 164 | 28 | -136 |
| Horse | 187 | 119 | 12 | -107 | 135 | 14 | -121 |
| Spider* | 174 | 160 | 12 | -148 | 196 | 16 | -180 |
| GreenBlob* | 100 | 92 | 0 | -92 | 102 | 0 | -102 |
| Shark | 132 | 87 | 9 | -78 | 135 | 11 | -124 |
| George | 289 | 246 | 32 | -214 | 475 | 41 | -434 |

`*` carrier-parity caution from section 3 applies.

Interpretation:

> F-B is stronger than the first residual-budget census implied.

A substantial fraction of prior same-patch multi-crossing was not a true representational failure; it was produced by a coarse local-patch lower bound.

But F-B still does **not** become complete. Even among the corrected cells, genuine same-qualified-patch multi-crossing remains across Bat, T-Rex, Horse, Shark and George.

The previous residual-budget table should therefore be rerun end-to-end with `cell_supported_patch` before its absolute percentages are treated as current authority.

---

## 5. Face topology separates cleanly

The same rerun reclassified face residual reasons under the qualified local patch partition.

`PATCH_GT2_STRANDS` drops sharply:

```text
Snake      10 ->  2
Bat        18 ->  4
T-Rex      18 ->  2
Horse      16 ->  8
Spider*    56 -> 34
GreenBlob*  2 ->  1
Shark      20 -> 14
George     54 ->  8
```

In the same reproduced court, the counts for the following geometric face phenomena did not change under patch splitting:

```text
CLOSED_LOOP
OPEN_INTERIOR_ENDPOINT
BRANCH_OR_NONMANIFOLD
```

This is exactly the distinction the architecture needs:

```text
false complexity caused by patch over-merge
!=
true face topology invisible to the base corner/scalar representation
```

Thus the argument for explicit local topology residual becomes **narrower but cleaner**.

---

## 6. Whole-template-only decoding remains weak on Bat

A deliberately conservative Bat census was run after cell-supported patch splitting.

To bias the result **in favor of a fixed template vocabulary**, the signature intentionally removed information:

```text
patch identity relations erased
edge-residual events ignored entirely
only rotation-canonical face-topology composition retained
```

Even under this aggressive collapse:

```text
Bat residual face-composition cells = 891
unique coarse rotation-canonical signatures = 163

top-1  coverage ~25.36%
top-4  coverage ~49.16%
top-8  coverage ~62.96%
top-16 coverage ~74.07%
top-32 coverage ~81.48%
top-64 coverage ~88.66%
top-128 coverage ~96.07%
```

Because this signature deliberately erases topology information, `163` is a **lower bound on exact boundary-graph diversity**, not an upper bound.

Therefore the previous conclusion survives the patch-authority correction:

> compact templates are useful proposal/compression mechanisms, but a fixed whole-template vocabulary is not a sufficient generic authority for morphology such as Bat.

The stronger generic representation remains a compositional residual language with explicit incidence and fail-closed unknown/abstain behavior.

---

## 7. Updated strongest F hypothesis

The refined candidate is simpler than the previous provisional F-R interpretation:

```text
BASE
  sparse active support
  + cell-supported local-sheet-conditioned DMC/FlexiCubes
  + direct teacher topology/geometry supervision

RESIDUAL
  only where the qualified base representation is insufficient

  EDGE
    extra crossings for one qualified local patch / edge

  FACE
    closed interior loop
    open interior endpoint
    branch/non-manifold incidence
    genuinely excessive strand multiplicity after patch qualification

  ASSEMBLY
    explicit strand/crossing incidence across neighboring cells
```

The residual remains attached to the same sparse extraction complex. It is not a second mesh.

---

## 8. Architecture consequence for S/F/T/CAO

`cell_supported_patch` is geometric local-sheet identity. It is still distinct from:

```text
connected component
mechanical owner
material
semantic owner
CAO presentation fragment
```

But it is a better lineage carrier than the previous coarse lower bound.

Preferred lineage:

```text
qualified source triangle carrier
  -> cell-supported patch / strand / crossing
  -> F high-fidelity surface binding
  -> T / Compiler canonical M binding
  -> CAO fragment split/merge/order qualification
```

This preserves the one-canonical-M rule while carrying enough local sheet identity to prevent geometry and appearance from anonymously rematching after extraction.

---

## 9. Required next courts

1. Re-run the complete N=128 F-A/F-B/F-R/F-C residual budget with `cell_supported_patch` as F-B authority.
2. Re-bind Spider and GreenBlob to the exact same qualified carrier/import transform used by the previous census before publishing updated absolute percentages.
3. Re-run exact rotation-canonical boundary-graph LOFO with the qualified patch partition; retain the conservative Bat lower-bound result regardless.
4. Build a teacher-only F-R **physical surface assembler**; metadata/component partition alone is not enough.
5. Compare candidate surface against topology invariants, source addressing, geometry, normal/depth/silhouette and later dynamic mechanics/CAO visible coherence.

No full image-to-scene IRIS training should begin until the extraction representation survives these oracle courts.
