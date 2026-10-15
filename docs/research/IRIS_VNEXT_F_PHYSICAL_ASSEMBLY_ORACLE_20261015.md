# IRIS-vNext F physical assembly oracle evidence

**Date:** 2026-10-15  
**Status:** architecture/oracle evidence — physical triangle carriers were materialized; no geometry-quality, dynamic, CAO, scientific, or product PASS  
**Scope:** N=128 local diagnostic carrier used by the current FIT8 extraction research branch

---

## 1. Why this court exists

Earlier F courts established local patch/crossing/face-strand/component-partition evidence, but metadata closure is not enough.

A stronger question is:

> Can the qualified local extraction incidence be turned back into an actual global triangle surface without copying the teacher's interior triangulation?

This court materializes candidate triangle carriers and then measures global topology.

The candidate is intentionally crude geometrically. Its purpose is to test **topological/assembly sufficiency**, not final surface quality.

---

## 2. Oracle construction

The oracle uses exact teacher/source evidence to derive local support and boundary incidence, but it does **not** copy the source triangle interiors into the candidate.

For each qualified cell-supported positive-area patch:

```text
source-qualified patch
    -> clipped local boundary
    -> source-address-aware boundary vertices
    -> fresh local triangulation
```

### Disk-like patch

A fresh center vertex is created and a new fan triangulation fills the single boundary loop.

### Annulus-like genus-0 patch

The two qualified boundary loops are connected with a fresh zipper triangulation.

### Zero-area support

`POINT_TOUCH` / zero-area local support does not instantiate a surface patch.

### Non-manifold source incidence

The first oracle deliberately abstains instead of inventing a manifold repair. Such patches require explicit residual/singularity handling.

---

## 3. Source-address identity is mandatory

An early T-Rex prototype welded candidate vertices using XYZ only.

That produced a false local non-manifold result even though the source-aware clipped patch was an ordinary disk.

The oracle was corrected to key weld identity by source address:

```text
SOURCE_VERTEX(vertex_id)
SOURCE_EDGE(edge_id, t)
SOURCE_TRIANGLE_BARY(triangle_id, barycentric_address)
```

Fresh patch-interior vertices remain cell/patch-local.

After this correction the T-Rex failure disappeared.

Therefore:

```text
same XYZ != same surface identity
```

is not merely a semantic/provenance preference. It is required for correct physical assembly.

This is directly relevant to F -> T -> canonical M -> CAO lineage.

---

## 4. Closed smooth control — Horse

Horse has only disk-like positive-area local patches in this court.

Source carrier:

```text
V = 1,093
F = 2,182
components = 1
Euler chi = 2
boundary = 0
non-manifold = 0
```

Fresh oracle candidate:

```text
assembled disk patches = 17,704
zero-area / point-touch patches skipped = 80
V = 51,979
F = 103,954
components = 1
Euler chi = 2
boundary = 0
non-manifold = 0
```

Topology closes despite a completely different tessellation.

---

## 5. Multi-component closed morphology — Snake

Snake includes three closed source components and two annulus-like local patches.

Oracle assembly:

```text
assembled disk patches    = 23,004
assembled annulus patches = 2
zero-area patches skipped = 67
candidate V               = 57,963
candidate F               = 115,914
```

Source topology:

```text
3 connected components
all components: Euler chi = 2, boundary = 0, non-manifold = 0
```

Candidate topology:

```text
3 connected components
all components: Euler chi = 2, boundary = 0, non-manifold = 0
```

Thus annulus-local boundary incidence can participate in a globally correct closed carrier without reusing source interior triangulation.

---

## 6. Annulus-heavy closed morphology — T-Rex

After source-address-aware welding:

```text
assembled disk patches    = 9,808
assembled annulus patches = 38
candidate V               = 29,783
candidate F               = 59,562
```

Source:

```text
components = 1
Euler chi = 2
boundary = 0
non-manifold = 0
```

Candidate:

```text
components = 1
Euler chi = 2
boundary = 0
non-manifold = 0
```

The earlier XYZ-only false failure is deliberately not counted as source evidence.

---

## 7. Open / layered / highly multipart morphology — George

George is the strongest open/multipart assembly test in this oracle.

```text
source components    = 61
candidate components = 61
assembled disk patches    = 21,741
assembled annulus patches = 2
zero-area patches skipped = 53
candidate V = 77,960
candidate F = 155,130
```

Candidate faces were assigned exact source geometric-component lineage from the teacher patch that generated them.

Under that exact correspondence there were **zero mismatches** for:

```text
Euler characteristic
boundary graph component count
boundary loop count
non-manifold state
genus
candidate connectedness inside the source component
```

Raw boundary segment counts changed:

```text
source boundary edges    = 216
candidate boundary edges = 622
```

This is legal retessellation/subdivision, not a topology change.

The observation falsifies the earlier `SurfaceTopologyComparisonCourt.v1` assumption that raw boundary-edge count is a retessellation invariant.

`SurfaceTopologyComparisonCourt.v2` therefore removes raw boundary/non-manifold segment counts from topology authority and requires separate incidence courts when singular topology is not qualified.

---

## 8. Expected fail — Bat non-manifold source incidence

Bat deliberately exposes the residual class that the first physical oracle does not implement.

Measured result:

```text
source components       = 12
candidate components    = 16
assembled disk patches  = 21,504
skipped non-manifold local patches = 128
zero-area patches skipped = 23
```

Exactly two source components carry authored/qualified non-manifold incidence in this carrier.

For each of those two components:

```text
source connected component -> three candidate pieces
```

The other source components do not exhibit this split.

Therefore the failure is not evidence against local incidence assembly in general. It localizes a missing capability:

```text
AUTHORED NON-MANIFOLD / SINGULAR INCIDENCE RESIDUAL
```

The oracle must not silently repair or manifoldize it.

---

## 9. What this changes in the F hypothesis

The strongest current interpretation is now narrower than the earlier provisional F-R design.

### Base physical assembly is stronger than expected

For qualified manifold genus-0 local patches:

```text
cell-supported patch identity
+ exact boundary/source-address incidence
+ disk / annulus local retriangulation
+ cross-cell source-address welding
```

is sufficient to reconstruct correct global topology on Horse, Snake, T-Rex and George in this oracle court.

### Residual capacity remains necessary but more localized

Current measured residual classes include:

```text
extra same-qualified-patch edge crossing
face-local closed loop / open endpoint / branch state beyond base scalar capacity
authored open-boundary incidence
authored non-manifold / singular incidence
future boundary-unexplained interior topology -> abstain / explicit residual
```

The residual is still not a second mesh. It is additional local topology state on the same sparse extraction complex.

---

## 10. Topology court correction

Legal retessellation can change:

```text
vertex count
edge count
face count
raw boundary-edge segment count
raw non-manifold-edge segment count
```

Therefore these are not exact topology invariants.

The topology court should instead preserve/qualify:

```text
connected component correspondence
Euler characteristic
boundary presence
qualified boundary component / loop topology
edge-manifold versus non-manifold state
orientability
qualified genus
```

If the source is non-manifold, exact singular-set equivalence requires a separate source-lineage incidence court; raw singular-edge count is not enough.

This is implemented separately in `fit8_surface_topology_comparison_court_v2.py`.

---

## 11. What is still unproven

This oracle deliberately does **not** prove:

- geometric fidelity of the fan/zipper surface;
- normal fidelity;
- silhouette/depth fidelity;
- self-intersection quality;
- differentiability/optimization stability;
- learned residual activation recall;
- motion/Jacobian/stretch behavior;
- T retopology quality;
- CAO appearance coverage or draw order;
- runtime triangle budget.

The candidate tessellation is intentionally high-density and diagnostic.

---

## 12. Next court

The next physical F court should add geometry authority without losing the topology result:

```text
physical oracle carrier
    -> source-addressed geometry samples
    -> point-to-surface / normal / depth / silhouette error
    -> false bridge / missing sheet / self-intersection
```

Then compare local reconstruction choices:

```text
fresh fan/zipper baseline
DMC/FlexiCubes base
F-R sparse residual
F-C explicit upper-capacity reference
```

Only after topology + geometry close should F be trained behind the expensive shared IRIS image/scene trunk.
