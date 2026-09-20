# RealSaS V2 — Adversarial Module Audit Protocol — 2026-09-20

Status: ACTIVE_ENGINEERING_DISCIPLINE
Scope: current V2 46-stage mainline
Rule: no module is trusted because it passed historically or because a downstream gate exists.

## Required audit for every authority boundary

For each module/stage family, record all of the following before product reliance:

1. **Input authority** — exact typed inputs and hashes actually consumed.
2. **Producer capability** — what the implementation can mathematically/algorithmically emit.
3. **Forbidden authority leakage** — teacher/source mesh, subject IDs, hidden camera/view, runtime generation, or other undeclared truth.
4. **Output semantics** — what the output proves and explicitly does not prove.
5. **Gate feasibility** — whether the frozen downstream thresholds are reachable by this producer under its own constraints.
6. **Adversarial synthetic tests** — smallest counterexample that should fail.
7. **Silent failure modes** — cases where aggregate metrics can hide a coherent/local defect.
8. **Repair ownership** — exact upstream stage that is allowed to change if the module fails.
9. **Knight measurement** — exact witness telemetry only after the above is frozen.
10. **Closure** — PASS/FAIL/INCONCLUSIVE plus hashes; no threshold change after Knight inspection.

## V2 audit order

A. source bytes / license / subject audit
B. source cameras and observation raster contract
C. normalization
D. IRIS fit/decode and Stage13 geometry objective
E. GSA dense-to-compact surface construction
F. GSA local-relation construction
G. mechanical partition and UNKNOWN policy
H. canonical relation baseline producer
I. CDT local-chart producer and stitch limitations
J. Stage18 mesh + SurfaceAddressingIR + AppearanceDomainIR birth transaction
K. Stage19 static carrier qualification
L. CAA source-lock / deterministic completion / bake / qualification
M. Geppetto proposal -> qualified skeleton
N. Arachne proposal -> qualified skin
O. deformation envelope and Stage35 dynamic mechanical mesh qualification
P. presentation structure
Q. complete puppet seal
R. motion source/compile/dynamic proof
S. runtime CAA binding and package load verification
T. native Dynamic Visual Integrity
U. Stage46 product closure

## First discovered producer/gate mismatch: relation-parent quality

Current product mesh does **not** directly remesh the dense marching-cubes surface.

Scene-first GSA compacts the dense decoded zero surface and derives a local edge graph from mapped dense faces. The canonical relation baseline then emits triangles as 3-cliques in that local relation graph. Therefore baseline product-parent faces are not literally raw marching-cubes parent triangles.

The current CDT V1 adapter uses those baseline 3-clique triangles as sealed local chart boundaries and forbids boundary splits. This creates a hard repairability fact:

> If any baseline parent corner angle is below the frozen G3 target, interior Steiner insertion without boundary splitting cannot make every child triangle meet that target, because child angles incident to the preserved parent corner sum to the parent angle.

Therefore Stage18 must measure relation-parent quality before CDT execution. A parent below the frozen minimum-angle threshold is a producer/gate incompatibility, not a late Stage35 surprise. The current adapter emits RealSaS.RelationParentQualityReport.v1 and fails closed with CDT_PARENT_MIN_ANGLE_UNREPAIRABLE_WITHOUT_BOUNDARY_SPLIT.

This report distinguishes:
- relation-graph 3-clique quality,
- CDT child quality,
- dense decoded zero-surface quality.

They may not be substituted for one another.

## Repair policy for the relation-parent mismatch

If the Knight relation-parent report is bad, thresholds remain frozen. Candidate repair families are evaluated separately:

1. improve GSA compaction/relation construction so the 3-clique complex satisfies the frozen conditioning floor; or
2. introduce synchronized boundary splits with an explicit stitch proof and a new adapter contract.

A generic remesher is not automatically admissible. Any new vertex must preserve explicit SurfaceSupportBinding, component/boundary semantics, source provenance, and the frozen mesh/appearance-domain invalidation rules.

No option is selected from Knight outcome alone.
