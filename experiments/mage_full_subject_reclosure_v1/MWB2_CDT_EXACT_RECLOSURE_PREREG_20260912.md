# Mage MWB2 CDT exact reclosure preregistration — 2026-09-12

**Status:** `PREREGISTERED__EXACT_KERNEL_REPLAY_NOT_YET_EXECUTED`

## Question

Can the current observation-grounded MWB2 contract close on the corrected Mage H1 zero surface when two already-localized discretization defects are removed without changing the frozen behavioral threshold?

The forensic precursor is `MWB2_CDT_CARRIER_RESOLUTION_FORENSIC_AUDIT_20260912.json`. That audit is causal evidence only and is explicitly not promotion evidence because its triangulation arm used a SciPy Delaunay equivalence mirror rather than the hash-verified historical kernel.

## Frozen authority

- H1 source run: `20260912T074348Z`.
- Product-clipped zero surface SHA-256: `56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925`.
- Input observations and cameras must match `INPUT_AUTHORITY_PREFLIGHT.json` exactly.
- Exact alpha authority is the same binary product mask used by the corrected H1/GSA replay (`RGBA alpha >= 8`).
- Teacher/source mesh topology remains forbidden from MWB2 product inference.
- Historical CDT source SHA-256 remains frozen at `dd21ae3fc570b2eb5d3c439a5c31fd25fed2ad0743d34a71119d5b391806854d` and is numerical machinery only.

## Prospective repair arm

Two changes are frozen together because the forensic audit showed that neither alone closes all eight views:

1. **Component semantics:** compute safe connected-component identity on the complete `RiggingSurfaceIR` relation graph first; only then take the visible/observation-supported carriers for the target view. A carrier that is temporarily invisible must not split one legal S component into unrelated view-local components. Unsafe/unknown relations remain forbidden and can never connect components.
2. **Carrier budget:** replay GSA from the same dense H1 zero surface with `target_nodes=8192`. This is a deterministic compaction-resolution arm, not IRIS retraining and not a threshold change. The existing 1024 result is preserved as the baseline lineage.

No Mage component names, source object labels, teacher mesh, skin, skeleton or downstream motion information may enter either change.

## Numerical contract

- GSA implementation remains `RealSaS.GSA.ZeroSurfaceAdaptiveVoxel.v1` plus the existing self-zbuffer and exact observation-support operators.
- `target_nodes=8192` is the only requested carrier-budget change.
- MWB2 must execute the hash-verified historical v0.5 CDT kernel through the current typed adapter.
- Quality-Steiner refinement remains disabled.
- Any kernel vertex that cannot be rebound under the declared current support-binding policy must fail closed.
- Every admitted triangle must satisfy the exact `ObservationRasterDomain.triangle_inside` fence.
- Deterministic replay is required.

## Frozen gates

For each of eight views report at minimum:

- visible/observation-supported carrier count;
- safe full-S component count with visible support;
- kernel triangle count;
- alpha-rejected triangle count;
- admitted face and used-vertex counts;
- source-alpha recall;
- precision inside source alpha.

Hard gate:

- min-view source-alpha recall **>= 0.90**;
- every-view precision inside source alpha **>= 0.995**;
- no face spans two distinct full safe-S components;
- no unsafe/unknown relation is used to establish component identity;
- exact historical CDT source hash matches;
- no teacher/source mesh authority enters MWB2;
- identical-input replay is deterministic.

The threshold is unchanged from the already-written current MWB2 CDT qualifier. Failure does not authorize widening it.

## Decision policy

- PASS closes only the corrected Mage MWB2/CDT behavioral gate for this lineage.
- PASS does **not** by itself promote H1, GSA, ARAP, BBW, XPBD, PRODUCT_PASS or unseen generalization.
- FAIL returns to surface-carrier/binding semantics. It does not authorize blind IRIS retraining.
- ARAP/BBW/XPBD remain locked until this exact replay and the H1 authority reconciliation are both closed.
