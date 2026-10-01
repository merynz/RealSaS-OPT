# RealSaS Structural System Index

## 2026-10-01 recovery note

Current continuation details are in `canonical/RECOVERY_CANONICAL_HANDOFF_20261001.md`. The former Stage42 source-owned visual runtime seam is closed: Stage18 provides visual substrate, Stage37 owns final qualified presentation, and typed transport continues through Stage42/RSS/native/Stage45/46. VF-11 R512 remains a separate scientific/witness-readiness blocker.

> Navigation only. Current continuation authority is `canonical/V2_IMPLEMENTATION_READINESS.json`.

| Product authority | Canonical home | V2 rule |
|---|---|---|
| Observation | observation/camera authority | exact source evidence; source cardinality is not output-direction authority |
| Geometry / IRIS | `models/iris/` + geometry adapters | signed field/surface evidence; strict geometry proof; never RGB authority |
| Geometry / GSA | `compiler/realsas_compiler_core/substrate/` | topology-local compact surface/relation producer with exact dense-face provenance |
| Canonical mesh domain | product mesh + `surface_addressing_v1.py` | common address domain for geometry, mechanics and appearance |
| Appearance / CAA | `appearance_authority_v2.py`, compile/bake/quality modules | total source-preserving art; source wins; holdout/seam/sampling/exposure proof |
| Presentation | `visual_presentation_v1.py` + Stage37 product-state adapter | final qualified source-owned presentation; role-free editable grouping |
| Mechanics / Rig | `models/geppetto/` | proposal only; Compiler owns qualified skeleton |
| Mechanics / Skin | `models/arachne/` | proposal only; Compiler owns qualified skin |
| Dynamic mechanics | mesh conditioning/deformation proof | stress-test frozen canonical carrier; repair mints new lineage |
| Motion | `motion_compile_v2.py` + dynamic proof | full-3D rotations and local translations; exact source semantics are retained |
| Visibility | V2 reference/native renderer | posed canonical geometry + camera depth |
| Runtime | `runtime_authority_v2.py` + `runtime_visual_authority_v1.py` + `runtime_package_v2.py` + native player | typed source-owned visual consumer; no mechanical render fallback or generative correction |
| Dynamic Visual Integrity | Stage45/native proof | native parity + provenance + exposure + intrinsic appearance conditioning |
| Orchestration | `orchestrator/mainline.py` | 46-stage dependency DAG; ordinal is display only |
| Closure | Stage46 | Geometry + Mechanics + Appearance + Presentation + native dynamic integrity |

## Historical code

V1 modules, Mage/FIT artifacts and the 2026-09-30 Stage42 fail-close handoff remain provenance. They are not current continuation authority.

## Witness

No subject witness is active merely because engineering recovery is green. Knight requires exact `READY_FOR_WITNESS_EXECUTION`, VF-11 R512 reclosure and explicit user approval.

## Platform transition

After canonical-main recovery, professional platformization may proceed independently of named-witness readiness: Artifact Registry, Attempt, ProductRevision, semantic invalidation and durable workflow become first-class system infrastructure.

## Machine-readable architecture ownership

The exact current system/domain/module/stage ownership map is implemented by `platform/internal/architecture/registry.go` and can be queried with:

```bash
cd platform
go run ./cmd/realsas-architecture
go run ./cmd/realsas-architecture -stage 35_DYNAMIC_MECHANICAL_MESH_QUALIFIED
go run ./cmd/realsas-architecture -module platform.product
```

Every canonical compiler stage must resolve to exactly one owning Engine module. Platform modules separately own durable state and orchestration; native Runtime modules own package/playback/render hot paths. This registry is validated against the canonical 46-stage plan in CI.
