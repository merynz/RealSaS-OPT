# RealSaS Structural System Index

> Navigation only. Current continuation authority is `canonical/V2_IMPLEMENTATION_READINESS.json`.

| Product authority | Canonical home | V2 rule |
|---|---|---|
| Observation | observation/camera authority | source evidence; not identical to output direction cardinality |
| Geometry / IRIS | `models/iris/` + geometry adapters | signed field/surface evidence; strict geometry proof; never RGB authority |
| Geometry / GSA | `compiler/realsas_compiler_core/substrate/` | compact surface/relation producer; relation quality independently audited |
| Canonical mesh domain | product mesh + `surface_addressing_v1.py` | common address domain for geometry, mechanics and appearance |
| Appearance / CAA | `appearance_authority_v2.py` + `appearance_compile_v2.py` + `appearance_bake_v2.py` + `appearance_quality_v2.py` | first-class total 2D-art authority; source wins; holdout/seam/exposure/sampling proof |
| Mechanics / Rig | `models/geppetto/` | proposal only; Compiler owns qualified skeleton |
| Mechanics / Skin | `models/arachne/` | proposal only; Compiler owns qualified skin |
| Dynamic mechanics | mesh conditioning/deformation proof | stress-test frozen canonical carrier; explicit repair ownership |
| Visibility | V2 reference/native renderer | posed canonical XYZ + camera depth; appearance does not choose front surface |
| Presentation | V2 presentation structure | slots/attachments/order/visibility/clipping; no appearance/geometry minting |
| Motion | full-3D motion core | operates sealed puppet; compiled-unobserved exposure is measured/budgeted |
| Runtime | `runtime_authority_v2.py` + `runtime_package_v2.py` + native V2 CAA player | deterministic CAA consumer; no donor search, completion, PBR or relighting |
| Visual integrity | V2 native proof | attributes holes/pepper/seams/flicker to geometry, visibility, appearance or sampling |
| Orchestration | `orchestrator/mainline.py` | dependency DAG; ordinal is human display only |
| Closure | Stage46 | Geometry + Mechanics + Appearance + native visual integrity |

## Historical code

V1 modules and artifacts may remain for scientific provenance and reusable generic helpers. They are not automatically current product authority. Current V2 plan/runtime may not depend on obsolete donor/UNSEEN/linear-execution semantics merely for compatibility.

## Witness

No subject witness is active while readiness is closed. Subject-2 Knight is minted fresh only from a valid implementation-closure-bound readiness seal and a run-local ledger.


## Current stage-level red-team

The 46-stage executable contract is green, but implementation readiness is reopened by `canonical/V2_STAGE_BY_STAGE_REDTEAM_20260920.md`.

- **Stage37:** presentation segmentation is role-free but presently limited to disconnected mesh-face islands inside mechanical components; connected mechanically equivalent visual regions are not automatically independently addressable.
- **Stage45/46:** native/reference parity, provenance totality and compiled-unobserved exposure are exhaustive, but dynamic 2D-art deformation quality is not yet an independent quantitative appearance gate; product closure inherits that claim limitation.

Knight execution remains forbidden until these findings are resolved/narrowed and readiness is resealed.
