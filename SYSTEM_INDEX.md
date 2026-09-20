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

## Stage-level red-team hardening

The second-pass audit is canonical at `canonical/V2_STAGE_BY_STAGE_REDTEAM_20260920.md`.

- **Stage37:** current automatic presentation grouping is explicitly scoped to connected face islands inside mechanical components. Artist-layer recovery and connected mechanically-equivalent visual-region auto-splitting are not claimed.
- **Stage45:** Dynamic Visual Integrity is explicitly a `DYNAMIC_RUNTIME_INTEGRITY_V2` proof. Native/reference parity, provenance totality and compiled-unobserved exposure do not claim perceptual Spine-class quality.
- **Stage46:** product PASS is explicitly `V2_EXECUTABLE_CONTRACT_CLOSURE`; controlled-witness visual-quality evaluation remains required.

The product target is unchanged: source-faithful Spine-class 2D art remains co-equal with geometry and mechanics. Knight remains forbidden until the exact hardening closure is green, readiness is resealed, and the user approves execution.
