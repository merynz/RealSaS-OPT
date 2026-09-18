# RealSaS Structural System Index

> Navigation only. `CURRENT_STATE.md` and `canonical/ACTIVE_RUN_V1.json` are current continuation/progress authority.

## Current mainline stack

| Layer | Canonical home | Rule |
|---|---|---|
| Observation contract | `compiler/realsas_compiler_core/observation_contract_v1.py` | exact 8-view, border-safe, OOF=UNKNOWN |
| IRIS | `models/iris/` | learned geometry evidence; fresh checkpoint per admitted FIT witness |
| GSA | `compiler/realsas_compiler_core/substrate/` | Compiler-owned `RiggingSurfaceIR` |
| Geppetto | `models/geppetto/` | proposal only; Compiler owns qualified skeleton |
| Arachne | `models/arachne/` | proposal only; Compiler owns qualified skin |
| Structural partition | `compiler/realsas_compiler_core/product_authority_v1.py` | immutable S + SEPARATE/PRESERVE_CONTINUITY/UNKNOWN constraints |
| Product mesh | `compiler/realsas_compiler_core/product_authority_v1.py` | view-independent canonical `QualifiedMeshIR`; single product geometry authority |
| Presentation | `compiler/realsas_compiler_core/product_authority_v1.py` | qualified slots/attachments/carriers/evidence/view overlays |
| Motion | `compiler/realsas_compiler_core/motion_3d_*.py` | canonical 3D deformation |
| Runtime | `compiler/realsas_compiler_services/export/runtime_v4.py`, `runtime_v4_cache.py` | compact shared-XYZ Runtime-v4 |
| Native render/playback | `runtime/realsas_cpp/` | subordinate exact consumer |
| Orchestration | `compiler/realsas_compiler_services/orchestrator/` | 40-stage resumable hash-bound execution |

## Optimization spine
- shared canonical XYZ once per asset per frame; no per-view posed-XYZ duplication;
- streamed Runtime-v4 binary/archive writer;
- content-addressed runtime package cache;
- exact typed product-proof + motion-bake cache;
- content-addressed native reference-render cache;
- batch native miss transport for visual proof.

Historical synthetic measurement evidence is frozen in `canonical/MAINLINE_PERFORMANCE_BASELINE_V1.json`. It is a performance direction, not PRODUCT_PASS.

## Current run
`SUBJECT2_KNIGHT_V1` is the active witness, but Knight-specific data may exist only in run/source artifacts. Generic source must remain subject-agnostic. Progress is read only from `canonical/ACTIVE_RUN_V1.json`.

## Historical Mage
Mage FIT/FIT2 branches, reports and checkpoints remain scoped evidence. They may donate generic implementations only after semantic review. Temporary Mage CDT/Steiner repairs, continuity hacks, subject seals and subject presets are not current authority.
