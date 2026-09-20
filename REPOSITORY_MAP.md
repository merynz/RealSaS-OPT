# Repository Map

RealSaS-OPT is the current RealSaS V2 research/compiler/runtime repository. The sole continuation branch is `main`.

## Current authority spine

1. `canonical/V2_IMPLEMENTATION_READINESS.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
3. `CURRENT_STATE.md`
4. `canonical/CONTEXT_STATE_V2.json`
5. `canonical/V2_STAGE_BY_STAGE_REDTEAM_20260920.md`
6. `canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md`
7. `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
8. `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
9. `canonical/CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json`
10. `canonical/PRESENTATION_PARTITION_POLICY_V1_20260921.json`
11. `canonical/DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json`
12. `canonical/AUTHORITY_MAP_V1.json`
13. `canonical/EXPERIMENT_REGISTRY_V3.json`

Generated `canonical/REHYDRATION_PACKET.md` and `canonical/LIVE_AUTHORITY_MAP.md` are navigation/cache views, not independent scientific authority.

## Current executable lineage

The current executable product path is the 46-stage V2 dependency DAG in `canonical/MAINLINE_EXECUTION_PLAN_V2.json`. The second-pass stage red-team implementation blockers RT-37 and RT-45/46 are repaired subject-free. Witness execution still requires the exact current readiness seal plus explicit user approval.

There is no alternate repair branch with current product authority.

Real execution ledgers are run-local under:

`$REALSAS_AUTHORITY_ROOT/runs/<run_id>/ACTIVE_RUN_V2.json`

The repository `canonical/ACTIVE_RUN_V2.json` is implementation-governance state, not a subject witness ledger.

## Main areas

| Path | Meaning |
|---|---|
| `models/iris/` | learned geometry/surface evidence producer |
| `models/geppetto/` | skeleton proposal model lineage |
| `models/arachne/` | skin proposal model lineage |
| `compiler/realsas_compiler_core/presentation_partition_v2.py` | role-free evidence-supported presentation partition authority |
| `compiler/realsas_compiler_core/dynamic_appearance_conditioning_v2.py` | intrinsic dynamic texture/line-art deformation conditioning |
| `compiler/realsas_compiler_core/` | canonical typed authority, qualification, CAA, motion and runtime contracts |
| `compiler/realsas_compiler_services/orchestrator/` | 46-stage V2 dependency-DAG execution |
| `runtime/realsas_cpp/` | deterministic native CAA/depth consumer |
| `tests/` | subject-free regression/adversarial qualification |
| `experiments/` | active/historical scientific experiments and diagnostics |
| `canonical/` | current decisions, policies, calibration, evidence seals and continuity state |
| `historical/` | preserved provenance only |

## V2 product architecture

`Observation -> Geometry/IRIS -> GSA -> Canonical Mesh Domain`

The canonical mesh domain splits into co-equal mechanics and appearance branches, then rejoins through evidence-supported Presentation/Complete Puppet, full-3D Motion, deterministic Runtime, Dynamic Visual Integrity and Stage46 Closure.

## Red-team additions

- `PRESENTATION_PARTITION_POLICY_V1_20260921.json` freezes role-free CAA-bound presentation boundary policy.
- `DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json` freezes subject-free intrinsic deformation thresholds.
- Stage46 refuses closure if either repaired authority is missing or drifted.

## Historical lineage

Old 40-stage V1, Mage FIT1/FIT2 and donor-era appearance/runtime paths remain evidence only. They are not imported into current V2 closure merely for compatibility.
