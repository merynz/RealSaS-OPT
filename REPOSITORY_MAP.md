# Repository Map

RealSaS-OPT is the current RealSaS V2 research/compiler/runtime repository. The sole continuation branch is `main`.

## Current authority spine

1. `canonical/V2_IMPLEMENTATION_READINESS.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
3. `CURRENT_STATE.md`
4. `canonical/CONTEXT_STATE_V2.json`
5. `canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md`
6. `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
7. `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
8. `canonical/CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json`
9. `canonical/V2_STAGE_BY_STAGE_REDTEAM_20260920.md`
10. `canonical/AUTHORITY_MAP_V1.json`
11. `canonical/EXPERIMENT_REGISTRY_V3.json`

Generated `canonical/REHYDRATION_PACKET.md` and `canonical/LIVE_AUTHORITY_MAP.md` are navigation/cache views, not independent scientific authority.

## Current executable lineage

The current executable product path is the 46-stage V2 DAG in `canonical/MAINLINE_EXECUTION_PLAN_V2.json`. The executable contract is self-hosted green, while the stage-level red-team has reopened witness readiness on RT-37 and RT-45/46.

There is no alternate repair branch with current product authority. Historical branches remain evidence only unless explicitly promoted by the current authority spine.

Real execution ledgers are run-local under:

`$REALSAS_AUTHORITY_ROOT/runs/<run_id>/ACTIVE_RUN_V2.json`

The repository `canonical/ACTIVE_RUN_V2.json` is implementation-governance state, not a subject witness ledger.

## Main areas

| Path | Meaning |
|---|---|
| `models/iris/` | learned geometry/surface evidence producer |
| `models/geppetto/` | skeleton proposal model lineage |
| `models/arachne/` | skin proposal model lineage |
| `compiler/realsas_compiler_core/` | canonical typed authority, qualification, CAA, motion and runtime contracts |
| `compiler/realsas_compiler_services/orchestrator/` | 46-stage V2 dependency-DAG execution |
| `runtime/realsas_cpp/` | deterministic native CAA/depth consumer |
| `tests/` | subject-free regression/adversarial qualification |
| `experiments/` | active/historical scientific experiments and diagnostics |
| `canonical/` | current decisions, preregistrations, evidence seals and continuity state |
| `historical/` | preserved provenance only |

## V2 product architecture

`Observation -> Geometry/IRIS -> GSA -> Canonical Mesh Domain`

The canonical mesh domain then supports two independent authorities:

- Mechanics: Geppetto -> qualified skeleton -> Arachne -> qualified skin -> dynamic mechanical mesh.
- Appearance: Complete Appearance Authority -> total directional art + provenance + quality proof.

They reunite in the complete puppet seal, then motion, deterministic runtime, Dynamic Visual Integrity and Stage46 product closure.

## Hard invariants

- Geometry, mechanics and appearance are co-equal product-quality authorities.
- Mesh is the common canonical mechanics/appearance address domain.
- Visibility is posed canonical geometry + camera depth.
- Runtime performs no donor search, generative appearance correction, PBR character relighting or hidden retriangulation.
- A visually incorrect puppet is not accepted because mechanics pass.
- Current V2 may reuse a generic historical helper only when its semantics still match; obsolete donor/UNSEEN/linear-product semantics may not enter the current import closure.
- Subject-specific compiler/runtime branches are forbidden.
