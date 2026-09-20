# RealSaS-OPT Agent Entry Contract

This repository must be resumable without conversational memory.

## Mandatory first read

1. `canonical/V2_IMPLEMENTATION_READINESS.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
3. `CURRENT_STATE.md`
4. `canonical/V2_STAGE_BY_STAGE_REDTEAM_20260920.md`
5. `canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md`
6. `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
7. `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
8. `canonical/CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json`
9. `canonical/PRESENTATION_PARTITION_POLICY_V1_20260921.json`
10. `canonical/DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json`
11. `canonical/AUTHORITY_MAP_V1.json`
12. `canonical/EXPERIMENT_REGISTRY_V3.json`
13. `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl`
14. `canonical/ACTIVE_RUN_V2.json` — implementation governance only
15. Historical provenance only: `canonical/EXPERIMENT_REGISTRY_V2.json`, then `canonical/SCIENTIFIC_JOURNAL_V1.jsonl`

## Current execution rule

The 46-stage second-pass red-team implementation blockers are closed subject-free. A subject witness still may not execute unless:

1. `V2_IMPLEMENTATION_READINESS.json` is exactly `READY_FOR_WITNESS_EXECUTION`;
2. its implementation-closure fingerprint equals exact current `main`;
3. the user explicitly approves Knight execution.

Never infer approval from a green CI run, a READY seal or historical activation records. `canonical/ACTIVE_RUN_V2.json` is never a subject-run ledger.

## Execution semantics

The mainline is a dependency DAG. `depends_on` defines readiness; `ordinal` is documentation order only. Verified independent upstream outputs survive downstream failures only under exact hash identity.

No subject-specific compiler/runtime branch. No moving aliases in authority inputs. No silent threshold relaxation after witness inspection.

## Product-quality rule

**Geometry, Mechanics and Appearance are co-equal product authorities; Presentation is a first-class editable addressing authority over them.**

- Geometry: surface, silhouette capacity, topology, `SurfaceAddressing`, rasterizable conditioning.
- Mechanics: rig, skin, deformation, contacts, motion.
- Appearance: source-faithful total 2D art, provenance, holdout/seam quality, alpha/sampling, exposure and intrinsic dynamic deformation conditioning.
- Presentation: role-free slots/attachments/grouping. It may use mechanical/topological/source-appearance evidence but may not invent category labels.

A visually incorrect puppet is not accepted because its mesh, rig and skin are mechanically valid.

## RT-37 invariant

`presentation_partition_v2.py` may cut adjacency inside a mechanical component only from frozen observable evidence. `PRESENTATION_PARTITION_POLICY_V1_20260921.json` is subject-free and Knight may not tune it. Missing evidence means continuity, not semantic invention.

## RT-45 invariant

Shipping dynamic appearance gates are **intrinsic textured-surface metrics**, not raw screen-space distortion metrics. Rigid 3D rotation/translation must not fail merely because projection foreshortens a face. Thresholds are frozen by `DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json`.

## Runtime invariant

Visibility = posed canonical geometry + camera depth. Appearance = sealed CAA. Runtime may not do donor search, appearance generation/correction, hidden retriangulation, skin re-solving or character relighting.

## Scientific claim discipline

A FIT or witness PASS is scoped evidence for the exact subject/apparatus. It is not unseen generalization. Subject-free implementation gates are not Knight performance evidence. Semantic recognition and human aesthetic optimality are not implied by V2 closure.

## Repository hygiene

Use one canonical continuation branch: `main`. Preserve historical evidence, but do not allow historical donor/runtime modules into current V2 closure. Current `canonical/EXPERIMENT_REGISTRY_V3.json` and `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl` must be read before historical `canonical/EXPERIMENT_REGISTRY_V2.json` and `canonical/SCIENTIFIC_JOURNAL_V1.jsonl`.

The current execution plan is `canonical/MAINLINE_EXECUTION_PLAN_V2.json`. Product architecture authority is `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`; appearance authority is `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`; repository governance state is `canonical/ACTIVE_RUN_V2.json`.
