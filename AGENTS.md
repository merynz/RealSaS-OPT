# RealSaS-OPT Agent Entry Contract

## Mandatory 2026-10-01 handoff

**First read:** `canonical/RECOVERY_CANONICAL_HANDOFF_20261001.md`.

If the user says only “continue / devam et”, resume from current canonical `main`. The old Stage18/38 → Stage42 carrier gap is historical and closed in the recovered lineage; do not reopen it without new evidence. The current scientific/witness blocker is VF-11 R512, while professional platform/backend work may proceed independently after canonical-main recovery.

This repository must be resumable without conversational memory.

## Mandatory first read

1. `canonical/RECOVERY_CANONICAL_HANDOFF_20261001.md`
2. `canonical/RECOVERY_ENGINEERING_CERTIFICATION_V1_20261001.json`
3. `canonical/V2_IMPLEMENTATION_READINESS.json`
4. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
5. `CURRENT_STATE.md`
6. `canonical/CONTEXT_STATE_V2.json`
7. `canonical/AUTHORITY_MAP_V1.json`
8. `canonical/EXPERIMENT_REGISTRY_V3.json`
9. `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl`
10. `canonical/ACTIVE_RUN_V2.json` — implementation governance only
11. `canonical/V2_STAGE_BY_STAGE_REDTEAM_20260920.md`
12. `canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md`
13. `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
14. `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
15. Historical provenance only: `canonical/EXPERIMENT_REGISTRY_V2.json`, then `canonical/SCIENTIFIC_JOURNAL_V1.jsonl`

## Current execution rule

Engineering recovery is exact-head certified, but a subject witness still may not execute unless:
1. `V2_IMPLEMENTATION_READINESS.json` is exactly `READY_FOR_WITNESS_EXECUTION`;
2. its implementation-closure fingerprint equals exact current `main`;
3. the user explicitly approves Knight execution.

VF-11 R512 remains a scientific/witness-readiness blocker. Never infer witness permission from canonical-main promotion or engineering-green CI.

## Execution semantics

The mainline is a dependency DAG. `depends_on` defines readiness; `ordinal` is documentation order only. Verified independent upstream outputs survive downstream failures only under exact hash identity.

No subject-specific compiler/runtime branch. No moving aliases in authority inputs. No silent threshold relaxation after witness inspection.

## Product-quality rule

**Geometry, Mechanics and Appearance are co-equal product authorities; Presentation is first-class editable addressing authority.**

- Geometry owns surface, silhouette capacity, topology and `SurfaceAddressing`.
- Mechanics owns rig, skin, deformation, contacts and full-3D motion.
- Appearance owns source-faithful total 2D art, provenance, holdout/seam quality, alpha/sampling and exposure.
- Presentation owns Stage37-qualified source-owned visual presentation and role-free grouping; categorical identity is not invented.

A visually incorrect puppet is not accepted because mechanics are valid.

## Runtime invariant

Stage18 visual output is substrate, Stage37 is final presentation owner, and Stage42+ must consume typed source-owned visual presentation. Mechanical render fallback, donor search, appearance generation/correction, hidden retriangulation, skin re-solving and character relighting are forbidden.

## Research / product separation

Research work may create branches and experiments, but normal product state must not be inferred from Git chronology. The next platform program introduces immutable `Artifact`, `Attempt`, `ProductRevision`, qualification and durable workflow state.

## Scientific claim discipline

A FIT or witness PASS is scoped evidence for the exact subject/apparatus. It is not unseen generalization. Engineering implementation gates are not Knight performance evidence.

## Repository hygiene

Use one canonical continuation branch: `main`. Current `canonical/EXPERIMENT_REGISTRY_V3.json` and `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl` must be read before historical `canonical/EXPERIMENT_REGISTRY_V2.json` and `canonical/SCIENTIFIC_JOURNAL_V1.jsonl`.
