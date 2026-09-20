# RealSaS-OPT Agent Entry Contract

This repository must be resumable without conversational memory.

## Mandatory first read

1. `canonical/V2_IMPLEMENTATION_READINESS.json`
2. `canonical/MAINLINE_EXECUTION_PLAN_V2.json`
3. `canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md`
4. `CURRENT_STATE.md`
5. `canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json`
6. `canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json`
7. `canonical/V2_ADVERSARIAL_MODULE_AUDIT_PROTOCOL_20260920.md`
8. `canonical/V2_STAGE_BY_STAGE_REDTEAM_20260920.md`
9. `canonical/AUTHORITY_MAP_V1.json`
10. `canonical/EXPERIMENT_REGISTRY_V3.json`
11. `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl`
12. `canonical/ACTIVE_RUN_V2.json` — implementation governance ledger only
13. `canonical/CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json`
14. Historical provenance only (including donor-era presentation/runtime contracts): `canonical/EXPERIMENT_REGISTRY_V2.json`, then `canonical/SCIENTIFIC_JOURNAL_V1.jsonl`

## Current mode

Current execution authority is determined by `canonical/V2_IMPLEMENTATION_READINESS.json`, the stage-level red-team record and the exact implementation-closure fingerprint. RT-37/45/46 are hardened as explicit claim boundaries: Stage37 does not claim artist-layer recovery or connected equivalent-region auto-splitting; Stage45 proves runtime integrity, not perceptual art quality; Stage46 product PASS is executable-contract closure. **Do not execute a subject witness until the exact hardening closure is green, readiness is resealed, and the user explicitly approves it.** The repository governance ledger is never a subject-run ledger.

## Execution semantics

The 46-stage mainline is a DAG. `depends_on` defines readiness. `ordinal` is display/documentation order only. Do not resume by finding the first non-PASS ordinal. Independent branches may progress independently; a branch failure invalidates only its dependency subgraph.

## Product-quality rule

**Geometry, Mechanics and Appearance are co-equal product authorities.**

A visually incorrect puppet is not accepted because mesh, rig and skin are mechanically valid. Spine-class 2D art is a primary product requirement.

- Geometry: surface, silhouette capacity, topology, addressability, rasterizable conditioning.
- Mechanics: rig, skin, deformation, contacts, motion.
- Appearance: source-faithful total 2D art, provenance, completion quality, seams, alpha/sampling and temporal continuity.
- Visibility: posed canonical XYZ + camera depth.
- Presentation/runtime: deterministic consumers; may not invent missing authority.

No layer may compensate for another layer's failure silently.

## Appearance invariants

- CAA is total over supported renderable surface states.
- Qualified source appearance wins and is provenance-locked.
- CAA may not mutate canonical geometry.
- Presentation geometry warp is forbidden in CAA V1.
- Internal filtering/compositing is premultiplied alpha with mandatory atlas bleed.
- Texture alpha is art coverage on a visible surface, not a replacement for z-buffer ownership.
- Runtime performs no generative/corrective appearance inference, donor search, PBR, relighting or normal-derived character shading.
- Totality alone is insufficient: structured holdout, seam continuity and compiled-unobserved screen exposure are explicit gates.
- Runtime-integrity proof is not perceptual-quality proof. Native/reference equality may not be promoted into a Spine-class aesthetic claim.

## Genericity

Compiler/runtime/orchestrator code must not branch on Mage, Knight, filenames, topology counts or subject IDs. Subject facts belong in run manifests and sealed artifacts.

## Adversarial science

Before relying on a module, prove producer capability against downstream gate feasibility, define smallest counterexamples, identify silent failure modes and repair ownership, then inspect the Knight outcome. Never choose thresholds or architecture from Knight results after the fact.

## Historical V1

V1 code/artifacts remain scientific provenance. Shared generic helpers may be reused only when semantics still match V2. Obsolete V1 product semantics must not remain on the current path for compatibility alone.

## Branch

`main` is the sole current continuation branch.
