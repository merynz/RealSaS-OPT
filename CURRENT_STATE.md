# RealSaS-OPT — Current State

**Date:** 2026-10-18  
**Canonical production line:** `main` until reviewed vNext is merged  
**Active review source:** `research/iris-vnext-post-v8-quality-plan-20261018`  
**Review vehicle:** PR #116 (`research/iris-vnext-single-carrier-graphics`)  
**Verified executable pretraining source:** `726768813c523081166cc1ac49f08453260b66b9`  
**Verification:** pretraining source gate `39137955380` PASS; full vNext suite `39137955347` PASS; PR #116 exact-head gate set **15/15 PASS**  
**Active program:** **POST-V8 QUALITY / TRUTH / CONSUMPTION QUALIFICATION**  
**Scientific status:** **NO IRIS FIT8 SCIENTIFIC PASS**  
**Focused post-V8 training steps executed:** **0**

## Authority separation

1. `main` remains merged production-code authority.
2. Completed V8 remains an immutable architecture/measurement baseline bound to source `5083c9915533ec8ec6530c7d229c635887abfea7` plus exact checkpoint/receipt lineage.
3. The active post-V8 branch is review source only; later source hardening does not rewrite V8 history.
4. Green CI proves source contracts, not scientific quality.
5. Artifact Registry / Attempt / ProductRevision and sealed experiment artifacts remain durable research/product authority.

## Preserved product continuity

The post-V8 IRIS program does not replace current V2 product/control-plane authority on `main`.

- `canonical/V2_IMPLEMENTATION_READINESS.json` remains the implementation-readiness pointer.
- The 43-stage dependency DAG and `PLATFORM_EXECUTION_LANES` remain existing control-plane continuity.
- `QualifiedMeshIR`, `SurfaceAddressing`, and Complete Appearance Authority retain their scoped product meanings.
- Existing `CARRIER_NATIVE_FIT1_SCOPED_PASS__PLATFORM_LANES_IMPLEMENTED` continuity is not a new IRIS/Fit8 PASS.

## Training lineage and invariant

V5 exposed unsafe downstream rewriting of shared H. V6 showed loss retention alone did not prevent Geometry/extraction drift while `scene_core` remained trainable. V7 repaired TESSA coordinate compatibility. V8 migrated exact V6 M1-end, sealed `scene_core`, completed step 12000, and preserved Geometry/extraction while downstream specialist objectives remained learnable.

Current default:

```text
G / M establish shared carrier H
        ↓
M1 seal
        ↓
A / R / T / J specialist learning
        ↓
scene_core remains sealed
```

Post-V8 `optimization_v2.py` separates future specialist parameter groups and telemetry; it does not rewrite historical V8 optimizer provenance.

## Verified post-V8 source stack

At `726768813c523081166cc1ac49f08453260b66b9`, exact source CI is green for:

- canonical post-M1 stage IDs and fail-closed carrier guard;
- Graphics Truth V2 authority/provenance/validity and validity-aware loss;
- Relation Truth V2 producer and support/class-balance eligibility;
- `GraphicsRelationHeadV2` conditioning plus deterministic V1→V2 migration semantics;
- head-separated specialist optimizer/telemetry;
- source-defensible Boundary Causality V1 materializer;
- exact completed-V8 relation-head migration tool;
- SHA-scoped candidate preparation lane and full remote round-trip verifier;
- materialization seal and preregistration/readiness gate;
- self-hosted writer safety contracts;
- single-carrier Compiler/runtime behavior, native Graphics ordering/E2E, and decoder regressions.

No training is invoked by this lane.

## Quarantined remote forensic result

Cancelled stale run `39121291171` wrote partially to:

`gdrive:RealSaS_Fit8_IRIS_vNext_20261016`

Read-only Drive inspection now establishes **`INCOMPLETE_REMOTE_SNAPSHOT`**:

- remote folder id `1QgrZSGKxnhFWKq-HYiycUi3i_0oIJp2B`;
- top-level contains only `teachers/`;
- `prepared/` and `receipts/` are absent;
- `TEACHER_BATCH_RECEIPT.json` SHA256 is `7ac1c66cc5f749e0f6f8cac43a9b79a135c8c45ed1138303d42c102f04c15d27`, exactly matching stale-run forensic evidence;
- George teacher receipt is present but George `.teacher.npz` is absent.

Therefore the remote is both incomplete and stale-lineage. It remains **QUARANTINED / CONSUMPTION FORBIDDEN**. Machine-readable evidence is in `docs/research/IRIS_VNEXT_QUARANTINE_CONNECTOR_AUDIT_20261018.json`.

## Pretraining materialization lane

The old writer is retired. A new manual, fail-closed candidate writer is now source-implemented:

`.github/workflows/iris_vnext_fit8_pretraining_candidate_self_hosted_v1.yml`

It is designed to:

1. bind one exact active-review SHA;
2. consume a read-only quarantine audit artifact without authorizing the old remote;
3. hydrate exact admitted Fit8 inputs;
4. export authored-motion teachers;
5. build the sealed V4 cohort;
6. materialize Relation Truth V2 and Boundary Causality V1;
7. hydrate the exact completed-V8 authority checkpoint;
8. perform zero-step relation-head V1→V2 migration;
9. write only to `gdrive:RealSaS_Fit8_IRIS_vNext_candidates/<exact_sha>`;
10. refuse overwrite of a non-empty candidate remote;
11. require full Drive round-trip byte equality;
12. seal a pretraining materialization receipt with `training_authorized=false`.

**Source-ready does not mean artifact-ready.** This workflow has not yet materialized the final candidate artifact for the current final source SHA.

## Exact V8 migration authority

Drive folder `v8_m1_sealed_5083c991` contains the completed V8 lineage. Authoritative `LATEST.json` selects:

- checkpoint `step_00012000_ac23a9ea448e465cb668773a131981be.pt`;
- step `12000`;
- checkpoint SHA256 `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`;
- binding SHA256 `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`.

The migration source is pinned to that exact artifact. Historical V8 remains `scientific_fit_pass_claimed=false`.

## Relation truth / Boundary Causality

Replacement source is implemented, but the new final-source corpus is **not yet sealed**.

- static same/cut/attachment abstain when independent authority is absent;
- view-conditioned front/visibility and projected boundaries are camera-conditioned;
- current deterministic occlusion rows are single-class positive support and are not binary-court eligible;
- Boundary Causality accepts only source-defensible causes and explicit abstention.

Do not fabricate negative truth or semantic/drawable intent unavailable from the source.

## Compiler/runtime consumption

Source evidence currently supports fragment clustering from admissible same/cut/boundary evidence, front+occlusion order edges, precedence overriding contradictory physical depth, and visibility changing final native RGBA.

Open behavioral debt remains for attachment and support-band application in the native reference renderer. Serialization is not behavioral use.

## TESSA/topology

V8 topology loss was not an independent topology court. Held-out qualification must measure actual valid/degenerate faces, component error, holes/boundary loops, meaningful manifold diagnostics, canonical-surface deviation, source support and cohort worst cases.

## Immediate execution state

Completed now:

- source pretraining lane implemented;
- executable source gate green;
- full vNext source suite green;
- PR #116 exact executable head 15/15 green;
- quarantined Drive remote classified read-only as incomplete/stale-lineage;
- exact completed-V8 migration checkpoint identity established.

Still required before **any** focused optimizer step:

1. run the exact-SHA read-only audit artifact on the final source head;
2. materialize the SHA-scoped candidate remote;
3. verify full remote round-trip hashes;
4. seal actual Relation Truth V2 + Boundary Causality manifests;
5. seal actual zero-step V8→HeadV2 migration artifact/receipt;
6. seal pretraining materialization receipt;
7. preregister held-out and non-regression gates against exact artifact hashes;
8. require readiness status `READY_FOR_FOCUSED_TRAINING_ENTRY__NO_STEP_EXECUTED`.

At that point the project has reached the training boundary. No focused training has started yet, and no broad 12k/A100 rerun is authorized.

## Required read order

1. `CURRENT_STATE.md`
2. `CURRENT_PLAN.md`
3. `docs/research/IRIS_VNEXT_PRETRAINING_EXECUTION_STATE_20261018.md`
4. `docs/research/IRIS_VNEXT_QUARANTINE_CONNECTOR_AUDIT_20261018.json`
5. `docs/research/IRIS_VNEXT_DECISION_LEDGER_20261018.md`
6. `docs/research/IRIS_VNEXT_RELATION_TRUTH_AUDIT_20261018.md`
7. `docs/research/IRIS_VNEXT_POST_V8_QUALITY_PLAN_20261018.md`
8. `docs/research/IRIS_VNEXT_STDGENPP_REAUDIT_20261018.md`
9. `docs/research/IRIS_VNEXT_GRAPHICS_CONSUMPTION_LINEAGE_V1.json`
10. `AGENTS.md`
11. `SYSTEM_INDEX.md`
