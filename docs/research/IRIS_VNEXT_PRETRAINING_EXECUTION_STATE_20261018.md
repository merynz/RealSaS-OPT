# IRIS vNext — Pretraining Execution State — 2026-10-18

Status: **SOURCE-READY / ARTIFACT MATERIALIZATION PENDING / TRAINING NOT STARTED**

## Executable source baseline

Pretraining machinery was implemented at source commit:

`726768813c523081166cc1ac49f08453260b66b9`

That exact source passed:

- `iris-vnext-pretraining-candidate-source-gate-v1` push run `39137955380`;
- full `iris-vnext-single-carrier-v1` push run `39137955347`;
- PR #116 broad gate set: **15/15 completed success** on the same head.

The source lane contains:

- exact-SHA candidate preparation remote isolation;
- full local↔Drive byte round-trip verification;
- Relation Truth V2 materialization;
- source-defensible Boundary Causality V1 materialization;
- exact completed-V8 relation-head → `GraphicsRelationHeadV2` zero-step migration;
- pretraining materialization seal;
- artifact-bound preregistration/readiness gate;
- explicit `training_steps_executed = 0` until a separate training entry is invoked.

## Quarantined remote forensic result

Read-only Google Drive inspection of:

`RealSaS_Fit8_IRIS_vNext_20261016`

folder id:

`1QgrZSGKxnhFWKq-HYiycUi3i_0oIJp2B`

established:

- top-level contains only `teachers/`;
- `prepared/` is missing;
- `receipts/` is missing;
- `teachers/TEACHER_BATCH_RECEIPT.json` SHA256 is `7ac1c66cc5f749e0f6f8cac43a9b79a135c8c45ed1138303d42c102f04c15d27`, exactly matching cancelled stale run `39121291171` forensic lineage;
- seven subjects have teacher NPZ + receipt pairs;
- George has its receipt but its `.teacher.npz` is absent.

Classification:

`INCOMPLETE_REMOTE_SNAPSHOT`

The remote remains **QUARANTINED** and **MUST NOT BE CONSUMED**.

Machine-readable forensic record:

`docs/research/IRIS_VNEXT_QUARANTINE_CONNECTOR_AUDIT_20261018.json`

## Exact completed V8 authority bound for migration

Drive authority folder:

`v8_m1_sealed_5083c991`

Completed V8 source:

`5083c9915533ec8ec6530c7d229c635887abfea7`

Authoritative latest pointer selects:

- file: `step_00012000_ac23a9ea448e465cb668773a131981be.pt`
- global step: `12000`
- checkpoint SHA256: `0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5`
- trainer binding SHA256: `735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2`
- checkpoint Drive file id: `1xttwuxgC8y9EVmf7xxMEN4yj84yhm4zH`

A second step-12000 file exists in the folder but is **not** selected by authoritative `LATEST.json`; migration source therefore remains the `ac23...` file above.

The completed V8 receipt still states `scientific_fit_pass_claimed=false`.

## What is still pending before training

Source readiness is not artifact readiness. The following must still be produced from one exact final source SHA:

1. read-only quarantine audit artifact bound to that SHA;
2. new candidate preparation under `gdrive:RealSaS_Fit8_IRIS_vNext_candidates/<exact_sha>`;
3. full remote round-trip SHA verification;
4. actual Relation Truth V2 manifest/corpus;
5. actual Boundary Causality V1 manifest/corpus;
6. actual zero-step V8→RelationHeadV2 migrated artifact/receipt;
7. pretraining materialization receipt;
8. artifact-bound preregistration with held-out gates and non-regression gates;
9. readiness audit returning `READY_FOR_FOCUSED_TRAINING_ENTRY__NO_STEP_EXECUTED`.

Only after item 9 is PASS does the work reach the training boundary requested by the operator.

## Training state

```text
optimizer steps in the post-V8 focused program = 0
scene_core reopening = NOT AUTHORIZED
broad 12k rerun = NOT AUTHORIZED
focused relation/boundary training = NOT STARTED
scientific Fit8 PASS = NOT CLAIMED
```
