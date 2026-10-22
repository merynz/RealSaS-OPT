# Operational Spine

Status: P0 implementation contract.

The Platform must answer current operational truth without Git-history, branch, notebook, run-directory, or conversation archaeology.

## Authority model

| State | Owner | Meaning |
|---|---|---|
| Source | Git `main` | Canonical shipped source code only. Git is not an experiment database. |
| Executable snapshot | `EngineRelease` | Immutable code/config/DAG identity selected for execution. |
| Scientific intent | `Experiment` | Research question, hypothesis, preregistered intent and lifecycle. |
| Execution | `Attempt` | One actual subject-bound execution. |
| Artifact | Artifact Registry | Immutable typed bytes and semantic dependencies. |
| Model identity | `ModelRelease` | Human-readable checkpoint identity and qualification/evaluation envelope. |
| Model selection | `model_channels` | Explicit RESEARCH/CANDIDATE/PRODUCTION selection, never a floating `latest` alias. |
| Product | `ProductRevision` | Qualified subject product state. |
| Scientific/product meaning | Compiler + Proof/Qualification | Validation, courts, scientific claims and product-pass authority. |

`Experiment` does not replace `Attempt`. `ModelRelease` does not replace an Artifact, EngineRelease, Proof, or Qualification. These records make intent and deployable identity queryable while preserving the existing scientific authority chain.

## Experiment lifecycle

```text
DRAFT
  -> PREREGISTERED
       -> RUNNING
            -> COMPLETED
            -> ABORTED
            -> SUPERSEDED
       -> ABORTED
       -> SUPERSEDED
  -> ABORTED
```

Attempts and Campaigns may be linked to an Experiment, but the Experiment itself never claims PASS/FAIL merely because an execution exists.

## Model release lifecycle

A `ModelRelease` binds an exact checkpoint Artifact to an exact EngineRelease and optionally to its originating Experiment and Attempt. Registration is identity sealing only: the caller must supply the checkpoint Artifact's expected content SHA256 and the Platform verifies that exact hash against the stored Artifact before the release is created.

Every ModelRelease is born `DRAFT`. Qualification is never accepted as a registration-time label.

```text
DRAFT
  -- explicit PASS evaluation scopes --> RESEARCH_QUALIFIED
  -- explicit PASS evaluation scopes --> PRODUCT_CANDIDATE
  -- explicit PASS evaluation scopes --> PRODUCTION
```

A rejected or superseded release is not promotable. Promotion to **any** model channel requires an explicit, non-empty set of evaluation scopes and every required scope must already be recorded `PASS`. The Platform does not invent those evaluation results; they must point to existing evidence and/or Experiment authority.

Human-readable channels are `(family, channel)` pairs such as:

```text
IRIS  / RESEARCH
IRIS  / CANDIDATE
IRIS  / PRODUCTION
TESSA / RESEARCH
AXIS  / PRODUCTION
MIRA  / PRODUCTION
```

They are versioned selections with optimistic lock versions. They are not mutable `latest` artifact aliases.

## Single operational query

`GET /v1/ops/status` and `realsasctl status` expose one read model containing:

- preregistered/running Experiments;
- open Attempts and their Experiment links;
- active Campaigns and their Experiment links;
- current model channels with exact ModelRelease/evaluation state;
- current ProductRevision per subject.

This is a read model only. It creates no scientific or product authority.

## Research execution rule

Normal research code lands through short reviewed PRs onto `main`. An Experiment/Attempt/EngineRelease owns continuation state. Long-lived research branches, notebooks and mutable canonical JSON files must not become alternate continuation authorities.

GPU notebooks and accelerator jobs are compute workers. They may return checkpoint/metric/receipt bytes; imported bytes remain non-authoritative until existing consumers validate and reseal them.

## P1 migration rule

Legacy research state is migrated fail-closed:

1. preserve exact branch/SHA/binding lineage as metadata/provenance;
2. create the durable Experiment without changing scientific thresholds or results;
3. link only real existing Attempts/Campaigns;
4. create a **DRAFT** ModelRelease only when the real checkpoint Artifact and EngineRelease identities are available and the exact checkpoint content SHA matches;
5. preserve historical qualification as explicit evaluation evidence before any channel promotion; never infer qualification from a legacy model name or branch;
6. never fabricate UUIDs, PASS results, ProductRevisions or promotion state;
7. archive immutable JSON records as provenance after Platform state becomes continuation authority;
8. retire the long-lived branch only after generic source needed by active execution is promoted through reviewed `main` changes.

## Runtime-consumption follow-up

P0 establishes lifecycle and model truth first. Shipping semantic fields must subsequently gain an explicit behavioral-consumption contract (`DEFINED -> PRODUCED -> QUALIFIED -> SERIALIZED -> CONSUMED -> BEHAVIORALLY_EFFECTIVE`) so serialization alone cannot masquerade as runtime use. That work must preserve the same rule: evidence production and qualification are distinct from behaviorally demonstrated consumption.
