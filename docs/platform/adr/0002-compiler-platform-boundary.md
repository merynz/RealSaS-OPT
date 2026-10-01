# ADR-0002 — Compiler Owns Meaning; Platform Owns Durable State

Status: ACCEPTED
Date: 2026-10-01

## Context

RealSaS already has a sophisticated compiler/orchestrator:
- a canonical 46-stage DAG;
- typed stage adapters and policies;
- implementation-closure hashing;
- stage input fingerprints;
- stale PASS/FAIL invalidation;
- dependency-aware resume;
- fail-closed output sealing;
- scientific/product proof boundaries.

The professional backend must not replace this with a second, weaker interpretation of the compiler.

At the same time, normal development and product operation must stop depending on:
- branch archaeology;
- run-directory archaeology;
- manually comparing commit hashes;
- reconstructing which artifact is current;
- replaying unrelated stages after a local code change.

## Decision

### Compiler owns meaning

The compiler remains authoritative for:
- stage semantics and scientific contracts;
- adapter invocation semantics;
- stage policies;
- implementation-closure identity;
- validation/proof rules;
- stage failure semantics;
- scientific and product-pass authority.

The platform must call or import those compiler authorities rather than reimplement them independently.

### Platform owns durable state and time

The platform is authoritative for:
- immutable EngineRelease snapshots;
- Subject / Attempt / ProductRevision lifecycle;
- Artifact Registry and content storage;
- durable workflow scheduling;
- cross-run reuse;
- transactional promotion;
- worker crash/retry/resume;
- indexed failure signatures and repair directives;
- user/agent-facing explanation of impact and current state.

In shorthand:

> Compiler owns meaning. Platform owns time, durability, and product state.

## Developer change model

A developer change is interpreted semantically:

```text
canonical main checkout
        |
        v
compiler implementation-closure snapshot
        |
        v
Research EngineRelease
        |
        v
compare to baseline EngineRelease
        |
        +--> direct_changed_stages
        |
        +--> true DAG descendants
        |
        v
Research Attempt
        |
        v
execute only affected semantic subgraph
```

A Git file change is not itself an invalidation rule.
The compiler's adapter implementation closure determines which stage implementations changed.

If one shared core file is consumed by multiple adapters, all affected adapter closures may change and each direct stage is reported explicitly.

## Failure localization model

A stage failure produces persistent diagnostic state:

```text
FailureSignature
    -> OwnerAttribution
    -> RepairDirective
    -> invalidated semantic subgraph
    -> child/retry Attempt
```

Owner attribution may target the failing stage or a true dependency ancestor.
It may not blame an unrelated/downstream stage.

The normal developer question becomes:

> Which semantic owner changed or failed, why, and what exact subgraph must resume?

—not:

> Which of the last hundreds of commits contained the thing we need?

## Existing compiler compatibility

During platform migration, existing compiler runs may be bound to an Attempt through an immutable CompilerRunBinding.

The platform may invoke the compiler's existing target/resume behavior.
After invocation it compares the compiler ledger execution delta with the platform's allowed semantic plan.

Unexpected execution outside the planned scope is a fail-closed error:

`COMPILER_PLATFORM_PLAN_DRIFT`

This prevents the backend from silently hiding a compiler/platform disagreement.

## Temporal granularity

The target workflow model is one durable orchestration unit per semantic stage decision:

- REUSE -> bind qualified immutable artifact;
- EXECUTE -> execute compiler stage contract;
- FAIL -> persist localized failure and stop the dependent path;
- PASS -> register/seal output identities and continue.

A monolithic 12-hour “compile everything” Temporal Activity is not the intended final architecture.

## Main branch policy

`main` is the canonical understandable engineering baseline, not a frozen product snapshot and not an experiment database.

Research changes may start from main, but their state lives in:
- Research EngineRelease;
- Attempt;
- Artifact graph;
- proofs/diagnostics.

Product behavior remains pinned to a Product EngineRelease and promoted ProductRevision until explicitly changed.

## Consequences

Positive:
- current compiler strengths are preserved;
- local changes remain local in semantic ownership;
- research can move quickly without mutating product current;
- crash recovery no longer requires conversational or Git archaeology;
- diagnostics become queryable system state.

Constraint:
- platform/compiler disagreement must be surfaced, never auto-healed by broad rerun.
- platform services may not invent scientific stage semantics that do not exist in compiler authority.
