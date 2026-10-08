# ADR-0003 — Go Control Plane, Python Engine, C++ Runtime

Status: **ACCEPTED**  
Date: 2026-10-01

## Decision

RealSaS uses three deliberate implementation domains:

- **Go** — professional platform/control plane.
- **Python** — scientific compiler, ML/model integration, research tooling, and engine workers.
- **C++** — runtime/render/deformation/hot geometry paths.

PostgreSQL remains transactional product-state truth and an S3-compatible immutable store remains artifact-byte truth.

## Why

The product backend is primarily systems engineering: transactions, APIs, concurrency, durable workflow orchestration, object storage, observability, lifecycle/state machines, idempotency, and service operation. Go is the default language for this layer.

The existing RealSaS compiler is already advanced and Python-native. Rewriting scientific/compiler semantics in Go would add risk without product value. Python remains the right language for IRIS/model code, geometry/ML research, compiler adapters, proof logic, and fast research iteration.

The runtime remains native C++ where latency and rendering/deformation performance matter.

## Authority boundary

```text
Go Platform
  owns durable product/research state, API, Artifact Registry metadata,
  ProductRevision, Attempt lifecycle, promotion, outbox, dependency scheduling,
  Temporal workflows, and query/explanation surfaces.

Python Engine
  owns compiler semantics, stage adapters, model inference/training,
  implementation-closure evidence, scientific proof, stage execution,
  and typed failure evidence.

C++ Runtime
  owns playback, deformation/render hot paths, native package consumption.
```

This extends ADR-0002:

> Compiler owns meaning. Platform owns time, durability, and product state.

## Temporal interoperability

Go owns top-level durable workflows.

Python workers may implement named Temporal Activities for compiler/model work. Activities are addressed by stable typed protocol/name rather than by importing Python application code into the Go service.

This keeps the control plane deployable as native Go services while retaining the existing Python compiler.

## PostgreSQL access

Go control-plane persistence uses `pgx/v5`.

The platform must not maintain two production migration authorities. Existing Alembic migrations created during the Python reference phase are executable specification only until the Go migration cutover is completed and parity-proven. The cutover must be explicit.

## Python backend retirement

The temporary `backend/realsas_platform` reference control plane was retired after the Go control plane passed its PostgreSQL, migration, artifact, capability-execution and architecture-contract gates on 2026-10-01.

Python remains authoritative only for scientific/compiler/model/runtime-worker concerns. A future Python engine worker must expose stable typed activity contracts to Go; it must not recreate product-state persistence, promotion, migration or orchestration authority in Python.

## Go baseline

- Go 1.27.x
- Temporal Go SDK v1.49.x
- pgx/v5
- standard library HTTP initially; add a router/framework only when it provides concrete value.

## Migration acceptance

The Go control plane is the sole live platform/control-plane implementation on this branch after parity and closure gates proved at minimum:

1. semantic Artifact identity parity;
2. exact 46-stage graph/invalidation parity;
3. immutable EngineRelease parity;
4. transactional Attempt/ProductRevision/Promotion semantics;
5. outbox idempotency;
6. deterministic Temporal workflow identity;
7. render cannot schedule fit/train/calibrate/promote;
8. failure/repair explanation can be queried without Git archaeology.

The Go control plane is on canonical main. Product authority remains an explicit qualified ProductRevision, independent of a Git merge. Operational development/product paths and outstanding executor limits are documented in `docs/platform/EXECUTION_LANES.md`.
