# ADR-0001 — Professional Backend Foundation

Status: **ACCEPTED FOR PLATFORM DESIGN**
Date: 2026-10-01

## Context

RealSaS currently carries too much product/execution truth in Git refs, hashes, generated files and CI runs. This works for research but does not scale to 10k+ unseen subjects, stable rendering, bounded invalidation or separate product/research operation.

## Decision

Adopt:
- PostgreSQL 18.x for relational metadata/product state;
- Temporal for durable workflows;
- S3-compatible immutable object storage behind a RealSaS adapter;
- SQLAlchemy 2.1 + psycopg3 + Alembic;
- pinned FastAPI/Uvicorn for the control-plane API;
- OpenTelemetry traces/metrics plus structured logs.

Use the existing Python compiler/model ecosystem for orchestration/domain services and retain C++ for runtime/performance-critical code.

Do not introduce Redis, Kafka/NATS, Kubernetes or a generic ML registry until a concrete requirement justifies them.

## Consequences

Positive:
- transactional ProductRevision promotion;
- durable crash-resumable execution;
- cross-run artifact reuse;
- bounded semantic invalidation;
- stable product current;
- explicit research Attempts;
- queryable provenance;
- simple Studio/Forge interfaces over one Engine.

Costs:
- schema migrations and operational database discipline become first-class;
- Temporal workflow determinism/versioning must be respected;
- artifact GC/backups/restore testing become required operations;
- platform code becomes a maintained subsystem.

## Guard

Implementation must integrate from the final proven canonical recovery head. This ADR may be committed on the isolated design/memory branch before recovery promotion.
