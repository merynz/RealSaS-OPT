# RealSaS Platform Backend Architecture V1

> DESIGN AUTHORITY CANDIDATE. This document starts the systems-engineering program. It does not promote production state and must not move `main` while canonical recovery remains open.

## Mission

Turn RealSaS from a repository/run/hash-driven research system into a professional compiler platform where:
- Git is source-code history, not product state;
- product truth is an atomic immutable `ProductRevision`;
- research truth is an `Attempt`;
- generated assets are immutable typed `Artifact` records backed by content-addressed object storage;
- workflow state is durable and resumable;
- render never silently trains/fits/recalibrates/promotes;
- dependency invalidation is semantic and minimal;
- exact provenance remains expandable without being required for normal operation.

## Chosen platform boundary

```text
RealSaS Studio / Forge / CLI
            |
      Control Plane API
            |
   +--------+---------+
   |                  |
PostgreSQL         Temporal
product/meta       durable execution
   |                  |
   +--------+---------+
            |
     Engine Workers
 Models / Compiler / Proof / Runtime
            |
   Immutable Artifact Store
       S3 API / Local CAS
```

The Engine remains the only scientific execution authority. Forge and Studio are different interfaces over the same Engine, registry and promotion rules.

## Technology decisions

### Metadata and transactional state
- PostgreSQL 18.x.
- SQLAlchemy 2.1 ORM/Core.
- psycopg 3 PostgreSQL driver.
- Alembic schema migrations.
- PostgreSQL constraints/FKs/unique indexes are part of the correctness model, not optional conveniences.

### Durable execution
- Temporal.
- Workflows coordinate deterministic stage execution.
- External I/O, model inference, compiler execution, database access and object-store operations occur in Activities.
- Workflow IDs are deterministic and idempotent.

### API / control plane
- Python service with pinned FastAPI/Uvicorn.
- Pydantic v2 DTOs/contracts.
- API handlers are thin command/query adapters; scientific/compiler logic stays in Engine packages.
- API does not run long jobs in-process.

### Artifact bytes
- Provider-neutral S3-compatible interface for production.
- AWS S3 or Cloudflare R2 are acceptable providers behind the same adapter.
- Local developer mode uses a filesystem content-addressed store.
- MinIO is not the default due current licensing constraints.
- Artifact object keys are content-derived and never overwritten.

### Observability
- OpenTelemetry traces and metrics.
- Structured JSON application logs.
- Correlation IDs: request_id, workflow_id, attempt_id, execution_id, artifact_id, product_revision_id.
- Metrics must distinguish cache hit/reuse, recomputation, queue delay, stage duration, proof failure and object-store transfer.

### Deployment
- Docker Compose for local reproducible development.
- Do not introduce Kubernetes until multi-node scheduling/availability requires it.
- Temporal Cloud vs self-hosted remains a deployment choice; application contracts must not depend on either.
- PostgreSQL and Temporal persistence must use separate logical databases/users even if they share one physical cluster.

## Deliberately not selected initially

- No custom queue/scheduler.
- No Redis as product truth or artifact registry.
- No Kafka/NATS unless an actual asynchronous fan-out/event-stream requirement appears.
- No generic ML experiment platform as product authority.
- No mutable "latest/" artifact directories.
- No branch name as product identity.
- No Git SHA as user-facing artifact identity.
- No second pipeline for Studio.

## Core domain objects

### Subject
Stable identity for an imported character/project subject.

### Artifact
Immutable typed semantic output.

Identity has two independent hashes:

```text
SemanticIdentity =
  artifact_type
  + schema_version
  + producer_contract_version
  + implementation_identity
  + policy_identity
  + ordered typed input artifact identities
  + explicit semantic parameters

ContentIdentity =
  sha256(exact stored bytes)
```

Execution/run identity is not part of artifact reuse identity.

### Attempt
Research/developer candidate graph.
- Can inherit artifacts from a parent attempt.
- May invalidate one owner and true descendants.
- Cannot silently change product current.
- Repair directives create child attempts.

### ProductRevision
Immutable compatible set of promoted product artifacts.

Conceptual roles include:
- observation
- geometry
- mechanical_mesh
- skeleton
- skin
- appearance
- visual_presentation
- motion_library
- runtime_compatibility

A revision never changes after sealing.

### Promotion
Append-only decision that makes one qualified ProductRevision current for a Subject.
Product current changes only through one transactional promotion operation.

### Execution
One attempt to produce or verify an artifact.
Contains worker/machine/time/retry/log provenance.
Execution identity never makes a semantically equivalent artifact unreusable.

### Proof / Qualification
Append-only evidence tied to exact artifact identities, policy identity and producer implementation.
Only explicit qualification may make artifacts eligible for ProductRevision construction/promotion.

### RenderRequest
Always binds an exact ProductRevision and explicit motion/view/settings.
A render request cannot resolve "whatever research state is newest."

## Product invariants

1. `RENDER != COMPILE`.
2. Product render may execute no fit/train/calibration/promotion adapters.
3. Repeated identical render request resolves the same ProductRevision and semantic render identity.
4. Product-current changes only through explicit promotion.
5. Artifacts and ProductRevisions are immutable.
6. Research Attempts may fail without affecting current product.
7. Runtime-only implementation changes invalidate runtime descendants, not model fits or upstream geometry.
8. A Stage35 change invalidates Stage35's true descendants, not unrelated upstream fits.
9. Same semantic inputs across different workflow/run IDs can reuse one artifact.
10. Hash/provenance details are expandable forensic data, not normal operating inputs.
11. Object-store bytes are considered usable only after content hash verification and registry commit.
12. No database row may point to an unverified missing artifact object.

## Transaction boundaries

### Artifact commit
1. Worker produces bytes into temporary local/workspace storage.
2. Compute exact SHA-256.
3. Upload to immutable CAS key.
4. Verify object metadata/size/hash expectations.
5. In one DB transaction:
   - insert artifact if semantic identity is new;
   - insert ordered input edges;
   - insert producer execution link;
   - mark object as registry-visible.
6. Duplicate semantic production resolves idempotently to the existing artifact when compatible.

### Product promotion
One serializable transaction:
1. lock SubjectCurrent row;
2. validate target ProductRevision is sealed and fully qualified;
3. append Promotion record;
4. compare-and-swap current revision pointer;
5. append audit/outbox event;
6. commit.

No artifact production occurs inside promotion.

### API command -> workflow
Use transactional outbox:
1. validate command and idempotency key;
2. insert command + outbox event in one PostgreSQL transaction;
3. dispatcher starts Temporal workflow using deterministic workflow ID;
4. dispatcher records delivery; repeated delivery is harmless.

This avoids a DB-success/Temporal-start-failed split-brain.

## Workflow model

### CompileSubjectWorkflow
- resolves desired ProductRevision graph;
- asks resolver which semantic artifacts already exist;
- schedules only missing/invalid artifacts;
- executes proof/qualification;
- creates sealed candidate ProductRevision;
- does not auto-promote unless command explicitly authorizes promotion semantics.

### RenderWorkflow
- accepts exact ProductRevision ID;
- resolves sealed motion/view/settings;
- checks render artifact semantic identity;
- returns cache hit or runs only render/runtime tail;
- mechanically rejects fit/train/calibrate/promote activity types.

### RepairAttemptWorkflow
- creates child Attempt from exact parent;
- carries typed RepairDirective owner;
- inherits unaffected artifacts;
- recomputes owner + dependency descendants only;
- qualifies candidate without changing SubjectCurrent.

## Dependency resolver

The existing 46-stage map becomes executable static metadata:
- stage contract version;
- typed input roles;
- typed output roles;
- implementation digest source;
- policy digest source;
- invalidation descendants;
- qualification predicates.

The resolver returns:
```text
REUSE artifact_id
or
EXECUTE stage_contract_id
```

It must explain every decision.

Example:
```text
renderer implementation changed
  Stage01-41  REUSE
  Stage42-46  INVALID / EXECUTE
```

## Storage layout

Production logical keys:

```text
cas/sha256/ab/cd/<full_sha256>
```

No mutable aliases are required in object storage.
Human/logical names live in PostgreSQL.

Large artifacts use multipart upload through the provider SDK.
Presigned URLs may be used for Studio upload/download without proxying large bytes through the API service.

## Database correctness posture

- all foreign keys enabled;
- explicit unique constraints for idempotency and semantic artifact identity;
- append-only proof/promotion/audit records;
- no hard-delete of promoted revisions or referenced artifacts;
- retention/GC state separated from immutable artifact metadata;
- migrations are reviewed and reversible where technically meaningful;
- destructive data migrations are separate operational jobs, not hidden inside schema migration code;
- production backups and restore drills are acceptance requirements.

## Artifact garbage collection

Mark-and-sweep with a grace period.

Roots:
- current/promoted ProductRevisions;
- explicitly pinned ProductRevisions;
- retained Attempts;
- active workflow executions;
- legal/audit retention pins.

Unreachable object deletion is a separate auditable operation. DB metadata tombstones remain sufficient for forensic explanation.

## Security boundaries

Initial system is single-organization but must be OIDC-ready.
- credentials never stored in Artifact metadata;
- object store uses least-privilege service credentials;
- Studio uses presigned artifact access where appropriate;
- worker capability is separated by task queue (model/compiler/runtime);
- promotion permission is separate from research execution permission.

## Developer / product separation

### Forge
Allowed:
- create Attempts;
- invalidate subgraphs;
- run individual stages/courts;
- train/fit models;
- change candidate policies/checkpoints;
- inspect raw lineage/proofs;
- request explicit promotion.

### Studio
Allowed:
- import Subject;
- compile using promoted model/policy releases;
- edit supported puppet state;
- select motion;
- preview/render/export.

Forbidden in product mode:
- fit/train;
- policy calibration;
- research promotion;
- implicit lineage switching.

## Repository shape target

```text
/engine        existing scientific/compiler/runtime packages
/platform
  /api
  /domain
  /persistence
  /artifacts
  /workflows
  /workers
  /observability
  /migrations
/forge
/studio
/infra
/tests/platform
/docs/platform
```

This is one repository and one Engine, not duplicate product/research pipelines.

## Acceptance bar

Platform V1 is not accepted until all are mechanically demonstrated:
1. identical render -> identical ProductRevision and semantic render identity;
2. render invokes zero fit/train adapters;
3. new workflow ID reuses existing semantic artifacts;
4. runtime-only change reuses upstream artifacts;
5. research Attempt cannot mutate SubjectCurrent;
6. promotion is atomic and audited;
7. worker crash resumes without reconstructing valid upstream work;
8. any output expands to exact hashes, producer/policy versions and proof records;
9. object-store loss/missing byte is detected before use;
10. PostgreSQL backup can be restored and registry integrity re-audited.

## Integration guard

Canonical recovery is still open on motion qualification and broad exact-head recertification. Platform implementation may be designed/scaffolded in isolation, but production integration must start from the proven canonical recovery head rather than encoding today's transient branch topology.
