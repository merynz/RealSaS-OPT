# RealSaS Platform Data Model V1

> Reference relational model for the professional backend. PostgreSQL is metadata/product-state truth; artifact bytes live outside the database.

## Identity principles

- Database entity IDs: UUID.
- Artifact byte identity: SHA-256.
- Artifact semantic identity: SHA-256 over canonical semantic descriptor.
- Workflow/execution IDs are provenance, never semantic artifact identity.
- ProductRevision and its artifact membership are immutable once sealed.

## Core relational model

### subjects
```text
id uuid PK
slug text UNIQUE NOT NULL
display_name text NOT NULL
created_at timestamptz NOT NULL
archived_at timestamptz NULL
```

### artifact_types
```text
id uuid PK
name text NOT NULL
schema_version text NOT NULL
domain text NOT NULL
UNIQUE(name, schema_version)
```

Examples:
`RealSaS.QualifiedMeshIR.v1`,
`RealSaS.QualifiedVisualPresentationSetIR.v1`,
`RealSaS.RuntimePackage.v2`.

### artifacts
```text
id uuid PK
artifact_type_id uuid FK artifact_types
semantic_sha256 char(64) NOT NULL
content_sha256 char(64) NOT NULL
storage_key text NOT NULL
size_bytes bigint NOT NULL CHECK(size_bytes >= 0)
producer_contract text NOT NULL
implementation_sha256 char(64) NOT NULL
policy_sha256 char(64) NOT NULL
semantic_parameters jsonb NOT NULL
created_at timestamptz NOT NULL
verified_at timestamptz NOT NULL
UNIQUE(artifact_type_id, semantic_sha256)
UNIQUE(content_sha256, storage_key)
```

No update of semantic/content fields after insert.

### artifact_inputs
```text
artifact_id uuid FK artifacts
input_artifact_id uuid FK artifacts
input_role text NOT NULL
ordinal integer NOT NULL
PRIMARY KEY(artifact_id, input_role, ordinal)
```

### attempts
```text
id uuid PK
subject_id uuid FK subjects
parent_attempt_id uuid FK attempts NULL
kind text NOT NULL CHECK(kind IN ('research','repair','compile_candidate'))
spec_sha256 char(64) NOT NULL
created_by text NOT NULL
created_at timestamptz NOT NULL
final_state text NOT NULL
```

Attempt spec is immutable. State transition history is append-only in `attempt_events`.

### attempt_events
```text
id bigint identity PK
attempt_id uuid FK attempts
event_type text NOT NULL
payload jsonb NOT NULL
created_at timestamptz NOT NULL
```

### attempt_artifacts
```text
attempt_id uuid FK attempts
role text NOT NULL
artifact_id uuid FK artifacts
origin text NOT NULL CHECK(origin IN ('inherited','produced'))
PRIMARY KEY(attempt_id, role)
```

### executions
```text
id uuid PK
attempt_id uuid FK attempts NULL
workflow_id text NOT NULL
workflow_run_id text NULL
stage_contract text NOT NULL
status text NOT NULL
worker_identity text NULL
started_at timestamptz NULL
finished_at timestamptz NULL
retry_number integer NOT NULL DEFAULT 0
error_code text NULL
error_payload jsonb NULL
UNIQUE(workflow_id, stage_contract, retry_number)
```

### execution_artifacts
```text
execution_id uuid FK executions
artifact_id uuid FK artifacts
relation text NOT NULL CHECK(relation IN ('input','output'))
role text NOT NULL
PRIMARY KEY(execution_id, relation, role, artifact_id)
```

### proofs
```text
id uuid PK
proof_type text NOT NULL
subject_artifact_id uuid FK artifacts NULL
policy_sha256 char(64) NOT NULL
implementation_sha256 char(64) NOT NULL
result text NOT NULL CHECK(result IN ('PASS','FAIL','ABSTAIN'))
report_artifact_id uuid FK artifacts NULL
created_at timestamptz NOT NULL
```

### qualifications
```text
id uuid PK
artifact_id uuid FK artifacts
qualification_type text NOT NULL
result text NOT NULL CHECK(result IN ('PASS','FAIL','ABSTAIN'))
proof_id uuid FK proofs NULL
created_at timestamptz NOT NULL
UNIQUE(artifact_id, qualification_type, proof_id)
```

### product_revisions
```text
id uuid PK
subject_id uuid FK subjects
revision_number bigint NOT NULL
manifest_sha256 char(64) NOT NULL
created_from_attempt_id uuid FK attempts
sealed_at timestamptz NOT NULL
UNIQUE(subject_id, revision_number)
UNIQUE(subject_id, manifest_sha256)
```

No mutation after insert.

### product_revision_artifacts
```text
product_revision_id uuid FK product_revisions
role text NOT NULL
artifact_id uuid FK artifacts
PRIMARY KEY(product_revision_id, role)
```

Required roles are validated by domain service before seal and again before promotion.

### promotions
```text
id uuid PK
subject_id uuid FK subjects
from_revision_id uuid FK product_revisions NULL
to_revision_id uuid FK product_revisions
reason text NOT NULL
requested_by text NOT NULL
qualification_snapshot jsonb NOT NULL
created_at timestamptz NOT NULL
```

Append-only.

### subject_current_revision
```text
subject_id uuid PK FK subjects
product_revision_id uuid FK product_revisions
lock_version bigint NOT NULL
updated_at timestamptz NOT NULL
```

Only promotion transaction may update this table.

### render_requests
```text
id uuid PK
subject_id uuid FK subjects
product_revision_id uuid FK product_revisions
motion_artifact_id uuid FK artifacts
view_spec jsonb NOT NULL
render_settings jsonb NOT NULL
semantic_sha256 char(64) NOT NULL UNIQUE
idempotency_key text NOT NULL UNIQUE
created_at timestamptz NOT NULL
```

### render_outputs
```text
render_request_id uuid FK render_requests
artifact_id uuid FK artifacts
created_at timestamptz NOT NULL
PRIMARY KEY(render_request_id, artifact_id)
```

### commands
```text
id uuid PK
command_type text NOT NULL
subject_id uuid FK subjects NULL
idempotency_key text NOT NULL UNIQUE
payload jsonb NOT NULL
created_at timestamptz NOT NULL
```

### outbox_events
```text
id bigint identity PK
aggregate_type text NOT NULL
aggregate_id uuid NOT NULL
event_type text NOT NULL
payload jsonb NOT NULL
created_at timestamptz NOT NULL
delivered_at timestamptz NULL
delivery_attempts integer NOT NULL DEFAULT 0
```

### audit_events
```text
id bigint identity PK
actor text NOT NULL
action text NOT NULL
subject_id uuid NULL
attempt_id uuid NULL
product_revision_id uuid NULL
artifact_id uuid NULL
payload jsonb NOT NULL
created_at timestamptz NOT NULL
```

## Referential and immutability rules

1. Artifact inputs must exist before an Artifact becomes registry-visible.
2. ProductRevision artifact membership cannot change after seal.
3. Promoted ProductRevisions cannot be hard-deleted.
4. Artifacts reachable from promoted revisions cannot be hard-deleted.
5. Research Attempts cannot update `subject_current_revision`.
6. RenderRequest always stores exact ProductRevision ID.
7. Product mode rejects stage contracts classified as fit/train/calibrate/promote.
8. A proof/qualification result is never overwritten; a new evaluation inserts a new row.
9. Failed execution may retry, but successful semantic artifact reuse resolves by artifact semantic identity.
10. Outbox consumers must be idempotent.

## Serializable promotion algorithm

```text
BEGIN ISOLATION LEVEL SERIALIZABLE

SELECT current revision FOR UPDATE

validate target revision:
  - same subject
  - sealed
  - required artifact roles present
  - required qualifications PASS
  - no blocked compatibility relation

INSERT promotion

UPDATE subject_current_revision
SET product_revision_id = target,
    lock_version = lock_version + 1

INSERT audit_event
INSERT outbox_event

COMMIT
```

On serialization failure, retry transaction. Never partially promote.

## Artifact registration algorithm

```text
produce temp bytes
compute content_sha256
compute semantic descriptor + semantic_sha256

if registry already contains same semantic identity:
    verify compatibility/content policy
    reuse artifact
else:
    upload CAS object
    verify object
    BEGIN
      INSERT artifact
      INSERT artifact_inputs
      INSERT execution_artifacts
      INSERT audit event
    COMMIT
```

If DB commit fails after upload, object is orphaned and later garbage-collected.
If upload fails, no Artifact row becomes visible.

## Minimal invalidation query model

Each stage contract declares:
- input artifact roles;
- output artifact type;
- implementation identity;
- policy identity;
- semantic parameters.

Resolver computes expected `semantic_sha256`.
If that identity exists and is qualified, REUSE.
If not, EXECUTE.
No run/timestamp participates in the semantic digest.

This is the central mechanism that prevents full Stage01–46 replay.

## Retention

Separate immutable metadata from retention decisions.

Suggested later tables:
- `artifact_pins`
- `retention_policies`
- `gc_marks`
- `gc_runs`

GC must never infer liveness from filesystem directory names.

## Migration policy

- Alembic owns schema migration history.
- Migrations run once before application deployment, not independently from every API worker.
- Large/destructive data backfills run as explicit resumable jobs.
- Production migration gates include backup checkpoint + dry-run/verification.
