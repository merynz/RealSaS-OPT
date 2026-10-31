-- +goose Up

-- Requalification is a distinct production Attempt class. It may reuse exact
-- qualified authority from one immutable base ProductRevision, but it can only
-- mint a new ProductRevision after the terminal Proof capability passes.
ALTER TABLE attempts DROP CONSTRAINT attempts_kind_check;
ALTER TABLE attempts ADD CONSTRAINT attempts_kind_check
    CHECK (kind IN ('research','repair','compile_candidate','requalification_candidate'));

CREATE TABLE studio_requalification_requests (
    id uuid PRIMARY KEY,
    public_id varchar(64) NOT NULL UNIQUE,
    project_id uuid NOT NULL REFERENCES studio_projects(id) ON DELETE RESTRICT,
    authoring_revision_id uuid NOT NULL REFERENCES studio_authoring_revisions(id) ON DELETE RESTRICT,
    base_product_revision_id uuid NOT NULL REFERENCES product_revisions(id) ON DELETE RESTRICT,
    authoring_artifact_id uuid NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
    attempt_id uuid NOT NULL UNIQUE REFERENCES attempts(id) ON DELETE RESTRICT,
    invalidated_domains jsonb NOT NULL,
    satisfied_capability_ids jsonb NOT NULL,
    state varchar(24) NOT NULL CHECK (state IN ('QUEUED','RUNNING','PASS','FAILED')),
    result_product_revision_id uuid UNIQUE REFERENCES product_revisions(id) ON DELETE RESTRICT,
    result_compile_session_id uuid UNIQUE REFERENCES public_product_compile_sessions(id) ON DELETE RESTRICT,
    failure_code varchar(200),
    created_by varchar(300) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_studio_requalification_revision UNIQUE (project_id, authoring_revision_id),
    CHECK (jsonb_typeof(invalidated_domains) = 'array' AND jsonb_array_length(invalidated_domains) > 0),
    CHECK (jsonb_typeof(satisfied_capability_ids) = 'array' AND jsonb_array_length(satisfied_capability_ids) > 0),
    CHECK ((state = 'PASS') = (result_product_revision_id IS NOT NULL AND result_compile_session_id IS NOT NULL)),
    CHECK (state <> 'FAILED' OR failure_code IS NOT NULL)
);
CREATE INDEX ix_studio_requalification_requests_project_created
    ON studio_requalification_requests(project_id, created_at DESC);
CREATE INDEX ix_studio_requalification_requests_state_created
    ON studio_requalification_requests(state, created_at)
    WHERE state IN ('QUEUED','RUNNING');

-- Request identity and its captured edit/base lineage are immutable. Only the
-- operational state and terminal result references may advance.
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION realsas_studio_requalification_identity_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.id IS DISTINCT FROM OLD.id
       OR NEW.public_id IS DISTINCT FROM OLD.public_id
       OR NEW.project_id IS DISTINCT FROM OLD.project_id
       OR NEW.authoring_revision_id IS DISTINCT FROM OLD.authoring_revision_id
       OR NEW.base_product_revision_id IS DISTINCT FROM OLD.base_product_revision_id
       OR NEW.authoring_artifact_id IS DISTINCT FROM OLD.authoring_artifact_id
       OR NEW.attempt_id IS DISTINCT FROM OLD.attempt_id
       OR NEW.invalidated_domains IS DISTINCT FROM OLD.invalidated_domains
       OR NEW.satisfied_capability_ids IS DISTINCT FROM OLD.satisfied_capability_ids
       OR NEW.created_by IS DISTINCT FROM OLD.created_by
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'STUDIO_REQUALIFICATION_IDENTITY_IMMUTABLE';
    END IF;
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_studio_requalification_identity_immutable
BEFORE UPDATE ON studio_requalification_requests
FOR EACH ROW
EXECUTE FUNCTION realsas_studio_requalification_identity_immutable();

-- +goose Down

DROP TRIGGER trg_studio_requalification_identity_immutable ON studio_requalification_requests;
DROP FUNCTION realsas_studio_requalification_identity_immutable();
DROP INDEX ix_studio_requalification_requests_state_created;
DROP INDEX ix_studio_requalification_requests_project_created;
DROP TABLE studio_requalification_requests;
ALTER TABLE attempts DROP CONSTRAINT attempts_kind_check;
ALTER TABLE attempts ADD CONSTRAINT attempts_kind_check
    CHECK (kind IN ('research','repair','compile_candidate'));
