-- +goose Up

-- Studio edits are not ProductRevision authority. One Studio project is anchored
-- to one proof-qualified immutable ProductRevision and carries a separate
-- append-only authoring history. Requalification may later produce a new
-- ProductRevision, but Studio never mutates the sealed base revision in place.
CREATE TABLE studio_projects (
    id uuid PRIMARY KEY,
    public_id varchar(64) NOT NULL UNIQUE,
    source_compile_session_id uuid NOT NULL UNIQUE REFERENCES public_product_compile_sessions(id) ON DELETE RESTRICT,
    product_revision_id uuid NOT NULL REFERENCES product_revisions(id) ON DELETE RESTRICT,
    display_name varchar(200) NOT NULL,
    state varchar(20) NOT NULL DEFAULT 'ACTIVE' CHECK (state IN ('ACTIVE','ARCHIVED')),
    created_by varchar(300) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_studio_projects_revision ON studio_projects(product_revision_id);

CREATE TABLE studio_authoring_revisions (
    id uuid PRIMARY KEY,
    project_id uuid NOT NULL REFERENCES studio_projects(id) ON DELETE RESTRICT,
    revision_number bigint NOT NULL CHECK (revision_number > 0),
    parent_revision_id uuid,
    source_product_revision_id uuid NOT NULL REFERENCES product_revisions(id) ON DELETE RESTRICT,
    source_manifest_sha256 char(64) NOT NULL,
    payload_sha256 char(64) NOT NULL,
    authoring_payload jsonb NOT NULL,
    requires_requalification boolean NOT NULL DEFAULT true CHECK (requires_requalification),
    created_by varchar(300) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_studio_authoring_revision_number UNIQUE (project_id, revision_number),
    CONSTRAINT uq_studio_authoring_revision_identity UNIQUE (id, project_id),
    CONSTRAINT fk_studio_authoring_parent_same_project
        FOREIGN KEY (parent_revision_id, project_id)
        REFERENCES studio_authoring_revisions(id, project_id)
        ON DELETE RESTRICT
);
CREATE INDEX ix_studio_authoring_revisions_project_created
    ON studio_authoring_revisions(project_id, created_at DESC);

CREATE TABLE studio_project_heads (
    project_id uuid PRIMARY KEY REFERENCES studio_projects(id) ON DELETE RESTRICT,
    authoring_revision_id uuid,
    lock_version bigint NOT NULL DEFAULT 0 CHECK (lock_version >= 0),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT fk_studio_project_head_same_project
        FOREIGN KEY (authoring_revision_id, project_id)
        REFERENCES studio_authoring_revisions(id, project_id)
        ON DELETE RESTRICT
);

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION realsas_studio_project_identity_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.id IS DISTINCT FROM OLD.id
       OR NEW.public_id IS DISTINCT FROM OLD.public_id
       OR NEW.source_compile_session_id IS DISTINCT FROM OLD.source_compile_session_id
       OR NEW.product_revision_id IS DISTINCT FROM OLD.product_revision_id
       OR NEW.created_by IS DISTINCT FROM OLD.created_by
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'STUDIO_PROJECT_IDENTITY_IMMUTABLE';
    END IF;
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_studio_project_identity_immutable
BEFORE UPDATE ON studio_projects
FOR EACH ROW
EXECUTE FUNCTION realsas_studio_project_identity_immutable();

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION realsas_studio_authoring_revision_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'STUDIO_AUTHORING_REVISION_IMMUTABLE';
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_studio_authoring_revision_immutable
BEFORE UPDATE OR DELETE ON studio_authoring_revisions
FOR EACH ROW
EXECUTE FUNCTION realsas_studio_authoring_revision_immutable();

-- +goose Down

DROP TRIGGER trg_studio_authoring_revision_immutable ON studio_authoring_revisions;
DROP FUNCTION realsas_studio_authoring_revision_immutable();
DROP TRIGGER trg_studio_project_identity_immutable ON studio_projects;
DROP FUNCTION realsas_studio_project_identity_immutable();
DROP TABLE studio_project_heads;
DROP INDEX ix_studio_authoring_revisions_project_created;
DROP TABLE studio_authoring_revisions;
DROP INDEX ix_studio_projects_revision;
DROP TABLE studio_projects;
