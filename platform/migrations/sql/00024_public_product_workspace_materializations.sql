-- +goose Up

-- ProductProof PASS seals ProductRevision authority, but that does not imply the
-- browser/editor delivery bundle already exists. Workspace materialization is a
-- deterministic derived job over one immutable ProductRevision.
CREATE TABLE public_product_workspace_materializations (
    session_id uuid PRIMARY KEY REFERENCES public_product_compile_sessions(id) ON DELETE RESTRICT,
    product_revision_id uuid NOT NULL UNIQUE REFERENCES product_revisions(id) ON DELETE RESTRICT,
    state varchar(24) NOT NULL CHECK (state IN ('QUEUED','LEASED','PASS','FAILED')),
    lease_token uuid,
    lease_owner varchar(300),
    lease_expires_at timestamptz,
    attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    bundle_artifact_id uuid UNIQUE REFERENCES artifacts(id) ON DELETE RESTRICT,
    last_error_code varchar(200),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (
        (state = 'LEASED' AND lease_token IS NOT NULL AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR
        (state <> 'LEASED' AND lease_token IS NULL AND lease_owner IS NULL AND lease_expires_at IS NULL)
    ),
    CHECK ((state = 'PASS') = (bundle_artifact_id IS NOT NULL))
);
CREATE INDEX ix_public_product_workspace_materializations_claim
    ON public_product_workspace_materializations(state, lease_expires_at, created_at);

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION realsas_public_product_workspace_materialization_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.session_id IS DISTINCT FROM OLD.session_id
       OR NEW.product_revision_id IS DISTINCT FROM OLD.product_revision_id
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'PUBLIC_PRODUCT_WORKSPACE_IDENTITY_IMMUTABLE';
    END IF;
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_public_product_workspace_materialization_updated_at
BEFORE UPDATE ON public_product_workspace_materializations
FOR EACH ROW
EXECUTE FUNCTION realsas_public_product_workspace_materialization_updated_at();

-- +goose Down

DROP TRIGGER trg_public_product_workspace_materialization_updated_at ON public_product_workspace_materializations;
DROP FUNCTION realsas_public_product_workspace_materialization_updated_at();
DROP INDEX ix_public_product_workspace_materializations_claim;
DROP TABLE public_product_workspace_materializations;
