-- +goose Up

-- Public product traffic must never choose an EngineRelease implicitly by
-- creation time and must never be allowed to supply an EngineRelease id. The
-- operator explicitly pins the one PRODUCT release admitted for public use.
CREATE TABLE product_release_channels (
    channel varchar(32) PRIMARY KEY CHECK (channel = 'PRODUCTION'),
    engine_release_id uuid NOT NULL REFERENCES engine_releases(id) ON DELETE RESTRICT,
    lock_version bigint NOT NULL DEFAULT 0 CHECK (lock_version >= 0),
    updated_by varchar(300) NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION realsas_enforce_public_product_release_channel()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    release_purpose varchar(20);
BEGIN
    SELECT purpose INTO release_purpose
    FROM engine_releases
    WHERE id = NEW.engine_release_id;

    IF release_purpose IS DISTINCT FROM 'PRODUCT' THEN
        RAISE EXCEPTION 'PUBLIC_PRODUCT_RELEASE_MUST_BE_PRODUCT';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM engine_release_source_bindings b
        WHERE b.release_id = NEW.engine_release_id
          AND b.authority = 'CANONICAL_MAIN'
    ) THEN
        RAISE EXCEPTION 'PUBLIC_PRODUCT_RELEASE_CANONICAL_MAIN_REQUIRED';
    END IF;

    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_public_product_release_channel
BEFORE INSERT OR UPDATE OF engine_release_id ON product_release_channels
FOR EACH ROW
EXECUTE FUNCTION realsas_enforce_public_product_release_channel();

-- A public compile id is deliberately a separate identity from command,
-- attempt, artifact, release and ProductRevision ids. The browser only sees
-- public_id. Internal authority remains fully relational and immutable.
CREATE TABLE public_product_compile_sessions (
    id uuid PRIMARY KEY,
    public_id varchar(96) NOT NULL UNIQUE,
    subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE RESTRICT,
    engine_release_id uuid NOT NULL REFERENCES engine_releases(id) ON DELETE RESTRICT,
    release_channel_lock_version bigint NOT NULL CHECK (release_channel_lock_version >= 0),
    source_observation_artifact_id uuid REFERENCES artifacts(id) ON DELETE RESTRICT,
    attempt_id uuid UNIQUE REFERENCES attempts(id) ON DELETE RESTRICT,
    product_revision_id uuid UNIQUE REFERENCES product_revisions(id) ON DELETE RESTRICT,
    state varchar(48) NOT NULL CHECK (state IN (
        'WAITING_SOURCE_MATERIALIZATION',
        'QUEUED',
        'RUNNING',
        'PASS',
        'FAILED'
    )),
    current_capability varchar(100),
    error_code varchar(200),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((attempt_id IS NULL) = (source_observation_artifact_id IS NULL)),
    CHECK (product_revision_id IS NULL OR attempt_id IS NOT NULL)
);
CREATE INDEX ix_public_product_compile_sessions_subject_created
    ON public_product_compile_sessions(subject_id, created_at DESC);
CREATE INDEX ix_public_product_compile_sessions_attempt
    ON public_product_compile_sessions(attempt_id)
    WHERE attempt_id IS NOT NULL;

-- Raw user images are external evidence only. This binding does not qualify,
-- normalize, mirror, synthesize or promote them to SourceObservation. A later
-- intake materializer must produce the qualified fixed-view observation that
-- product.input consumes.
CREATE TABLE public_product_compile_views (
    session_id uuid NOT NULL REFERENCES public_product_compile_sessions(id) ON DELETE RESTRICT,
    slot varchar(16) NOT NULL CHECK (slot IN ('front','back')),
    artifact_id uuid NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
    mime_type varchar(80) NOT NULL CHECK (mime_type IN ('image/png','image/jpeg','image/webp')),
    original_name varchar(300) NOT NULL,
    PRIMARY KEY (session_id, slot),
    CONSTRAINT uq_public_product_compile_view_artifact UNIQUE (session_id, artifact_id)
);

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION realsas_public_product_session_identity_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.id IS DISTINCT FROM OLD.id
       OR NEW.public_id IS DISTINCT FROM OLD.public_id
       OR NEW.subject_id IS DISTINCT FROM OLD.subject_id
       OR NEW.engine_release_id IS DISTINCT FROM OLD.engine_release_id
       OR NEW.release_channel_lock_version IS DISTINCT FROM OLD.release_channel_lock_version
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'PUBLIC_PRODUCT_COMPILE_SESSION_IDENTITY_IMMUTABLE';
    END IF;
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_public_product_session_identity_immutable
BEFORE UPDATE ON public_product_compile_sessions
FOR EACH ROW
EXECUTE FUNCTION realsas_public_product_session_identity_immutable();

-- +goose Down

DROP TRIGGER trg_public_product_session_identity_immutable ON public_product_compile_sessions;
DROP FUNCTION realsas_public_product_session_identity_immutable();
DROP TABLE public_product_compile_views;
DROP INDEX ix_public_product_compile_sessions_attempt;
DROP INDEX ix_public_product_compile_sessions_subject_created;
DROP TABLE public_product_compile_sessions;
DROP TRIGGER trg_public_product_release_channel ON product_release_channels;
DROP FUNCTION realsas_enforce_public_product_release_channel();
DROP TABLE product_release_channels;
