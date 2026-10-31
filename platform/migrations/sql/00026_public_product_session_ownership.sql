-- +goose Up

-- Browser/user ownership is distinct from the Product Gateway service identity.
-- The public shell keeps the raw high-entropy browser session secret in an
-- HttpOnly cookie and sends only its SHA-256 digest to Platform. Operator-only
-- exact SourceObservation compiles deliberately keep this column NULL and are
-- therefore not addressable through user-scoped public routes.
ALTER TABLE public_product_compile_sessions
    ADD COLUMN owner_key_sha256 char(64);

ALTER TABLE public_product_compile_sessions
    ADD CONSTRAINT ck_public_product_owner_key_sha256
    CHECK (
        owner_key_sha256 IS NULL
        OR owner_key_sha256 ~ '^[0-9a-f]{64}$'
    );

CREATE INDEX ix_public_product_compile_sessions_owner_created
    ON public_product_compile_sessions(owner_key_sha256, created_at DESC)
    WHERE owner_key_sha256 IS NOT NULL;

-- Ownership is part of the immutable public compile identity. A compile may
-- never be transferred by updating the session row; future account linking
-- must use an explicit, audited ownership layer instead.
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
       OR NEW.owner_key_sha256 IS DISTINCT FROM OLD.owner_key_sha256
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'PUBLIC_PRODUCT_COMPILE_SESSION_IDENTITY_IMMUTABLE';
    END IF;
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

-- +goose Down

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

DROP INDEX ix_public_product_compile_sessions_owner_created;
ALTER TABLE public_product_compile_sessions DROP CONSTRAINT ck_public_product_owner_key_sha256;
ALTER TABLE public_product_compile_sessions DROP COLUMN owner_key_sha256;
