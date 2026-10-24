-- +goose Up

-- ModelRelease provenance must be anchored to the exact canonical-main source
-- admitted by its EngineRelease. A client-supplied 40-char SHA is not authority.
-- This closes the remaining gap between EngineRelease source truth and model
-- lifecycle state: legacy-frozen or unrelated SHAs may be preserved elsewhere,
-- but they cannot mint a new ModelRelease.

-- Fail closed if pre-existing rows already violate the invariant. We do not
-- silently backfill CANONICAL_MAIN authority because that would manufacture
-- provenance that Platform never observed.
-- +goose StatementBegin
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM model_releases mr
        LEFT JOIN engine_release_source_bindings b
          ON b.release_id = mr.engine_release_id
         AND b.code_sha = mr.source_code_sha
         AND b.authority = 'CANONICAL_MAIN'
        WHERE b.release_id IS NULL
    ) THEN
        RAISE EXCEPTION 'MODEL_RELEASE_EXISTING_CANONICAL_SOURCE_VIOLATION';
    END IF;
END;
$$;
-- +goose StatementEnd

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION enforce_model_release_canonical_source()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM engine_release_source_bindings b
        WHERE b.release_id = NEW.engine_release_id
          AND b.code_sha = NEW.source_code_sha
          AND b.authority = 'CANONICAL_MAIN'
    ) THEN
        RAISE EXCEPTION 'MODEL_RELEASE_CANONICAL_MAIN_SOURCE_REQUIRED';
    END IF;

    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_model_release_canonical_source
BEFORE INSERT OR UPDATE OF engine_release_id, source_code_sha ON model_releases
FOR EACH ROW
EXECUTE FUNCTION enforce_model_release_canonical_source();

-- +goose Down

DROP TRIGGER trg_model_release_canonical_source ON model_releases;
DROP FUNCTION enforce_model_release_canonical_source();
