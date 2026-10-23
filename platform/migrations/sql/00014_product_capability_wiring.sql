-- +goose Up

-- ProductRevision must identify the exact immutable execution wiring that
-- produced it. Historical revisions remain replayable even if they predate a
-- capability snapshot; every revision inserted after this migration is bound
-- fail-closed to its Attempt -> EngineRelease -> CapabilitySnapshot chain.

ALTER TABLE product_revisions
    ADD COLUMN engine_release_id uuid
    REFERENCES engine_releases(id) ON DELETE RESTRICT;

ALTER TABLE product_revisions
    ADD COLUMN capability_set_sha256 char(64);

-- Backfill historical rows whenever their originating release already has a
-- sealed capability snapshot. Rows predating that authority may remain NULL.
UPDATE product_revisions pr
SET engine_release_id = a.engine_release_id,
    capability_set_sha256 = er.capability_set_sha256
FROM attempts a
JOIN engine_releases er ON er.id = a.engine_release_id
WHERE pr.created_from_attempt_id = a.id
  AND a.engine_release_id IS NOT NULL
  AND er.capability_set_sha256 IS NOT NULL;

ALTER TABLE product_revisions
    ADD CONSTRAINT ck_product_revision_release_wiring_pair
    CHECK (
        (engine_release_id IS NULL AND capability_set_sha256 IS NULL)
        OR
        (engine_release_id IS NOT NULL AND capability_set_sha256 IS NOT NULL)
    );

CREATE INDEX ix_product_revisions_engine_release
    ON product_revisions(engine_release_id)
    WHERE engine_release_id IS NOT NULL;

CREATE INDEX ix_product_revisions_capability_set
    ON product_revisions(capability_set_sha256)
    WHERE capability_set_sha256 IS NOT NULL;

-- +goose StatementBegin
CREATE FUNCTION bind_product_revision_release_wiring_v1()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    resolved_release_id uuid;
    resolved_capability_set_sha256 char(64);
BEGIN
    SELECT a.engine_release_id, er.capability_set_sha256
    INTO resolved_release_id, resolved_capability_set_sha256
    FROM attempts a
    LEFT JOIN engine_releases er ON er.id = a.engine_release_id
    WHERE a.id = NEW.created_from_attempt_id;

    IF resolved_release_id IS NULL THEN
        RAISE EXCEPTION 'PRODUCT_REVISION_ENGINE_RELEASE_BINDING_REQUIRED';
    END IF;
    IF resolved_capability_set_sha256 IS NULL THEN
        RAISE EXCEPTION 'PRODUCT_REVISION_CAPABILITY_SNAPSHOT_REQUIRED';
    END IF;
    IF NEW.engine_release_id IS NOT NULL
       AND NEW.engine_release_id <> resolved_release_id THEN
        RAISE EXCEPTION 'PRODUCT_REVISION_ENGINE_RELEASE_BINDING_DRIFT';
    END IF;
    IF NEW.capability_set_sha256 IS NOT NULL
       AND NEW.capability_set_sha256 <> resolved_capability_set_sha256 THEN
        RAISE EXCEPTION 'PRODUCT_REVISION_CAPABILITY_SET_BINDING_DRIFT';
    END IF;

    NEW.engine_release_id := resolved_release_id;
    NEW.capability_set_sha256 := resolved_capability_set_sha256;
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_product_revision_bind_release_wiring
BEFORE INSERT ON product_revisions
FOR EACH ROW
EXECUTE FUNCTION bind_product_revision_release_wiring_v1();

-- +goose StatementBegin
CREATE FUNCTION forbid_product_revision_release_wiring_mutation_v1()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.created_from_attempt_id IS DISTINCT FROM OLD.created_from_attempt_id
       OR NEW.engine_release_id IS DISTINCT FROM OLD.engine_release_id
       OR NEW.capability_set_sha256 IS DISTINCT FROM OLD.capability_set_sha256 THEN
        RAISE EXCEPTION 'PRODUCT_REVISION_RELEASE_WIRING_IMMUTABLE';
    END IF;
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_product_revision_release_wiring_immutable
BEFORE UPDATE OF created_from_attempt_id, engine_release_id, capability_set_sha256 ON product_revisions
FOR EACH ROW
EXECUTE FUNCTION forbid_product_revision_release_wiring_mutation_v1();

-- +goose Down

DROP TRIGGER trg_product_revision_release_wiring_immutable ON product_revisions;
DROP FUNCTION forbid_product_revision_release_wiring_mutation_v1();
DROP TRIGGER trg_product_revision_bind_release_wiring ON product_revisions;
DROP FUNCTION bind_product_revision_release_wiring_v1();
DROP INDEX ix_product_revisions_capability_set;
DROP INDEX ix_product_revisions_engine_release;
ALTER TABLE product_revisions DROP CONSTRAINT ck_product_revision_release_wiring_pair;
ALTER TABLE product_revisions DROP COLUMN capability_set_sha256;
ALTER TABLE product_revisions DROP COLUMN engine_release_id;
