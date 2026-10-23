-- +goose Up

-- Historical engineering witnesses predate ProductRevision/ModelRelease wiring.
-- They are preserved as immutable CAS artifacts and may serve only as a
-- regression baseline while no promoted ProductRevision exists for the subject.
-- This is a migration bridge, not a second product authority system.
CREATE TABLE legacy_engineering_witnesses (
    id uuid PRIMARY KEY,
    subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE RESTRICT,
    witness_artifact_id uuid NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
    source_ref text NOT NULL,
    source_sha256 char(64) NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    superseded_by_product_revision_id uuid REFERENCES product_revisions(id) ON DELETE RESTRICT,
    created_by varchar(300) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_legacy_engineering_witness_identity UNIQUE (subject_id, witness_artifact_id)
);

CREATE UNIQUE INDEX uq_legacy_engineering_witness_active_subject
    ON legacy_engineering_witnesses(subject_id)
    WHERE superseded_by_product_revision_id IS NULL;

-- +goose StatementBegin
CREATE FUNCTION validate_legacy_engineering_witness_v1()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    artifact_content_sha char(64);
    artifact_type_name varchar(300);
    artifact_schema varchar(100);
    current_revision uuid;
BEGIN
    SELECT a.content_sha256,t.name,t.schema_version
    INTO artifact_content_sha,artifact_type_name,artifact_schema
    FROM artifacts a
    JOIN artifact_types t ON t.id=a.artifact_type_id
    WHERE a.id=NEW.witness_artifact_id;

    IF artifact_content_sha IS NULL THEN
        RAISE EXCEPTION 'LEGACY_WITNESS_ARTIFACT_REQUIRED';
    END IF;
    IF artifact_content_sha <> NEW.source_sha256 THEN
        RAISE EXCEPTION 'LEGACY_WITNESS_CONTENT_SHA_MISMATCH';
    END IF;
    IF artifact_type_name <> 'RealSaS.ProductNearestWitnessInventory'
       OR artifact_schema <> 'v1' THEN
        RAISE EXCEPTION 'LEGACY_WITNESS_ARTIFACT_TYPE_MISMATCH:%:%', artifact_type_name, artifact_schema;
    END IF;

    SELECT product_revision_id
    INTO current_revision
    FROM subject_current_revision
    WHERE subject_id=NEW.subject_id;
    IF current_revision IS NOT NULL THEN
        NEW.superseded_by_product_revision_id := current_revision;
    END IF;
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_validate_legacy_engineering_witness
BEFORE INSERT OR UPDATE OF subject_id,witness_artifact_id,source_sha256
ON legacy_engineering_witnesses
FOR EACH ROW
EXECUTE FUNCTION validate_legacy_engineering_witness_v1();

-- A real promoted ProductRevision always outranks a legacy engineering witness.
-- Promotion automatically retires the bridge so callers cannot accidentally
-- keep using historical FIT/demo evidence as current product truth.
-- +goose StatementBegin
CREATE FUNCTION supersede_legacy_witness_on_product_promotion_v1()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE legacy_engineering_witnesses
    SET superseded_by_product_revision_id=NEW.product_revision_id
    WHERE subject_id=NEW.subject_id
      AND superseded_by_product_revision_id IS NULL;
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_supersede_legacy_witness_on_product_promotion
AFTER INSERT OR UPDATE OF product_revision_id
ON subject_current_revision
FOR EACH ROW
EXECUTE FUNCTION supersede_legacy_witness_on_product_promotion_v1();

-- +goose Down

DROP TRIGGER trg_supersede_legacy_witness_on_product_promotion ON subject_current_revision;
DROP FUNCTION supersede_legacy_witness_on_product_promotion_v1();
DROP TRIGGER trg_validate_legacy_engineering_witness ON legacy_engineering_witnesses;
DROP FUNCTION validate_legacy_engineering_witness_v1();
DROP INDEX uq_legacy_engineering_witness_active_subject;
DROP TABLE legacy_engineering_witnesses;
