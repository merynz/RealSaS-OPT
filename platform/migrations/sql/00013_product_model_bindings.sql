-- +goose Up

-- P3 product inference authority: a product EngineRelease may declare the model
-- families it requires. A compile snapshots exact channel selections once, and
-- a sealed ProductRevision preserves that immutable model provenance.
CREATE TABLE engine_release_model_requirements (
    release_id uuid NOT NULL REFERENCES engine_releases(id) ON DELETE RESTRICT,
    family varchar(80) NOT NULL,
    architecture_id varchar(300) NOT NULL,
    channel varchar(32) NOT NULL CHECK (channel IN ('RESEARCH','CANDIDATE','PRODUCTION')),
    PRIMARY KEY (release_id, family)
);

CREATE TABLE attempt_model_releases (
    attempt_id uuid NOT NULL REFERENCES attempts(id) ON DELETE RESTRICT,
    family varchar(80) NOT NULL,
    model_release_id uuid NOT NULL REFERENCES model_releases(id) ON DELETE RESTRICT,
    selected_channel varchar(32) NOT NULL CHECK (selected_channel IN ('RESEARCH','CANDIDATE','PRODUCTION')),
    selected_channel_lock_version bigint NOT NULL CHECK (selected_channel_lock_version >= 0),
    PRIMARY KEY (attempt_id, family)
);
CREATE INDEX ix_attempt_model_releases_release
    ON attempt_model_releases(model_release_id);

CREATE TABLE product_revision_model_releases (
    product_revision_id uuid NOT NULL REFERENCES product_revisions(id) ON DELETE RESTRICT,
    family varchar(80) NOT NULL,
    model_release_id uuid NOT NULL REFERENCES model_releases(id) ON DELETE RESTRICT,
    PRIMARY KEY (product_revision_id, family)
);
CREATE INDEX ix_product_revision_model_releases_release
    ON product_revision_model_releases(model_release_id);

-- Snapshot required model channels in the exact transaction that creates the
-- product compile Attempt. Locking the selected channel row prevents a channel
-- promotion from racing the snapshot.
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION realsas_snapshot_product_attempt_models()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    requirement record;
    selected_model_release_id uuid;
    selected_architecture_id varchar(300);
    selected_lifecycle_state varchar(32);
    selected_lock_version bigint;
BEGIN
    IF NEW.kind <> 'compile_candidate' THEN
        RETURN NEW;
    END IF;

    FOR requirement IN
        SELECT family, architecture_id, channel
        FROM engine_release_model_requirements
        WHERE release_id = NEW.engine_release_id
        ORDER BY family
    LOOP
        SELECT mc.model_release_id, mc.lock_version,
               mr.architecture_id, mr.lifecycle_state
        INTO selected_model_release_id, selected_lock_version,
             selected_architecture_id, selected_lifecycle_state
        FROM model_channels mc
        JOIN model_releases mr ON mr.id = mc.model_release_id
        WHERE mc.family = requirement.family
          AND mc.channel = requirement.channel
        FOR SHARE OF mc, mr;

        IF NOT FOUND THEN
            RAISE EXCEPTION 'MODEL_SELECTION_CHANNEL_MISSING:%:%',
                requirement.family, requirement.channel;
        END IF;
        IF selected_architecture_id <> requirement.architecture_id THEN
            RAISE EXCEPTION 'MODEL_SELECTION_ARCHITECTURE_MISMATCH:%:%!=%',
                requirement.family, selected_architecture_id, requirement.architecture_id;
        END IF;
        IF requirement.channel = 'PRODUCTION'
           AND selected_lifecycle_state <> 'PRODUCTION' THEN
            RAISE EXCEPTION 'MODEL_SELECTION_NOT_PRODUCTION:%:%',
                requirement.family, selected_lifecycle_state;
        END IF;

        INSERT INTO attempt_model_releases(
            attempt_id, family, model_release_id,
            selected_channel, selected_channel_lock_version
        ) VALUES (
            NEW.id, requirement.family, selected_model_release_id,
            requirement.channel, selected_lock_version
        );
    END LOOP;
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_snapshot_product_attempt_models
AFTER INSERT ON attempts
FOR EACH ROW
EXECUTE FUNCTION realsas_snapshot_product_attempt_models();

-- +goose Down

DROP TRIGGER trg_snapshot_product_attempt_models ON attempts;
DROP FUNCTION realsas_snapshot_product_attempt_models();
DROP INDEX ix_product_revision_model_releases_release;
DROP TABLE product_revision_model_releases;
DROP INDEX ix_attempt_model_releases_release;
DROP TABLE attempt_model_releases;
DROP TABLE engine_release_model_requirements;
