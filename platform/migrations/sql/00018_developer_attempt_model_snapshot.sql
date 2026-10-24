-- +goose Up

-- Developer capability runs may execute the same sealed product capability
-- closure as a compile candidate, but with promotion policy NEVER.  They still
-- need the exact EngineRelease ModelRelease selection frozen at Attempt
-- acceptance; capability execution must never resolve live model channels.
-- Generic developer releases with no model requirements remain unchanged.
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
    IF NEW.kind NOT IN ('compile_candidate', 'developer') THEN
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

-- +goose Down

-- Restore the migration-13 behavior: only compile candidates snapshot models.
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
