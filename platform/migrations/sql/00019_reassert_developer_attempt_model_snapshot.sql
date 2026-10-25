-- +goose Up

-- Reassert the acceptance-time model snapshot trigger for developer capability
-- runs. Migration 00018 introduced this semantic, but a live developer host was
-- observed accepting a fresh developer Attempt without freezing its required
-- ModelReleases. This migration is intentionally a forward-only schema repair:
-- it changes no historical Attempt rows, because resolving live channels after
-- acceptance would invent provenance. Future Attempt inserts must snapshot the
-- exact channel selections transactionally.
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

-- No historical rows were changed by Up. Migration 00018 owns the semantic
-- change, so reverting this reassertion is intentionally a no-op; rolling back
-- 00018 subsequently restores migration-13 compile-candidate-only behavior.
SELECT 1;
