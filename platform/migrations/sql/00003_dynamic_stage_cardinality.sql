-- +goose Up

ALTER TABLE engine_release_stages
    DROP CONSTRAINT IF EXISTS engine_release_stages_ordinal_check;

ALTER TABLE engine_release_stages
    ADD CONSTRAINT ck_engine_release_stage_ordinal_positive
    CHECK (ordinal >= 1);

-- +goose Down

ALTER TABLE engine_release_stages
    DROP CONSTRAINT IF EXISTS ck_engine_release_stage_ordinal_positive;

ALTER TABLE engine_release_stages
    ADD CONSTRAINT engine_release_stages_ordinal_check
    CHECK (ordinal BETWEEN 1 AND 46);
