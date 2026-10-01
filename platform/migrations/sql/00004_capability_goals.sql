-- +goose Up

ALTER TABLE engine_releases
    ADD COLUMN capability_set_sha256 char(64);

CREATE TABLE engine_release_capabilities (
    release_id uuid NOT NULL REFERENCES engine_releases(id) ON DELETE RESTRICT,
    capability_id varchar(300) NOT NULL,
    kind varchar(40) NOT NULL,
    owner_module_id varchar(300) NOT NULL,
    executor_activity varchar(300) NOT NULL,
    descriptor_sha256 char(64) NOT NULL,
    implementation_sha256 char(64) NOT NULL,
    policy_sha256 char(64) NOT NULL,
    parameters_sha256 char(64) NOT NULL,
    descriptor jsonb NOT NULL,
    PRIMARY KEY (release_id, capability_id)
);
CREATE INDEX ix_engine_release_capabilities_kind
    ON engine_release_capabilities(release_id, kind);

ALTER TABLE attempts
    ALTER COLUMN subject_id DROP NOT NULL;
ALTER TABLE attempts
    DROP CONSTRAINT IF EXISTS attempts_kind_check;
ALTER TABLE attempts
    DROP CONSTRAINT IF EXISTS ck_attempt_kind;
ALTER TABLE attempts
    DROP CONSTRAINT IF EXISTS ck_attempt_kind_dynamic;
ALTER TABLE attempts
    ADD CONSTRAINT ck_attempt_kind_dynamic
    CHECK (kind IN ('research','repair','compile_candidate','developer','proof'));

CREATE TABLE execution_goals (
    id uuid PRIMARY KEY,
    attempt_id uuid NOT NULL UNIQUE REFERENCES attempts(id) ON DELETE RESTRICT,
    goal_type varchar(50) NOT NULL,
    promotion_policy varchar(50) NOT NULL,
    target_capability_ids jsonb NOT NULL,
    resolved_capability_ids jsonb NOT NULL,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    spec_sha256 char(64) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_execution_goals_type_created
    ON execution_goals(goal_type, created_at);

-- +goose Down

DROP TABLE execution_goals;

ALTER TABLE attempts
    DROP CONSTRAINT IF EXISTS ck_attempt_kind_dynamic;
ALTER TABLE attempts
    ADD CONSTRAINT attempts_kind_check
    CHECK (kind IN ('research','repair','compile_candidate'));
ALTER TABLE attempts
    ALTER COLUMN subject_id SET NOT NULL;

DROP TABLE engine_release_capabilities;
ALTER TABLE engine_releases
    DROP COLUMN capability_set_sha256;
