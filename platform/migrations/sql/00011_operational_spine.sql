-- +goose Up

-- P0 Operational Spine: make scientific intent and deployable model identity
-- first-class Platform state. Git remains source truth; these tables own durable
-- research/model lifecycle state and human-readable current-model channels.
CREATE TABLE experiments (
    id uuid PRIMARY KEY,
    slug varchar(200) NOT NULL UNIQUE,
    name varchar(300) NOT NULL,
    research_question text NOT NULL,
    hypothesis text NOT NULL,
    lifecycle_state varchar(32) NOT NULL CHECK (lifecycle_state IN (
        'DRAFT','PREREGISTERED','RUNNING','COMPLETED','SUPERSEDED','ABORTED'
    )),
    spec_sha256 char(64) NOT NULL UNIQUE,
    supersedes_experiment_id uuid REFERENCES experiments(id) ON DELETE RESTRICT,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_by varchar(300) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    preregistered_at timestamptz,
    completed_at timestamptz
);
CREATE INDEX ix_experiments_state_created ON experiments(lifecycle_state, created_at DESC);

CREATE TABLE experiment_artifacts (
    experiment_id uuid NOT NULL REFERENCES experiments(id) ON DELETE RESTRICT,
    role varchar(150) NOT NULL,
    artifact_id uuid NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
    PRIMARY KEY (experiment_id, role)
);

ALTER TABLE attempts
    ADD COLUMN experiment_id uuid REFERENCES experiments(id) ON DELETE RESTRICT;
CREATE INDEX ix_attempts_experiment_created ON attempts(experiment_id, created_at DESC)
    WHERE experiment_id IS NOT NULL;

ALTER TABLE research_campaigns
    ADD COLUMN experiment_id uuid REFERENCES experiments(id) ON DELETE RESTRICT;
CREATE INDEX ix_research_campaigns_experiment_created
    ON research_campaigns(experiment_id, created_at DESC)
    WHERE experiment_id IS NOT NULL;

CREATE TABLE model_releases (
    id uuid PRIMARY KEY,
    family varchar(80) NOT NULL,
    name varchar(300) NOT NULL,
    architecture_id varchar(300) NOT NULL,
    checkpoint_artifact_id uuid NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
    engine_release_id uuid NOT NULL REFERENCES engine_releases(id) ON DELETE RESTRICT,
    originating_experiment_id uuid REFERENCES experiments(id) ON DELETE RESTRICT,
    originating_attempt_id uuid REFERENCES attempts(id) ON DELETE RESTRICT,
    source_code_sha char(40) NOT NULL,
    lifecycle_state varchar(40) NOT NULL CHECK (lifecycle_state IN (
        'DRAFT','RESEARCH_QUALIFIED','PRODUCT_CANDIDATE','PRODUCTION','SUPERSEDED','REJECTED'
    )),
    supersedes_model_release_id uuid REFERENCES model_releases(id) ON DELETE RESTRICT,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_by varchar(300) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    sealed_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_model_release_family_name UNIQUE (family, name)
);
CREATE INDEX ix_model_releases_family_state_created
    ON model_releases(family, lifecycle_state, created_at DESC);
CREATE INDEX ix_model_releases_checkpoint ON model_releases(checkpoint_artifact_id);

CREATE TABLE model_release_evaluations (
    model_release_id uuid NOT NULL REFERENCES model_releases(id) ON DELETE RESTRICT,
    scope varchar(100) NOT NULL,
    result varchar(20) NOT NULL CHECK (result IN ('NOT_RUN','PASS','FAIL','ABSTAIN')),
    evidence_artifact_id uuid REFERENCES artifacts(id) ON DELETE RESTRICT,
    experiment_id uuid REFERENCES experiments(id) ON DELETE RESTRICT,
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    observed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model_release_id, scope),
    CHECK (result = 'NOT_RUN' OR evidence_artifact_id IS NOT NULL OR experiment_id IS NOT NULL)
);
CREATE INDEX ix_model_release_evaluations_scope_result
    ON model_release_evaluations(scope, result);

CREATE TABLE model_channels (
    family varchar(80) NOT NULL,
    channel varchar(32) NOT NULL CHECK (channel IN ('RESEARCH','CANDIDATE','PRODUCTION')),
    model_release_id uuid NOT NULL REFERENCES model_releases(id) ON DELETE RESTRICT,
    lock_version bigint NOT NULL DEFAULT 0 CHECK (lock_version >= 0),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (family, channel)
);

-- +goose Down

DROP TABLE model_channels;
DROP TABLE model_release_evaluations;
DROP TABLE model_releases;
DROP INDEX ix_research_campaigns_experiment_created;
ALTER TABLE research_campaigns DROP COLUMN experiment_id;
DROP INDEX ix_attempts_experiment_created;
ALTER TABLE attempts DROP COLUMN experiment_id;
DROP TABLE experiment_artifacts;
DROP TABLE experiments;
