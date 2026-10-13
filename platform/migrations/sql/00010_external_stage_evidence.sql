-- +goose Up

-- External GPU/model execution evidence is raw immutable input to a released
-- stage. It is not a qualification and it never replaces the subject manifest.
CREATE TABLE external_stage_evidence (
    attempt_id uuid NOT NULL REFERENCES attempts(id) ON DELETE RESTRICT,
    stage_id varchar(100) NOT NULL,
    role varchar(150) NOT NULL,
    artifact_id uuid NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
    bound_by varchar(300) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (attempt_id, stage_id, role)
);
CREATE INDEX ix_external_stage_evidence_artifact
    ON external_stage_evidence(artifact_id, attempt_id, stage_id);

-- +goose Down

DROP TABLE external_stage_evidence;
