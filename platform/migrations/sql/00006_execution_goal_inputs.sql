-- +goose Up

CREATE TABLE execution_goal_inputs (
    execution_goal_id uuid NOT NULL REFERENCES execution_goals(id) ON DELETE RESTRICT,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    artifact_id uuid NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
    PRIMARY KEY (execution_goal_id, ordinal),
    CONSTRAINT uq_execution_goal_input_artifact UNIQUE (execution_goal_id, artifact_id)
);
CREATE INDEX ix_execution_goal_inputs_artifact
    ON execution_goal_inputs(artifact_id);

-- +goose Down

DROP TABLE execution_goal_inputs;
