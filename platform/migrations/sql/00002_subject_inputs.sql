-- +goose Up

CREATE TABLE subject_inputs (
    id uuid PRIMARY KEY,
    subject_id uuid NOT NULL REFERENCES subjects(id) ON DELETE RESTRICT,
    manifest_sha256 char(64) NOT NULL,
    created_by varchar(300) NOT NULL,
    sealed_at timestamptz NOT NULL,
    CONSTRAINT uq_subject_input_manifest UNIQUE (subject_id, manifest_sha256)
);

CREATE TABLE subject_input_artifacts (
    subject_input_id uuid NOT NULL REFERENCES subject_inputs(id) ON DELETE RESTRICT,
    role varchar(150) NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    artifact_id uuid NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
    PRIMARY KEY (subject_input_id, role),
    CONSTRAINT uq_subject_input_ordinal UNIQUE (subject_input_id, ordinal)
);

-- +goose Down

DROP TABLE subject_input_artifacts;
DROP TABLE subject_inputs;
