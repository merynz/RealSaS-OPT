-- +goose Up

-- Registry semantic identity and immutable byte identity are different axes.
-- Exact identical bytes may be inputs/results of multiple typed contracts.
-- Keep uq_artifact_semantic_identity; the CAS store still verifies every hash.
ALTER TABLE artifacts DROP CONSTRAINT uq_artifact_content_storage;
CREATE INDEX ix_artifact_content_storage ON artifacts(content_sha256, storage_key);

-- +goose Down

-- PostgreSQL must reject this downgrade if semantic aliases now share bytes.
-- Never delete or collapse their qualifications, dependency edges or provenance.
ALTER TABLE artifacts ADD CONSTRAINT uq_artifact_content_storage UNIQUE (content_sha256, storage_key);
DROP INDEX ix_artifact_content_storage;
