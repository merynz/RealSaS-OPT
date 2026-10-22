-- +goose Up

-- Canonical source authority for executable EngineReleases.
--
-- New research execution is legal only when the candidate EngineRelease has an
-- explicit CANONICAL_MAIN binding to the exact code SHA currently deployed by
-- the Platform. Historical branch-bound work may be recorded separately as
-- LEGACY_FROZEN provenance, but that authority can never satisfy a new
-- research Attempt/Campaign execution gate.
--
-- EngineRelease identity is the semantic release manifest. The same release may
-- remain valid across multiple reviewed main revisions when its stage/policy/DAG
-- manifest is unchanged. Each execution still records one exact canonical code
-- SHA, so provenance is exact without forcing meaningless release reseals.

CREATE TABLE engine_release_source_bindings (
    release_id uuid NOT NULL REFERENCES engine_releases(id) ON DELETE RESTRICT,
    code_sha char(40) NOT NULL,
    authority varchar(24) NOT NULL CHECK (authority IN ('CANONICAL_MAIN','LEGACY_FROZEN')),
    bound_by varchar(200) NOT NULL,
    bound_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (release_id, code_sha),
    CHECK (code_sha ~ '^[0-9a-f]{40}$')
);

CREATE INDEX idx_engine_release_source_bindings_code_sha
    ON engine_release_source_bindings(code_sha, authority);

-- Historical Attempts remain readable. New research Attempts must carry the
-- deployed canonical code identity that admitted their EngineRelease.
ALTER TABLE attempts ADD COLUMN canonical_code_sha char(40);
ALTER TABLE attempts ADD CONSTRAINT attempts_canonical_code_sha_shape
    CHECK (canonical_code_sha IS NULL OR canonical_code_sha ~ '^[0-9a-f]{40}$');

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION enforce_research_attempt_canonical_source()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.kind <> 'research' THEN
        RETURN NEW;
    END IF;

    IF NEW.canonical_code_sha IS NULL THEN
        RAISE EXCEPTION 'RESEARCH_ATTEMPT_CANONICAL_SOURCE_REQUIRED';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM engine_release_source_bindings b
        WHERE b.release_id = NEW.engine_release_id
          AND b.code_sha = NEW.canonical_code_sha
          AND b.authority = 'CANONICAL_MAIN'
    ) THEN
        RAISE EXCEPTION 'RESEARCH_ATTEMPT_OFF_MAIN_SOURCE_FORBIDDEN';
    END IF;

    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_research_attempt_canonical_source
BEFORE INSERT ON attempts
FOR EACH ROW
EXECUTE FUNCTION enforce_research_attempt_canonical_source();

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION enforce_research_campaign_canonical_source()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM engine_release_source_bindings b
        WHERE b.release_id = NEW.candidate_engine_release_id
          AND b.code_sha = NEW.canonical_code_sha
          AND b.authority = 'CANONICAL_MAIN'
    ) THEN
        RAISE EXCEPTION 'RESEARCH_CAMPAIGN_OFF_MAIN_SOURCE_FORBIDDEN';
    END IF;

    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_research_campaign_canonical_source
BEFORE INSERT ON research_campaigns
FOR EACH ROW
EXECUTE FUNCTION enforce_research_campaign_canonical_source();

-- +goose Down

DROP TRIGGER trg_research_campaign_canonical_source ON research_campaigns;
DROP FUNCTION enforce_research_campaign_canonical_source();

DROP TRIGGER trg_research_attempt_canonical_source ON attempts;
DROP FUNCTION enforce_research_attempt_canonical_source();

ALTER TABLE attempts DROP CONSTRAINT attempts_canonical_code_sha_shape;
ALTER TABLE attempts DROP COLUMN canonical_code_sha;

DROP TABLE engine_release_source_bindings;
