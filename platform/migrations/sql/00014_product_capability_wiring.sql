-- New product revisions bind directly to the exact immutable execution wiring.
-- Columns remain nullable only so historical revisions sealed before this
-- migration remain replayable. Application sealing requires both values.

ALTER TABLE product_revisions
    ADD COLUMN engine_release_id uuid
    REFERENCES engine_releases(id) ON DELETE RESTRICT;

ALTER TABLE product_revisions
    ADD COLUMN capability_set_sha256 char(64);

ALTER TABLE product_revisions
    ADD CONSTRAINT ck_product_revision_release_wiring_pair
    CHECK (
        (engine_release_id IS NULL AND capability_set_sha256 IS NULL)
        OR
        (engine_release_id IS NOT NULL AND capability_set_sha256 IS NOT NULL)
    );

CREATE INDEX ix_product_revisions_engine_release
    ON product_revisions(engine_release_id)
    WHERE engine_release_id IS NOT NULL;

CREATE INDEX ix_product_revisions_capability_set
    ON product_revisions(capability_set_sha256)
    WHERE capability_set_sha256 IS NOT NULL;
