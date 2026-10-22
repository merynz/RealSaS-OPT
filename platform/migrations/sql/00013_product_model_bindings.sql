-- +goose Up

-- P3 product inference authority: a product EngineRelease may declare the model
-- families it requires. A compile snapshots exact channel selections once, and
-- a sealed ProductRevision preserves that immutable model provenance.
CREATE TABLE engine_release_model_requirements (
    release_id uuid NOT NULL REFERENCES engine_releases(id) ON DELETE RESTRICT,
    family varchar(80) NOT NULL,
    architecture_id varchar(300) NOT NULL,
    channel varchar(32) NOT NULL CHECK (channel IN ('RESEARCH','CANDIDATE','PRODUCTION')),
    PRIMARY KEY (release_id, family)
);

CREATE TABLE attempt_model_releases (
    attempt_id uuid NOT NULL REFERENCES attempts(id) ON DELETE RESTRICT,
    family varchar(80) NOT NULL,
    model_release_id uuid NOT NULL REFERENCES model_releases(id) ON DELETE RESTRICT,
    selected_channel varchar(32) NOT NULL CHECK (selected_channel IN ('RESEARCH','CANDIDATE','PRODUCTION')),
    selected_channel_lock_version bigint NOT NULL CHECK (selected_channel_lock_version >= 0),
    PRIMARY KEY (attempt_id, family)
);
CREATE INDEX ix_attempt_model_releases_release
    ON attempt_model_releases(model_release_id);

CREATE TABLE product_revision_model_releases (
    product_revision_id uuid NOT NULL REFERENCES product_revisions(id) ON DELETE RESTRICT,
    family varchar(80) NOT NULL,
    model_release_id uuid NOT NULL REFERENCES model_releases(id) ON DELETE RESTRICT,
    PRIMARY KEY (product_revision_id, family)
);
CREATE INDEX ix_product_revision_model_releases_release
    ON product_revision_model_releases(model_release_id);

-- +goose Down

DROP INDEX ix_product_revision_model_releases_release;
DROP TABLE product_revision_model_releases;
DROP INDEX ix_attempt_model_releases_release;
DROP TABLE attempt_model_releases;
DROP TABLE engine_release_model_requirements;
