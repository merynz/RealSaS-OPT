-- +goose Up

CREATE TABLE product_contracts (
    id uuid PRIMARY KEY,
    name varchar(200) NOT NULL,
    version varchar(100) NOT NULL,
    contract_sha256 char(64) NOT NULL UNIQUE,
    contract jsonb NOT NULL,
    created_by varchar(300) NOT NULL,
    sealed_at timestamptz NOT NULL,
    CONSTRAINT uq_product_contract_name_version UNIQUE (name, version)
);

ALTER TABLE product_revisions
    ADD COLUMN product_contract_id uuid
    REFERENCES product_contracts(id) ON DELETE RESTRICT;

CREATE INDEX ix_product_revisions_contract
    ON product_revisions(product_contract_id, sealed_at);

-- +goose Down

ALTER TABLE product_revisions
    DROP COLUMN product_contract_id;
DROP TABLE product_contracts;
