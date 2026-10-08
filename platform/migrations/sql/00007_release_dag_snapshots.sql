-- +goose Up
CREATE TABLE engine_release_graphs (
    release_id uuid PRIMARY KEY REFERENCES engine_releases(id) ON DELETE RESTRICT,
    graph_sha256 char(64) NOT NULL,
    pipeline_plan_sha256 varchar(64) NOT NULL DEFAULT '',
    snapshot jsonb NOT NULL,
    CHECK (pipeline_plan_sha256 = '' OR pipeline_plan_sha256 ~ '^[0-9a-f]{64}$')
);

-- +goose StatementBegin
CREATE FUNCTION reject_release_graph_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'sealed release DAG is immutable';
END;
$$ LANGUAGE plpgsql;
-- +goose StatementEnd
CREATE TRIGGER immutable_release_graph
BEFORE UPDATE OR DELETE ON engine_release_graphs
FOR EACH ROW EXECUTE FUNCTION reject_release_graph_mutation();

-- +goose Down
DROP TABLE engine_release_graphs;
DROP FUNCTION reject_release_graph_mutation();
