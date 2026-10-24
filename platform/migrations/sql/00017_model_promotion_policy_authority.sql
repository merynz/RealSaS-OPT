-- +goose Up

-- Model-channel promotion is a Platform authority decision. A caller may ask to
-- promote a release, but it must not be able to choose the qualification exam
-- that makes the promotion legal. Promotion policy is therefore sealed as
-- immutable content and selected explicitly by a typed policy binding.
--
-- No production policies are seeded by this migration. Missing policy is a
-- fail-closed state until a reviewed policy is deliberately sealed and bound.
CREATE TABLE model_promotion_policies (
    id uuid PRIMARY KEY,
    family varchar(80) NOT NULL,
    architecture_id varchar(300) NOT NULL,
    channel varchar(32) NOT NULL CHECK (channel IN ('RESEARCH','CANDIDATE','PRODUCTION')),
    policy_version bigint NOT NULL CHECK (policy_version >= 1),
    required_evaluation_scopes text[] NOT NULL,
    policy_sha256 char(64) NOT NULL UNIQUE CHECK (policy_sha256 ~ '^[0-9a-f]{64}$'),
    sealed_by varchar(300) NOT NULL,
    sealed_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (family, architecture_id, channel, policy_version),
    UNIQUE (id, family, architecture_id, channel),
    CHECK (cardinality(required_evaluation_scopes) > 0)
);

-- The binding is the single current-policy pointer. Policy content itself is
-- immutable; a future reviewed operation may advance this pointer without
-- rewriting historical policy identity.
CREATE TABLE model_promotion_policy_bindings (
    family varchar(80) NOT NULL,
    architecture_id varchar(300) NOT NULL,
    channel varchar(32) NOT NULL CHECK (channel IN ('RESEARCH','CANDIDATE','PRODUCTION')),
    policy_id uuid NOT NULL,
    lock_version bigint NOT NULL DEFAULT 0 CHECK (lock_version >= 0),
    bound_by varchar(300) NOT NULL,
    bound_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (family, architecture_id, channel),
    FOREIGN KEY (policy_id, family, architecture_id, channel)
        REFERENCES model_promotion_policies(id, family, architecture_id, channel)
        ON DELETE RESTRICT
);

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION enforce_model_promotion_policy_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'MODEL_PROMOTION_POLICY_IMMUTABLE';
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_model_promotion_policy_immutable
BEFORE UPDATE OR DELETE ON model_promotion_policies
FOR EACH ROW
EXECUTE FUNCTION enforce_model_promotion_policy_immutable();

-- Validate policy shape once at seal time so promotion does not interpret
-- duplicate or blank scope names differently across callers.
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION validate_model_promotion_policy_scopes()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    scope_name text;
    distinct_count bigint;
BEGIN
    FOREACH scope_name IN ARRAY NEW.required_evaluation_scopes LOOP
        IF btrim(scope_name) = '' OR scope_name IS DISTINCT FROM btrim(scope_name) THEN
            RAISE EXCEPTION 'MODEL_PROMOTION_POLICY_SCOPE_INVALID';
        END IF;
    END LOOP;
    SELECT count(DISTINCT value) INTO distinct_count
    FROM unnest(NEW.required_evaluation_scopes) AS value;
    IF distinct_count <> cardinality(NEW.required_evaluation_scopes) THEN
        RAISE EXCEPTION 'MODEL_PROMOTION_POLICY_SCOPE_DUPLICATE';
    END IF;
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_validate_model_promotion_policy_scopes
BEFORE INSERT ON model_promotion_policies
FOR EACH ROW
EXECUTE FUNCTION validate_model_promotion_policy_scopes();

-- This trigger is the final authority boundary. Even if an API/client supplies
-- a weaker list of evaluation scopes, the model channel cannot be inserted or
-- repointed unless every scope selected by Platform policy is PASS.
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION enforce_model_channel_promotion_policy()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    release_architecture varchar(300);
    required_scopes text[];
    scope_name text;
    evaluation_result varchar(20);
BEGIN
    SELECT architecture_id INTO release_architecture
    FROM model_releases
    WHERE id = NEW.model_release_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'MODEL_PROMOTION_RELEASE_NOT_FOUND';
    END IF;

    SELECT p.required_evaluation_scopes INTO required_scopes
    FROM model_promotion_policy_bindings b
    JOIN model_promotion_policies p ON p.id = b.policy_id
    WHERE b.family = NEW.family
      AND b.architecture_id = release_architecture
      AND b.channel = NEW.channel;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'MODEL_PROMOTION_POLICY_MISSING:%:%:%',
            NEW.family, release_architecture, NEW.channel;
    END IF;

    FOREACH scope_name IN ARRAY required_scopes LOOP
        SELECT result INTO evaluation_result
        FROM model_release_evaluations
        WHERE model_release_id = NEW.model_release_id
          AND scope = scope_name;

        IF NOT FOUND THEN
            RAISE EXCEPTION 'MODEL_PROMOTION_POLICY_EVALUATION_MISSING:%', scope_name;
        END IF;
        IF evaluation_result <> 'PASS' THEN
            RAISE EXCEPTION 'MODEL_PROMOTION_POLICY_EVALUATION_NOT_PASS:%:%',
                scope_name, evaluation_result;
        END IF;
    END LOOP;

    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_enforce_model_channel_promotion_policy
BEFORE INSERT OR UPDATE OF model_release_id ON model_channels
FOR EACH ROW
EXECUTE FUNCTION enforce_model_channel_promotion_policy();

-- +goose Down

DROP TRIGGER trg_enforce_model_channel_promotion_policy ON model_channels;
DROP FUNCTION enforce_model_channel_promotion_policy();
DROP TRIGGER trg_validate_model_promotion_policy_scopes ON model_promotion_policies;
DROP FUNCTION validate_model_promotion_policy_scopes();
DROP TRIGGER trg_model_promotion_policy_immutable ON model_promotion_policies;
DROP FUNCTION enforce_model_promotion_policy_immutable();
DROP TABLE model_promotion_policy_bindings;
DROP TABLE model_promotion_policies;
