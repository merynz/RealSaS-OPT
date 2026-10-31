-- +goose Up

-- Public source intake is intentionally a pre-compile job. Raw front/back
-- uploads are not SourceObservation authority. A qualified materializer leases
-- one job, produces an exact RealSaS.SourceObservation/v1 artifact, then hands
-- that artifact back to Platform before the shipping product workflow starts.
CREATE TABLE public_product_intake_jobs (
    session_id uuid PRIMARY KEY REFERENCES public_product_compile_sessions(id) ON DELETE RESTRICT,
    state varchar(24) NOT NULL CHECK (state IN ('QUEUED','LEASED','PASS','FAILED')),
    lease_token uuid,
    lease_owner varchar(300),
    lease_expires_at timestamptz,
    attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    last_error_code varchar(200),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (
        (state = 'LEASED' AND lease_token IS NOT NULL AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR
        (state <> 'LEASED' AND lease_token IS NULL AND lease_owner IS NULL AND lease_expires_at IS NULL)
    )
);
CREATE INDEX ix_public_product_intake_jobs_claim
    ON public_product_intake_jobs(state, lease_expires_at, created_at);

-- +goose StatementBegin
CREATE OR REPLACE FUNCTION realsas_public_product_intake_job_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;
-- +goose StatementEnd

CREATE TRIGGER trg_public_product_intake_job_updated_at
BEFORE UPDATE ON public_product_intake_jobs
FOR EACH ROW
EXECUTE FUNCTION realsas_public_product_intake_job_updated_at();

-- +goose Down

DROP TRIGGER trg_public_product_intake_job_updated_at ON public_product_intake_jobs;
DROP FUNCTION realsas_public_product_intake_job_updated_at();
DROP INDEX ix_public_product_intake_jobs_claim;
DROP TABLE public_product_intake_jobs;
