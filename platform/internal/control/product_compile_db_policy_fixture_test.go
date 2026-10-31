package control

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

func installProductE2EProductionPolicy(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	requirement release.ModelRequirement,
	modelID uuid.UUID,
	checkpointArtifactID uuid.UUID,
	suffix string,
) {
	t.Helper()
	policyID := uuid.New()
	scope := "PRODUCT_E2E_GATE"
	if _, err := pool.Exec(ctx, `
		INSERT INTO model_promotion_policies(
			id,family,architecture_id,channel,policy_version,required_evaluation_scopes,
			policy_sha256,sealed_by
		) VALUES ($1,$2,$3,'PRODUCTION',1,$4,$5,'product-e2e')
	`, policyID, requirement.Family, requirement.ArchitectureID, []string{scope},
		productE2ESHA("promotion-policy:"+requirement.Family+":"+requirement.ArchitectureID+":"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO model_promotion_policy_bindings(
			family,architecture_id,channel,policy_id,lock_version,bound_by
		) VALUES ($1,$2,'PRODUCTION',$3,0,'product-e2e')
	`, requirement.Family, requirement.ArchitectureID, policyID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO model_release_evaluations(
			model_release_id,scope,result,evidence_artifact_id,details,observed_at,updated_at
		) VALUES ($1,$2,'PASS',$3,'{"court":"shipping-product-db"}'::jsonb,now(),now())
	`, modelID, scope, checkpointArtifactID); err != nil {
		t.Fatal(err)
	}
}
