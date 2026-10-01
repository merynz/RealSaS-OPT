package control

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestProductRevisionRejectsDemoOnlyStage46Qualification(t *testing.T) {
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		t.Skip("REALSAS_DATABASE_URL not set")
	}
	ctx := context.Background()
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	subjectID := uuid.New()
	attemptID := uuid.New()
	typeID := uuid.New()
	artifactID := uuid.New()
	if _, err := pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)", subjectID, "demo-guard-"+subjectID.String(), "Demo guard"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,'compile_candidate',$3,'ci','OPEN')
	`, attemptID, subjectID, strings.Repeat("a", 64)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifact_types(id,name,schema_version,domain)
		VALUES ($1,$2,'v1','stage')
	`, typeID, "RealSaS.DemoGuard."+subjectID.String()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifacts(
			id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
			producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at
		)
		VALUES ($1,$2,$3,$4,$5,1,'46_PRODUCT_CLOSURE_SEAL',$6,$7,'{}',$8)
	`, artifactID, typeID, strings.Repeat("b", 64), strings.Repeat("c", 64), "cas/test/"+artifactID.String(), strings.Repeat("d", 64), strings.Repeat("e", 64), time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempt_artifacts(attempt_id,role,artifact_id,origin)
		VALUES ($1,'stage:46_PRODUCT_CLOSURE_SEAL',$2,'produced')
	`, attemptID, artifactID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO qualifications(id,artifact_id,qualification_type,result)
		VALUES ($1,$2,'DEMO_REUSE_ELIGIBLE','PASS')
	`, uuid.New(), artifactID); err != nil {
		t.Fatal(err)
	}

	activities := Activities{Pool: pool, Graph: controlGraph(t)}
	err = activities.requireProductQualifiedClosure(ctx, attemptID)
	if err == nil || err.Error() != "PRODUCT_REVISION_STAGE46_PRODUCT_QUALIFICATION_REQUIRED" {
		t.Fatalf("demo-only Stage46 must not qualify product revision, got %v", err)
	}

	if _, err := pool.Exec(ctx, `
		INSERT INTO qualifications(id,artifact_id,qualification_type,result)
		VALUES ($1,$2,'REUSE_ELIGIBLE','PASS')
	`, uuid.New(), artifactID); err != nil {
		t.Fatal(err)
	}
	if err := activities.requireProductQualifiedClosure(ctx, attemptID); err != nil {
		t.Fatalf("product-qualified Stage46 should pass guard: %v", err)
	}
}
