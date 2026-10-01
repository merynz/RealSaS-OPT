package product

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func integrationPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		t.Skip("REALSAS_DATABASE_URL not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return pool
}

func TestAtomicPromotion(t *testing.T) {
	pool := integrationPool(t)
	ctx := context.Background()
	subjectID := uuid.New()
	attemptID := uuid.New()
	revisionID := uuid.New()
	artifactTypeID := uuid.New()

	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx,
		"INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)",
		subjectID, "promotion-"+subjectID.String(), "Promotion Integration",
	); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx,
		"INSERT INTO attempts(id,subject_id,kind,spec_sha256,created_by,final_state) VALUES ($1,$2,'compile_candidate',$3,'ci','QUALIFIED')",
		attemptID, subjectID, repeatHex("a"),
	); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx,
		"INSERT INTO artifact_types(id,name,schema_version,domain) VALUES ($1,$2,'v1','test')",
		artifactTypeID, "RealSaS.GoPromotionSmoke."+subjectID.String(),
	); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx,
		"INSERT INTO product_revisions(id,subject_id,revision_number,manifest_sha256,created_from_attempt_id,sealed_at) VALUES ($1,$2,1,$3,$4,now())",
		revisionID, subjectID, repeatHex("d"), attemptID,
	); err != nil {
		t.Fatal(err)
	}

	for i, role := range requiredProductRoles {
		artifactID := uuid.New()
		qualificationID := uuid.New()
		if _, err := tx.Exec(ctx, `
			INSERT INTO artifacts
			  (id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at)
			VALUES ($1,$2,$3,$4,$5,1,$6,$7,$8,'{}',now())
		`,
			artifactID, artifactTypeID,
			hex64(i+1), hex64(i+101), "cas/test/"+role,
			"TEST_"+role, repeatHex("b"), repeatHex("c"),
		); err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Exec(ctx,
			"INSERT INTO qualifications(id,artifact_id,qualification_type,result) VALUES ($1,$2,'PRODUCT_PROMOTION','PASS')",
			qualificationID, artifactID,
		); err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Exec(ctx,
			"INSERT INTO product_revision_artifacts(product_revision_id,role,artifact_id) VALUES ($1,$2,$3)",
			revisionID, role, artifactID,
		); err != nil {
			t.Fatal(err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	result, err := Promote(ctx, pool, PromotionRequest{
		SubjectID: subjectID,
		TargetRevisionID: revisionID,
		RequestedBy: "ci",
		Reason: "integration smoke",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ToRevision != revisionID || result.NewLockVersion != 1 || result.FromRevision != nil {
		t.Fatalf("unexpected result: %+v", result)
	}

	var current uuid.UUID
	var lockVersion int64
	if err := pool.QueryRow(ctx,
		"SELECT product_revision_id,lock_version FROM subject_current_revision WHERE subject_id=$1",
		subjectID,
	).Scan(&current, &lockVersion); err != nil {
		t.Fatal(err)
	}
	if current != revisionID || lockVersion != 1 {
		t.Fatalf("current=%s lock=%d", current, lockVersion)
	}
	var auditCount, outboxCount int
	if err := pool.QueryRow(ctx,
		"SELECT count(*) FROM audit_events WHERE subject_id=$1 AND action='PRODUCT_REVISION_PROMOTED'",
		subjectID,
	).Scan(&auditCount); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx,
		"SELECT count(*) FROM outbox_events WHERE aggregate_id=$1 AND event_type='ProductRevisionPromoted'",
		subjectID,
	).Scan(&outboxCount); err != nil {
		t.Fatal(err)
	}
	if auditCount != 1 || outboxCount != 1 {
		t.Fatalf("audit=%d outbox=%d", auditCount, outboxCount)
	}
	if _, err := Promote(ctx, pool, PromotionRequest{
		SubjectID: subjectID,
		TargetRevisionID: revisionID,
		RequestedBy: "ci",
		Reason: "duplicate",
	}); err != ErrAlreadyCurrent {
		t.Fatalf("duplicate promotion error=%v", err)
	}
}

func repeatHex(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}

func hex64(v int) string {
	const digits = "0123456789abcdef"
	out := make([]byte, 64)
	for i := range out {
		out[i] = '0'
	}
	for i := 63; v > 0 && i >= 0; i-- {
		out[i] = digits[v&15]
		v >>= 4
	}
	return string(out)
}
