package registry

import (
	"context"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestCapabilityOutputRegistrationDetectsSemanticNondeterminism(t *testing.T) {
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
	store, err := artifactstore.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	firstObject, err := store.PutBytes(ctx, []byte("first"))
	if err != nil {
		t.Fatal(err)
	}
	secondObject, err := store.PutBytes(ctx, []byte("second"))
	if err != nil {
		t.Fatal(err)
	}

	subjectID := uuid.New()
	attemptID := uuid.New()
	executionID := uuid.New()
	if _, err := pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)", subjectID, "registry-"+subjectID.String(), "Registry"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,'developer',$3,'ci','OPEN')
	`, attemptID, subjectID, hex64("a")); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO executions(id,attempt_id,workflow_id,stage_contract,status,retry_number)
		VALUES ($1,$2,$3,'model/test','RUNNING',0)
	`, executionID, attemptID, "wf-"+executionID.String()); err != nil {
		t.Fatal(err)
	}

	registrar := Registrar{Store: store}
	capability := CapabilityIdentity{
		ID:                   "model/test",
		ImplementationSHA256: hex64("b"),
		PolicySHA256:         hex64("c"),
		ParametersSHA256:     hex64("d"),
	}
	output := ProducedObject{
		Role: "result", ArtifactType: "RealSaS.TestResult", SchemaVersion: "v1",
		StorageKey: firstObject.StorageKey, ContentSHA256: firstObject.ContentSHA256, SizeBytes: firstObject.SizeBytes,
		AuthorityClass: "DEVELOPER",
	}
	if err := registrar.VerifyOutputs(ctx, []ProducedObject{output}); err != nil {
		t.Fatal(err)
	}
	tx, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		t.Fatal(err)
	}
	ids, err := registrar.RegisterCapabilityOutputs(ctx, tx, executionID, attemptID, capability, []ProducedObject{output})
	if err != nil {
		tx.Rollback(ctx)
		t.Fatal(err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 {
		t.Fatalf("ids=%v", ids)
	}

	second := output
	second.StorageKey = secondObject.StorageKey
	second.ContentSHA256 = secondObject.ContentSHA256
	second.SizeBytes = secondObject.SizeBytes
	tx2, err := pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.Serializable})
	if err != nil {
		t.Fatal(err)
	}
	_, err = registrar.RegisterCapabilityOutputs(ctx, tx2, executionID, attemptID, capability, []ProducedObject{second})
	_ = tx2.Rollback(ctx)
	if err == nil {
		t.Fatal("expected semantic nondeterminism rejection")
	}
}

func hex64(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}
