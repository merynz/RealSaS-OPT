package registry

import (
	"context"
	"errors"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestArtifactReadSurfacePreservesTypedSubjectLineage(t *testing.T) {
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

	payload := []byte(`{"schema":"RealSaS.TestReadArtifact.v1","value":1}`)
	obj, err := store.PutBytes(ctx, payload)
	if err != nil {
		t.Fatal(err)
	}
	imported, err := Import(ctx, pool, store, ImportRequest{
		ArtifactType:  "RealSaS.TestReadArtifact",
		SchemaVersion: "v1",
		Object:        obj,
		SourceURI:     "urn:test:artifact-read:" + uuid.NewString(),
		CreatedBy:     "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO qualifications(id,artifact_id,qualification_type,result)
		VALUES ($1,$2,'READ_SURFACE_TEST','PASS')
	`, uuid.New(), imported.ArtifactID); err != nil {
		t.Fatal(err)
	}

	subject, err := input.EnsureSubject(ctx, pool, input.SubjectRequest{
		Slug:        "artifact-read-" + uuid.NewString(),
		DisplayName: "Artifact read test",
		CreatedBy:   "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	sealed, err := input.Seal(ctx, pool, subject.SubjectID, []input.Binding{{
		Role:       "manifest:read-test",
		ArtifactID: imported.ArtifactID,
	}}, "ci")
	if err != nil {
		t.Fatal(err)
	}

	record, err := GetArtifact(ctx, pool, imported.ArtifactID)
	if err != nil {
		t.Fatal(err)
	}
	if record.ArtifactType != "RealSaS.TestReadArtifact" || record.SchemaVersion != "v1" || record.ContentSHA256 != obj.ContentSHA256 || record.StorageKey != obj.StorageKey || record.SizeBytes != obj.SizeBytes {
		t.Fatalf("artifact metadata drift: %+v", record)
	}
	if len(record.Qualifications) != 1 || record.Qualifications[0].QualificationType != "READ_SURFACE_TEST" || record.Qualifications[0].Result != "PASS" {
		t.Fatalf("qualification lineage drift: %+v", record.Qualifications)
	}

	rows, err := ListSubjectArtifacts(ctx, pool, subject.SubjectID, "RealSaS.TestReadArtifact", "v1", 20)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].ArtifactID != imported.ArtifactID || rows[0].SourceKind != "SUBJECT_INPUT" || rows[0].SourceID != sealed.SubjectInputID || rows[0].Origin != "sealed" {
		t.Fatalf("subject artifact lineage drift: %+v", rows)
	}

	if _, err := ListSubjectArtifacts(ctx, pool, uuid.New(), "", "", 20); !errors.Is(err, ErrSubjectNotFound) {
		t.Fatalf("missing subject did not fail closed: %v", err)
	}
	if _, err := ListSubjectArtifacts(ctx, pool, subject.SubjectID, "", "", 501); err == nil {
		t.Fatal("unbounded subject artifact list admitted")
	}
	if _, err := GetArtifact(ctx, pool, uuid.New()); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("missing artifact did not fail closed: %v", err)
	}
}
