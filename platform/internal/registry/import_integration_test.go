package registry

import (
	"context"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/input"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestImportExactEvidenceSealsInputsWithoutMintingQualification(t *testing.T) {
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
	obj, err := store.PutBytes(ctx, []byte("exact source "+uuid.NewString()))
	if err != nil {
		t.Fatal(err)
	}
	req := ImportRequest{ArtifactType: "RealSaS.TestExternalEvidence", SchemaVersion: "v1", Object: obj, SourceURI: "urn:test:external-evidence", CreatedBy: "ci"}
	first, err := Import(ctx, pool, store, req)
	if err != nil {
		t.Fatal(err)
	}
	req.SourceURI = "urn:test:same-bytes-moved"
	second, err := Import(ctx, pool, store, req)
	if err != nil {
		t.Fatal(err)
	}
	if !second.Reused || first.ArtifactID != second.ArtifactID || first.SemanticSHA256 != second.SemanticSHA256 {
		t.Fatalf("identity drift: %+v %+v", first, second)
	}
    otherType := req
    otherType.ArtifactType = "RealSaS.TestOtherEvidenceContract"
    alias, err := Import(ctx, pool, store, otherType)
    if err != nil || alias.ArtifactID == first.ArtifactID || alias.SemanticSHA256 == first.SemanticSHA256 {
        t.Fatalf("distinct typed semantics could not share exact CAS bytes: %+v %v", alias, err)
    }
    var sharedBytes int
    if err := pool.QueryRow(ctx, "SELECT count(*) FROM artifacts WHERE content_sha256=$1 AND storage_key=$2", obj.ContentSHA256, obj.StorageKey).Scan(&sharedBytes); err != nil || sharedBytes != 2 {
        t.Fatalf("shared byte identities=%d err=%v", sharedBytes, err)
    }
	for _, demo := range []bool{false, true} {
		_, found, err := (QualifiedCatalog{Pool: pool, AllowDemo: demo}).FindQualified(ctx, req.ArtifactType, req.SchemaVersion, first.SemanticSHA256)
		if err != nil || found {
			t.Fatalf("import acquired reuse qualification: %v %v", found, err)
		}
	}
	broken := req
	broken.Object.SizeBytes++
	if _, err := Import(ctx, pool, store, broken); err == nil {
		t.Fatal("size drift imported")
	}
	broken = req
	broken.Object.ContentSHA256 = hex64("0")
	if _, err := Import(ctx, pool, store, broken); err == nil {
		t.Fatal("hash drift imported")
	}
	subjectRequest := input.SubjectRequest{Slug: "import-" + uuid.NewString(), DisplayName: "Imported subject", CreatedBy: "ci"}
	sub, err := input.EnsureSubject(ctx, pool, subjectRequest)
	if err != nil {
		t.Fatal(err)
	}
	sameSubject, err := input.EnsureSubject(ctx, pool, subjectRequest)
	if err != nil || !sameSubject.Reused || sameSubject.SubjectID != sub.SubjectID {
		t.Fatalf("subject idempotence failed: %+v %v", sameSubject, err)
	}
	conflictingSubject := subjectRequest
	conflictingSubject.DisplayName = "Conflicting subject"
	if _, err := input.EnsureSubject(ctx, pool, conflictingSubject); err == nil {
		t.Fatal("subject identity was silently overwritten")
	}
	sealed, err := input.Seal(ctx, pool, sub.SubjectID, []input.Binding{{Role: "manifest:model", ArtifactID: first.ArtifactID}}, "ci")
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := input.Load(ctx, pool, sealed.SubjectInputID, sub.SubjectID)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.RootInputs) != 1 || loaded.RootInputs[0].Role != "subject:manifest:model" || loaded.ArtifactIDs[0] != first.ArtifactID {
		t.Fatalf("load=%+v", loaded)
	}
	var qualifications int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM qualifications WHERE artifact_id=$1", first.ArtifactID).Scan(&qualifications); err != nil || qualifications != 0 {
		t.Fatalf("qualifications=%d err=%v", qualifications, err)
	}
    if err := pool.QueryRow(ctx, "SELECT count(*) FROM qualifications WHERE artifact_id=$1", alias.ArtifactID).Scan(&qualifications); err != nil || qualifications != 0 {
        t.Fatalf("sharing bytes minted qualification: %d %v", qualifications, err)
    }
}
