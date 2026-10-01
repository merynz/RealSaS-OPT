package input

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestSubjectInputIsImmutableArtifactSet(t *testing.T) {
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
	typeID := uuid.New()
	artifactID := uuid.New()
	if _, err := pool.Exec(ctx,
		"INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)",
		subjectID, "input-"+subjectID.String(), "Subject Input",
	); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx,
		"INSERT INTO artifact_types(id,name,schema_version,domain) VALUES ($1,'RealSaS.SourceImage','v1','source')",
		typeID,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifacts
		  (id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,1,'SOURCE_IMPORT',$6,$7,'{}',$8)
	`, artifactID, typeID, repeatInput("1"), repeatInput("2"), "cas/source/"+artifactID.String(), repeatInput("3"), repeatInput("4"), time.Now().UTC()); err != nil {
		t.Fatal(err)
	}

	first, err := Seal(ctx, pool, subjectID, []Binding{{Role: "source", ArtifactID: artifactID}}, "ci")
	if err != nil {
		t.Fatal(err)
	}
	second, err := Seal(ctx, pool, subjectID, []Binding{{Role: "source", ArtifactID: artifactID}}, "ci")
	if err != nil {
		t.Fatal(err)
	}
	if !second.Reused || first.SubjectInputID != second.SubjectInputID || first.ManifestSHA256 != second.ManifestSHA256 {
		t.Fatalf("first=%+v second=%+v", first, second)
	}
	loaded, err := Load(ctx, pool, first.SubjectInputID, subjectID)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.RootInputs) != 1 || loaded.RootInputs[0].Role != "subject:source" || loaded.ArtifactIDs[0] != artifactID {
		t.Fatalf("loaded=%+v", loaded)
	}
}

func repeatInput(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}
