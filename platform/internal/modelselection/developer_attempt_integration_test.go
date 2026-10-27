package modelselection

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestRawDeveloperAttemptDoesNotSnapshotReleaseWideModels(t *testing.T) {
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

	suffix := uuid.NewString()
	family := "developer-target-scope-" + suffix
	releaseID := uuid.New()
	subjectID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)
	`, subjectID, "developer-target-scope-"+suffix, "Developer target scope test"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'RESEARCH','ci',now())
	`, releaseID, "developer-target-scope-"+suffix, selectionHash("developer-target-release:"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_release_model_requirements(release_id,family,architecture_id,channel)
		VALUES ($1,$2,'unused-architecture','RESEARCH')
	`, releaseID, family); err != nil {
		t.Fatal(err)
	}

	attemptID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'developer',$4,'ci','OPEN')
	`, attemptID, subjectID, releaseID, selectionHash("developer-target-attempt:"+suffix)); err != nil {
		t.Fatalf("raw developer attempt was incorrectly forced through release-wide model selection: %v", err)
	}
	selected, err := LoadAttempt(ctx, pool, attemptID)
	if err != nil {
		t.Fatal(err)
	}
	if len(selected) != 0 {
		t.Fatalf("raw developer attempt unexpectedly snapshotted release-wide models=%+v", selected)
	}
}

func TestCompileCandidateStillSnapshotsReleaseWideModelsFailClosed(t *testing.T) {
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

	suffix := uuid.NewString()
	family := "compile-target-scope-" + suffix
	releaseID := uuid.New()
	subjectID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)
	`, subjectID, "compile-target-scope-"+suffix, "Compile target scope test"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'PRODUCT','ci',now())
	`, releaseID, "compile-target-scope-"+suffix, selectionHash("compile-target-release:"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_release_model_requirements(release_id,family,architecture_id,channel)
		VALUES ($1,$2,'required-architecture','PRODUCTION')
	`, releaseID, family); err != nil {
		t.Fatal(err)
	}

	_, err = pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'compile_candidate',$4,'ci','OPEN')
	`, uuid.New(), subjectID, releaseID, selectionHash("compile-target-attempt:"+suffix))
	if err == nil || !strings.Contains(err.Error(), "MODEL_SELECTION_CHANNEL_MISSING:"+family+":PRODUCTION") {
		t.Fatalf("compile_candidate no longer fails closed on full release model envelope: %v", err)
	}
}
