package modelselection

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

func selectionHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return fmt.Sprintf("%x", sum[:])
}

func TestProductAttemptSnapshotsExactProductionModelAndRejectsInvalidSelection(t *testing.T) {
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
	family := "p3-test-" + suffix
	architecture := "arch-v1"
	productReleaseID := uuid.New()
	subjectID := uuid.New()
	artifactTypeID := uuid.New()
	sourceCodeSHA := strings.Repeat("c", 40)

	if _, err := pool.Exec(ctx, `
		INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)
	`, subjectID, "p3-subject-"+suffix, "P3 Model Selection Test"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'PRODUCT','ci',now())
	`, productReleaseID, "p3-product-"+suffix, selectionHash("product-release:"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := release.BindCanonicalMain(ctx, pool, productReleaseID, sourceCodeSHA, "ci"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifact_types(id,name,schema_version,domain)
		VALUES ($1,$2,'v1','model')
	`, artifactTypeID, "RealSaS.P3Checkpoint."+suffix); err != nil {
		t.Fatal(err)
	}

	createModel := func(label string) (uuid.UUID, uuid.UUID, string) {
		t.Helper()
		artifactID := uuid.New()
		modelID := uuid.New()
		contentSHA := selectionHash("checkpoint:" + label + ":" + suffix)
		if _, err := pool.Exec(ctx, `
			INSERT INTO artifacts(
				id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
				producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at
			) VALUES ($1,$2,$3,$4,$5,1,'p3-test',$6,$7,'{}'::jsonb,now())
		`, artifactID, artifactTypeID, selectionHash("semantic:"+label+":"+suffix), contentSHA,
			"p3/"+suffix+"/"+label+".ckpt", strings.Repeat("a", 64), strings.Repeat("b", 64)); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO model_releases(
				id,family,name,architecture_id,checkpoint_artifact_id,engine_release_id,
				source_code_sha,lifecycle_state,metadata,created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,'PRODUCTION','{}'::jsonb,'ci')
		`, modelID, family, "p3-model-"+label+"-"+suffix, architecture, artifactID,
			productReleaseID, sourceCodeSHA); err != nil {
			t.Fatal(err)
		}
		return modelID, artifactID, contentSHA
	}

	modelA, artifactA, contentA := createModel("a")
	modelB, _, _ := createModel("b")
	if _, err := pool.Exec(ctx, `
		INSERT INTO model_channels(family,channel,model_release_id,lock_version)
		VALUES ($1,'PRODUCTION',$2,0)
	`, family, modelA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_release_model_requirements(release_id,family,architecture_id,channel)
		VALUES ($1,$2,$3,'PRODUCTION')
	`, productReleaseID, family, architecture); err != nil {
		t.Fatal(err)
	}

	attemptA := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'compile_candidate',$4,'ci','OPEN')
	`, attemptA, subjectID, productReleaseID, selectionHash("attempt-a:"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		UPDATE model_channels
		SET model_release_id=$2,lock_version=lock_version+1,updated_at=now()
		WHERE family=$1 AND channel='PRODUCTION'
	`, family, modelB); err != nil {
		t.Fatal(err)
	}

	// Lifecycle/maturity is intentionally mutable, but the exact identity behind
	// a selected ModelRelease is not. A running Attempt must survive a later
	// lifecycle change without changing its model/checkpoint identity.
	if _, err := pool.Exec(ctx, `
		UPDATE model_releases SET lifecycle_state='SUPERSEDED' WHERE id=$1
	`, modelA); err != nil {
		t.Fatalf("lifecycle promotion axis was accidentally frozen: %v", err)
	}
	_, err = pool.Exec(ctx, `
		UPDATE model_releases SET architecture_id='mutated-identity' WHERE id=$1
	`, modelA)
	if err == nil || !strings.Contains(err.Error(), "MODEL_RELEASE_IDENTITY_IMMUTABLE") {
		t.Fatalf("ModelRelease identity mutation was admitted: %v", err)
	}

	snapA, err := LoadAttempt(ctx, pool, attemptA)
	if err != nil {
		t.Fatal(err)
	}
	if len(snapA) != 1 || snapA[0].ModelReleaseID != modelA || snapA[0].CheckpointArtifactID != artifactA || snapA[0].CheckpointContentSHA256 != contentA {
		t.Fatalf("attempt snapshot drifted after channel/lifecycle movement: %+v", snapA)
	}
	if snapA[0].ArchitectureID != architecture {
		t.Fatalf("attempt model identity mutated after acceptance: %+v", snapA[0])
	}
	if snapA[0].ChannelLockVersion != 0 {
		t.Fatalf("attempt did not preserve accepted channel lock version: %+v", snapA[0])
	}

	attemptB := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'compile_candidate',$4,'ci','OPEN')
	`, attemptB, subjectID, productReleaseID, selectionHash("attempt-b:"+suffix)); err != nil {
		t.Fatal(err)
	}
	snapB, err := LoadAttempt(ctx, pool, attemptB)
	if err != nil {
		t.Fatal(err)
	}
	if len(snapB) != 1 || snapB[0].ModelReleaseID != modelB || snapB[0].ChannelLockVersion != 1 {
		t.Fatalf("new attempt did not snapshot new production selection: %+v", snapB)
	}

	missingReleaseID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'PRODUCT','ci',now())
	`, missingReleaseID, "p3-missing-"+suffix, selectionHash("missing-release:"+suffix)); err != nil {
		t.Fatal(err)
	}
	missingFamily := "p3-missing-family-" + suffix
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_release_model_requirements(release_id,family,architecture_id,channel)
		VALUES ($1,$2,'arch-missing','PRODUCTION')
	`, missingReleaseID, missingFamily); err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'compile_candidate',$4,'ci','OPEN')
	`, uuid.New(), subjectID, missingReleaseID, selectionHash("attempt-missing:"+suffix))
	if err == nil || !strings.Contains(err.Error(), "MODEL_SELECTION_CHANNEL_MISSING") {
		t.Fatalf("missing production channel was admitted: %v", err)
	}

	mismatchReleaseID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'PRODUCT','ci',now())
	`, mismatchReleaseID, "p3-mismatch-"+suffix, selectionHash("mismatch-release:"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_release_model_requirements(release_id,family,architecture_id,channel)
		VALUES ($1,$2,'wrong-architecture','PRODUCTION')
	`, mismatchReleaseID, family); err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'compile_candidate',$4,'ci','OPEN')
	`, uuid.New(), subjectID, mismatchReleaseID, selectionHash("attempt-mismatch:"+suffix))
	if err == nil || !strings.Contains(err.Error(), "MODEL_SELECTION_ARCHITECTURE_MISMATCH") {
		t.Fatalf("architecture mismatch was admitted: %v", err)
	}
}
