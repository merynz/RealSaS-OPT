package modelselection

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

func TestDeveloperAttemptSnapshotsReleaseModelRequirementsAtAcceptance(t *testing.T) {
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
	family := "developer-model-snapshot-" + suffix
	architecture := "developer-arch-v1"
	releaseID := uuid.New()
	subjectID := uuid.New()
	artifactTypeID := uuid.New()
	sourceCodeSHA := strings.Repeat("d", 40)

	if _, err := pool.Exec(ctx, `
		INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)
	`, subjectID, "developer-model-subject-"+suffix, "Developer Model Snapshot Test"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'PRODUCT','ci',now())
	`, releaseID, "developer-model-release-"+suffix, selectionHash("developer-release:"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := release.BindCanonicalMain(ctx, pool, releaseID, sourceCodeSHA, "ci"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifact_types(id,name,schema_version,domain)
		VALUES ($1,$2,'v1','model')
	`, artifactTypeID, "RealSaS.DeveloperModelCheckpoint."+suffix); err != nil {
		t.Fatal(err)
	}

	createModel := func(label string) uuid.UUID {
		t.Helper()
		artifactID := uuid.New()
		modelID := uuid.New()
		if _, err := pool.Exec(ctx, `
			INSERT INTO artifacts(
				id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
				producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at
			) VALUES ($1,$2,$3,$4,$5,1,'developer-model-snapshot',$6,$7,'{}'::jsonb,now())
		`, artifactID, artifactTypeID,
			selectionHash("developer-semantic:"+label+":"+suffix),
			selectionHash("developer-content:"+label+":"+suffix),
			"developer-model/"+suffix+"/"+label+".ckpt",
			strings.Repeat("a", 64), strings.Repeat("b", 64)); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO model_releases(
				id,family,name,architecture_id,checkpoint_artifact_id,engine_release_id,
				source_code_sha,lifecycle_state,metadata,created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,'PRODUCTION','{}'::jsonb,'ci')
		`, modelID, family, "developer-model-"+label+"-"+suffix, architecture,
			artifactID, releaseID, sourceCodeSHA); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO model_release_evaluations(
				model_release_id,scope,result,evidence_artifact_id,details,observed_at,updated_at
			) VALUES ($1,'DEVELOPER_MODEL_SNAPSHOT_FIXTURE','PASS',$2,'{}'::jsonb,now(),now())
		`, modelID, artifactID); err != nil {
			t.Fatal(err)
		}
		return modelID
	}

	modelA := createModel("a")
	modelB := createModel("b")
	policyID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO model_promotion_policies(
			id,family,architecture_id,channel,policy_version,required_evaluation_scopes,policy_sha256,sealed_by
		) VALUES ($1,$2,$3,'PRODUCTION',1,$4,$5,'ci')
	`, policyID, family, architecture, []string{"DEVELOPER_MODEL_SNAPSHOT_FIXTURE"},
		selectionHash("developer-promotion-policy:"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO model_promotion_policy_bindings(
			family,architecture_id,channel,policy_id,bound_by
		) VALUES ($1,$2,'PRODUCTION',$3,'ci')
	`, family, architecture, policyID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO model_channels(family,channel,model_release_id,lock_version)
		VALUES ($1,'PRODUCTION',$2,0)
	`, family, modelA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_release_model_requirements(release_id,family,architecture_id,channel)
		VALUES ($1,$2,$3,'PRODUCTION')
	`, releaseID, family, architecture); err != nil {
		t.Fatal(err)
	}

	attemptID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'developer',$4,'ci','OPEN')
	`, attemptID, subjectID, releaseID, selectionHash("developer-attempt:"+suffix)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		UPDATE model_channels
		SET model_release_id=$2,lock_version=lock_version+1,updated_at=now()
		WHERE family=$1 AND channel='PRODUCTION'
	`, family, modelB); err != nil {
		t.Fatal(err)
	}

	selected, err := LoadAttempt(ctx, pool, attemptID)
	if err != nil {
		t.Fatal(err)
	}
	if len(selected) != 1 {
		t.Fatalf("developer attempt model snapshot=%+v", selected)
	}
	if selected[0].Family != family || selected[0].ArchitectureID != architecture {
		t.Fatalf("developer attempt model identity=%+v", selected[0])
	}
	if selected[0].ModelReleaseID != modelA || selected[0].Channel != "PRODUCTION" || selected[0].ChannelLockVersion != 0 {
		t.Fatalf("developer attempt did not preserve acceptance-time selection: %+v", selected[0])
	}
}
