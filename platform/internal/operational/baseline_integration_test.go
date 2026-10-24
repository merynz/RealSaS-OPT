package operational

import (
	"context"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
)

func TestBaselineResolverPrefersPromotedProductOverLegacyWitness(t *testing.T) {
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

	suffix := uuid.New().String()
	subjectID := uuid.New()
	slug := "baseline-" + suffix
	if _, err := pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)", subjectID, slug, "Baseline resolver test"); err != nil {
		t.Fatal(err)
	}

	witnessTypeID := uuid.New()
	witnessArtifactID := uuid.New()
	witnessContentSHA := uniqueHash("legacy-witness-content", suffix)
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifact_types(id,name,schema_version,domain)
		VALUES ($1,'RealSaS.ProductNearestWitnessInventory','v1','witness')
		ON CONFLICT(name,schema_version) DO NOTHING
	`, witnessTypeID); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT id FROM artifact_types WHERE name='RealSaS.ProductNearestWitnessInventory' AND schema_version='v1'`).Scan(&witnessTypeID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifacts
		(id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,
		 implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,1,'RealSaS.ExternalEvidenceImport.v1',$6,$7,'{}',now())
	`, witnessArtifactID, witnessTypeID, uniqueHash("legacy-witness-semantic", suffix), witnessContentSHA,
		"cas/test/"+witnessContentSHA, uniqueHash("legacy-witness-implementation", suffix), uniqueHash("legacy-witness-policy", suffix)); err != nil {
		t.Fatal(err)
	}

	legacy, err := RegisterLegacyWitness(ctx, pool, LegacyWitnessRegisterRequest{
		SubjectID:         subjectID,
		WitnessArtifactID: witnessArtifactID,
		SourceRef:         "canonical/PRODUCT_NEAREST_KNIGHT_WITNESS_V1.json",
		SourceSHA256:      witnessContentSHA,
		Metadata: map[string]any{
			"axis_version": "V5.4.1",
			"mira_version": "V5.5",
			"authority":    "ENGINEERING_REGRESSION_WITNESS__NOT_PRODUCT_AUTHORITY",
		},
		CreatedBy: "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	if legacy.SupersededByProductRevisionID != nil {
		t.Fatal("legacy witness unexpectedly superseded before product promotion")
	}
	baseline, err := ResolveBaseline(ctx, pool, slug)
	if err != nil {
		t.Fatal(err)
	}
	if baseline.Kind != BaselineLegacyEngineeringWitness || baseline.LegacyWitness == nil || baseline.Product != nil {
		t.Fatalf("legacy baseline=%+v", baseline)
	}

	releaseID := uuid.New()
	capabilitySetSHA := uniqueHash("capability-set", suffix)
	sourceCodeSHA := repeated("a", 40)
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at,capability_set_sha256)
		VALUES ($1,$2,$3,'PRODUCT','ci',now(),$4)
	`, releaseID, "baseline-product-"+suffix, uniqueHash("product-release", suffix), capabilitySetSHA); err != nil {
		t.Fatal(err)
	}
	if _, err := release.BindCanonicalMain(ctx, pool, releaseID, sourceCodeSHA, "ci"); err != nil {
		t.Fatal(err)
	}
	attemptID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'compile_candidate',$4,'ci','QUALIFIED')
	`, attemptID, subjectID, releaseID, uniqueHash("product-attempt", suffix)); err != nil {
		t.Fatal(err)
	}
	revisionID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO product_revisions(id,subject_id,revision_number,manifest_sha256,created_from_attempt_id,sealed_at)
		VALUES ($1,$2,1,$3,$4,now())
	`, revisionID, subjectID, uniqueHash("product-manifest", suffix), attemptID); err != nil {
		t.Fatal(err)
	}

	checkpointTypeID := uuid.New()
	checkpointArtifactID := uuid.New()
	checkpointContentSHA := uniqueHash("axis-checkpoint-content", suffix)
	if _, err := pool.Exec(ctx, `INSERT INTO artifact_types(id,name,schema_version,domain) VALUES ($1,$2,'v1','model')`, checkpointTypeID, "BaselineModelCheckpoint."+suffix); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifacts
		(id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,producer_contract,
		 implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,1,'ci',$6,$7,'{}',now())
	`, checkpointArtifactID, checkpointTypeID, uniqueHash("axis-checkpoint-semantic", suffix), checkpointContentSHA,
		"cas/test/"+checkpointContentSHA, uniqueHash("axis-checkpoint-implementation", suffix), uniqueHash("axis-checkpoint-policy", suffix)); err != nil {
		t.Fatal(err)
	}
	modelReleaseID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO model_releases
		(id,family,name,architecture_id,checkpoint_artifact_id,engine_release_id,source_code_sha,lifecycle_state,metadata,created_by)
		VALUES ($1,'AXIS',$2,'RealSaS.AXIS.Test.v1',$3,$4,$5,'PRODUCTION','{}','ci')
	`, modelReleaseID, "axis-baseline-"+suffix, checkpointArtifactID, releaseID, sourceCodeSHA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO product_revision_model_releases(product_revision_id,family,model_release_id)
		VALUES ($1,'AXIS',$2)
	`, revisionID, modelReleaseID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO product_revision_artifacts(product_revision_id,role,artifact_id)
		VALUES ($1,'mechanical_baseline',$2)
	`, revisionID, witnessArtifactID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO subject_current_revision(subject_id,product_revision_id)
		VALUES ($1,$2)
	`, subjectID, revisionID); err != nil {
		t.Fatal(err)
	}

	baseline, err = ResolveBaseline(ctx, pool, slug)
	if err != nil {
		t.Fatal(err)
	}
	if baseline.Kind != BaselineProductRevision || baseline.Product == nil || baseline.LegacyWitness != nil {
		t.Fatalf("product baseline=%+v", baseline)
	}
	if baseline.Product.ProductRevisionID != revisionID || baseline.Product.EngineReleaseID == nil || *baseline.Product.EngineReleaseID != releaseID {
		t.Fatalf("product wiring=%+v", baseline.Product)
	}
	if len(baseline.Product.Models) != 1 || baseline.Product.Models[0].Family != "AXIS" || baseline.Product.Models[0].CheckpointContentSHA256 != checkpointContentSHA {
		t.Fatalf("product models=%+v", baseline.Product.Models)
	}
	if len(baseline.Product.Artifacts) != 1 || baseline.Product.Artifacts[0].Role != "mechanical_baseline" || baseline.Product.Artifacts[0].ArtifactID != witnessArtifactID {
		t.Fatalf("product artifacts=%+v", baseline.Product.Artifacts)
	}

	var supersededBy uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT superseded_by_product_revision_id FROM legacy_engineering_witnesses WHERE id=$1`, legacy.ID).Scan(&supersededBy); err != nil {
		t.Fatal(err)
	}
	if supersededBy != revisionID {
		t.Fatalf("legacy witness superseded_by=%s want=%s", supersededBy, revisionID)
	}
}
