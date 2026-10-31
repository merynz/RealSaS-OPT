package control

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

type productRevisionFixture struct {
	CapabilityID  string
	AttemptRole   string
	ArtifactType  string
	SchemaVersion string
}

var productRevisionFixtures = []productRevisionFixture{
	{"product.mesh", "capability:product.mesh:canonical_mesh", "RealSaS.CanonicalMeshIR", "v1"},
	{"product.rig", "capability:product.rig:canonical_rig", "RealSaS.CanonicalRigIR", "v2"},
	{"product.skin", "capability:product.skin:canonical_skin", "RealSaS.CanonicalSkinIR", "v2"},
	{"product.mechanics", "capability:product.mechanics:canonical_mechanics", "RealSaS.CanonicalMechanicsIR", "v1"},
	{"product.graphics", "capability:product.graphics:canonical_graphics", "RealSaS.CanonicalGraphicsIR", "v1"},
	{"product.motion", "capability:product.motion:canonical_motion", "RealSaS.CanonicalMotionIR", "v2"},
	{"product.proof", "capability:product.proof:product_proof", "RealSaS.ProductProofBundleIR", "v1"},
}

func productE2ESHA(value string) string {
	sum := sha256.Sum256([]byte(value))
	return fmt.Sprintf("%x", sum[:])
}

func productE2EGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
	raw, err := os.ReadFile(filepath.Join(root, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json"))
	if err != nil {
		t.Fatal(err)
	}
	graph, err := stagegraph.ParseCanonicalPlan(raw)
	if err != nil {
		t.Fatal(err)
	}
	return graph
}

func TestShippingProductCompileDBCourt(t *testing.T) {
	if os.Getenv("REALSAS_PRODUCT_E2E_DB_COURT") != "1" {
		t.Skip("REALSAS_PRODUCT_E2E_DB_COURT not enabled")
	}
	dsn := os.Getenv("REALSAS_DATABASE_URL")
	if dsn == "" {
		t.Fatal("REALSAS_DATABASE_URL is required for shipping product DB court")
	}
	ctx := context.Background()
	pool, err := persistence.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	graph := productE2EGraph(t)
	registry, err := capability.BuildEngineReleaseRegistry(ctx, graph)
	if err != nil {
		t.Fatal(err)
	}
	modelRows, err := capability.ProductModelRequirements(registry)
	if err != nil {
		t.Fatal(err)
	}
	modelRequirements := make([]release.ModelRequirement, 0, len(modelRows))
	for _, row := range modelRows {
		modelRequirements = append(modelRequirements, release.ModelRequirement{
			Family: row.Family, ArchitectureID: row.ArchitectureID, Channel: "PRODUCTION",
		})
	}
	sort.Slice(modelRequirements, func(i, j int) bool {
		return modelRequirements[i].Family < modelRequirements[j].Family
	})

	implementationHashes := map[string]string{}
	moduleImplementations := map[string]capability.ModuleImplementationIdentity{}
	for _, descriptor := range registry.Descriptors() {
		if descriptor.Kind == capability.KindStage || !strings.HasPrefix(descriptor.ID, "product.") {
			continue
		}
		implementationSHA := productE2ESHA("product-implementation:" + descriptor.ID)
		implementationHashes[descriptor.ID] = implementationSHA
		moduleImplementations[descriptor.ID] = capability.ModuleImplementationIdentity{
			ImplementationSHA256: implementationSHA,
		}
	}
	if len(implementationHashes) != 9 {
		t.Fatalf("shipping product role count=%d want=9", len(implementationHashes))
	}
	if len(modelRequirements) == 0 {
		t.Fatal("shipping product release unexpectedly has no learned model requirements")
	}

	stageBindings := make([]release.StageBinding, 0, len(graph.Stages()))
	for _, stage := range graph.Stages() {
		stageBindings = append(stageBindings, release.StageBinding{
			Ordinal:              stage.Ordinal,
			StageID:              stage.ID,
			ImplementationSHA256: productE2ESHA("stage-implementation:" + stage.ID),
			PolicySHA256:         productE2ESHA("stage-policy:" + stage.ID),
			SemanticParameters:   map[string]any{"product_e2e": true},
		})
	}
	graphSnapshot := graph.Snapshot()
	sealedRelease, err := release.Seal(ctx, pool, graph, release.Manifest{
		ContractVersion:           "RealSaS.EngineReleaseManifest.v2",
		Name:                      "product-e2e-" + uuid.NewString(),
		Purpose:                   release.PurposeProduct,
		Stages:                    stageBindings,
		DAG:                       &graphSnapshot,
		ModelRequirements:         modelRequirements,
		CapabilityImplementations: implementationHashes,
	}, "product-e2e")
	if err != nil {
		t.Fatal(err)
	}
	capabilitySetSHA, err := capability.SealEngineReleaseSnapshot(
		ctx, pool, sealedRelease.ReleaseID, moduleImplementations, "product-e2e",
	)
	if err != nil {
		t.Fatal(err)
	}
	if capabilitySetSHA == "" {
		t.Fatal("shipping capability set sha missing")
	}

	sourceCodeSHA := strings.Repeat("c", 40)
	if _, err := release.BindCanonicalMain(ctx, pool, sealedRelease.ReleaseID, sourceCodeSHA, "product-e2e"); err != nil {
		t.Fatal(err)
	}
	suffix := uuid.NewString()
	modelIDs := map[string]uuid.UUID{}
	for _, requirement := range modelRequirements {
		artifactTypeID := uuid.New()
		artifactID := uuid.New()
		modelID := uuid.New()
		if _, err := pool.Exec(ctx, `
			INSERT INTO artifact_types(id,name,schema_version,domain)
			VALUES ($1,$2,'v1','model')
		`, artifactTypeID, "RealSaS.ProductE2E."+requirement.Family+"Checkpoint."+suffix); err != nil {
			t.Fatal(err)
		}
		checkpointSHA := productE2ESHA("checkpoint:" + requirement.Family + ":" + suffix)
		if _, err := pool.Exec(ctx, `
			INSERT INTO artifacts(
				id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
				producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at
			) VALUES ($1,$2,$3,$4,$5,1,'product-e2e',$6,$7,'{}'::jsonb,now())
		`, artifactID, artifactTypeID,
			productE2ESHA("model-semantic:"+requirement.Family+":"+suffix), checkpointSHA,
			"product-e2e/models/"+requirement.Family+"/"+suffix,
			productE2ESHA("model-impl:"+requirement.Family+":"+suffix),
			productE2ESHA("model-policy:"+requirement.Family+":"+suffix)); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO model_releases(
				id,family,name,architecture_id,checkpoint_artifact_id,engine_release_id,
				source_code_sha,lifecycle_state,metadata,created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,'PRODUCTION','{}'::jsonb,'product-e2e')
		`, modelID, requirement.Family, "product-e2e-"+requirement.Family+"-"+suffix,
			requirement.ArchitectureID, artifactID, sealedRelease.ReleaseID, sourceCodeSHA); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO model_channels(family,channel,model_release_id,lock_version)
			VALUES ($1,'PRODUCTION',$2,0)
		`, requirement.Family, modelID); err != nil {
			t.Fatal(err)
		}
		modelIDs[requirement.Family] = modelID
	}

	subjectID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)
	`, subjectID, "product-e2e-"+suffix, "Product E2E"); err != nil {
		t.Fatal(err)
	}
	sourceTypeID := uuid.New()
	sourceID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifact_types(id,name,schema_version,domain)
		VALUES ($1,'RealSaS.SourceObservation','v1','source')
	`, sourceTypeID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifacts(
			id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
			producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at
		) VALUES ($1,$2,$3,$4,$5,1,'RealSaS.SourceObservation.v1',$6,$7,'{}'::jsonb,now())
	`, sourceID, sourceTypeID,
		productE2ESHA("source-semantic:"+suffix), productE2ESHA("source-content:"+suffix),
		"product-e2e/source/"+suffix,
		productE2ESHA("source-impl:"+suffix), productE2ESHA("source-policy:"+suffix)); err != nil {
		t.Fatal(err)
	}

	receipt, err := command.SubmitProductCapabilityCompile(ctx, pool, command.ProductCapabilityCompileRequest{
		EngineReleaseID:  sealedRelease.ReleaseID,
		SubjectID:        subjectID,
		InputArtifactIDs: []uuid.UUID{sourceID},
		Parameters:       map[string]any{"court": "shipping-product-db"},
		IdempotencyKey:   "product-e2e:" + suffix,
		RequestedBy:      "product-e2e",
	})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.CommandID == uuid.Nil || receipt.AttemptID == uuid.Nil || receipt.ExecutionGoalID == uuid.Nil {
		t.Fatalf("incomplete product compile receipt: %+v", receipt)
	}

	rows, err := pool.Query(ctx, `
		SELECT family,model_release_id,selected_channel,selected_channel_lock_version
		FROM attempt_model_releases WHERE attempt_id=$1 ORDER BY family
	`, receipt.AttemptID)
	if err != nil {
		t.Fatal(err)
	}
	selectedCount := 0
	for rows.Next() {
		var family, channel string
		var modelID uuid.UUID
		var lockVersion int64
		if err := rows.Scan(&family, &modelID, &channel, &lockVersion); err != nil {
			rows.Close()
			t.Fatal(err)
		}
		if modelIDs[family] != modelID || channel != "PRODUCTION" || lockVersion != 0 {
			rows.Close()
			t.Fatalf("attempt model snapshot drift: family=%s model=%s channel=%s lock=%d", family, modelID, channel, lockVersion)
		}
		selectedCount++
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		t.Fatal(err)
	}
	rows.Close()
	if selectedCount != len(modelRequirements) {
		t.Fatalf("attempt model snapshot count=%d want=%d", selectedCount, len(modelRequirements))
	}

	var goalSpecSHA string
	if err := pool.QueryRow(ctx, `SELECT spec_sha256 FROM execution_goals WHERE id=$1`, receipt.ExecutionGoalID).Scan(&goalSpecSHA); err != nil {
		t.Fatal(err)
	}
	workflowInput := orchestration.CapabilityWorkflowInput{
		CommandID:       receipt.CommandID.String(),
		AttemptID:       receipt.AttemptID.String(),
		ExecutionGoalID: receipt.ExecutionGoalID.String(),
		EngineReleaseID: sealedRelease.ReleaseID.String(),
		GoalSpecSHA256:  goalSpecSHA,
	}
	activities := Activities{Pool: pool, Graph: graph}
	resolved, err := activities.ResolveCapabilityGoal(ctx, workflowInput)
	if err != nil {
		t.Fatal(err)
	}
	wantCapabilities := []string{
		"product.input", "product.shape", "product.mesh", "product.rig", "product.skin",
		"product.mechanics", "product.graphics", "product.motion", "product.proof",
	}
	if len(resolved.Steps) != len(wantCapabilities) {
		t.Fatalf("resolved product capability count=%d want=%d", len(resolved.Steps), len(wantCapabilities))
	}
	for index, want := range wantCapabilities {
		if resolved.Steps[index].CapabilityID != want {
			t.Fatalf("resolved[%d]=%s want=%s", index, resolved.Steps[index].CapabilityID, want)
		}
	}
	if resolved.GoalType != string(capability.GoalProductCompile) ||
		resolved.PromotionPolicy != string(capability.PromotionQualifiedProductOnly) {
		t.Fatalf("shipping goal policy drift: type=%s promotion=%s", resolved.GoalType, resolved.PromotionPolicy)
	}
	if resolved.CapabilitySetSHA256 != capabilitySetSHA {
		t.Fatalf("capability set drift: %s != %s", resolved.CapabilitySetSHA256, capabilitySetSHA)
	}

	inputRequest, err := activities.PrepareCapabilityExecution(ctx, orchestration.PrepareCapabilityExecutionRequest{
		CommandID:       receipt.CommandID.String(),
		AttemptID:       receipt.AttemptID.String(),
		ExecutionGoalID: receipt.ExecutionGoalID.String(),
		EngineReleaseID: sealedRelease.ReleaseID.String(),
		Step:            resolved.Steps[0],
		GoalParameters:  resolved.Parameters,
	})
	if err != nil {
		t.Fatal(err)
	}
	if inputRequest.CapabilityID != capability.ProductInputCapabilityID || len(inputRequest.InputArtifacts) != 1 {
		t.Fatalf("product.input request=%+v", inputRequest)
	}
	if inputRequest.InputArtifacts[0].ID != sourceID.String() ||
		inputRequest.InputArtifacts[0].ArtifactType != "RealSaS.SourceObservation" ||
		inputRequest.InputArtifacts[0].SchemaVersion != "v1" {
		t.Fatalf("product.input source authority drift: %+v", inputRequest.InputArtifacts[0])
	}
	if len(inputRequest.ModelReleases) != 0 || inputRequest.ReleasedGraph != nil {
		t.Fatalf("product.input leaked model/stage authority: models=%d graph=%v", len(inputRequest.ModelReleases), inputRequest.ReleasedGraph != nil)
	}

	inputExecutionID, err := uuid.Parse(inputRequest.ExecutionID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE executions SET status='PASS',finished_at=now() WHERE id=$1`, inputExecutionID); err != nil {
		t.Fatal(err)
	}
	workflowID := "realsas:capability:" + receipt.CommandID.String()
	for _, capabilityID := range wantCapabilities[1:] {
		if _, err := pool.Exec(ctx, `
			INSERT INTO executions(
				id,attempt_id,workflow_id,stage_contract,status,retry_number,started_at,finished_at
			) VALUES ($1,$2,$3,$4,'PASS',0,now(),now())
		`, uuid.New(), receipt.AttemptID, workflowID, capabilityID); err != nil {
			t.Fatal(err)
		}
	}

	for index, fixture := range productRevisionFixtures {
		artifactTypeID := uuid.New()
		artifactID := uuid.New()
		if _, err := pool.Exec(ctx, `
			INSERT INTO artifact_types(id,name,schema_version,domain)
			VALUES ($1,$2,$3,'product')
		`, artifactTypeID, fixture.ArtifactType, fixture.SchemaVersion); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO artifacts(
				id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
				producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at
			) VALUES ($1,$2,$3,$4,$5,1,$6,$7,$8,'{}'::jsonb,now())
		`, artifactID, artifactTypeID,
			productE2ESHA(fmt.Sprintf("revision-semantic:%d:%s", index, suffix)),
			productE2ESHA(fmt.Sprintf("revision-content:%d:%s", index, suffix)),
			fmt.Sprintf("product-e2e/revision/%d/%s", index, suffix), fixture.CapabilityID,
			implementationHashes[fixture.CapabilityID], productE2ESHA("revision-policy:"+fixture.CapabilityID+":"+suffix)); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO attempt_artifacts(attempt_id,role,artifact_id,origin)
			VALUES ($1,$2,$3,'produced')
		`, receipt.AttemptID, fixture.AttemptRole, artifactID); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO qualifications(id,artifact_id,qualification_type,result)
			VALUES ($1,$2,'CAPABILITY_EXECUTION_PASS','PASS')
		`, uuid.New(), artifactID); err != nil {
			t.Fatal(err)
		}
	}

	result, err := activities.FinalizeCapabilityProduct(ctx, map[string]any{
		"command": workflowInput,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != "PASS" || result.ProductRevisionID == nil || *result.ProductRevisionID == "" {
		t.Fatalf("shipping product finalization result=%+v", result)
	}
	revisionID, err := uuid.Parse(*result.ProductRevisionID)
	if err != nil {
		t.Fatal(err)
	}

	var attemptState string
	if err := pool.QueryRow(ctx, `SELECT final_state FROM attempts WHERE id=$1`, receipt.AttemptID).Scan(&attemptState); err != nil {
		t.Fatal(err)
	}
	if attemptState != "QUALIFIED" {
		t.Fatalf("product attempt state=%s want=QUALIFIED", attemptState)
	}
	var revisionArtifacts, revisionModels int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM product_revision_artifacts WHERE product_revision_id=$1`, revisionID).Scan(&revisionArtifacts); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM product_revision_model_releases WHERE product_revision_id=$1`, revisionID).Scan(&revisionModels); err != nil {
		t.Fatal(err)
	}
	if revisionArtifacts != len(productRevisionFixtures) {
		t.Fatalf("product revision artifact count=%d want=%d", revisionArtifacts, len(productRevisionFixtures))
	}
	if revisionModels != len(modelRequirements) {
		t.Fatalf("product revision model count=%d want=%d", revisionModels, len(modelRequirements))
	}
}
