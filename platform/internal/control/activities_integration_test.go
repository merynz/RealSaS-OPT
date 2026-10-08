package control

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"

	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func controlGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
	data, err := os.ReadFile(filepath.Join(root, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json"))
	if err != nil {
		t.Fatal(err)
	}
	g, err := stagegraph.ParseCanonicalPlan(data)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestPrepareAndFailExecutionPersistsLocalizedRepair(t *testing.T) {
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
	g := controlGraph(t)
	a := Activities{Pool: pool, Graph: g}
	a.Store, err = artifactstore.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}

	subjectID := uuid.New()
	releaseID := uuid.New()
	attemptID := uuid.New()
	commandID := uuid.New()
	subjectInputID := uuid.New()
	sourceTypeID := uuid.New()
	sourceArtifactID := uuid.New()
	if _, err := pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)", subjectID, "control-"+subjectID.String(), "Control"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'PRODUCT','ci',now())
	`, releaseID, "control-release-"+subjectID.String(), repeatHex("1")); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'compile_candidate',$4,'ci','OPEN')
	`, attemptID, subjectID, releaseID, repeatHex("2")); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifact_types(id,name,schema_version,domain)
		VALUES ($1,$2,'v1','source')
	`, sourceTypeID, "RealSaS.ControlTestSource."+subjectID.String()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO artifacts
		  (id,artifact_type_id,semantic_sha256,content_sha256,storage_key,size_bytes,
		   producer_contract,implementation_sha256,policy_sha256,semantic_parameters,verified_at)
		VALUES ($1,$2,$3,$4,$5,1,'TEST_SOURCE',$6,$7,'{}',now())
	`, sourceArtifactID, sourceTypeID, repeatHex("6"), repeatHex("7"), "cas/test/"+sourceArtifactID.String(), repeatHex("8"), repeatHex("9")); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO subject_inputs(id,subject_id,manifest_sha256,created_by,sealed_at)
		VALUES ($1,$2,$3,'ci',now())
	`, subjectInputID, subjectID, repeatHex("a")); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO subject_input_artifacts(subject_input_id,role,ordinal,artifact_id)
		VALUES ($1,'source',0,$2)
	`, subjectInputID, sourceArtifactID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO compiler_run_bindings(attempt_id,compiler_run_id,run_manifest_path,run_ledger_path,pipeline_plan_sha256)
		VALUES ($1,$2,'manifest.json','ledger.json',$3)
	`, attemptID, "run-"+attemptID.String(), repeatHex("3")); err != nil {
		t.Fatal(err)
	}
	commandPayload, err := json.Marshal(map[string]any{
		"schema":            "RealSaS.CompileSubjectCommand.v1",
		"command_id":        commandID.String(),
		"attempt_id":        attemptID.String(),
		"subject_id":        subjectID.String(),
		"engine_release_id": releaseID.String(),
		"subject_input_id":  subjectInputID.String(),
		"target_stage_id":   "46_PRODUCT_CLOSURE_SEAL",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO commands(id,command_type,subject_id,idempotency_key,payload)
		VALUES ($1,'COMPILE_SUBJECT',$2,$3,$4)
	`, commandID, subjectID, "control-test:"+commandID.String(), commandPayload); err != nil {
		t.Fatal(err)
	}
	stageID := "37_QUALIFIED_PRESENTATION_STRUCTURE"
	stage, ok := g.Get(stageID)
	if !ok {
		t.Fatal("stage37 missing")
	}
	if _, err := pool.Exec(ctx, `INSERT INTO engine_release_stages
		(release_id,stage_id,ordinal,implementation_sha256,policy_sha256,semantic_parameters)
		VALUES ($1,$2,$3,$4,$5,'{}')`, releaseID, stageID, stage.Ordinal, repeatHex("b"), repeatHex("c")); err != nil {
		t.Fatal(err)
	}
	for _, dependency := range stage.DependsOn {
		if _, err := pool.Exec(ctx, `
			INSERT INTO attempt_artifacts(attempt_id,role,artifact_id,origin)
			VALUES ($1,$2,$3,'inherited')
		`, attemptID, "stage:"+dependency, sourceArtifactID); err != nil {
			t.Fatal(err)
		}
	}
	allowed, err := g.DescendantsIncluding("35_DYNAMIC_MECHANICAL_MESH_QUALIFIED")
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := a.PrepareStageExecution(ctx, orchestration.PrepareStageExecutionRequest{
		CommandID:              commandID.String(),
		AttemptID:              attemptID.String(),
		SubjectID:              subjectID.String(),
		SubjectInputID:         subjectInputID.String(),
		EngineReleaseID:        releaseID.String(),
		StageID:                stageID,
		ExpectedSemanticSHA256: repeatHex("4"),
		AllowedExecuteStageIDs: allowed,
	})
	if err != nil {
		t.Fatal(err)
	}
	if prepared.CompilerRunID != "run-"+attemptID.String() || prepared.ExecutionID == "" {
		t.Fatalf("prepared=%+v", prepared)
	}
	result, err := a.CommitStageResult(ctx, orchestration.StageCommitRequest{
		ExecutionID:            prepared.ExecutionID,
		CommandID:              prepared.CommandID,
		AttemptID:              prepared.AttemptID,
		StageID:                stageID,
		ExpectedSemanticSHA256: repeatHex("4"),
		AllowedExecuteStageIDs: allowed,
		EngineResult: orchestration.EngineStageResult{
			StageID:          stageID,
			Status:           "FAIL",
			ExecutedStageIDs: []string{stageID},
			DiagnosticsHash:  repeatHex("5"),
			Failure: &orchestration.EngineFailureEvidence{
				Code:                 "VISUAL_BINDING_MECHANICAL_INCOMPATIBILITY",
				Class:                "CODE",
				ReportedOwnerStageID: "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
				Diagnostics:          map[string]any{"face": 17},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != "FAIL" {
		t.Fatalf("result=%+v", result)
	}
	var executionStatus string
	if err := pool.QueryRow(ctx, "SELECT status FROM executions WHERE id=$1", prepared.ExecutionID).Scan(&executionStatus); err != nil {
		t.Fatal(err)
	}
	if executionStatus != "FAIL" {
		t.Fatalf("status=%s", executionStatus)
	}
	var owner string
	var invalidatedRaw []byte
	if err := pool.QueryRow(ctx, `
		SELECT oa.owner_stage_id,rd.invalidated_stage_ids
		FROM failure_signatures fs
		JOIN owner_attributions oa ON oa.failure_signature_id=fs.id
		JOIN repair_directives rd ON rd.failure_signature_id=fs.id
		WHERE fs.attempt_id=$1
	`, attemptID).Scan(&owner, &invalidatedRaw); err != nil {
		t.Fatal(err)
	}
	if owner != "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED" {
		t.Fatalf("owner=%s", owner)
	}
	var invalidated struct {
		StageIDs []string `json:"stage_ids"`
	}
	if err := json.Unmarshal(invalidatedRaw, &invalidated); err != nil {
		t.Fatal(err)
	}
	want, _ := g.DescendantsIncluding(owner)
	if len(invalidated.StageIDs) != len(want) {
		t.Fatalf("invalidated=%v want=%v", invalidated.StageIDs, want)
	}
	_ = time.Now()
	// Exercise the real PostgreSQL PASS commit with bound dependencies, not
	// only failure localization. This catches connection-busy registry writes.
	uniqueSHA := sha256.Sum256([]byte(subjectID.String()))
	repaired, err := a.PrepareStageExecution(ctx, orchestration.PrepareStageExecutionRequest{
		CommandID: uuid.New().String(), AttemptID: attemptID.String(), SubjectID: subjectID.String(),
		SubjectInputID: subjectInputID.String(), EngineReleaseID: releaseID.String(), StageID: stageID,
		ExpectedSemanticSHA256: fmt.Sprintf("%x", uniqueSHA), AllowedExecuteStageIDs: allowed,
	})
	if err != nil {
		t.Fatal(err)
	}
	object, err := a.Store.PutBytes(ctx, []byte(`{"schema":"RealSaS.TestProof.v1"}`))
	if err != nil {
		t.Fatal(err)
	}
	passed, err := a.CommitStageResult(ctx, orchestration.StageCommitRequest{
		ExecutionID: repaired.ExecutionID, CommandID: repaired.CommandID, AttemptID: attemptID.String(),
		StageID: stageID, ExpectedSemanticSHA256: fmt.Sprintf("%x", uniqueSHA), AllowedExecuteStageIDs: allowed,
		EngineResult: orchestration.EngineStageResult{StageID: stageID, Status: "PASS", ExecutedStageIDs: []string{stageID},
			Outputs: []orchestration.EngineOutput{{Role: "proof", ArtifactType: "RealSaS.TestProof", SchemaVersion: "v1",
				RelativePath: "proof.json", PayloadSchema: "RealSaS.TestProof.v1", AuthorityClass: "TEST",
				StorageKey: object.StorageKey, ContentSHA256: object.ContentSHA256, SizeBytes: object.SizeBytes}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var dependencyCount int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM artifact_inputs WHERE artifact_id=$1", *passed.ArtifactID).Scan(&dependencyCount); err != nil {
		t.Fatal(err)
	}
	if dependencyCount != len(stage.DependsOn)+1 {
		t.Fatalf("dependency count=%d", dependencyCount)
	}
}

func repeatHex(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}
