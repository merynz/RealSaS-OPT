package control

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/google/uuid"

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

	subjectID := uuid.New()
	releaseID := uuid.New()
	attemptID := uuid.New()
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
		INSERT INTO compiler_run_bindings(attempt_id,compiler_run_id,run_manifest_path,run_ledger_path,pipeline_plan_sha256)
		VALUES ($1,$2,'manifest.json','ledger.json',$3)
	`, attemptID, "run-"+attemptID.String(), repeatHex("3")); err != nil {
		t.Fatal(err)
	}
	stageID := "37_QUALIFIED_PRESENTATION_STRUCTURE"
	allowed, err := g.DescendantsIncluding("35_DYNAMIC_MECHANICAL_MESH_QUALIFIED")
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := a.PrepareStageExecution(ctx, orchestration.PrepareStageExecutionRequest{
		CommandID:              "cmd-" + attemptID.String(),
		AttemptID:              attemptID.String(),
		SubjectID:              subjectID.String(),
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
}

func repeatHex(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}
