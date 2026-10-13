package control

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestExternalEvidenceWaitIsResumableAndDoesNotMintFailure(t *testing.T) {
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
	executionID := uuid.New()
	stageID := "10_IRIS_FIT"
	releaseHex := strings.ReplaceAll(subjectID.String(), "-", "")
	releaseSHA := releaseHex + releaseHex
	specHex := strings.ReplaceAll(attemptID.String(), "-", "")
	specSHA := specHex + specHex

	if _, err := pool.Exec(ctx, "INSERT INTO subjects(id,slug,display_name) VALUES ($1,$2,$3)", subjectID, "external-wait-"+subjectID.String(), "External wait"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'RESEARCH','ci',now())
	`, releaseID, "external-wait-release-"+subjectID.String(), releaseSHA); err != nil {
		t.Fatal(err)
	}
	graphSHA, err := g.SHA256()
	if err != nil {
		t.Fatal(err)
	}
	graphJSON, err := json.Marshal(g.Snapshot())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO engine_release_graphs
		(release_id,graph_sha256,snapshot) VALUES ($1,$2,$3)`, releaseID, graphSHA, graphJSON); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO attempts(id,subject_id,engine_release_id,kind,spec_sha256,created_by,final_state)
		VALUES ($1,$2,$3,'research',$4,'ci','OPEN')
	`, attemptID, subjectID, releaseID, specSHA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO executions(id,attempt_id,workflow_id,stage_contract,status,retry_number,started_at)
		VALUES ($1,$2,$3,$4,'RUNNING',0,now())
	`, executionID, attemptID, "realsas:compile:external-wait-"+attemptID.String(), stageID); err != nil {
		t.Fatal(err)
	}

	result, err := a.CommitStageResult(ctx, orchestration.StageCommitRequest{
		ExecutionID: executionID.String(),
		AttemptID:   attemptID.String(),
		StageID:     stageID,
		EngineResult: orchestration.EngineStageResult{
			StageID: stageID,
			Status:  "WAITING_EXTERNAL_EVIDENCE",
			Failure: &orchestration.EngineFailureEvidence{
				Code:  "EXTERNAL_STAGE_EVIDENCE_REQUIRED",
				Class: "EXTERNAL_EVIDENCE",
				Diagnostics: map[string]any{
					"missing_roles": []string{"checkpoint", "result"},
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != "WAITING_EXTERNAL_EVIDENCE" || result.ArtifactID != nil {
		t.Fatalf("wait result=%+v", result)
	}

	var status string
	var finished bool
	if err := pool.QueryRow(ctx, `SELECT status,finished_at IS NOT NULL FROM executions WHERE id=$1`, executionID).Scan(&status, &finished); err != nil {
		t.Fatal(err)
	}
	if status != "WAITING_EXTERNAL_EVIDENCE" || finished {
		t.Fatalf("execution status=%s finished=%v", status, finished)
	}
	var failures int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM failure_signatures WHERE attempt_id=$1`, attemptID).Scan(&failures); err != nil {
		t.Fatal(err)
	}
	if failures != 0 {
		t.Fatalf("wait minted %d failure signatures", failures)
	}
	var state string
	if err := pool.QueryRow(ctx, `SELECT final_state FROM attempts WHERE id=$1`, attemptID).Scan(&state); err != nil {
		t.Fatal(err)
	}
	if state != "OPEN" {
		t.Fatalf("wait finalized attempt: %s", state)
	}
	var events int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM attempt_events WHERE attempt_id=$1 AND event_type='STAGE_WAITING_EXTERNAL_EVIDENCE'`, attemptID).Scan(&events); err != nil {
		t.Fatal(err)
	}
	if events != 1 {
		t.Fatalf("wait event count=%d", events)
	}
}
