package control

import (
	"context"
	"os"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/artifactstore"
	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/command"
	"github.com/merynz/RealSaS-OPT/platform/internal/orchestration"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestDeveloperCapabilityGoalPersistsDependencyArtifactsAndFinalizes(t *testing.T) {
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

	releaseID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'RESEARCH','ci',now())
	`, releaseID, "cap-e2e-"+releaseID.String(), repeat64("a")); err != nil {
		t.Fatal(err)
	}

	firstID := "model/test.source." + releaseID.String()
	secondID := "module/test.consume." + releaseID.String()
	first := capability.Descriptor{
		ID: firstID, Kind: capability.KindModel, OwnerModuleID: "engine.test",
		ExecutorActivity: "engine.execute_capability.v1",
		AllowedModes:     []capability.Mode{capability.ModeDeveloper},
	}
	second := capability.Descriptor{
		ID: secondID, Kind: capability.KindModule, OwnerModuleID: "engine.test",
		ExecutorActivity: "engine.execute_capability.v1",
		Dependencies:     []string{firstID},
		AllowedModes:     []capability.Mode{capability.ModeDeveloper},
	}
	bindings := []capability.ReleaseBinding{
		{Descriptor: first, Version: capability.VersionIdentity{
			ImplementationSHA256: repeat64("b"), PolicySHA256: repeat64("c"), ParametersSHA256: repeat64("d"),
		}},
		{Descriptor: second, Version: capability.VersionIdentity{
			ImplementationSHA256: repeat64("e"), PolicySHA256: repeat64("f"), ParametersSHA256: repeat64("1"),
		}},
	}
	if _, err := capability.SealReleaseSnapshot(ctx, pool, releaseID, bindings, "ci"); err != nil {
		t.Fatal(err)
	}

	receipt, err := command.SubmitCapabilityRun(ctx, pool, command.CapabilityRunRequest{
		EngineReleaseID: releaseID,
		Targets:         []string{secondID},
		Parameters:      map[string]any{"probe": "knight"},
		IdempotencyKey:  "cap-e2e-" + releaseID.String(),
		RequestedBy:     "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		rows, _ := pool.Query(ctx, "SELECT artifact_id FROM attempt_artifacts WHERE attempt_id=$1", receipt.AttemptID)
		var artifactIDs []uuid.UUID
		if rows != nil {
			for rows.Next() {
				var id uuid.UUID
				if rows.Scan(&id) == nil {
					artifactIDs = append(artifactIDs, id)
				}
			}
			rows.Close()
		}
		_, _ = pool.Exec(ctx, "DELETE FROM qualifications WHERE artifact_id = ANY($1)", artifactIDs)
		_, _ = pool.Exec(ctx, "DELETE FROM artifact_inputs WHERE artifact_id = ANY($1) OR input_artifact_id = ANY($1)", artifactIDs)
		_, _ = pool.Exec(ctx, "DELETE FROM execution_artifacts WHERE execution_id IN (SELECT id FROM executions WHERE attempt_id=$1)", receipt.AttemptID)
		_, _ = pool.Exec(ctx, "DELETE FROM attempt_artifacts WHERE attempt_id=$1", receipt.AttemptID)
		_, _ = pool.Exec(ctx, "DELETE FROM executions WHERE attempt_id=$1", receipt.AttemptID)
		_, _ = pool.Exec(ctx, "DELETE FROM attempt_events WHERE attempt_id=$1", receipt.AttemptID)
		_, _ = pool.Exec(ctx, "DELETE FROM execution_goal_inputs WHERE execution_goal_id=$1", receipt.ExecutionGoalID)
		_, _ = pool.Exec(ctx, "DELETE FROM execution_goals WHERE id=$1", receipt.ExecutionGoalID)
		_, _ = pool.Exec(ctx, "DELETE FROM audit_events WHERE attempt_id=$1", receipt.AttemptID)
		_, _ = pool.Exec(ctx, "DELETE FROM outbox_events WHERE aggregate_id=$1", receipt.CommandID)
		_, _ = pool.Exec(ctx, "DELETE FROM commands WHERE id=$1", receipt.CommandID)
		_, _ = pool.Exec(ctx, "DELETE FROM attempts WHERE id=$1", receipt.AttemptID)
		for _, id := range artifactIDs {
			_, _ = pool.Exec(ctx, "DELETE FROM artifacts WHERE id=$1", id)
		}
		_, _ = pool.Exec(ctx, "DELETE FROM engine_release_capabilities WHERE release_id=$1", releaseID)
		_, _ = pool.Exec(ctx, "DELETE FROM engine_releases WHERE id=$1", releaseID)
	}()

	var specSHA string
	if err := pool.QueryRow(ctx,
		"SELECT spec_sha256 FROM execution_goals WHERE id=$1",
		receipt.ExecutionGoalID,
	).Scan(&specSHA); err != nil {
		t.Fatal(err)
	}

	activities := Activities{Pool: pool, Store: store}
	workflowInput := orchestration.CapabilityWorkflowInput{
		CommandID:       receipt.CommandID.String(),
		AttemptID:       receipt.AttemptID.String(),
		ExecutionGoalID: receipt.ExecutionGoalID.String(),
		EngineReleaseID: releaseID.String(),
		GoalSpecSHA256:  specSHA,
	}
	goal, err := activities.ResolveCapabilityGoal(ctx, workflowInput)
	if err != nil {
		t.Fatal(err)
	}
	if len(goal.Steps) != 2 || goal.Steps[0].CapabilityID != firstID || goal.Steps[1].CapabilityID != secondID {
		t.Fatalf("resolved steps=%v", goal.Steps)
	}

	firstReq, err := activities.PrepareCapabilityExecution(ctx, orchestration.PrepareCapabilityExecutionRequest{
		CommandID:       workflowInput.CommandID,
		AttemptID:       workflowInput.AttemptID,
		ExecutionGoalID: workflowInput.ExecutionGoalID,
		EngineReleaseID: workflowInput.EngineReleaseID,
		Step:            goal.Steps[0],
		GoalParameters:  goal.Parameters,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(firstReq.InputArtifacts) != 0 {
		t.Fatalf("first capability inputs=%v", firstReq.InputArtifacts)
	}
	firstObject, err := store.PutBytes(ctx, []byte("first-capability-result"))
	if err != nil {
		t.Fatal(err)
	}
	firstCommit, err := activities.CommitCapabilityResult(ctx, orchestration.CapabilityCommitRequest{
		Request: firstReq,
		Result: orchestration.EngineCapabilityResult{
			CapabilityID: firstID,
			Status:       "PASS",
			Outputs: []orchestration.EngineOutput{{
				Role: "result", ArtifactType: "RealSaS.CapabilityE2E.First." + releaseID.String(),
				SchemaVersion: "v1", StorageKey: firstObject.StorageKey,
				ContentSHA256: firstObject.ContentSHA256, SizeBytes: firstObject.SizeBytes,
				AuthorityClass: "DEVELOPER",
			}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(firstCommit.OutputArtifactIDs) != 1 {
		t.Fatalf("first outputs=%v", firstCommit.OutputArtifactIDs)
	}

	secondReq, err := activities.PrepareCapabilityExecution(ctx, orchestration.PrepareCapabilityExecutionRequest{
		CommandID:       workflowInput.CommandID,
		AttemptID:       workflowInput.AttemptID,
		ExecutionGoalID: workflowInput.ExecutionGoalID,
		EngineReleaseID: workflowInput.EngineReleaseID,
		Step:            goal.Steps[1],
		GoalParameters:  goal.Parameters,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(secondReq.InputArtifacts) != 1 || secondReq.InputArtifacts[0].ID != firstCommit.OutputArtifactIDs[0] {
		t.Fatalf("second inputs=%v first output=%v", secondReq.InputArtifacts, firstCommit.OutputArtifactIDs)
	}
	secondObject, err := store.PutBytes(ctx, []byte("second-capability-result"))
	if err != nil {
		t.Fatal(err)
	}
	secondCommit, err := activities.CommitCapabilityResult(ctx, orchestration.CapabilityCommitRequest{
		Request: secondReq,
		Result: orchestration.EngineCapabilityResult{
			CapabilityID: secondID,
			Status:       "PASS",
			Outputs: []orchestration.EngineOutput{{
				Role: "result", ArtifactType: "RealSaS.CapabilityE2E.Second." + releaseID.String(),
				SchemaVersion: "v1", StorageKey: secondObject.StorageKey,
				ContentSHA256: secondObject.ContentSHA256, SizeBytes: secondObject.SizeBytes,
				AuthorityClass: "DEVELOPER",
			}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(secondCommit.OutputArtifactIDs) != 1 {
		t.Fatalf("second outputs=%v", secondCommit.OutputArtifactIDs)
	}

	finalized, err := activities.FinalizeCapabilityGoal(ctx, map[string]any{"command": workflowInput})
	if err != nil {
		t.Fatal(err)
	}
	if finalized.Status != "PASS" {
		t.Fatalf("finalized=%+v", finalized)
	}
	var state string
	if err := pool.QueryRow(ctx, "SELECT final_state FROM attempts WHERE id=$1", receipt.AttemptID).Scan(&state); err != nil {
		t.Fatal(err)
	}
	if state != "PASS" {
		t.Fatalf("attempt state=%s", state)
	}
}

func repeat64(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}
