package command

import (
	"context"
	"os"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestGlobalDeveloperCapabilityRunNeedsNoSyntheticSubject(t *testing.T) {
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

	releaseID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO engine_releases(id,name,release_sha256,purpose,created_by,sealed_at)
		VALUES ($1,$2,$3,'RESEARCH','ci',now())
	`, releaseID, "developer-"+releaseID.String(), hex64("a")); err != nil {
		t.Fatal(err)
	}

	descriptor := capability.Descriptor{
		ID:               "model/iris.infer",
		Kind:             capability.KindModel,
		OwnerModuleID:    "engine.model.iris",
		ExecutorActivity: "engine.execute_model.v1",
		AllowedModes:     []capability.Mode{capability.ModeDeveloper},
	}
	setSHA, err := capability.SealReleaseSnapshot(ctx, pool, releaseID, []capability.ReleaseBinding{{
		Descriptor: descriptor,
		Version: capability.VersionIdentity{
			ImplementationSHA256: hex64("b"),
			PolicySHA256:         hex64("c"),
			ParametersSHA256:     hex64("d"),
		},
	}}, "ci")
	if err != nil {
		t.Fatal(err)
	}
	if setSHA == "" {
		t.Fatal("capability set sha missing")
	}

	receipt, err := SubmitCapabilityRun(ctx, pool, CapabilityRunRequest{
		EngineReleaseID: releaseID,
		Targets:         []string{"model/iris.infer"},
		Parameters:      map[string]any{"view_count": 2},
		IdempotencyKey:  "developer-" + releaseID.String(),
		RequestedBy:     "ci",
	})
	if err != nil {
		t.Fatal(err)
	}
	var subject *uuid.UUID
	var kind string
	if err := pool.QueryRow(ctx, "SELECT subject_id,kind FROM attempts WHERE id=$1", receipt.AttemptID).Scan(&subject, &kind); err != nil {
		t.Fatal(err)
	}
	if subject != nil || kind != "developer" {
		t.Fatalf("subject=%v kind=%s", subject, kind)
	}
	var goalType, policy string
	if err := pool.QueryRow(ctx, "SELECT goal_type,promotion_policy FROM execution_goals WHERE id=$1", receipt.ExecutionGoalID).Scan(&goalType, &policy); err != nil {
		t.Fatal(err)
	}
	if goalType != "DEVELOPER_RUN" || policy != "NEVER" {
		t.Fatalf("goal=%s policy=%s", goalType, policy)
	}

	// Clean up so migration-down reversibility can restore subject_id NOT NULL.
	_, _ = pool.Exec(ctx, "DELETE FROM audit_events WHERE attempt_id=$1", receipt.AttemptID)
	_, _ = pool.Exec(ctx, "DELETE FROM outbox_events WHERE aggregate_id=$1", receipt.CommandID)
	_, _ = pool.Exec(ctx, "DELETE FROM commands WHERE id=$1", receipt.CommandID)
	_, _ = pool.Exec(ctx, "DELETE FROM execution_goals WHERE id=$1", receipt.ExecutionGoalID)
	_, _ = pool.Exec(ctx, "DELETE FROM attempts WHERE id=$1", receipt.AttemptID)
	_, _ = pool.Exec(ctx, "DELETE FROM engine_release_capabilities WHERE release_id=$1", releaseID)
	_, _ = pool.Exec(ctx, "DELETE FROM engine_releases WHERE id=$1", releaseID)
}

func hex64(ch string) string {
	out := ""
	for len(out) < 64 {
		out += ch
	}
	return out[:64]
}
