package command

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
)

func TestDeveloperCapabilityRunSnapshotsOnlyResolvedTargetModels(t *testing.T) {
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
	`, releaseID, "target-models-"+releaseID.String(), hex64("1")); err != nil {
		t.Fatal(err)
	}

	inputRows, err := (capability.ProductInputProvider{}).Capabilities(ctx)
	if err != nil || len(inputRows) != 1 {
		t.Fatalf("input provider rows=%d err=%v", len(inputRows), err)
	}
	shapeRows, err := (capability.ProductShapeProvider{}).Capabilities(ctx)
	if err != nil || len(shapeRows) != 1 {
		t.Fatalf("shape provider rows=%d err=%v", len(shapeRows), err)
	}
	bindings := []capability.ReleaseBinding{
		{Descriptor: inputRows[0], Version: capability.VersionIdentity{
			ImplementationSHA256: hex64("2"), PolicySHA256: hex64("3"), ParametersSHA256: hex64("4"),
		}},
		{Descriptor: shapeRows[0], Version: capability.VersionIdentity{
			ImplementationSHA256: hex64("5"), PolicySHA256: hex64("6"), ParametersSHA256: hex64("7"),
		}},
	}
	if _, err := capability.SealReleaseSnapshot(ctx, pool, releaseID, bindings, "ci"); err != nil {
		t.Fatal(err)
	}

	for _, row := range []struct {
		family string
		arch   string
	}{
		{"AXIS", "unused-axis-architecture"},
		{"IRIS", "RealSaS.IRIS.VNextEvidenceModel.v3"},
		{"TESSA", "RealSaS.TESSA.Structuralizer.v3"},
	} {
		if _, err := pool.Exec(ctx, `
			INSERT INTO engine_release_model_requirements(release_id,family,architecture_id,channel)
			VALUES ($1,$2,$3,'RESEARCH')
		`, releaseID, row.family, row.arch); err != nil {
			t.Fatal(err)
		}
	}

	inputReceipt, err := SubmitCapabilityRun(ctx, pool, CapabilityRunRequest{
		EngineReleaseID: releaseID,
		Targets:         []string{capability.ProductInputCapabilityID},
		IdempotencyKey:  "target-input-" + releaseID.String(),
		RequestedBy:     "ci",
	})
	if err != nil {
		t.Fatalf("input target was blocked by unrelated release models: %v", err)
	}
	var modelCount int
	if err := pool.QueryRow(ctx,
		"SELECT count(*) FROM attempt_model_releases WHERE attempt_id=$1",
		inputReceipt.AttemptID,
	).Scan(&modelCount); err != nil {
		t.Fatal(err)
	}
	if modelCount != 0 {
		t.Fatalf("input target snapshotted unrelated models=%d", modelCount)
	}

	_, err = SubmitCapabilityRun(ctx, pool, CapabilityRunRequest{
		EngineReleaseID: releaseID,
		Targets:         []string{capability.ProductShapeCapabilityID},
		IdempotencyKey:  "target-shape-" + releaseID.String(),
		RequestedBy:     "ci",
	})
	if err == nil || !strings.Contains(err.Error(), "MODEL_SELECTION_CHANNEL_MISSING:IRIS:RESEARCH") {
		t.Fatalf("shape target did not fail closed on its required IRIS channel: %v", err)
	}

	// Clean up so migration-down reversibility can restore historical constraints.
	_, _ = pool.Exec(ctx, "DELETE FROM audit_events WHERE attempt_id=$1", inputReceipt.AttemptID)
	_, _ = pool.Exec(ctx, "DELETE FROM outbox_events WHERE aggregate_id=$1", inputReceipt.CommandID)
	_, _ = pool.Exec(ctx, "DELETE FROM commands WHERE id=$1", inputReceipt.CommandID)
	_, _ = pool.Exec(ctx, "DELETE FROM execution_goal_inputs WHERE execution_goal_id=$1", inputReceipt.ExecutionGoalID)
	_, _ = pool.Exec(ctx, "DELETE FROM execution_goals WHERE id=$1", inputReceipt.ExecutionGoalID)
	_, _ = pool.Exec(ctx, "DELETE FROM attempt_model_releases WHERE attempt_id=$1", inputReceipt.AttemptID)
	_, _ = pool.Exec(ctx, "DELETE FROM attempts WHERE id=$1", inputReceipt.AttemptID)
	_, _ = pool.Exec(ctx, "DELETE FROM engine_release_model_requirements WHERE release_id=$1", releaseID)
	_, _ = pool.Exec(ctx, "DELETE FROM engine_release_capabilities WHERE release_id=$1", releaseID)
	_, _ = pool.Exec(ctx, "DELETE FROM engine_releases WHERE id=$1", releaseID)
}
