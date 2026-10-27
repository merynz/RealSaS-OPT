package command

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/persistence"
	"github.com/merynz/RealSaS-OPT/platform/internal/release"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func targetScopedResearchGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	snapshot := stagegraph.Snapshot{
		Schema:     "RealSaS.StageGraph.v1",
		StageCount: 1,
		Stages: []stagegraph.Stage{{
			Ordinal: 1,
			ID:      "01_TARGET_SCOPED_MODEL_FIXTURE",
			Title:   "Target-scoped model fixture",
			Group:   "platform",
			Adapter: "fixture:noop",
			Policy:  stagegraph.Policy{FailClosed: true},
		}},
	}
	raw, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	graph, err := stagegraph.ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	return graph
}

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

	inputRows, err := (capability.ProductInputProvider{}).Capabilities(ctx)
	if err != nil || len(inputRows) != 1 {
		t.Fatalf("input provider rows=%d err=%v", len(inputRows), err)
	}
	shapeRows, err := (capability.ProductShapeProvider{}).Capabilities(ctx)
	if err != nil || len(shapeRows) != 1 {
		t.Fatalf("shape provider rows=%d err=%v", len(shapeRows), err)
	}
	inputImpl := hex64("2")
	shapeImpl := hex64("5")
	graph := targetScopedResearchGraph(t)
	graphSnapshot := graph.Snapshot()
	sealed, err := release.Seal(ctx, pool, graph, release.Manifest{
		ContractVersion: "RealSaS.EngineReleaseManifest.v2",
		Name:            "target-models-" + uuid.NewString(),
		Purpose:         release.PurposeResearch,
		Stages: []release.StageBinding{{
			Ordinal: 1, StageID: "01_TARGET_SCOPED_MODEL_FIXTURE",
			ImplementationSHA256: hex64("a"), PolicySHA256: hex64("b"),
			SemanticParameters: map[string]any{},
		}},
		DAG: &graphSnapshot,
		ModelRequirements: []release.ModelRequirement{
			{Family: "AXIS", ArchitectureID: "unused-axis-architecture", Channel: "RESEARCH"},
			{Family: "IRIS", ArchitectureID: "RealSaS.IRIS.VNextEvidenceModel.v3", Channel: "RESEARCH"},
			{Family: "TESSA", ArchitectureID: "RealSaS.TESSA.Structuralizer.v3", Channel: "RESEARCH"},
		},
		CapabilityImplementations: map[string]string{
			capability.ProductInputCapabilityID: inputImpl,
			capability.ProductShapeCapabilityID: shapeImpl,
		},
	}, "ci")
	if err != nil {
		t.Fatal(err)
	}
	releaseID := sealed.ReleaseID

	bindings := []capability.ReleaseBinding{
		{Descriptor: inputRows[0], Version: capability.VersionIdentity{
			ImplementationSHA256: inputImpl, PolicySHA256: hex64("3"), ParametersSHA256: hex64("4"),
		}},
		{Descriptor: shapeRows[0], Version: capability.VersionIdentity{
			ImplementationSHA256: shapeImpl, PolicySHA256: hex64("6"), ParametersSHA256: hex64("7"),
		}},
	}
	if _, err := capability.SealReleaseSnapshot(ctx, pool, releaseID, bindings, "ci"); err != nil {
		t.Fatal(err)
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
}
