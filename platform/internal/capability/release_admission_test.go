package capability

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func partialReleaseTestGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	raw, err := json.Marshal(stagegraph.Snapshot{
		Schema:     "RealSaS.StageGraph.v1",
		StageCount: 1,
		Stages: []stagegraph.Stage{{
			Ordinal: 1,
			ID:      "05_CAMERA_CONTRACT_SOLVED",
			Title:   "camera",
			Group:   "observation",
			Adapter: "x:y",
			Policy:  stagegraph.Policy{FailClosed: true},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	graph, err := stagegraph.ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	return graph
}

func TestPartialEngineReleaseAdmitsOnlyExecutableProductRoles(t *testing.T) {
	registry, err := BuildEngineReleaseRegistry(context.Background(), partialReleaseTestGraph(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, capabilityID := range []string{ProductInputCapabilityID, ProductShapeCapabilityID} {
		if _, ok := registry.Get(capabilityID); !ok {
			t.Fatalf("partial release lost executable role %s", capabilityID)
		}
	}
	for _, capabilityID := range []string{ProductMeshCapabilityID, ProductRigCapabilityID} {
		if _, ok := registry.Get(capabilityID); ok {
			t.Fatalf("partial release advertised unavailable role %s", capabilityID)
		}
	}

	modelRequirements, err := ProductModelRequirements(registry)
	if err != nil {
		t.Fatal(err)
	}
	families := map[string]bool{}
	for _, row := range modelRequirements {
		families[row.Family] = true
	}
	if !families["IRIS"] || !families["TESSA"] || len(families) != 2 {
		t.Fatalf("partial release model requirements=%+v", modelRequirements)
	}
	if families["AXIS"] {
		t.Fatalf("partial release required AXIS without admitted Rig: %+v", modelRequirements)
	}
}
