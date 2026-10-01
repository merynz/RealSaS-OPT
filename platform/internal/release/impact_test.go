package release

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func graph(t *testing.T) *stagegraph.Graph {
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

func versions(g *stagegraph.Graph, stage35 string) map[string]StageVersion {
	out := map[string]StageVersion{}
	for _, s := range g.Stages() {
		impl := "same"
		if s.ID == "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED" {
			impl = stage35
		}
		out[s.ID] = StageVersion{ImplementationSHA256: impl, PolicySHA256: "policy", ParametersSHA256: "params"}
	}
	return out
}

func TestStage35ChangeIsLocalized(t *testing.T) {
	g := graph(t)
	impact, err := Compare(g, versions(g, "old"), versions(g, "new"))
	if err != nil {
		t.Fatal(err)
	}
	if len(impact.DirectChanges) != 1 || impact.DirectChanges[0].StageID != "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED" {
		t.Fatalf("direct changes=%v", impact.DirectChanges)
	}
	for _, id := range impact.UnchangedStageIDs {
		if id == "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED" {
			t.Fatal("Stage35 must not be unchanged")
		}
	}
	foundIris := false
	for _, id := range impact.UnchangedStageIDs {
		if id == "10_IRIS_FIT" {
			foundIris = true
		}
	}
	if !foundIris {
		t.Fatal("IRIS fit must remain reusable after Stage35-only change")
	}
}
