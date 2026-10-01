package architecture

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func testRegistry(t *testing.T) Registry {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
	data, err := os.ReadFile(filepath.Join(root, "canonical", "MAINLINE_EXECUTION_PLAN_V2.json"))
	if err != nil {
		t.Fatal(err)
	}
	graph, err := stagegraph.ParseCanonicalPlan(data)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := Build(graph)
	if err != nil {
		t.Fatal(err)
	}
	if err := registry.Validate(); err != nil {
		t.Fatal(err)
	}
	return registry
}

func TestEveryCanonicalStageHasOneExactOwner(t *testing.T) {
	r := testRegistry(t)
	if len(r.Stages) != 46 {
		t.Fatalf("stages=%d", len(r.Stages))
	}
	cases := map[string]string{
		"10_IRIS_FIT":                          "engine.geometry",
		"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED": "engine.mesh",
		"37_QUALIFIED_PRESENTATION_STRUCTURE":  "engine.presentation",
		"42_RUNTIME_PROJECTION_AND_CAA_BINDING": "engine.runtime_binding",
		"46_PRODUCT_CLOSURE_SEAL":              "engine.closure",
	}
	for stageID, wantOwner := range cases {
		stage, ok := r.Stage(stageID)
		if !ok {
			t.Fatalf("missing %s", stageID)
		}
		if stage.OwnerModuleID != wantOwner {
			t.Fatalf("%s owner=%s want=%s", stageID, stage.OwnerModuleID, wantOwner)
		}
	}
}

func TestPlatformModulesExposeStateOwnership(t *testing.T) {
	r := testRegistry(t)
	for _, id := range []string{
		"platform.artifact",
		"platform.release",
		"platform.research",
		"platform.workflow",
		"platform.proof",
		"platform.product",
		"platform.render",
	} {
		module, ok := r.Module(id)
		if !ok {
			t.Fatalf("missing module %s", id)
		}
		if len(module.StateTables) == 0 {
			t.Fatalf("%s has no state-table ownership", id)
		}
	}
}
