package capability

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/architecture"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func currentRegistry(t *testing.T) (*Registry, *stagegraph.Graph) {
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
	arch, err := architecture.Build(graph)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := Build(context.Background(), CompilePlanProvider{Graph: graph, Architecture: arch})
	if err != nil {
		t.Fatal(err)
	}
	return registry, graph
}

func TestCompilePlanIsJustOneCapabilityProvider(t *testing.T) {
	registry, graph := currentRegistry(t)
	if len(registry.Descriptors()) != graph.StageCount() {
		t.Fatalf("capabilities=%d graph=%d", len(registry.Descriptors()), graph.StageCount())
	}
	productStage, ok := graph.ProductPassStageID()
	if !ok {
		t.Fatal("product pass stage missing")
	}
	goal := Goal{
		Type:            GoalProductCompile,
		Targets:         []string{StageCapabilityID(productStage)},
		PromotionPolicy: PromotionQualifiedProductOnly,
	}
	resolved, err := goal.Validate(registry)
	if err != nil {
		t.Fatal(err)
	}
	required, err := graph.AncestorsIncluding(productStage)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != len(required) {
		t.Fatalf("resolved=%d required=%d graph=%d", len(resolved), len(required), graph.StageCount())
	}
}

func TestDeveloperCanRunOnlyOneSubgraphWithoutProductPromotion(t *testing.T) {
	registry, _ := currentRegistry(t)
	target := StageCapabilityID("35_DYNAMIC_MECHANICAL_MESH_QUALIFIED")
	resolved, err := (Goal{
		Type:            GoalDeveloperRun,
		Targets:         []string{target},
		PromotionPolicy: PromotionNever,
	}).Validate(registry)
	if err != nil {
		t.Fatal(err)
	}
	foundTarget := false
	foundRuntime := false
	for _, descriptor := range resolved {
		if descriptor.ID == target {
			foundTarget = true
		}
		if descriptor.ID == StageCapabilityID("42_RUNTIME_PROJECTION_AND_CAA_BINDING") {
			foundRuntime = true
		}
	}
	if !foundTarget || foundRuntime {
		t.Fatalf("target=%v runtime=%v resolved=%v", foundTarget, foundRuntime, resolved)
	}
}

func TestExternalModelOrModuleProviderCanExtendRegistryWithoutChangingCore(t *testing.T) {
	base, _ := currentRegistry(t)
	baseRows := base.Descriptors()
	extra := ProviderFunc(func(context.Context) ([]Descriptor, error) {
		return []Descriptor{
			{
				ID:               "model/experimental.depth",
				Kind:             KindModel,
				OwnerModuleID:    "engine.experimental_depth",
				ExecutorActivity: "engine.execute_model.v1",
				AllowedModes:     []Mode{ModeDeveloper},
				Aliases:          []string{"depth", "experimental model"},
			},
			{
				ID:               "module/experimental.inspect",
				Kind:             KindModule,
				OwnerModuleID:    "engine.experimental_depth",
				ExecutorActivity: "engine.execute_module.v1",
				Dependencies:     []string{"model/experimental.depth"},
				AllowedModes:     []Mode{ModeDeveloper},
			},
		}, nil
	})
	all := ProviderFunc(func(context.Context) ([]Descriptor, error) {
		return baseRows, nil
	})
	registry, err := Build(context.Background(), all, extra)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := (Goal{
		Type:            GoalDeveloperRun,
		Targets:         []string{"module/experimental.inspect"},
		PromotionPolicy: PromotionNever,
	}).Validate(registry)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 2 || resolved[0].ID != "model/experimental.depth" || resolved[1].ID != "module/experimental.inspect" {
		t.Fatalf("resolved=%v", resolved)
	}
}
