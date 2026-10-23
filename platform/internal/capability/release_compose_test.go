package capability

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func TestBuildEngineReleaseRegistryIncludesStagesAndProductModules(t *testing.T) {
	raw, err := json.Marshal(stagegraph.Snapshot{
		Schema:     "RealSaS.StageGraph.v1",
		StageCount: 2,
		Stages: []stagegraph.Stage{
			{Ordinal: 1, ID: "A", Title: "A", Group: "source", Adapter: "x:y", Policy: stagegraph.Policy{FailClosed: true}},
			{Ordinal: 2, ID: "B", Title: "B", Group: "mesh", DependsOn: []string{"A"}, Adapter: "x:z", Policy: stagegraph.Policy{FailClosed: true}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	graph, err := stagegraph.ParsePlan(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := BuildEngineReleaseRegistry(context.Background(), graph)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{
		StageCapabilityID("A"),
		StageCapabilityID("B"),
		ProductInputCapabilityID,
		ProductShapeCapabilityID,
	} {
		if _, ok := registry.Get(id); !ok {
			t.Fatalf("release registry missing %s", id)
		}
	}
	resolved, err := registry.Resolve(ProductShapeCapabilityID)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 2 || resolved[0].ID != ProductInputCapabilityID || resolved[1].ID != ProductShapeCapabilityID {
		t.Fatalf("shape product dependency resolution=%+v", resolved)
	}
}

func TestProductProvidersAreSingleSourceOfNonStageProductGraph(t *testing.T) {
	providers := ProductProviders()
	if len(providers) != 2 {
		t.Fatalf("product provider count=%d", len(providers))
	}
	registry, err := Build(context.Background(), providers...)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := registry.Get(ProductInputCapabilityID); !ok {
		t.Fatal("product.input missing")
	}
	shape, ok := registry.Get(ProductShapeCapabilityID)
	if !ok || len(shape.Dependencies) != 1 || shape.Dependencies[0] != ProductInputCapabilityID {
		t.Fatalf("shape descriptor=%+v", shape)
	}
}

func TestModuleVersionPolicyAndParametersAreRederivedFromDescriptor(t *testing.T) {
	registry, err := Build(context.Background(), ProductInputProvider{}, ProductShapeProvider{})
	if err != nil {
		t.Fatal(err)
	}
	for _, capabilityID := range []string{ProductInputCapabilityID, ProductShapeCapabilityID} {
		descriptor, ok := registry.Get(capabilityID)
		if !ok {
			t.Fatal(capabilityID)
		}
		policySHA, err := modulePolicySHA256(descriptor)
		if err != nil {
			t.Fatal(err)
		}
		parametersSHA, err := semantic.JSONSHA256(descriptor.Metadata)
		if err != nil {
			t.Fatal(err)
		}
		valid := VersionIdentity{
			ImplementationSHA256: strings.Repeat("a", 64),
			PolicySHA256:         policySHA,
			ParametersSHA256:     parametersSHA,
		}
		if err := validateModuleVersionIdentity(descriptor, valid); err != nil {
			t.Fatalf("valid %s version rejected: %v", capabilityID, err)
		}
		badPolicy := valid
		badPolicy.PolicySHA256 = strings.Repeat("b", 64)
		if err := validateModuleVersionIdentity(descriptor, badPolicy); err == nil {
			t.Fatalf("%s policy drift accepted", capabilityID)
		}
		badParameters := valid
		badParameters.ParametersSHA256 = strings.Repeat("c", 64)
		if err := validateModuleVersionIdentity(descriptor, badParameters); err == nil {
			t.Fatalf("%s parameter drift accepted", capabilityID)
		}
	}
}
