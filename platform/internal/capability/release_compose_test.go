package capability

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/merynz/RealSaS-OPT/platform/internal/semantic"
	"github.com/merynz/RealSaS-OPT/platform/internal/stagegraph"
)

func meshReleaseTestGraph(t *testing.T) *stagegraph.Graph {
	t.Helper()
	ids := []struct {
		id    string
		group string
	}{
		{"05_CAMERA_CONTRACT_SOLVED", "observation"},
		{"07_OBSERVATION_CONTRACT_QUALIFIED", "observation"},
		{"08_NORMALIZATION_DOMAIN_QUALIFIED", "geometry"},
		{"13_GEOMETRY_SUBSTRATE_QUALIFIED", "geometry"},
		{"15_RIGGING_SURFACE_QUALIFIED", "rig"},
		{"17_MECHANICAL_PARTITION_QUALIFIED", "components"},
	}
	stages := make([]stagegraph.Stage, 0, len(ids))
	for index, row := range ids {
		stages = append(stages, stagegraph.Stage{
			Ordinal: index + 1,
			ID:      row.id,
			Title:   row.id,
			Group:   row.group,
			Adapter: "x:y",
			Policy:  stagegraph.Policy{FailClosed: true},
		})
	}
	raw, err := json.Marshal(stagegraph.Snapshot{
		Schema:     "RealSaS.StageGraph.v1",
		StageCount: len(stages),
		Stages:     stages,
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

func TestBuildEngineReleaseRegistryIncludesStagesAndProductModules(t *testing.T) {
	graph := meshReleaseTestGraph(t)
	registry, err := BuildEngineReleaseRegistry(context.Background(), graph)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{
		StageCapabilityID("05_CAMERA_CONTRACT_SOLVED"),
		StageCapabilityID("07_OBSERVATION_CONTRACT_QUALIFIED"),
		StageCapabilityID("08_NORMALIZATION_DOMAIN_QUALIFIED"),
		StageCapabilityID("13_GEOMETRY_SUBSTRATE_QUALIFIED"),
		StageCapabilityID("15_RIGGING_SURFACE_QUALIFIED"),
		StageCapabilityID("17_MECHANICAL_PARTITION_QUALIFIED"),
		ProductInputCapabilityID,
		ProductShapeCapabilityID,
		ProductMeshCapabilityID,
	} {
		if _, ok := registry.Get(id); !ok {
			t.Fatalf("release registry missing %s", id)
		}
	}
	resolved, err := registry.Resolve(ProductMeshCapabilityID)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, descriptor := range resolved {
		seen[descriptor.ID] = true
	}
	for _, id := range []string{
		ProductInputCapabilityID,
		ProductShapeCapabilityID,
		StageCapabilityID("05_CAMERA_CONTRACT_SOLVED"),
		StageCapabilityID("07_OBSERVATION_CONTRACT_QUALIFIED"),
		StageCapabilityID("08_NORMALIZATION_DOMAIN_QUALIFIED"),
		StageCapabilityID("13_GEOMETRY_SUBSTRATE_QUALIFIED"),
		StageCapabilityID("15_RIGGING_SURFACE_QUALIFIED"),
		StageCapabilityID("17_MECHANICAL_PARTITION_QUALIFIED"),
		ProductMeshCapabilityID,
	} {
		if !seen[id] {
			t.Fatalf("mesh resolution missing %s: %+v", id, resolved)
		}
	}
	if resolved[len(resolved)-1].ID != ProductMeshCapabilityID {
		t.Fatalf("mesh must resolve after all dependencies: %+v", resolved)
	}
}

func TestProductProvidersAreSingleSourceOfNonStageProductGraph(t *testing.T) {
	providers := ProductProviders()
	if len(providers) != 3 {
		t.Fatalf("product provider count=%d", len(providers))
	}
	registry := productProvidersTestRegistry(t)
	if _, ok := registry.Get(ProductInputCapabilityID); !ok {
		t.Fatal("product.input missing")
	}
	shape, ok := registry.Get(ProductShapeCapabilityID)
	if !ok || len(shape.Dependencies) != 1 || shape.Dependencies[0] != ProductInputCapabilityID {
		t.Fatalf("shape descriptor=%+v", shape)
	}
	mesh, ok := registry.Get(ProductMeshCapabilityID)
	if !ok {
		t.Fatal("product.mesh missing")
	}
	if len(mesh.Dependencies) != 7 || mesh.Dependencies[0] != ProductShapeCapabilityID {
		t.Fatalf("mesh descriptor=%+v", mesh)
	}
	for _, dependency := range mesh.Dependencies {
		if dependency == StageCapabilityID("16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED") {
			t.Fatal("presentation direction leaked into Mesh birth dependency")
		}
	}
}

func TestModuleVersionPolicyAndParametersAreRederivedFromDescriptor(t *testing.T) {
	registry := productProvidersTestRegistry(t)
	for _, capabilityID := range []string{
		ProductInputCapabilityID,
		ProductShapeCapabilityID,
		ProductMeshCapabilityID,
	} {
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
