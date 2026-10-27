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
		{"15_RIGGING_SURFACE_QUALIFIED", "geometry"},
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

func TestBuildEngineReleaseRegistryIncludesCurrentProductClosure(t *testing.T) {
	graph := meshReleaseTestGraph(t)
	registry, err := BuildEngineReleaseRegistry(context.Background(), graph)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{
		StageCapabilityID("05_CAMERA_CONTRACT_SOLVED"),
		StageCapabilityID("07_OBSERVATION_CONTRACT_QUALIFIED"),
		StageCapabilityID("08_NORMALIZATION_DOMAIN_QUALIFIED"),
		StageCapabilityID("15_RIGGING_SURFACE_QUALIFIED"),
		ProductInputCapabilityID,
		ProductShapeCapabilityID,
		ProductMeshCapabilityID,
		ProductRigCapabilityID,
		ProductSkinCapabilityID,
	} {
		if _, ok := registry.Get(id); !ok {
			t.Fatalf("release registry missing %s", id)
		}
	}
	resolved, err := registry.Resolve(ProductRigCapabilityID)
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
		ProductMeshCapabilityID,
		ProductRigCapabilityID,
	} {
		if !seen[id] {
			t.Fatalf("rig resolution missing %s: %+v", id, resolved)
		}
	}
	for _, forbidden := range []string{
		"09_IRIS_FIT_PREREGISTERED",
		"10_IRIS_FIT",
		"11_IRIS_CHECKPOINT_SEALED",
		"12_ZERO_SURFACE_DECODED",
		"13_GEOMETRY_SUBSTRATE_QUALIFIED",
		"14_GSA_BUILD",
		"15_RIGGING_SURFACE_QUALIFIED",
		"17_MECHANICAL_PARTITION_QUALIFIED",
	} {
		if seen[StageCapabilityID(forbidden)] {
			t.Fatalf("legacy fit-era stage leaked into product.rig closure: %s", forbidden)
		}
	}
	if resolved[len(resolved)-1].ID != ProductRigCapabilityID {
		t.Fatalf("rig must resolve after all dependencies: %+v", resolved)
	}

	skinResolved, err := registry.Resolve(ProductSkinCapabilityID)
	if err != nil {
		t.Fatal(err)
	}
	skinSeen := map[string]bool{}
	for _, descriptor := range skinResolved {
		skinSeen[descriptor.ID] = true
	}
	for _, id := range []string{
		ProductMeshCapabilityID,
		ProductRigCapabilityID,
		StageCapabilityID("15_RIGGING_SURFACE_QUALIFIED"),
		ProductSkinCapabilityID,
	} {
		if !skinSeen[id] {
			t.Fatalf("skin resolution missing %s: %+v", id, skinResolved)
		}
	}
	if skinResolved[len(skinResolved)-1].ID != ProductSkinCapabilityID {
		t.Fatalf("skin must resolve after all dependencies: %+v", skinResolved)
	}
}

func TestProductProvidersAreSingleSourceOfNonStageProductGraph(t *testing.T) {
	providers := ProductProviders()
	if len(providers) != 5 {
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
	wantMeshDeps := []string{
		ProductShapeCapabilityID,
		StageCapabilityID("05_CAMERA_CONTRACT_SOLVED"),
		StageCapabilityID("07_OBSERVATION_CONTRACT_QUALIFIED"),
		StageCapabilityID("08_NORMALIZATION_DOMAIN_QUALIFIED"),
	}
	if len(mesh.Dependencies) != len(wantMeshDeps) {
		t.Fatalf("mesh descriptor=%+v", mesh)
	}
	for index, want := range wantMeshDeps {
		if mesh.Dependencies[index] != want {
			t.Fatalf("mesh dependency[%d]=%s want=%s", index, mesh.Dependencies[index], want)
		}
	}
	for _, dependency := range mesh.Dependencies {
		if dependency == StageCapabilityID("16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED") {
			t.Fatal("presentation direction leaked into Mesh birth dependency")
		}
		for _, forbidden := range []string{"IRIS_FIT", "CHECKPOINT_SEALED", "ZERO_SURFACE", "GEOMETRY_SUBSTRATE", "RIGGING_SURFACE", "MECHANICAL_PARTITION"} {
			if strings.Contains(dependency, forbidden) {
				t.Fatalf("legacy derived-stage dependency leaked into product.mesh: %s", dependency)
			}
		}
	}
	if mesh.Metadata["fit_allowed_during_product_execution"] != false || mesh.Metadata["legacy_ir_fit_stage_dependency"] != false {
		t.Fatalf("mesh fit policy drift=%+v", mesh.Metadata)
	}
	if mesh.Metadata["surface_born_inside_mesh_court"] != true || mesh.Metadata["surface_is_canonical_mesh_authority"] != false {
		t.Fatalf("mesh surface boundary drift=%+v", mesh.Metadata)
	}

	rig, ok := registry.Get(ProductRigCapabilityID)
	if !ok {
		t.Fatal("product.rig missing")
	}
	if len(rig.Dependencies) != 1 || rig.Dependencies[0] != ProductMeshCapabilityID {
		t.Fatalf("rig descriptor=%+v", rig)
	}
	if rig.Metadata["feature_evidence_source"] != "PRODUCT_MESH_OUTPUT" || rig.Metadata["legacy_stage15_dependency"] != false {
		t.Fatalf("rig feature evidence source drift=%+v", rig.Metadata)
	}
	if rig.Metadata["mesh_is_canonical_parent"] != true {
		t.Fatalf("rig mesh parent metadata=%+v", rig.Metadata)
	}
	if rig.Metadata["feature_evidence_is_canonical_parent"] != false {
		t.Fatalf("rig feature evidence became canonical parent: %+v", rig.Metadata)
	}
	if rig.Metadata["frame_observability_claimed"] != false {
		t.Fatalf("rig prematurely claims post-bind observability: %+v", rig.Metadata)
	}

	skin, ok := registry.Get(ProductSkinCapabilityID)
	if !ok {
		t.Fatal("product.skin missing")
	}
	if skin.Metadata["mesh_and_rig_are_canonical_parents"] != true {
		t.Fatalf("skin canonical parent metadata=%+v", skin.Metadata)
	}
	if skin.Metadata["operational_model_state_embedded_in_source"] != false {
		t.Fatalf("skin source metadata leaked live model state=%+v", skin.Metadata)
	}
}

func TestModuleVersionPolicyAndParametersAreMintedFromDescriptor(t *testing.T) {
	registry := productProvidersTestRegistry(t)
	for _, capabilityID := range []string{
		ProductInputCapabilityID,
		ProductShapeCapabilityID,
		ProductMeshCapabilityID,
		ProductRigCapabilityID,
		ProductSkinCapabilityID,
	} {
		descriptor, ok := registry.Get(capabilityID)
		if !ok {
			t.Fatal(capabilityID)
		}
		implementation := ModuleImplementationIdentity{
			ImplementationSHA256: strings.Repeat("a", 64),
		}
		version, err := mintModuleVersionIdentity(descriptor, implementation)
		if err != nil {
			t.Fatalf("%s mint failed: %v", capabilityID, err)
		}
		policySHA, err := modulePolicySHA256(descriptor)
		if err != nil {
			t.Fatal(err)
		}
		parametersSHA, err := semantic.JSONSHA256(descriptor.Metadata)
		if err != nil {
			t.Fatal(err)
		}
		if version.ImplementationSHA256 != implementation.ImplementationSHA256 {
			t.Fatalf("%s implementation identity drift", capabilityID)
		}
		if version.PolicySHA256 != policySHA {
			t.Fatalf("%s policy identity not minted from descriptor", capabilityID)
		}
		if version.ParametersSHA256 != parametersSHA {
			t.Fatalf("%s parameter identity not minted from descriptor", capabilityID)
		}
	}
}
