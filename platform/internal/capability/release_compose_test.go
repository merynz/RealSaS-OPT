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
	registry, err := BuildEngineReleaseRegistry(context.Background(), meshReleaseTestGraph(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{
		StageCapabilityID("05_CAMERA_CONTRACT_SOLVED"),
		StageCapabilityID("07_OBSERVATION_CONTRACT_QUALIFIED"),
		StageCapabilityID("08_NORMALIZATION_DOMAIN_QUALIFIED"),
		ProductInputCapabilityID, ProductShapeCapabilityID, ProductMeshCapabilityID,
		ProductRigCapabilityID, ProductSkinCapabilityID, "product.mechanics",
		"product.graphics", "product.motion", "product.proof",
	} {
		if _, ok := registry.Get(id); !ok {
			t.Fatalf("release registry missing %s", id)
		}
	}
	for _, testcase := range []struct {
		target string
		want   []string
	}{
		{ProductRigCapabilityID, []string{ProductInputCapabilityID, ProductShapeCapabilityID,
			ProductMeshCapabilityID, ProductRigCapabilityID}},
		{ProductSkinCapabilityID, []string{ProductInputCapabilityID, ProductShapeCapabilityID,
			ProductMeshCapabilityID, ProductRigCapabilityID, ProductSkinCapabilityID}},
	} {
		resolved, err := registry.Resolve(testcase.target)
		if err != nil {
			t.Fatal(err)
		}
		if len(resolved) != len(testcase.want) {
			t.Fatalf("%s stage-free dependency closure drift: %+v", testcase.target, resolved)
		}
		for index, want := range testcase.want {
			if resolved[index].ID != want {
				t.Fatalf("%s dependency[%d]=%s want=%s", testcase.target, index, resolved[index].ID, want)
			}
		}
	}
	// A staged research graph must not silently insert retired eight-view,
	// fit or zero-surface compiler stages into new RSS product dependencies.
	for _, role := range []string{ProductMeshCapabilityID, ProductRigCapabilityID, ProductSkinCapabilityID} {
		closure, err := registry.Resolve(role)
		if err != nil {
			t.Fatal(err)
		}
		for _, row := range closure {
			if strings.HasPrefix(row.ID, "compiler.stage/") {
				t.Fatalf("%s unexpectedly depends on historical stage %s", role, row.ID)
			}
		}
	}
}

func TestProductProvidersAreSingleSourceOfNonStageProductGraph(t *testing.T) {
	if providers := ProductProviders(); len(providers) != 9 {
		t.Fatalf("product provider count=%d want=9", len(providers))
	}
	registry := productProvidersTestRegistry(t)
	for _, testcase := range []struct {
		id           string
		dependencies []string
		contract     string
		modelCount   int
	}{
		{ProductInputCapabilityID, nil, RSSvNextInputContractV3, 0},
		{ProductShapeCapabilityID, []string{ProductInputCapabilityID}, RSSvNextShapeContractV4, 1},
		{ProductMeshCapabilityID, []string{ProductShapeCapabilityID}, RSSvNextMeshContractV3, 0},
		{ProductRigCapabilityID, []string{ProductShapeCapabilityID, ProductMeshCapabilityID}, RSSvNextRigContractV4, 0},
		{ProductSkinCapabilityID, []string{ProductShapeCapabilityID, ProductMeshCapabilityID, ProductRigCapabilityID}, RSSvNextSkinContractV4, 0},
	} {
		row, ok := registry.Get(testcase.id)
		if !ok {
			t.Fatalf("missing product role %s", testcase.id)
		}
		if len(row.Dependencies) != len(testcase.dependencies) {
			t.Fatalf("%s dependency count drift: %+v", testcase.id, row.Dependencies)
		}
		for i, want := range testcase.dependencies {
			if row.Dependencies[i] != want {
				t.Fatalf("%s dependency[%d]=%s want=%s", testcase.id, i, row.Dependencies[i], want)
			}
		}
		if row.Metadata["executor_contract"] != testcase.contract {
			t.Fatalf("%s executor contract drift: %+v", testcase.id, row.Metadata)
		}
		models, err := descriptorRequiredModels(row)
		if err != nil || len(models) != testcase.modelCount {
			t.Fatalf("%s independently selected model release: %+v / %v", testcase.id, models, err)
		}
		if row.ProductPassAuthority {
			t.Fatalf("%s claimed ProductPASS authority", testcase.id)
		}
	}
	mesh, _ := registry.Get(ProductMeshCapabilityID)
	if mesh.Metadata["output_artifact_type"] != "RealSaS.CanonicalMeshIR" ||
		mesh.Metadata["canonical_mesh_required_for_product_execution"] != true ||
		mesh.Metadata["requires_legacy_eight_view_stages"] != false ||
		mesh.Metadata["independent_model_inference_forbidden"] != true {
		t.Fatalf("RSS canonical Mesh source admission drift: %+v", mesh.Metadata)
	}
	rig, _ := registry.Get(ProductRigCapabilityID)
	if rig.Metadata["compiler_control_synthesis_allowed"] != false ||
		rig.Metadata["compiler_constraint_synthesis_allowed"] != false ||
		rig.Metadata["single_checkpoint_inference_reused"] != true ||
		rig.Metadata["canonical_mesh_parent_required"] != true {
		t.Fatalf("RSS Rig rewrote learned control program: %+v", rig.Metadata)
	}
	skin, _ := registry.Get(ProductSkinCapabilityID)
	if skin.Metadata["input_skin_code_artifact_type"] != "RealSaS.SkinCodeProposalIR" ||
		skin.Metadata["exact_embedded_mira_decoder_replayed"] != true ||
		skin.Metadata["compiler_code_repair_allowed"] != false ||
		skin.Metadata["post_hoc_skin_transfer_allowed"] != false {
		t.Fatalf("RSS Skin rewrote generated FSQ evidence: %+v", skin.Metadata)
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
