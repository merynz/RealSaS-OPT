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

func TestBuildEngineReleaseRegistryIncludesControlReadyProductClosure(t *testing.T) {
	registry, err := BuildEngineReleaseRegistry(context.Background(), meshReleaseTestGraph(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{
		StageCapabilityID("05_CAMERA_CONTRACT_SOLVED"),
		StageCapabilityID("07_OBSERVATION_CONTRACT_QUALIFIED"),
		StageCapabilityID("08_NORMALIZATION_DOMAIN_QUALIFIED"),
		ProductInputCapabilityID,
		ProductShapeCapabilityID,
		ProductMeshCapabilityID,
		ProductRigCapabilityID,
		ProductSkinCapabilityID,
		ProductMechanicsCapabilityID,
		ProductMotionCapabilityID,
	} {
		if _, ok := registry.Get(id); !ok {
			t.Fatalf("release registry missing %s", id)
		}
	}

	motionResolved, err := registry.Resolve(ProductMotionCapabilityID)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, descriptor := range motionResolved {
		seen[descriptor.ID] = true
	}
	for _, id := range []string{
		ProductInputCapabilityID,
		ProductShapeCapabilityID,
		ProductMeshCapabilityID,
		ProductRigCapabilityID,
		ProductSkinCapabilityID,
		ProductMechanicsCapabilityID,
		ProductMotionCapabilityID,
	} {
		if !seen[id] {
			t.Fatalf("motion closure missing %s: %+v", id, motionResolved)
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
			t.Fatalf("legacy fit-era stage leaked into product.motion closure: %s", forbidden)
		}
	}
	if motionResolved[len(motionResolved)-1].ID != ProductMotionCapabilityID {
		t.Fatalf("motion must resolve after all dependencies: %+v", motionResolved)
	}
}

func TestProductProvidersAreSingleSourceOfControlReadyProductGraph(t *testing.T) {
	providers := ProductProviders()
	if len(providers) != 7 {
		t.Fatalf("product provider count=%d", len(providers))
	}
	registry := productProvidersTestRegistry(t)

	rig, ok := registry.Get(ProductRigCapabilityID)
	if !ok {
		t.Fatal("product.rig missing")
	}
	if len(rig.Dependencies) != 1 || rig.Dependencies[0] != ProductMeshCapabilityID {
		t.Fatalf("rig descriptor=%+v", rig)
	}
	if rig.Metadata["frame_observability_claimed"] != false ||
		rig.Metadata["post_bind_frame_observability_required"] != true ||
		rig.Metadata["frame_observability_owner"] != "MECHANICS" ||
		rig.Metadata["constraint_execution_owner"] != "MECHANICS" {
		t.Fatalf("rig mechanics boundary drift: %+v", rig.Metadata)
	}

	skin, ok := registry.Get(ProductSkinCapabilityID)
	if !ok {
		t.Fatal("product.skin missing")
	}
	parents, ok := skin.Metadata["canonical_parents"].([]any)
	if !ok || len(parents) != 2 || parents[0] != "CANONICAL_MESH" || parents[1] != "CANONICAL_RIG_DEFORM_STRUCTURE" {
		t.Fatalf("skin canonical parents=%+v", skin.Metadata)
	}
	if skin.Metadata["full_control_program_is_skin_parent"] != false ||
		skin.Metadata["axis_rerun_allowed"] != false ||
		skin.Metadata["compiler_code_repair_allowed"] != false {
		t.Fatalf("skin v2 execution boundary drift: %+v", skin.Metadata)
	}

	mechanics, ok := registry.Get(ProductMechanicsCapabilityID)
	if !ok {
		t.Fatal("product.mechanics missing")
	}
	wantMechanics := []string{ProductMeshCapabilityID, ProductRigCapabilityID, ProductSkinCapabilityID}
	if len(mechanics.Dependencies) != len(wantMechanics) {
		t.Fatalf("mechanics dependencies=%v", mechanics.Dependencies)
	}
	for index, want := range wantMechanics {
		if mechanics.Dependencies[index] != want {
			t.Fatalf("mechanics dependency[%d]=%s want=%s", index, mechanics.Dependencies[index], want)
		}
	}
	if mechanics.Metadata["learned_inference_allowed"] != false || mechanics.Metadata["rig_program_mutation_allowed"] != false {
		t.Fatalf("mechanics authority drift: %+v", mechanics.Metadata)
	}

	motion, ok := registry.Get(ProductMotionCapabilityID)
	if !ok {
		t.Fatal("product.motion missing")
	}
	if len(motion.Dependencies) != 2 || motion.Dependencies[0] != ProductRigCapabilityID || motion.Dependencies[1] != ProductMechanicsCapabilityID {
		t.Fatalf("motion dependencies=%v", motion.Dependencies)
	}
	if motion.Metadata["editable_motion_space"] != "ANIMATOR_CONTROLS" ||
		motion.Metadata["direct_joint_track_authority"] != false ||
		motion.Metadata["constraint_execution_owner"] != "MECHANICS" {
		t.Fatalf("motion control-space boundary drift: %+v", motion.Metadata)
	}
	if motion.Metadata["compiler_retargeting_allowed"] != false || motion.Metadata["compiler_control_synthesis_allowed"] != false {
		t.Fatalf("motion Compiler overreach policy drift: %+v", motion.Metadata)
	}

	mesh, ok := registry.Get(ProductMeshCapabilityID)
	if !ok {
		t.Fatal("product.mesh missing")
	}
	for _, dependency := range mesh.Dependencies {
		for _, forbidden := range []string{"IRIS_FIT", "CHECKPOINT_SEALED", "ZERO_SURFACE", "GEOMETRY_SUBSTRATE", "RIGGING_SURFACE", "MECHANICAL_PARTITION"} {
			if strings.Contains(dependency, forbidden) {
				t.Fatalf("legacy derived-stage dependency leaked into product.mesh: %s", dependency)
			}
		}
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
		ProductMechanicsCapabilityID,
		ProductMotionCapabilityID,
	} {
		descriptor, ok := registry.Get(capabilityID)
		if !ok {
			t.Fatal(capabilityID)
		}
		implementation := ModuleImplementationIdentity{ImplementationSHA256: strings.Repeat("a", 64)}
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
		if version.ImplementationSHA256 != implementation.ImplementationSHA256 ||
			version.PolicySHA256 != policySHA || version.ParametersSHA256 != parametersSHA {
			t.Fatalf("%s module identity drift", capabilityID)
		}
	}
}
