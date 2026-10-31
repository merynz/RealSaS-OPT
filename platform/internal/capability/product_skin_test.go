package capability

import (
	"context"
	"testing"
)

func TestProductSkinDescriptorUsesDeformStructureParentClosure(t *testing.T) {
	descriptors, err := (ProductSkinProvider{}).Capabilities(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(descriptors) != 1 {
		t.Fatalf("skin descriptor count = %d, want 1", len(descriptors))
	}
	d := descriptors[0]
	if d.ID != ProductSkinCapabilityID {
		t.Fatalf("skin id = %q", d.ID)
	}
	if d.ProductPassAuthority {
		t.Fatal("Skin must not own ProductRevision PASS authority")
	}
	if len(d.AllowedModes) != 2 || d.AllowedModes[0] != ModeDeveloper || d.AllowedModes[1] != ModeProduct {
		t.Fatalf("Skin execution modes = %v", d.AllowedModes)
	}
	wantDeps := map[string]bool{
		ProductMeshCapabilityID: true,
		ProductRigCapabilityID:  true,
	}
	if len(d.Dependencies) != len(wantDeps) {
		t.Fatalf("skin dependencies = %v", d.Dependencies)
	}
	for _, dep := range d.Dependencies {
		if !wantDeps[dep] {
			t.Fatalf("unexpected Skin dependency %q", dep)
		}
		if dep == StageCapabilityID("15_RIGGING_SURFACE_QUALIFIED") {
			t.Fatal("legacy Stage15 leaked into Skin parent closure")
		}
	}

	parents, ok := d.Metadata["canonical_parents"].([]any)
	if !ok || len(parents) != 2 || parents[0] != "CANONICAL_MESH" || parents[1] != "CANONICAL_RIG_DEFORM_STRUCTURE" {
		t.Fatalf("Skin canonical parent contract = %#v", d.Metadata["canonical_parents"])
	}
	if got := d.Metadata["full_control_program_is_skin_parent"]; got != false {
		t.Fatalf("full control program skin-parent claim = %v", got)
	}
	if got := d.Metadata["axis_rerun_allowed"]; got != false {
		t.Fatalf("AXIS rerun policy = %v", got)
	}
	if got := d.Metadata["compiler_code_search_allowed"]; got != false {
		t.Fatalf("Compiler code search policy = %v", got)
	}
	if got := d.Metadata["compiler_code_repair_allowed"]; got != false {
		t.Fatalf("Compiler code repair policy = %v", got)
	}
	if got := d.Metadata["post_hoc_surface_skin_transfer_allowed"]; got != false {
		t.Fatalf("post-hoc skin transfer policy = %v", got)
	}

	if got := d.Metadata["model_release_admission_owner"]; got != "PLATFORM_MODEL_REQUIREMENTS" {
		t.Fatalf("model release admission owner = %v", got)
	}
	wantResearch := []any{"RESEARCH", "CANDIDATE"}
	gotResearch, ok := d.Metadata["research_execution_model_channels"].([]any)
	if !ok || len(gotResearch) != len(wantResearch) {
		t.Fatalf("research model channels = %#v", d.Metadata["research_execution_model_channels"])
	}
	for i := range wantResearch {
		if gotResearch[i] != wantResearch[i] {
			t.Fatalf("research model channels = %#v", gotResearch)
		}
	}
	gotProduct, ok := d.Metadata["product_execution_model_channels"].([]any)
	if !ok || len(gotProduct) != 1 || gotProduct[0] != "PRODUCTION" {
		t.Fatalf("product model channels = %#v", d.Metadata["product_execution_model_channels"])
	}
	if got := d.Metadata["operational_model_state_embedded_in_source"]; got != false {
		t.Fatalf("operational model state source policy = %v", got)
	}
}

func TestSkinIsCataloguedWithStaticModelRequirementOnly(t *testing.T) {
	found := false
	for _, provider := range ProductProviders() {
		descriptors, err := provider.Capabilities(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		for _, descriptor := range descriptors {
			if descriptor.ID != ProductSkinCapabilityID {
				continue
			}
			found = true
			if descriptor.Metadata["model_release_admission_owner"] != "PLATFORM_MODEL_REQUIREMENTS" {
				t.Fatalf("Skin admission escaped Platform model selection: %+v", descriptor.Metadata)
			}
			required, ok := descriptor.Metadata["required_models"].([]any)
			if !ok || len(required) != 1 {
				t.Fatalf("Skin static model requirement = %#v", descriptor.Metadata["required_models"])
			}
			for _, forbidden := range []string{
				"activation_state",
				"activation_blocker",
				"current_platform_mira_modelrelease_bound",
			} {
				if _, exists := descriptor.Metadata[forbidden]; exists {
					t.Fatalf("operational Platform state leaked into Skin source metadata: %s", forbidden)
				}
			}
		}
	}
	if !found {
		t.Fatal("product.skin missing from first-class product catalog")
	}
}
