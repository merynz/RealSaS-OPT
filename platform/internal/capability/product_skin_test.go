package capability

import (
	"context"
	"testing"
)

func TestProductSkinDescriptorIsTruthfulButInactive(t *testing.T) {
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
	wantDeps := map[string]bool{
		ProductMeshCapabilityID:                           true,
		ProductRigCapabilityID:                            true,
		StageCapabilityID("15_RIGGING_SURFACE_QUALIFIED"): true,
	}
	if len(d.Dependencies) != len(wantDeps) {
		t.Fatalf("skin dependencies = %v", d.Dependencies)
	}
	for _, dep := range d.Dependencies {
		if !wantDeps[dep] {
			t.Fatalf("unexpected Skin dependency %q", dep)
		}
	}
	if got := d.Metadata["mesh_and_rig_are_canonical_parents"]; got != true {
		t.Fatalf("Mesh/Rig canonical parent contract = %v", got)
	}
	if got := d.Metadata["feature_evidence_is_canonical_parent"]; got != false {
		t.Fatalf("RiggingSurface canonical parent claim = %v", got)
	}
	if got := d.Metadata["material_support_is_canonical_parent"]; got != false {
		t.Fatalf("material-support canonical parent claim = %v", got)
	}
	if got := d.Metadata["inference_compute_contract"]; got != "CUDA_BF16_BACKBONE_FP32_READOUT_TF32_OFF" {
		t.Fatalf("MIRA compute contract = %v", got)
	}
	if got := d.Metadata["cuda_required"]; got != true {
		t.Fatalf("MIRA CUDA requirement = %v", got)
	}
	if got := d.Metadata["cuda_bf16_required"]; got != true {
		t.Fatalf("MIRA BF16 requirement = %v", got)
	}
	if got := d.Metadata["tf32_allowed"]; got != false {
		t.Fatalf("MIRA TF32 policy = %v", got)
	}
	if got := d.Metadata["model_release_admission_owner"]; got != "PLATFORM_MODEL_REQUIREMENTS" {
		t.Fatalf("model release admission owner = %v", got)
	}
	if got := d.Metadata["operational_model_state_embedded_in_source"]; got != false {
		t.Fatalf("operational model state source policy = %v", got)
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
	for _, forbidden := range []string{
		"activation_state",
		"activation_blocker",
		"current_platform_mira_modelrelease_bound",
	} {
		if _, exists := d.Metadata[forbidden]; exists {
			t.Fatalf("operational Platform state leaked into Skin source metadata: %s", forbidden)
		}
	}
	if got := d.Metadata["historical_checkpoint_backed_mira_v55_exists"]; got != true {
		t.Fatalf("historical MIRA checkpoint claim = %v", got)
	}
	if got := d.Metadata["historical_mira_v55_learned_directly_on_exact_m"]; got != true {
		t.Fatalf("historical direct-on-M learned claim = %v", got)
	}
	if got := d.Metadata["historical_mira_v55_terminal_pass"]; got != true {
		t.Fatalf("historical MIRA terminal pass = %v", got)
	}
	if got := d.Metadata["historical_mira_v55_product_authority"]; got != false {
		t.Fatalf("historical MIRA product authority = %v", got)
	}
	if got := d.Metadata["historical_mira_v55_generalization_claimed"]; got != false {
		t.Fatalf("historical MIRA generalization claim = %v", got)
	}
	if got := d.Metadata["historical_mira_v55_compute_profile_preserved"]; got != true {
		t.Fatalf("historical MIRA compute-profile claim = %v", got)
	}
	if got := d.Metadata["current_repo_mira_v55_runtime_source_recovered"]; got != true {
		t.Fatalf("current MIRA runtime source recovery claim = %v", got)
	}
	if got := d.Metadata["current_mira_v55_runtime_package_recovered"]; got != true {
		t.Fatalf("current MIRA runtime package recovery claim = %v", got)
	}
	if got := d.Metadata["post_hoc_surface_skin_transfer_allowed"]; got != false {
		t.Fatalf("post-hoc skin transfer policy = %v", got)
	}
}

func TestSkinIsNotAnActiveProductProviderBeforeQualifiedChannelAdmission(t *testing.T) {
	for _, provider := range ProductProviders() {
		descriptors, err := provider.Capabilities(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		for _, descriptor := range descriptors {
			if descriptor.ID == ProductSkinCapabilityID {
				t.Fatal("product.skin must not enter active EngineRelease graph before qualified ModelRelease channel admission")
			}
		}
	}
}
