package capability

import (
	"context"
	"testing"
)

func TestProductSkinDescriptorUsesModernParentClosure(t *testing.T) {
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
	if len(d.AllowedModes) != 1 || d.AllowedModes[0] != ModeDeveloper {
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
	if got := d.Metadata["mesh_and_rig_are_canonical_parents"]; got != true {
		t.Fatalf("Mesh/Rig canonical parent contract = %v", got)
	}
	if got := d.Metadata["executor_contract"]; got != ProductSkinContractV2 {
		t.Fatalf("active Skin executor contract = %v", got)
	}
	if got := d.Metadata["input_rig_schema_version"]; got != "v2" {
		t.Fatalf("canonical Rig parent version = %v", got)
	}
	if got := d.Metadata["input_skin_code_artifact_type"]; got != "RealSaS.SkinCodeProposalIR" {
		t.Fatalf("AXIS generated skin-code parent = %v", got)
	}
	if got := d.Metadata["input_skin_code_source"]; got != "PRODUCT_RIG_OUTPUT" {
		t.Fatalf("skin-code provenance = %v", got)
	}
	if got := d.Metadata["output_schema_version"]; got != "v2" {
		t.Fatalf("canonical Skin output version = %v", got)
	}
	if got := d.Metadata["compiler_code_search_allowed"]; got != false {
		t.Fatalf("Compiler must not synthesize or search learned codes = %v", got)
	}
	if got := d.Metadata["compiler_code_repair_allowed"]; got != false {
		t.Fatalf("Compiler must not repair learned codes = %v", got)
	}
	if got := d.Metadata["codec_checkpoint_must_match_axis_skin_code"]; got != true {
		t.Fatalf("MIRA and AXIS checkpoint binding policy = %v", got)
	}
	if got := d.Metadata["legacy_stage15_dependency"]; got != false {
		t.Fatalf("legacy Stage15 dependency = %v", got)
	}
	for _, obsolete := range []string{"feature_evidence_source", "material_support_source", "decode_chunk", "cuda_bf16_required"} {
		if _, exists := d.Metadata[obsolete]; exists {
			t.Fatalf("retired V5 skin contract leaked into V6 active descriptor: %s", obsolete)
		}
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
	// Historical V5 evidence belongs to archival receipts, not the active
	// checkpoint-to-product contract.
	for _, legacy := range []string{"historical_checkpoint_backed_mira_v55_exists", "current_repo_mira_v55_runtime_source_recovered"} {
		if _, exists := d.Metadata[legacy]; exists {
			t.Fatalf("archival V5 claim leaked into active V6 catalog: %s", legacy)
		}
	}
	if got := d.Metadata["post_hoc_surface_skin_transfer_allowed"]; got != false {
		t.Fatalf("post-hoc skin transfer policy = %v", got)
	}
}

func TestSkinIsCataloguedWithoutEmbeddingLiveActivationState(t *testing.T) {
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
			models, err := descriptorRequiredModels(descriptor)
			if err != nil || len(models) != 0 ||
				descriptor.Metadata["single_checkpoint_inference_reused"] != true ||
				descriptor.Metadata["independent_mira_model_release_allowed"] != false ||
				descriptor.Metadata["exact_embedded_mira_decoder_replayed"] != true {
				t.Fatalf("Skin must only consume embedded MIRA from the single RSS model: %+v / %v", descriptor.Metadata, err)
			}
			for _, forbidden := range []string{"activation_state", "activation_blocker", "current_platform_mira_modelrelease_bound"} {
				if _, exists := descriptor.Metadata[forbidden]; exists {
					t.Fatalf("live activation state leaked into Skin source: %s", forbidden)
				}
			}
		}
	}
	if !found {
		t.Fatal("product.skin missing from first-class product catalog")
	}
}
