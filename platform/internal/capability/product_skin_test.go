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
		ProductMeshCapabilityID:                         true,
		ProductRigCapabilityID:                          true,
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
	if got := d.Metadata["activation_state"]; got != "BLOCKED" {
		t.Fatalf("activation_state = %v", got)
	}
	if got := d.Metadata["activation_blocker"]; got != "CURRENT_MIRA_MODELRELEASE_NOT_REGISTERED" {
		t.Fatalf("activation_blocker = %v", got)
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
	if got := d.Metadata["current_repo_mira_v55_runtime_source_recovered"]; got != true {
		t.Fatalf("current MIRA runtime source recovery claim = %v", got)
	}
	if got := d.Metadata["current_mira_v55_runtime_package_recovered"]; got != true {
		t.Fatalf("current MIRA runtime package recovery claim = %v", got)
	}
	if got := d.Metadata["current_platform_mira_modelrelease_bound"]; got != false {
		t.Fatalf("current MIRA ModelRelease claim = %v", got)
	}
	if got := d.Metadata["post_hoc_surface_skin_transfer_allowed"]; got != false {
		t.Fatalf("post-hoc skin transfer policy = %v", got)
	}
}

func TestBlockedSkinIsNotAnActiveProductProvider(t *testing.T) {
	for _, provider := range ProductProviders() {
		descriptors, err := provider.Capabilities(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		for _, descriptor := range descriptors {
			if descriptor.ID == ProductSkinCapabilityID {
				t.Fatal("blocked product.skin must not enter active EngineRelease capability graph")
			}
		}
	}
}
