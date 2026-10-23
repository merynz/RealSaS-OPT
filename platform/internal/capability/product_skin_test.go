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
	}
	if got := d.Metadata["activation_state"]; got != "BLOCKED" {
		t.Fatalf("activation_state = %v", got)
	}
	if got := d.Metadata["activation_blocker"]; got != "CHECKPOINT_BACKED_MIRA_RUNTIME_MISSING" {
		t.Fatalf("activation_blocker = %v", got)
	}
	if got := d.Metadata["historical_carrier_witness_is_model"]; got != false {
		t.Fatalf("historical carrier witness model claim = %v", got)
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
