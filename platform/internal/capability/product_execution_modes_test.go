package capability

import (
	"context"
	"testing"
)

func TestCurrentProductProvidersAllowProductExecutionWithoutClaimingProductPass(t *testing.T) {
	providers := ProductProviders()
	if len(providers) != 8 {
		t.Fatalf("product provider count=%d want=8 before Proof joins the catalog", len(providers))
	}

	expected := map[string]bool{
		ProductInputCapabilityID:     true,
		ProductShapeCapabilityID:     true,
		ProductMeshCapabilityID:      true,
		ProductRigCapabilityID:       true,
		ProductSkinCapabilityID:      true,
		ProductMechanicsCapabilityID: true,
		ProductGraphicsCapabilityID:  true,
		ProductMotionCapabilityID:    true,
	}
	seen := map[string]bool{}
	for _, provider := range providers {
		descriptors, err := provider.Capabilities(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if len(descriptors) != 1 {
			t.Fatalf("provider emitted %d descriptors want=1", len(descriptors))
		}
		descriptor := descriptors[0]
		if !expected[descriptor.ID] {
			t.Fatalf("unexpected current product provider %s", descriptor.ID)
		}
		modes := map[Mode]bool{}
		for _, mode := range descriptor.AllowedModes {
			modes[mode] = true
		}
		if !modes[ModeDeveloper] || !modes[ModeProduct] {
			t.Fatalf("%s modes=%v must allow DEVELOPER and PRODUCT", descriptor.ID, descriptor.AllowedModes)
		}
		if descriptor.ProductPassAuthority {
			t.Fatalf("%s must not mint product-pass authority; Proof owns the terminal decision", descriptor.ID)
		}
		seen[descriptor.ID] = true
	}
	if len(seen) != len(expected) {
		t.Fatalf("seen current product providers=%v expected=%v", seen, expected)
	}
}
