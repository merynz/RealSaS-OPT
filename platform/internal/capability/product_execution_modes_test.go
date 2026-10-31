package capability

import (
	"context"
	"testing"
)

func TestProductProvidersAreProductExecutableAndOnlyProofOwnsProductPass(t *testing.T) {
	providers := ProductProviders()
	if len(providers) != 9 {
		t.Fatalf("product provider count=%d want=9", len(providers))
	}

	expected := map[string]bool{
		ProductInputCapabilityID:     false,
		ProductShapeCapabilityID:     false,
		ProductMeshCapabilityID:      false,
		ProductRigCapabilityID:       false,
		ProductSkinCapabilityID:      false,
		ProductMechanicsCapabilityID: false,
		ProductGraphicsCapabilityID:  false,
		ProductMotionCapabilityID:    false,
		ProductProofCapabilityID:     true,
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
		wantPass, ok := expected[descriptor.ID]
		if !ok {
			t.Fatalf("unexpected current product provider %s", descriptor.ID)
		}
		modes := map[Mode]bool{}
		for _, mode := range descriptor.AllowedModes {
			modes[mode] = true
		}
		if !modes[ModeDeveloper] || !modes[ModeProduct] {
			t.Fatalf("%s modes=%v must allow DEVELOPER and PRODUCT", descriptor.ID, descriptor.AllowedModes)
		}
		if descriptor.ProductPassAuthority != wantPass {
			t.Fatalf("%s product-pass authority=%v want=%v", descriptor.ID, descriptor.ProductPassAuthority, wantPass)
		}
		seen[descriptor.ID] = true
	}
	if len(seen) != len(expected) {
		t.Fatalf("seen current product providers=%v expected=%v", seen, expected)
	}
}
