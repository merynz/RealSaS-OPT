package capability

import (
	"context"
	"testing"
)

func TestProductInputCapabilityIsDeveloperOnlyAndReplaceable(t *testing.T) {
	registry, err := Build(context.Background(), ProductInputProvider{})
	if err != nil {
		t.Fatal(err)
	}
	descriptor, ok := registry.Get(ProductInputCapabilityID)
	if !ok {
		t.Fatal("product.input capability missing")
	}
	if descriptor.Kind != KindModule || descriptor.OwnerModuleID != "product.input" {
		t.Fatalf("descriptor=%+v", descriptor)
	}
	if descriptor.ProductPassAuthority {
		t.Fatal("Input may not claim product-pass authority")
	}
	if len(descriptor.AllowedModes) != 1 || descriptor.AllowedModes[0] != ModeDeveloper {
		t.Fatalf("allowed modes=%v", descriptor.AllowedModes)
	}
	if descriptor.Metadata["executor_contract"] != ProductInputContractV1 {
		t.Fatalf("metadata=%v", descriptor.Metadata)
	}
	if descriptor.Metadata["platform_view_count_invariant"] != false {
		t.Fatalf("view count leaked into Platform invariant: %v", descriptor.Metadata)
	}
	if descriptor.Metadata["scientific_contract_is_replaceable"] != true {
		t.Fatalf("replaceability contract missing: %v", descriptor.Metadata)
	}
}

func TestShapeResolutionIncludesInputFirst(t *testing.T) {
	registry, err := Build(context.Background(), ProductInputProvider{}, ProductShapeProvider{})
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := registry.Resolve(ProductShapeCapabilityID)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 2 || resolved[0].ID != ProductInputCapabilityID || resolved[1].ID != ProductShapeCapabilityID {
		t.Fatalf("resolved=%+v", resolved)
	}
}
