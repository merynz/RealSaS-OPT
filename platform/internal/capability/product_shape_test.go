package capability

import (
	"context"
	"strings"
	"testing"
)

func TestProductShapeCapabilityIsDeveloperOnlyAndCannotPromote(t *testing.T) {
	registry, err := Build(context.Background(), ProductInputProvider{}, ProductShapeProvider{})
	if err != nil {
		t.Fatal(err)
	}
	descriptor, ok := registry.Get(ProductShapeCapabilityID)
	if !ok {
		t.Fatal("product.shape capability missing")
	}
	if descriptor.Kind != KindModule || descriptor.OwnerModuleID != "product.shape" {
		t.Fatalf("descriptor=%+v", descriptor)
	}
	if len(descriptor.Dependencies) != 1 || descriptor.Dependencies[0] != ProductInputCapabilityID {
		t.Fatalf("dependencies=%v", descriptor.Dependencies)
	}
	if descriptor.ProductPassAuthority {
		t.Fatal("Shape may not claim product-pass authority")
	}
	if len(descriptor.AllowedModes) != 1 || descriptor.AllowedModes[0] != ModeDeveloper {
		t.Fatalf("allowed modes=%v", descriptor.AllowedModes)
	}
	if descriptor.Metadata["executor_contract"] != ProductShapeContractV1 {
		t.Fatalf("metadata=%v", descriptor.Metadata)
	}

	if _, err := (Goal{
		Type: GoalDeveloperRun, Targets: []string{ProductShapeCapabilityID}, PromotionPolicy: PromotionNever,
	}).Validate(registry); err != nil {
		t.Fatalf("developer Shape goal rejected: %v", err)
	}
	_, err = (Goal{
		Type: GoalProductCompile, Targets: []string{ProductShapeCapabilityID}, PromotionPolicy: PromotionQualifiedProductOnly,
	}).Validate(registry)
	if err == nil || !strings.Contains(err.Error(), "not product-pass authority") {
		t.Fatalf("product promotion should fail closed, err=%v", err)
	}
}
