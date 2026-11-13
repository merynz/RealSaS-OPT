package capability

import (
	"context"
	"strings"
	"testing"
)

// Product eligibility means the exact role graph is addressable, not that a
// checkpoint, source, Proof or ProductRevision has passed its release courts.
func TestRSSvNextProductGoalIsResolvableOnlyThroughTerminalProof(t *testing.T) {
	descriptors, err := productProviderCatalog(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(descriptors) != 9 {
		t.Fatalf("active product roles=%d, want 9", len(descriptors))
	}
	registry, err := NewRegistry(descriptors)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range descriptors {
		product, developer := false, false
		for _, mode := range row.AllowedModes {
			if mode == ModeProduct {
				product = true
			}
			if mode == ModeDeveloper {
				developer = true
			}
		}
		if !product || !developer {
			t.Fatalf("%s missing product/developer mode: %v", row.ID, row.AllowedModes)
		}
		if row.ProductPassAuthority != (row.ID == "product.proof") {
			t.Fatalf("%s has unexpected product-pass authority: %v", row.ID, row.ProductPassAuthority)
		}
	}

	goal := Goal{
		Type: GoalProductCompile, Targets: []string{"product.proof"},
		PromotionPolicy: PromotionQualifiedProductOnly,
	}
	resolved, err := goal.Validate(registry)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 9 || resolved[len(resolved)-1].ID != "product.proof" {
		t.Fatalf("resolved product graph must terminate with Proof: %+v", resolved)
	}
	if _, err := (Goal{
		Type: GoalProductCompile, Targets: []string{"product.graphics"},
		PromotionPolicy: PromotionQualifiedProductOnly,
	}).Validate(registry); err == nil || !strings.Contains(err.Error(), "product-pass authority") {
		t.Fatalf("non-Proof target unexpectedly qualified: %v", err)
	}
	if _, err := (Goal{
		Type: GoalProductCompile, Targets: []string{"product.proof"},
		PromotionPolicy: PromotionNever,
	}).Validate(registry); err == nil {
		t.Fatal("product goal must reject promotion NEVER")
	}
	if _, err := (Goal{
		Type: GoalDeveloperRun, Targets: []string{"product.proof"},
		PromotionPolicy: PromotionQualifiedProductOnly,
	}).Validate(registry); err == nil {
		t.Fatal("developer goal must never gain product promotion")
	}
}
