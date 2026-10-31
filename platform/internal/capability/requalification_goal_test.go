package capability

import (
	"context"
	"testing"
)

func productRegistryForRequalificationTest(t *testing.T) *Registry {
	t.Helper()
	registry, err := Build(context.Background(), ProductProviders()...)
	if err != nil {
		t.Fatal(err)
	}
	return registry
}

func TestResolveWithSatisfiedAnimationOnlyRunsMotionAndProof(t *testing.T) {
	registry := productRegistryForRequalificationTest(t)
	satisfied := []string{
		ProductMeshCapabilityID,
		ProductRigCapabilityID,
		ProductSkinCapabilityID,
		ProductMechanicsCapabilityID,
		ProductGraphicsCapabilityID,
	}
	resolved, err := registry.ResolveWithSatisfied(
		[]string{ProductProofCapabilityID},
		satisfied,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 2 {
		t.Fatalf("resolved count=%d want=2: %+v", len(resolved), resolved)
	}
	if resolved[0].ID != ProductMotionCapabilityID || resolved[1].ID != ProductProofCapabilityID {
		t.Fatalf("resolved=%v want motion->proof", []string{resolved[0].ID, resolved[1].ID})
	}
}

func TestProductRequalificationGoalRequiresQualifiedBaseLeaves(t *testing.T) {
	registry := productRegistryForRequalificationTest(t)
	goal := Goal{
		Type:            GoalProductRequalification,
		Targets:         []string{ProductProofCapabilityID},
		Satisfied:       []string{ProductMeshCapabilityID, ProductRigCapabilityID, ProductSkinCapabilityID, ProductMechanicsCapabilityID, ProductGraphicsCapabilityID},
		PromotionPolicy: PromotionQualifiedProductOnly,
	}
	resolved, err := goal.Validate(registry)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved) != 2 || resolved[0].ID != ProductMotionCapabilityID || resolved[1].ID != ProductProofCapabilityID {
		t.Fatalf("unexpected requalification plan: %+v", resolved)
	}

	goal.Satisfied = nil
	if _, err := goal.Validate(registry); err == nil {
		t.Fatal("requalification without satisfied base capabilities unexpectedly validated")
	}
}

func TestResolveWithSatisfiedRejectsTargetAndIrrelevantLeaves(t *testing.T) {
	registry := productRegistryForRequalificationTest(t)
	if _, err := registry.ResolveWithSatisfied(
		[]string{ProductProofCapabilityID},
		[]string{ProductProofCapabilityID},
	); err == nil {
		t.Fatal("target unexpectedly accepted as satisfied")
	}
}
