package capability

import (
	"context"
	"testing"
)

func TestProductProofIsTerminalProductPassAuthority(t *testing.T) {
	descriptors, err := (ProductProofProvider{}).Capabilities(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(descriptors) != 1 {
		t.Fatalf("proof descriptor count=%d want=1", len(descriptors))
	}
	d := descriptors[0]
	if d.ID != ProductProofCapabilityID {
		t.Fatalf("proof id=%q", d.ID)
	}
	if !d.ProductPassAuthority {
		t.Fatal("Proof must own terminal product-pass authority")
	}
	if len(d.AllowedModes) != 2 || d.AllowedModes[0] != ModeDeveloper || d.AllowedModes[1] != ModeProduct {
		t.Fatalf("Proof execution modes=%v", d.AllowedModes)
	}
	wantDeps := []string{
		ProductMeshCapabilityID,
		ProductRigCapabilityID,
		ProductSkinCapabilityID,
		ProductMechanicsCapabilityID,
		ProductGraphicsCapabilityID,
		ProductMotionCapabilityID,
	}
	if len(d.Dependencies) != len(wantDeps) {
		t.Fatalf("Proof dependencies=%v want=%v", d.Dependencies, wantDeps)
	}
	for i := range wantDeps {
		if d.Dependencies[i] != wantDeps[i] {
			t.Fatalf("Proof dependencies=%v want=%v", d.Dependencies, wantDeps)
		}
	}
	if d.Metadata["output_artifact_type"] != "RealSaS.ProductProofBundleIR" {
		t.Fatalf("Proof output=%v", d.Metadata["output_artifact_type"])
	}
	if d.Metadata["proof_evaluator"] != "RealSaS.ModularProductProofEvaluator.v1" {
		t.Fatalf("Proof evaluator=%v", d.Metadata["proof_evaluator"])
	}
	if d.Metadata["legacy_puppet_projection_required"] != false {
		t.Fatal("modular Proof must not require legacy CanonicalPuppetGraphV3 projection")
	}
	if d.Metadata["heuristic_fallback_allowed"] != false {
		t.Fatal("Proof may not admit heuristic fallback")
	}
	if d.Metadata["product_revision_minting_owned_here"] != false {
		t.Fatal("Proof must not mint ProductRevision directly")
	}
}

func TestProductCompileCanTerminateAtProof(t *testing.T) {
	registry, err := Build(context.Background(), ProductProviders()...)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := (Goal{
		Type:            GoalProductCompile,
		Targets:         []string{ProductProofCapabilityID},
		PromotionPolicy: PromotionQualifiedProductOnly,
	}).Validate(registry)
	if err != nil {
		t.Fatalf("product.proof terminal goal rejected: %v", err)
	}
	if len(resolved) == 0 {
		t.Fatal("product.proof goal resolved to an empty execution graph")
	}
	if resolved[len(resolved)-1].ID != ProductProofCapabilityID {
		t.Fatalf("terminal capability=%s want=%s", resolved[len(resolved)-1].ID, ProductProofCapabilityID)
	}
	for _, descriptor := range resolved {
		allowed := false
		for _, mode := range descriptor.AllowedModes {
			if mode == ModeProduct {
				allowed = true
				break
			}
		}
		if !allowed {
			t.Fatalf("resolved capability %s is not PRODUCT executable", descriptor.ID)
		}
	}
}
