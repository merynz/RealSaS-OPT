package orchestration

import "testing"

func TestCapabilityWorkflowAuthorityForProductAndDeveloper(t *testing.T) {
	developer := ResolvedCapabilityGoal{
		GoalType: "DEVELOPER_RUN", PromotionPolicy: "NEVER",
		Steps: []CapabilityStep{{CapabilityID: "product.proof"}},
	}
	if err := validateCapabilityWorkflowAuthority(developer); err != nil {
		t.Fatal(err)
	}
	product := ResolvedCapabilityGoal{
		GoalType: "PRODUCT_COMPILE", PromotionPolicy: "QUALIFIED_PRODUCT_ONLY",
		Steps: []CapabilityStep{{CapabilityID: "product.input"}, {CapabilityID: "product.proof"}},
	}
	if err := validateCapabilityWorkflowAuthority(product); err != nil {
		t.Fatal(err)
	}
	product.Steps[len(product.Steps)-1].CapabilityID = "product.graphics"
	if err := validateCapabilityWorkflowAuthority(product); err == nil {
		t.Fatal("product execution must terminate with Proof")
	}
	for _, invalid := range []ResolvedCapabilityGoal{
		{GoalType: "DEVELOPER_RUN", PromotionPolicy: "QUALIFIED_PRODUCT_ONLY"},
		{GoalType: "PRODUCT_COMPILE", PromotionPolicy: "NEVER"},
		{GoalType: "PROOF_RUN", PromotionPolicy: "QUALIFIED_PRODUCT_ONLY"},
	} {
		if err := validateCapabilityWorkflowAuthority(invalid); err == nil {
			t.Fatalf("unauthorized goal admitted: %+v", invalid)
		}
	}
}
