package capability

import "testing"

func TestProductRoleReadProjectionKeepsLegacySnapshotsInspectable(t *testing.T) {
	legacy, err := NewRegistry([]Descriptor{{
		ID: "product.shape", Kind: KindModule,
		OwnerModuleID: "legacy.shape", ExecutorActivity: "engine.execute_capability.v1",
		AllowedModes: []Mode{ModeDeveloper},
		Metadata: map[string]any{"executor_contract": "RealSaS.ProductShapeInference.v1"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	nodes, status, err := ProductRoleReadProjection(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if status != ProductRoleProjectionLegacyUndeclared || len(nodes) != 0 {
		t.Fatalf("status=%s nodes=%+v", status, nodes)
	}
}

func TestProductRoleReadProjectionRejectsMixedAuthorityEpoch(t *testing.T) {
	declared := roleDescriptor("product.input", ProductRoleInput, ImplementationHybrid)
	legacy := Descriptor{
		ID: "product.shape", Kind: KindModule,
		OwnerModuleID: "legacy.shape", ExecutorActivity: "engine.execute_capability.v1",
		Dependencies: []string{"product.input"}, AllowedModes: []Mode{ModeDeveloper},
		Metadata: map[string]any{"executor_contract": "RealSaS.ProductShapeInference.v1"},
	}
	registry, err := NewRegistry([]Descriptor{declared, legacy})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := ProductRoleReadProjection(registry); err == nil {
		t.Fatal("mixed product-role authority epoch accepted")
	}
}

func TestProductRoleReadProjectionUsesStrictGraphForDeclaredSnapshots(t *testing.T) {
	registry, err := NewRegistry([]Descriptor{
		roleDescriptor("product.input", ProductRoleInput, ImplementationHybrid),
		roleDescriptor("product.shape", ProductRoleShape, ImplementationModel, "product.input"),
	})
	if err != nil {
		t.Fatal(err)
	}
	nodes, status, err := ProductRoleReadProjection(registry)
	if err != nil {
		t.Fatal(err)
	}
	if status != ProductRoleProjectionDeclared || len(nodes) != 2 {
		t.Fatalf("status=%s nodes=%+v", status, nodes)
	}
}
