package capability

import (
	"context"
	"testing"
)

func TestProductRoleGraphSeparatesStableRoleFromImplementation(t *testing.T) {
	registry, err := Build(context.Background(), ProductInputProvider{}, ProductShapeProvider{})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := ProductRoleGraph(registry)
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) != 2 {
		t.Fatalf("roles=%d want=2", len(nodes))
	}
	byRole := map[ProductRoleID]ProductRoleNode{}
	for _, node := range nodes {
		byRole[node.RoleID] = node
	}
	input := byRole[ProductRoleInput]
	shape := byRole[ProductRoleShape]
	if input.CapabilityID != ProductInputCapabilityID || input.ImplementationClass != ImplementationHybrid {
		t.Fatalf("input=%+v", input)
	}
	if shape.CapabilityID != ProductShapeCapabilityID || shape.ImplementationClass != ImplementationModel {
		t.Fatalf("shape=%+v", shape)
	}
	if len(shape.ProductRoleDependencies) != 1 || shape.ProductRoleDependencies[0] != ProductRoleInput {
		t.Fatalf("shape deps=%v", shape.ProductRoleDependencies)
	}
}

func TestProductRoleMayMoveFromCompilerToModelWithoutChangingRoleID(t *testing.T) {
	compiler := Descriptor{
		ID: ProductShapeCapabilityID, Kind: KindModule,
		OwnerModuleID: "engine.shape", ExecutorActivity: "engine.execute_capability.v1",
		AllowedModes: []Mode{ModeDeveloper},
		Metadata: map[string]any{
			"product_role_id":      string(ProductRoleShape),
			"implementation_class": string(ImplementationCompiler),
			"interface_contract":   "RealSaS.ShapeContract.vX",
		},
	}
	model := compiler
	model.OwnerModuleID = "models.shape"
	model.Metadata = map[string]any{
		"product_role_id":      string(ProductRoleShape),
		"implementation_class": string(ImplementationModel),
		"interface_contract":   "RealSaS.ShapeContract.vX",
	}
	for _, descriptor := range []Descriptor{compiler, model} {
		registry, err := NewRegistry([]Descriptor{descriptor})
		if err != nil {
			t.Fatal(err)
		}
		nodes, err := ProductRoleGraph(registry)
		if err != nil {
			t.Fatal(err)
		}
		if len(nodes) != 1 || nodes[0].RoleID != ProductRoleShape || nodes[0].CapabilityID != ProductShapeCapabilityID {
			t.Fatalf("node=%+v", nodes)
		}
	}
}

func TestProductRoleCapabilityIdentityCannotDriftFromRole(t *testing.T) {
	registry, err := NewRegistry([]Descriptor{{
		ID: "product.mesh", Kind: KindModule,
		OwnerModuleID: "engine.mesh", ExecutorActivity: "engine.execute_capability.v1",
		AllowedModes: []Mode{ModeDeveloper},
		Metadata: map[string]any{
			"product_role_id":      string(ProductRoleShape),
			"implementation_class": string(ImplementationCompiler),
			"interface_contract":   "RealSaS.Mesh.v1",
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ProductRoleGraph(registry); err == nil {
		t.Fatal("expected product role/capability identity drift to fail")
	}
}
