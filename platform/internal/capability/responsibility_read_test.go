package capability

import (
	"context"
	"testing"
)

func TestResponsibilityReadProjectionDeclaresCurrentTerrain(t *testing.T) {
	registry, err := Build(context.Background(), ProductInputProvider{}, ProductShapeProvider{})
	if err != nil {
		t.Fatal(err)
	}
	terrain, status, err := ResponsibilityReadProjection(registry)
	if err != nil {
		t.Fatal(err)
	}
	if status != ResponsibilityProjectionDeclared || len(terrain) != 3 {
		t.Fatalf("status=%s terrain=%d", status, len(terrain))
	}
}

func TestResponsibilityReadProjectionKeepsLegacySnapshotInspectable(t *testing.T) {
	registry, err := NewRegistry([]Descriptor{{
		ID: ProductShapeCapabilityID,
		Kind: KindModule,
		OwnerModuleID: "legacy.shape",
		ExecutorActivity: "engine.execute_capability.v1",
		AllowedModes: []Mode{ModeDeveloper},
		Metadata: map[string]any{
			"product_role_id":      string(ProductRoleShape),
			"implementation_class": string(ImplementationModel),
			"interface_contract":   "RealSaS.LegacyShape.v1",
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	terrain, status, err := ResponsibilityReadProjection(registry)
	if err != nil {
		t.Fatal(err)
	}
	if status != ResponsibilityProjectionLegacyUndeclared || len(terrain) != 0 {
		t.Fatalf("status=%s terrain=%d", status, len(terrain))
	}
}
