package capability

import "testing"

func testVersion(ch string) VersionIdentity {
	value := ""
	for len(value) < 64 {
		value += ch
	}
	value = value[:64]
	return VersionIdentity{ImplementationSHA256: value, PolicySHA256: value, ParametersSHA256: value}
}

func roleDescriptor(id string, role ProductRoleID, implementation ImplementationClass, dependencies ...string) Descriptor {
	return Descriptor{
		ID: id, Kind: KindModule,
		OwnerModuleID:    "owner." + string(role),
		ExecutorActivity: "engine.execute_capability.v1",
		Dependencies:     append([]string(nil), dependencies...),
		AllowedModes:     []Mode{ModeDeveloper},
		Metadata: map[string]any{
			"product_role_id":      string(role),
			"implementation_class": string(implementation),
			"interface_contract":   "RealSaS." + string(role) + ".v1",
		},
	}
}

func TestCapabilityImpactTracksRolePreservingCompilerToModelMove(t *testing.T) {
	input := roleDescriptor("product.input", ProductRoleInput, ImplementationHybrid)
	oldShape := roleDescriptor("product.shape", ProductRoleShape, ImplementationCompiler, "product.input")
	newShape := roleDescriptor("product.shape", ProductRoleShape, ImplementationModel, "product.input")
	mesh := roleDescriptor("product.mesh", ProductRoleMesh, ImplementationCompiler, "product.shape")
	oldRegistry, err := NewRegistry([]Descriptor{input, oldShape, mesh})
	if err != nil {
		t.Fatal(err)
	}
	newRegistry, err := NewRegistry([]Descriptor{input, newShape, mesh})
	if err != nil {
		t.Fatal(err)
	}
	oldSnapshot := ReleaseSnapshot{
		Registry: oldRegistry,
		Versions: map[string]VersionIdentity{
			"product.input": testVersion("1"),
			"product.shape": testVersion("2"),
			"product.mesh":  testVersion("3"),
		},
	}
	newSnapshot := ReleaseSnapshot{
		Registry: newRegistry,
		Versions: map[string]VersionIdentity{
			"product.input": testVersion("1"),
			"product.shape": testVersion("4"),
			"product.mesh":  testVersion("3"),
		},
	}
	impact, err := CompareReleaseSnapshots(oldSnapshot, newSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	if len(impact.DirectChanges) != 1 {
		t.Fatalf("direct=%+v", impact.DirectChanges)
	}
	change := impact.DirectChanges[0]
	if change.CapabilityID != "product.shape" || change.ProductRoleID != ProductRoleShape || !change.DescriptorChanged || !change.ImplementationChanged {
		t.Fatalf("change=%+v", change)
	}
	if len(impact.ChangedProductRoleIDs) != 1 || impact.ChangedProductRoleIDs[0] != ProductRoleShape {
		t.Fatalf("changed roles=%v", impact.ChangedProductRoleIDs)
	}
	if len(impact.InvalidatedCapabilityIDs) != 2 || impact.InvalidatedCapabilityIDs[0] != "product.shape" || impact.InvalidatedCapabilityIDs[1] != "product.mesh" {
		t.Fatalf("invalidated=%v", impact.InvalidatedCapabilityIDs)
	}
	if len(impact.InvalidatedProductRoleIDs) != 2 || impact.InvalidatedProductRoleIDs[0] != ProductRoleMesh || impact.InvalidatedProductRoleIDs[1] != ProductRoleShape {
		t.Fatalf("invalidated roles=%v", impact.InvalidatedProductRoleIDs)
	}
	if len(impact.UnchangedCapabilityIDs) != 1 || impact.UnchangedCapabilityIDs[0] != "product.input" {
		t.Fatalf("unchanged=%v", impact.UnchangedCapabilityIDs)
	}
}
