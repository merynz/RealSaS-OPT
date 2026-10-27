package capability

import (
	"context"
	"reflect"
	"testing"
)

func providerDescriptor(t *testing.T, provider Provider) Descriptor {
	t.Helper()
	rows, err := provider.Capabilities(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("provider rows=%d", len(rows))
	}
	return rows[0]
}

func requirementFamilies(rows []ProductModelRequirement) []string {
	out := make([]string, 0, len(rows))
	for _, row := range rows {
		out = append(out, row.Family)
	}
	return out
}

func TestResolvedProductModelRequirementsFollowResolvedClosure(t *testing.T) {
	input := providerDescriptor(t, ProductInputProvider{})
	shape := providerDescriptor(t, ProductShapeProvider{})
	mesh := providerDescriptor(t, ProductMeshProvider{})
	rig := providerDescriptor(t, ProductRigProvider{})

	rows, err := ResolvedProductModelRequirements([]Descriptor{input})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 0 {
		t.Fatalf("input-only requirements=%+v", rows)
	}

	rows, err = ResolvedProductModelRequirements([]Descriptor{input, shape})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := requirementFamilies(rows), []string{"IRIS", "TESSA"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("shape requirements=%v want=%v", got, want)
	}

	rows, err = ResolvedProductModelRequirements([]Descriptor{input, shape, mesh, rig})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := requirementFamilies(rows), []string{"AXIS", "IRIS", "TESSA"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("rig requirements=%v want=%v", got, want)
	}
}
