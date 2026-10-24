package control

import (
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/merynz/RealSaS-OPT/platform/internal/capability"
	"github.com/merynz/RealSaS-OPT/platform/internal/modelselection"
)

func modelRoutingDescriptor(role string, implementation capability.ImplementationClass, required ...map[string]any) capability.Descriptor {
	items := make([]any, 0, len(required))
	for _, row := range required {
		items = append(items, row)
	}
	return capability.Descriptor{
		ID:   "product." + role,
		Kind: capability.KindModule,
		Metadata: map[string]any{
			"product_role_id":      role,
			"implementation_class": string(implementation),
			"interface_contract":   "RealSaS.Test." + role,
			"required_models":      items,
		},
	}
}

func modelRoutingSelection(family, architecture string) modelselection.Selection {
	return modelselection.Selection{
		Family:         family,
		ArchitectureID: architecture,
		ModelReleaseID: uuid.New(),
	}
}

func modelRoutingFamilies(rows []modelselection.Selection) []string {
	out := make([]string, 0, len(rows))
	for _, row := range rows {
		out = append(out, row.Family)
	}
	return out
}

func TestCapabilityModelRefsRoutesOnlyDescriptorOwnedFamilies(t *testing.T) {
	selected := []modelselection.Selection{
		modelRoutingSelection("AXIS", "axis-v1"),
		modelRoutingSelection("IRIS", "iris-v1"),
		modelRoutingSelection("TESSA", "tessa-v1"),
	}
	shape := modelRoutingDescriptor(
		"shape",
		capability.ImplementationModel,
		map[string]any{"family": "IRIS", "architecture_id": "iris-v1"},
		map[string]any{"family": "TESSA", "architecture_id": "tessa-v1"},
	)
	refs, err := capabilityModelRefs(shape, selected)
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != 2 || refs[0].Family != "IRIS" || refs[1].Family != "TESSA" {
		t.Fatalf("shape refs=%v", refs)
	}

	mesh := modelRoutingDescriptor("mesh", capability.ImplementationCompiler)
	refs, err = capabilityModelRefs(mesh, selected)
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != 0 {
		t.Fatalf("mesh must not receive learned releases: %v", refs)
	}

	rig := modelRoutingDescriptor(
		"rig",
		capability.ImplementationHybrid,
		map[string]any{"family": "AXIS", "architecture_id": "axis-v1"},
	)
	refs, err = capabilityModelRefs(rig, selected)
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != 1 || refs[0].Family != "AXIS" {
		t.Fatalf("rig refs=%v", refs)
	}
}

func TestCapabilityModelRefsFailsClosedOnMissingOrDriftedOwnedModel(t *testing.T) {
	shape := modelRoutingDescriptor(
		"shape",
		capability.ImplementationModel,
		map[string]any{"family": "IRIS", "architecture_id": "iris-v1"},
		map[string]any{"family": "TESSA", "architecture_id": "tessa-v1"},
	)
	_, err := capabilityModelRefs(shape, []modelselection.Selection{
		modelRoutingSelection("IRIS", "iris-v1"),
	})
	if err == nil || !strings.Contains(err.Error(), "CAPABILITY_MODEL_RELEASE_REQUIRED:product.shape:TESSA") {
		t.Fatalf("missing TESSA err=%v", err)
	}

	_, err = capabilityModelRefs(shape, []modelselection.Selection{
		modelRoutingSelection("IRIS", "iris-v2"),
		modelRoutingSelection("TESSA", "tessa-v1"),
	})
	if err == nil || !strings.Contains(err.Error(), "CAPABILITY_MODEL_ARCHITECTURE_DRIFT:product.shape:IRIS") {
		t.Fatalf("architecture drift err=%v", err)
	}
}

func TestCapabilityModelRefsPreservesNonProductCompatibility(t *testing.T) {
	selected := []modelselection.Selection{
		modelRoutingSelection("AXIS", "axis-v1"),
		modelRoutingSelection("IRIS", "iris-v1"),
	}
	descriptor := capability.Descriptor{ID: "module/test.compat", Kind: capability.KindModule}
	refs, err := capabilityModelRefs(descriptor, selected)
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != len(selected) {
		t.Fatalf("non-product refs=%v", refs)
	}
	for i, want := range modelRoutingFamilies(selected) {
		if refs[i].Family != want {
			t.Fatalf("non-product family[%d]=%s want=%s", i, refs[i].Family, want)
		}
	}
}
