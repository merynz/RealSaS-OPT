package capability

import (
	"context"
	"testing"
)

func TestProductProvidersMatchSharedSemanticContract(t *testing.T) {
	registry, err := Build(context.Background(), ProductProviders()...)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateProductDescriptorSemantics(registry); err != nil {
		t.Fatal(err)
	}
}

func TestSharedSemanticContractRejectsMetadataDrift(t *testing.T) {
	registry, err := Build(context.Background(), ProductProviders()...)
	if err != nil {
		t.Fatal(err)
	}
	rows := registry.Descriptors()
	for i := range rows {
		if rows[i].ID != ProductShapeCapabilityID {
			continue
		}
		metadata := map[string]any{}
		for key, value := range rows[i].Metadata {
			metadata[key] = value
		}
		metadata["output_schema_version"] = "DRIFTED"
		rows[i].Metadata = metadata
	}
	drifted, err := NewRegistry(rows)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateProductDescriptorSemantics(drifted); err == nil {
		t.Fatal("shared product semantic contract accepted metadata drift")
	}
}

func TestSharedSemanticContractIgnoresResponsibilityIDsBecauseLedgerOwnsThem(t *testing.T) {
	registry, err := Build(context.Background(), ProductProviders()...)
	if err != nil {
		t.Fatal(err)
	}
	rows := registry.Descriptors()
	for i := range rows {
		if rows[i].ID != ProductShapeCapabilityID {
			continue
		}
		metadata := map[string]any{}
		for key, value := range rows[i].Metadata {
			metadata[key] = value
		}
		metadata["responsibility_ids"] = []string{"shape.topology_proposal"}
		rows[i].Metadata = metadata
	}
	drifted, err := NewRegistry(rows)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateProductDescriptorSemantics(drifted); err != nil {
		t.Fatalf("capability semantics must not duplicate responsibility authority: %v", err)
	}
	if _, err := ResponsibilityTerrain(drifted); err == nil {
		t.Fatal("responsibility ledger failed to reject missing required responsibility")
	}
}
