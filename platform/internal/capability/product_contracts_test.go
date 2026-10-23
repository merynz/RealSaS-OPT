package capability

import (
	"context"
	"testing"
)

func TestProductProvidersMatchEmbeddedCrossLanguageContract(t *testing.T) {
	registry, err := Build(context.Background(), ProductProviders()...)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateProductDescriptorContracts(registry); err != nil {
		t.Fatal(err)
	}
}

func TestEmbeddedProductContractRejectsDescriptorMetadataDrift(t *testing.T) {
	registry, err := Build(context.Background(), ProductProviders()...)
	if err != nil {
		t.Fatal(err)
	}
	rows := registry.Descriptors()
	for i := range rows {
		if rows[i].ID != ProductMeshCapabilityID {
			continue
		}
		metadata := map[string]any{}
		for key, value := range rows[i].Metadata {
			metadata[key] = value
		}
		metadata["source_fidelity_policy"] = "DRIFTED"
		rows[i].Metadata = metadata
	}
	drifted, err := NewRegistry(rows)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateProductDescriptorContracts(drifted); err == nil {
		t.Fatal("shared product semantic contract accepted descriptor drift")
	}
}
