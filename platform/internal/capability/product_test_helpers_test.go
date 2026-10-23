package capability

import (
	"context"
	"testing"
)

func productProvidersTestRegistry(t *testing.T) *Registry {
	t.Helper()
	providers := ProductProviders()
	var descriptors []Descriptor
	for _, provider := range providers {
		rows, err := provider.Capabilities(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		descriptors = append(descriptors, rows...)
	}
	seen := map[string]bool{}
	for _, descriptor := range descriptors {
		seen[descriptor.ID] = true
	}
	for _, descriptor := range append([]Descriptor(nil), descriptors...) {
		for _, dependency := range descriptor.Dependencies {
			if seen[dependency] || dependency == ProductInputCapabilityID || dependency == ProductShapeCapabilityID {
				continue
			}
			descriptors = append(descriptors, Descriptor{
				ID:                   dependency,
				Kind:                 KindStage,
				OwnerModuleID:        "test.compiler",
				ExecutorActivity:     "test.execute",
				AllowedModes:         []Mode{ModeDeveloper},
				ProductPassAuthority: false,
				Metadata:             map[string]any{"stage_id": dependency},
			})
			seen[dependency] = true
		}
	}
	registry, err := NewRegistry(descriptors)
	if err != nil {
		t.Fatal(err)
	}
	return registry
}
