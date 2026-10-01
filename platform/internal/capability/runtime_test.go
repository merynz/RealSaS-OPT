package capability

import (
	"context"
	"testing"
)

func TestRuntimeRenderCapabilityIsDeveloperOnlyAndNonPromoting(t *testing.T) {
	registry, err := Build(context.Background(), RuntimeProvider{})
	if err != nil {
		t.Fatal(err)
	}
	d, ok := registry.Get(RuntimeRenderCompilerRunCapabilityID)
	if !ok {
		t.Fatal("runtime render capability missing")
	}
	if d.Kind != KindRuntime || d.OwnerModuleID != "engine.runtime_binding" {
		t.Fatalf("descriptor=%+v", d)
	}
	if len(d.AllowedModes) != 1 || d.AllowedModes[0] != ModeDeveloper {
		t.Fatalf("allowed_modes=%v", d.AllowedModes)
	}
	if d.ProductPassAuthority {
		t.Fatal("developer runtime render must never own product promotion")
	}
	if d.Metadata["operation"] != "RENDER_COMPILER_RUN" {
		t.Fatalf("metadata=%v", d.Metadata)
	}
}
