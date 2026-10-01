package capability

import "context"

const RuntimeRenderCompilerRunCapabilityID = "runtime.render_compiler_run"

type RuntimeProvider struct{}

func (RuntimeProvider) Capabilities(_ context.Context) ([]Descriptor, error) {
	return []Descriptor{
		{
			ID:               RuntimeRenderCompilerRunCapabilityID,
			Kind:             KindRuntime,
			OwnerModuleID:    "engine.runtime_binding",
			ExecutorActivity: "engine.execute_capability.v1",
			AllowedModes:     []Mode{ModeDeveloper},
			Aliases:          []string{"render", "runtime render", "compiler run render"},
			Metadata: map[string]any{
				"operation":              "RENDER_COMPILER_RUN",
				"compiler_run_bridge":    true,
				"product_promotion":      false,
				"fit_train_calibration":  false,
				"output_artifact_type":   "RealSaS.DeveloperRender",
				"output_schema_version":  "v1",
			},
		},
	}, nil
}
