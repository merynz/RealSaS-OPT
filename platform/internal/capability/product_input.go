package capability

import "context"

const (
	ProductInputCapabilityID = "product.input"
	ProductInputContractV1   = "RealSaS.ProductInputInference.v1"
)

// ProductInputProvider exposes the source-to-model-input lowering boundary and
// the deterministic source qualification authorities required by shipping Mesh.
// The capability identity is stable; its versioned source/output schemas carry
// replaceable scientific assumptions such as current view count and frontend.
type ProductInputProvider struct{}

func (ProductInputProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleInput)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:                   ProductInputCapabilityID,
		Kind:                 KindModule,
		OwnerModuleID:        "product.input",
		ExecutorActivity:     "engine.execute_capability.v1",
		AllowedModes:         []Mode{ModeDeveloper, ModeProduct},
		ProductPassAuthority: false,
		Aliases:              []string{"Input", "Observation"},
		Metadata: map[string]any{
			"product_role_id":                       string(ProductRoleInput),
			"implementation_class":                  string(ImplementationHybrid),
			"interface_contract":                    ProductInputContractV1,
			"executor_contract":                     ProductInputContractV1,
			"input_artifact_type":                   "RealSaS.SourceObservation",
			"input_schema_version":                  "v1",
			"output_artifact_type":                  "RealSaS.ShapeInput",
			"output_schema_version":                 "v1",
			"camera_output_artifact_type":           "RealSaS.QualifiedCameraSetIR",
			"camera_output_schema_version":          "v1",
			"observation_output_artifact_type":      "RealSaS.QualifiedObservationSetIR",
			"observation_output_schema_version":     "v1",
			"normalization_output_artifact_type":    "RealSaS.NormalizationDomainIR",
			"normalization_output_schema_version":   "v1",
			"source_qualification_owner":            "PRODUCT_INPUT",
			"legacy_compiler_stage_bridge_required": false,
			"platform_view_count_invariant":         false,
			"source_truth_copied_into_output":       false,
			"scientific_contract_is_replaceable":    true,
			"implementation_owner_may_change":       true,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids":                    responsibilities,
		},
	}}, nil
}
