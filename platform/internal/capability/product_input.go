package capability

import "context"

const (
	ProductInputCapabilityID = "product.input"
	ProductInputContractV1   = "RealSaS.ProductInputInference.v1"
)

// ProductInputProvider exposes the source-to-model-input lowering boundary.
// The capability identity is stable; its versioned source/output schemas carry
// replaceable scientific assumptions such as current view count and frontend.
type ProductInputProvider struct{}

func (ProductInputProvider) Capabilities(context.Context) ([]Descriptor, error) {
	return []Descriptor{{
		ID:                   ProductInputCapabilityID,
		Kind:                 KindModule,
		OwnerModuleID:        "product.input",
		ExecutorActivity:     "engine.execute_capability.v1",
		AllowedModes:         []Mode{ModeDeveloper},
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
			"platform_view_count_invariant":         false,
			"source_truth_copied_into_output":       false,
			"scientific_contract_is_replaceable":    true,
			"implementation_owner_may_change":       true,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids": []string{
				"input.model_observation_lowering",
			},
		},
	}}, nil
}
