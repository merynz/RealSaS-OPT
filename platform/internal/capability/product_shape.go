package capability

import "context"

const (
	ProductShapeCapabilityID = "product.shape"
	ProductShapeContractV1   = "RealSaS.ProductShapeInference.v1"
)

// ProductShapeProvider exposes the first inference-only product capability.
// It is deliberately developer-only and cannot mint ProductRevision authority.
// A future complete product graph may depend on this capability, but only the
// final Proof terminal is allowed to own product-pass authority.
type ProductShapeProvider struct{}

func (ProductShapeProvider) Capabilities(context.Context) ([]Descriptor, error) {
	return []Descriptor{{
		ID:               ProductShapeCapabilityID,
		Kind:             KindModule,
		OwnerModuleID:    "product.shape",
		ExecutorActivity: "engine.execute_capability.v1",
		AllowedModes:     []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:          []string{"Shape", "IRIS", "TESSA"},
		Metadata: map[string]any{
			"executor_contract":     ProductShapeContractV1,
			"input_artifact_type":   "RealSaS.ShapeInput",
			"input_schema_version":  "v1",
			"output_artifact_type":  "RealSaS.ShapeProposal",
			"output_schema_version": "v1",
		},
	}}, nil
}
