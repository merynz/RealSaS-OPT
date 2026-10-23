package capability

import "context"

const (
	ProductShapeCapabilityID = "product.shape"
	ProductShapeContractV1   = "RealSaS.ProductShapeInference.v1" // immutable legacy release contract
	ProductShapeContractV2   = "RealSaS.ProductShapeInference.v2"
)

// ProductShapeProvider exposes inference-only learned Shape evidence.
// Input lowering is an explicit dependency so source-view/frontend research can
// evolve without being hidden inside Shape. Shape still cannot mint Mesh or
// ProductRevision authority. V2 carries deterministic decoded TESSA topology so
// Compiler Mesh need not rerun the learned model.
type ProductShapeProvider struct{}

func (ProductShapeProvider) Capabilities(context.Context) ([]Descriptor, error) {
	return []Descriptor{{
		ID:                   ProductShapeCapabilityID,
		Kind:                 KindModule,
		OwnerModuleID:        "product.shape",
		ExecutorActivity:     "engine.execute_capability.v1",
		Dependencies:         []string{ProductInputCapabilityID},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"Shape", "IRIS", "TESSA"},
		Metadata: map[string]any{
			"executor_contract":     ProductShapeContractV2,
			"input_artifact_type":   "RealSaS.ShapeInput",
			"input_schema_version":  "v1",
			"output_artifact_type":  "RealSaS.ShapeProposal",
			"output_schema_version": "v2",
			"decoded_topology":      true,
			"mesh_authority":        false,
		},
	}}, nil
}
