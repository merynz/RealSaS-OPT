package capability

import "context"

const (
	ProductShapeCapabilityID = "product.shape"
	ProductShapeContractV1   = "RealSaS.ProductShapeInference.v1"
)

// ProductShapeProvider exposes inference-only learned Shape evidence.
// Input lowering is an explicit dependency so source-view/frontend research can
// evolve without being hidden inside Shape. Shape still cannot mint Mesh or
// ProductRevision authority.
type ProductShapeProvider struct{}

func (ProductShapeProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleShape)
	if err != nil {
		return nil, err
	}
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
			"product_role_id":       string(ProductRoleShape),
			"implementation_class":  string(ImplementationModel),
			"interface_contract":    ProductShapeContractV1,
			"executor_contract":     ProductShapeContractV1,
			"input_artifact_type":   "RealSaS.ShapeInput",
			"input_schema_version":  "v1",
			"output_artifact_type":  "RealSaS.ShapeProposal",
			"output_schema_version": "v1",
			"required_models": []any{
				map[string]any{"family": "IRIS", "architecture_id": "RealSaS.IRIS.VNextEvidenceModel.v3"},
				map[string]any{"family": "TESSA", "architecture_id": "RealSaS.TESSA.Structuralizer.v3"},
			},
			"implementation_owner_may_change":       true,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids":                    responsibilities,
		},
	}}, nil
}
