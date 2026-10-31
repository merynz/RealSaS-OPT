package capability

import "context"

const (
	ProductProofCapabilityID = "product.proof"
	ProductProofContractV1   = "RealSaS.ProductProofCompile.v1"
)

// ProductProofProvider is the terminal product-pass authority for the modular
// shipping graph. It consumes exact canonical product artifacts and executes
// deterministic domain courts. It does not run learned inference and it does
// not project the current product back into the legacy CanonicalPuppetGraphV3.
type ProductProofProvider struct{}

func (ProductProofProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleProof)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:               ProductProofCapabilityID,
		Kind:             KindModule,
		OwnerModuleID:    "product.proof",
		ExecutorActivity: "engine.execute_capability.v1",
		Dependencies: []string{
			ProductMeshCapabilityID,
			ProductRigCapabilityID,
			ProductSkinCapabilityID,
			ProductMechanicsCapabilityID,
			ProductGraphicsCapabilityID,
			ProductMotionCapabilityID,
		},
		AllowedModes:         []Mode{ModeDeveloper, ModeProduct},
		ProductPassAuthority: true,
		Aliases:              []string{"Proof", "ProductPass"},
		Metadata: map[string]any{
			"product_role_id":                       string(ProductRoleProof),
			"implementation_class":                  string(ImplementationCompiler),
			"interface_contract":                    ProductProofContractV1,
			"executor_contract":                     ProductProofContractV1,
			"input_mesh_artifact_type":              "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version":             "v1",
			"input_rig_artifact_type":               "RealSaS.CanonicalRigIR",
			"input_rig_schema_version":              "v2",
			"input_skin_artifact_type":              "RealSaS.CanonicalSkinIR",
			"input_skin_schema_version":             "v2",
			"input_mechanics_artifact_type":         "RealSaS.CanonicalMechanicsIR",
			"input_mechanics_schema_version":        "v1",
			"input_graphics_artifact_type":          "RealSaS.CanonicalGraphicsIR",
			"input_graphics_schema_version":         "v1",
			"input_motion_artifact_type":            "RealSaS.CanonicalMotionIR",
			"input_motion_schema_version":           "v2",
			"output_artifact_type":                  "RealSaS.ProductProofBundleIR",
			"output_schema_version":                 "v1",
			"required_models":                       []any{},
			"proof_evaluator":                       "RealSaS.ModularProductProofEvaluator.v1",
			"product_state_schema":                  "RealSaS.ModularProductState.v1",
			"legacy_puppet_projection_required":     false,
			"learned_inference_allowed":             false,
			"heuristic_fallback_allowed":            false,
			"product_revision_minting_owned_here":   false,
			"post_export_native_interlock_required": true,
			"implementation_owner_may_change":       true,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids":                    responsibilities,
		},
	}}, nil
}
