package capability

import "context"

const (
	ProductSkinCapabilityID = "product.skin"
	ProductSkinContractV2   = "RealSaS.ProductSkinInference.v2"
)

// ProductSkinProvider owns the stable Skin product boundary. AXIS already emitted
// joint-coupled discrete SkinTokens during product.rig. product.skin does not
// rerun AXIS: it verifies the exact codec binding, decodes the tokens through the
// admitted MIRA V6 codec on CanonicalMesh + Rig deform structure, then Compiler
// qualifies carrier coverage/simplex legality and mints CanonicalSkinIR.v2.
type ProductSkinProvider struct{}

func (ProductSkinProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleSkin)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:               ProductSkinCapabilityID,
		Kind:             KindModule,
		OwnerModuleID:    "product.skin",
		ExecutorActivity: "engine.execute_capability.v1",
		Dependencies: []string{ProductMeshCapabilityID, ProductRigCapabilityID},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"Skin", "MIRA"},
		Metadata: map[string]any{
			"product_role_id":                     string(ProductRoleSkin),
			"implementation_class":                string(ImplementationHybrid),
			"interface_contract":                  ProductSkinContractV2,
			"executor_contract":                   ProductSkinContractV2,
			"input_mesh_artifact_type":            "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version":           "v1",
			"input_rig_artifact_type":             "RealSaS.CanonicalRigIR",
			"input_rig_schema_version":            "v2",
			"input_skin_token_artifact_type":      "RealSaS.SkinTokenProposalIR",
			"input_skin_token_schema_version":     "v1",
			"output_artifact_type":                "RealSaS.CanonicalSkinIR",
			"output_schema_version":               "v2",
			"carrier_skin_qualification": map[string]any{
				"max_simplex_repair_l1":   1.0e-5,
				"max_total_correction_l1": 1.0e-4,
				"negative_tolerance":      1.0e-8,
				"max_influences":          256,
			},
			"required_models": []any{
				map[string]any{
					"family":          "MIRA",
					"architecture_id": "RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0",
				},
			},
			"model_checkpoint_artifact_type":       "RealSaS.MIRARuntimeCheckpoint",
			"model_checkpoint_schema_version":      "v2",
			"canonical_parents":                    []any{"CANONICAL_MESH", "CANONICAL_RIG_DEFORM_STRUCTURE"},
			"full_control_program_is_skin_parent":  false,
			"axis_rerun_allowed":                   false,
			"compiler_token_search_allowed":        false,
			"compiler_token_repair_allowed":        false,
			"semantic_skin_transfer_allowed":       false,
			"post_hoc_surface_skin_transfer_allowed": false,
			"fit_allowed_during_product_execution": false,
			"implementation_owner_may_change":      true,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids":                    responsibilities,
		},
	}}, nil
}