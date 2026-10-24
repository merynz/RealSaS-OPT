package capability

import "context"

const (
	ProductSkinCapabilityID = "product.skin"
	ProductSkinContractV1   = "RealSaS.ProductSkinInference.v1"
)

// ProductSkinProvider defines the stable Skin product boundary without making it
// an active release provider yet. Canonical parents are Mesh + Rig. Current MIRA
// V5.5 additionally consumes RiggingSurface and TESSA material-support evidence;
// neither is a canonical Skin parent and both remain replaceable feature inputs.
// MIRA proposes carrier-native influences; Compiler qualifies exact bindings and
// simplex legality, then mints CanonicalSkinIR.
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
		Dependencies: []string{
			ProductMeshCapabilityID,
			ProductRigCapabilityID,
			StageCapabilityID("15_RIGGING_SURFACE_QUALIFIED"),
		},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"Skin", "MIRA"},
		Metadata: map[string]any{
			"product_role_id":           string(ProductRoleSkin),
			"implementation_class":      string(ImplementationHybrid),
			"interface_contract":        ProductSkinContractV1,
			"executor_contract":         ProductSkinContractV1,
			"input_mesh_artifact_type":  "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version": "v1",
			"input_rig_artifact_type":   "RealSaS.CanonicalRigIR",
			"input_rig_schema_version":  "v1",
			"feature_evidence_type":     "RealSaS.RiggingSurfaceIR",
			"feature_evidence_version":  "v1",
			"material_support_type":     "RealSaS.TESSAMaterialSupportFieldIR",
			"material_support_version":  "v1",
			"output_artifact_type":      "RealSaS.CanonicalSkinIR",
			"output_schema_version":     "v1",
			"decode_chunk":              512,
			"carrier_skin_qualification": map[string]any{
				"max_simplex_repair_l1":   1.0e-5,
				"max_total_correction_l1": 1.0e-4,
				"negative_tolerance":      1.0e-8,
				"max_influences":          41,
			},
			"required_models": []any{
				map[string]any{
					"family":          "MIRA",
					"architecture_id": "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5",
				},
			},
			"model_checkpoint_artifact_type":       "RealSaS.MIRARuntimeCheckpoint",
			"model_checkpoint_schema_version":      "v1",
			"mesh_and_rig_are_canonical_parents":   true,
			"feature_evidence_is_canonical_parent": false,
			"material_support_is_canonical_parent": false,
			"inference_compute_contract":           "CUDA_BF16_BACKBONE_FP32_READOUT_TF32_OFF",
			"cuda_required":                        true,
			"cuda_bf16_required":                   true,
			"tf32_allowed":                         false,
			"activation_state":                     "BLOCKED",
			"activation_blocker":                   "CURRENT_MIRA_MODELRELEASE_NOT_REGISTERED",

			// Historical scientific truth, intentionally not upgraded to current
			// Platform/product authority merely by source/runtime recovery.
			"historical_checkpoint_backed_mira_v55_exists":    true,
			"historical_mira_v55_learned_directly_on_exact_m": true,
			"historical_mira_v55_terminal_pass":               true,
			"historical_mira_v55_composite_schema":            "RealSaS.MIRAMDirectComposite.v5_5",
			"historical_mira_v55_result_schema":               "RealSaS.MIRAMDirectResult.v5_5",
			"historical_mira_v55_product_authority":           false,
			"historical_mira_v55_generalization_claimed":      false,
			"historical_mira_v55_compute_profile_preserved":   true,
			"current_repo_mira_v55_runtime_source_recovered":  true,
			"current_mira_v55_runtime_package_recovered":      true,
			"current_platform_mira_modelrelease_bound":        false,
			"canonical_parents":                               []any{"CANONICAL_MESH", "CANONICAL_RIG"},
			"post_hoc_surface_skin_transfer_allowed":          false,
			"hidden_repair_allowed":                           false,
			"fit_allowed_during_product_execution":            false,
			"implementation_owner_may_change":                 true,
			"stable_product_role_survives_rewiring":           true,
			"responsibility_ids":                              responsibilities,
		},
	}}, nil
}
