package capability

import "context"

const (
	ProductSkinCapabilityID = "product.skin"
	ProductSkinContractV1   = "RealSaS.ProductSkinInference.v1"
)

// ProductSkinProvider defines the stable Skin product boundary without making it
// an active release provider yet. The canonical parents are Mesh + Rig. MIRA is
// the intended influence-proposal owner, while the Compiler owns legality,
// carrier binding and canonical Skin minting.
//
// Historical FIT1 evidence contains a real checkpoint-backed MIRA-M V5.5 run:
// it learned W_M directly on the exact frozen carrier M, reached TERMINAL_PASS,
// and sealed a RealSaS.MIRAMDirectComposite.v5_5 checkpoint. That historical
// scientific witness is not currently packaged/imported as a Platform
// ModelRelease/runtime provider in this repository. Therefore this provider MUST
// NOT enter ProductProviders until the sealed MIRA runtime is recovered/package-
// qualified and bound through the normal ModelRelease path.
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
			"output_artifact_type":      "RealSaS.CanonicalSkinIR",
			"output_schema_version":     "v1",
			"required_models": []any{
				map[string]any{
					"family":          "MIRA",
					"architecture_id": "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5",
				},
			},
			"activation_state":   "BLOCKED",
			"activation_blocker": "SEALED_CURRENT_MIRA_MODELRELEASE_RUNTIME_PACKAGE_MISSING",

			// Historical scientific truth, intentionally not upgraded to current
			// Platform/product authority.
			"historical_checkpoint_backed_mira_v55_exists":    true,
			"historical_mira_v55_learned_directly_on_exact_m": true,
			"historical_mira_v55_terminal_pass":               true,
			"historical_mira_v55_composite_schema":            "RealSaS.MIRAMDirectComposite.v5_5",
			"historical_mira_v55_result_schema":               "RealSaS.MIRAMDirectResult.v5_5",
			"historical_mira_v55_product_authority":           false,
			"historical_mira_v55_generalization_claimed":      false,
			"current_repo_mira_v55_runtime_provider_present":  false,
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
