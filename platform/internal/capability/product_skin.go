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
// Current repository evidence does not contain a checkpoint-backed MIRA V5.5
// runtime producer. The historical carrier-native V5.5 witness is an input-
// implementation promotion, not a learned-model promotion. Therefore this
// provider MUST NOT be added to ProductProviders until a sealed MIRA ModelRelease
// can satisfy this contract.
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
			"activation_state":                       "BLOCKED",
			"activation_blocker":                     "CHECKPOINT_BACKED_MIRA_RUNTIME_MISSING",
			"historical_carrier_witness_is_model":    false,
			"historical_promotion_kind":              "CARRIER_NATIVE_INPUT_IMPLEMENTATION_PROMOTION",
			"canonical_parents":                      []any{"CANONICAL_MESH", "CANONICAL_RIG"},
			"post_hoc_surface_skin_transfer_allowed": false,
			"hidden_repair_allowed":                  false,
			"fit_allowed_during_product_execution":   false,
			"implementation_owner_may_change":        true,
			"stable_product_role_survives_rewiring":  true,
			"responsibility_ids":                     responsibilities,
		},
	}}, nil
}
