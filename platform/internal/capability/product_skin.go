package capability

import "context"

const (
	ProductSkinCapabilityID = "product.skin"
	ProductSkinContractV1   = "RealSaS.ProductSkinInference.v1"
)

// ProductSkinProvider owns the stable Skin product boundary. MIRA is the current
// influence-proposal provider; the Compiler binds the proposal to the exact
// CanonicalMesh + CanonicalRig parents, qualifies legality/simplex budgets and
// mints CanonicalSkinIR. Stage15 is current learned feature evidence only and is
// deliberately not a canonical Skin parent.
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
			"product_role_id":             string(ProductRoleSkin),
			"implementation_class":        string(ImplementationHybrid),
			"interface_contract":          ProductSkinContractV1,
			"executor_contract":           ProductSkinContractV1,
			"input_mesh_artifact_type":    "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version":   "v1",
			"input_rig_artifact_type":     "RealSaS.CanonicalRigIR",
			"input_rig_schema_version":    "v1",
			"feature_evidence_type":       "RealSaS.RiggingSurfaceIR",
			"feature_evidence_version":    "v1",
			"output_artifact_type":        "RealSaS.CanonicalSkinIR",
			"output_schema_version":       "v1",
			"qualification_artifact_type": "RealSaS.CanonicalSkinQualificationIR",
			"proposal_artifact_type":      "RealSaS.CarrierSkinProposalIR",
			"max_simplex_repair_l1":       1.0e-5,
			"max_total_correction_l1":     1.0e-4,
			"negative_tolerance":          1.0e-9,
			"required_models": []any{
				map[string]any{
					"family":          "MIRA",
					"architecture_id": "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5",
				},
			},
			"model_checkpoint_artifact_type":         "RealSaS.MIRARuntimeCheckpoint",
			"model_checkpoint_schema_version":        "v1",
			"canonical_parent_set":                   []any{"CANONICAL_MESH", "CANONICAL_RIG"},
			"feature_evidence_is_canonical_parent":   false,
			"current_model_fit1_carrier_bound":       true,
			"current_model_fit1_rig_bound":           true,
			"generalization_claimed":                 false,
			"semantic_skin_transfer_allowed":         false,
			"gsa_skin_field_authority_allowed":       false,
			"post_bind_frame_observability_claimed":  false,
			"post_bind_frame_observability_required": true,
			"fit_allowed_during_product_execution":   false,
			"implementation_owner_may_change":        true,
			"stable_product_role_survives_rewiring":  true,
			"responsibility_ids":                     responsibilities,
		},
	}}, nil
}
