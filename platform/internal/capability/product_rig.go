package capability

import "context"

const (
	ProductRigCapabilityID = "product.rig"
	ProductRigContractV2   = "RealSaS.ProductRigInference.v2"
)

// ProductRigProvider is the single active Rig product boundary. AXIS V6 emits
// the typed causal articulation/control program AND the MIRA-bound SkinCode
// proposal in the same generation. Compiler validates this learned evidence,
// mints CanonicalRigIR.v2, and never synthesizes or repairs control programs.
// RiggingSurfaceIR is exact product.mesh feature evidence, not a canonical
// parent. No V5 fallback or model-release admission is implied by this source.
type ProductRigProvider struct{}

func (ProductRigProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleRig)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:               ProductRigCapabilityID,
		Kind:             KindModule,
		OwnerModuleID:    "product.rig",
		ExecutorActivity: "engine.execute_capability.v1",
		Dependencies: []string{
			ProductMeshCapabilityID,
		},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"Rig", "AXIS"},
		Metadata: map[string]any{
			"product_role_id":                 string(ProductRoleRig),
			"implementation_class":            string(ImplementationHybrid),
			"interface_contract":              ProductRigContractV2,
			"executor_contract":               ProductRigContractV2,
			"input_mesh_artifact_type":        "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version":       "v1",
			"feature_evidence_type":           "RealSaS.RiggingSurfaceIR",
			"feature_evidence_version":        "v1",
			"feature_evidence_source":         "PRODUCT_MESH_OUTPUT",
			"output_artifact_type":            "RealSaS.CanonicalRigIR",
			"output_schema_version":           "v2",
			"skin_code_output_artifact_type":  "RealSaS.SkinCodeProposalIR",
			"skin_code_output_schema_version": "v1",
			"skin_code_output_source":         "SAME_AXIS_CAUSAL_SEQUENCE",
			"resource_step_limit_max":         4096,
			"maximum_new_tokens":              4096,
			"required_models": []any{
				map[string]any{
					"family":          "AXIS",
					"architecture_id": "RealSaS.AXIS.RigProgram.TypedCausal.ControlSkin.v6_0",
				},
			},
			"model_checkpoint_artifact_type":         "RealSaS.AXISRuntimeCheckpoint",
			"model_checkpoint_schema_version":        "v2",
			"mesh_is_canonical_parent":               true,
			"feature_evidence_is_canonical_parent":   false,
			"legacy_stage15_dependency":              false,
			"canonical_root_semantics":               "NON_EMPTY_DEFORM_ROOT_SET",
			"current_axis_root_semantics":            "LEARNED_CAUSAL_PARENT_FOREST",
			"hard_parent_authority_preserved":        true,
			"compiler_parent_reselection_allowed":    false,
			"compiler_control_synthesis_allowed":     false,
			"compiler_constraint_synthesis_allowed":  false,
			"constraint_execution_authority":         "MECHANICS",
			"frame_observability_claimed":            false,
			"post_bind_frame_observability_required": true,
			"fit_allowed_during_product_execution":   false,
			"implementation_owner_may_change":        true,
			"stable_product_role_survives_rewiring":  true,
			"responsibility_ids":                     responsibilities,
		},
	}}, nil
}
