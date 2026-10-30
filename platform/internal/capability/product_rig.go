package capability

import "context"

const (
	ProductRigCapabilityID = "product.rig"
	ProductRigContractV2   = "RealSaS.ProductRigInference.v2"
)

// ProductRigProvider owns the stable Rig product boundary. AXIS V6 proposes the
// complete control-ready mechanical program and joint-coupled discrete skin
// codes in one causal execution. Compiler validates/materializes only; it may
// not reparent, synthesize controls/constraints, repair frames, search skin
// codes, or execute the constraint program. Constraint execution and post-bind
// frame observability belong to Mechanics. RiggingSurface remains replaceable
// feature evidence, not a canonical parent.
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
			"product_role_id":              string(ProductRoleRig),
			"implementation_class":         string(ImplementationHybrid),
			"interface_contract":           ProductRigContractV2,
			"executor_contract":            ProductRigContractV2,
			"input_mesh_artifact_type":     "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version":    "v1",
			"feature_evidence_type":        "RealSaS.RiggingSurfaceIR",
			"feature_evidence_version":     "v1",
			"feature_evidence_source":      "PRODUCT_MESH_OUTPUT",
			"output_artifact_type":         "RealSaS.CanonicalRigIR",
			"output_schema_version":        "v2",
			"rig_program_proposal_type":    "RealSaS.RigProgramProposalIR",
			"rig_program_proposal_version": "v1",
			"skin_token_output_type":       "RealSaS.SkinTokenProposalIR",
			"skin_token_output_version":    "v1",
			"maximum_new_tokens":           4096,
			"required_models": []any{
				map[string]any{
					"family":          "AXIS",
					"architecture_id": "RealSaS.AXIS.RigProgram.TypedCausal.ControlSkin.v6_0",
				},
			},
			"model_checkpoint_artifact_type":                "RealSaS.AXISRuntimeCheckpoint",
			"model_checkpoint_schema_version":               "v2",
			"mesh_is_canonical_parent":                      true,
			"feature_evidence_is_canonical_parent":          false,
			"legacy_stage15_dependency":                     false,
			"canonical_root_semantics":                      "NON_EMPTY_DEFORM_ROOT_SET",
			"compiler_parent_reselection_allowed":           false,
			"compiler_control_synthesis_allowed":            false,
			"compiler_constraint_synthesis_allowed":         false,
			"compiler_frame_repair_allowed":                 false,
			"compiler_constraint_solver_execution_allowed":  false,
			"skin_tokens_generated_in_same_causal_axis_run": true,
			"constraint_execution_owner":                    "MECHANICS",
			"frame_observability_claimed":                   false,
			"post_bind_frame_observability_required":        true,
			"frame_observability_owner":                     "MECHANICS",
			"fit_allowed_during_product_execution":          false,
			"implementation_owner_may_change":               true,
			"stable_product_role_survives_rewiring":         true,
			"responsibility_ids":                            responsibilities,
		},
	}}, nil
}
