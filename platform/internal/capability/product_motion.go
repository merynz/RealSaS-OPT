package capability

import "context"

const (
	ProductMotionCapabilityID = "product.motion"
	ProductMotionContractV2   = "RealSaS.ProductMotionCompile.v2"
)

// ProductMotionProvider owns editable animation authority in animator-control
// space. It admits a typed control-motion proposal against the exact canonical
// Rig/Mechanics program and mints CanonicalMotionIR.v2. Constraint execution is
// delegated to Mechanics; direct deform-joint tracks are derived bake/proof only.
type ProductMotionProvider struct{}

func (ProductMotionProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleMotion)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:               ProductMotionCapabilityID,
		Kind:             KindModule,
		OwnerModuleID:    "product.motion",
		ExecutorActivity: "engine.execute_capability.v1",
		Dependencies: []string{
			ProductRigCapabilityID,
			ProductMechanicsCapabilityID,
		},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"Motion", "ControlMotion"},
		Metadata: map[string]any{
			"product_role_id":                       string(ProductRoleMotion),
			"implementation_class":                  string(ImplementationHybrid),
			"interface_contract":                    ProductMotionContractV2,
			"executor_contract":                     ProductMotionContractV2,
			"input_rig_artifact_type":               "RealSaS.CanonicalRigIR",
			"input_rig_schema_version":              "v2",
			"input_mechanics_artifact_type":         "RealSaS.CanonicalMechanicsIR",
			"input_mechanics_schema_version":        "v1",
			"input_motion_proposal_artifact_type":   "RealSaS.ControlMotionClipProposalIR",
			"input_motion_proposal_schema_version":  "v1",
			"output_artifact_type":                  "RealSaS.CanonicalMotionIR",
			"output_schema_version":                 "v2",
			"required_models":                       []any{},
			"editable_motion_space":                 "ANIMATOR_CONTROLS",
			"direct_joint_track_authority":          false,
			"derived_joint_bakes_allowed":           true,
			"constraint_execution_owner":            "MECHANICS",
			"compiler_control_synthesis_allowed":    false,
			"compiler_retargeting_allowed":          false,
			"compiler_joint_curve_bake_allowed":     false,
			"compiler_constraint_execution_allowed": false,
			"learned_motion_provider_replaceable":   true,
			"fit_allowed_during_product_execution":  false,
			"implementation_owner_may_change":       true,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids":                    responsibilities,
		},
	}}, nil
}
