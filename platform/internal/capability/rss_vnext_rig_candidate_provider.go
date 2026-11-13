package capability

import "context"

// RSSvNextRigCandidateProvider is the one-checkpoint replacement for the
// historical independent AXIS ProductRigProvider. It must be activated only
// with RSSvNext Input/Shape/Mesh and the new Skin consumer; the stable role is
// product.rig, never a second parallel rig capability.
type RSSvNextRigCandidateProvider struct{}

const RSSvNextRigContractV3 = "RealSaS.ProductRigInference.v3"

func (RSSvNextRigCandidateProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleRig)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:                   ProductRigCapabilityID,
		Kind:                 KindModule,
		OwnerModuleID:        "product.rig",
		ExecutorActivity:     "engine.execute_capability.v1",
		Dependencies:         []string{ProductShapeCapabilityID, ProductMeshCapabilityID},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"RSS vNext Canonical Rig", "Unified AXIS Controls"},
		Metadata: map[string]any{
			"product_role_id":                  string(ProductRoleRig),
			"implementation_class":            string(ImplementationCompiler),
			"interface_contract":              RSSvNextRigContractV3,
			"executor_contract":               RSSvNextRigContractV3,
			"required_models":                 []any{},
			"input_source_artifact_type":      "RealSaS.RSSvNextTwoViewSource",
			"input_source_schema_version":     "v1",
			"input_fanout_artifact_type":      "RealSaS.RSSvNextProductEvidenceFanout",
			"input_fanout_schema_version":     "v1",
			"input_mesh_artifact_type":        "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version":       "v1",
			"input_candidate_artifact_type":   "RealSaS.RSSvNextComputedCandidateSurface",
			"input_candidate_schema_version":  "v1",
			"output_artifact_type":            "RealSaS.CanonicalRigIR",
			"output_schema_version":           "v2",
			"skin_code_output_artifact_type":  "RealSaS.SkinCodeProposalIR",
			"skin_code_output_schema_version": "v1",
			"canonical_mesh_parent_required":  true,
			"single_checkpoint_inference_reused": true,
			"independent_axis_model_release_allowed": false,
			"independent_mira_model_release_allowed": false,
			"mechanics_attachment_qualified_by_rig": false,
			"source_joint_truth_claimed": false,
			"compiler_parent_reselection_allowed": false,
			"compiler_control_synthesis_allowed": false,
			"compiler_constraint_synthesis_allowed": false,
			"post_bind_mechanics_and_skin_qualification_required": true,
			"can_mint_product_revision": false,
			"responsibility_ids": responsibilities,
		},
	}}, nil
}
