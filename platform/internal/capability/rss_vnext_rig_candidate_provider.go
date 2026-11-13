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
	metadata := map[string]any{}
	metadata["product_role_id"] = string(ProductRoleRig)
	metadata["implementation_class"] = string(ImplementationCompiler)
	metadata["interface_contract"] = RSSvNextRigContractV3
	metadata["executor_contract"] = RSSvNextRigContractV3
	metadata["required_models"] = []any{}
	metadata["input_source_artifact_type"] = "RealSaS.RSSvNextTwoViewSource"
	metadata["input_source_schema_version"] = "v1"
	metadata["input_fanout_artifact_type"] = "RealSaS.RSSvNextProductEvidenceFanout"
	metadata["input_fanout_schema_version"] = "v1"
	metadata["input_mesh_artifact_type"] = "RealSaS.CanonicalMeshIR"
	metadata["input_mesh_schema_version"] = "v1"
	metadata["input_candidate_artifact_type"] = "RealSaS.RSSvNextComputedCandidateSurface"
	metadata["input_candidate_schema_version"] = "v1"
	metadata["output_artifact_type"] = "RealSaS.CanonicalRigIR"
	metadata["output_schema_version"] = "v2"
	metadata["skin_code_output_artifact_type"] = "RealSaS.SkinCodeProposalIR"
	metadata["skin_code_output_schema_version"] = "v1"
	metadata["canonical_mesh_parent_required"] = true
	metadata["single_checkpoint_inference_reused"] = true
	metadata["independent_axis_model_release_allowed"] = false
	metadata["independent_mira_model_release_allowed"] = false
	metadata["mechanics_attachment_qualified_by_rig"] = false
	metadata["source_joint_truth_claimed"] = false
	metadata["compiler_parent_reselection_allowed"] = false
	metadata["compiler_control_synthesis_allowed"] = false
	metadata["compiler_constraint_synthesis_allowed"] = false
	metadata["post_bind_mechanics_and_skin_qualification_required"] = true
	metadata["can_mint_product_revision"] = false
	metadata["responsibility_ids"] = responsibilities
	return []Descriptor{{
		ID:                   ProductRigCapabilityID,
		Kind:                 KindModule,
		OwnerModuleID:        "product.rig",
		ExecutorActivity:     "engine.execute_capability.v1",
		Dependencies:         []string{ProductShapeCapabilityID, ProductMeshCapabilityID},
		AllowedModes:         []Mode{ModeProduct, ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"RSS vNext Canonical Rig", "Unified AXIS Controls"},
		Metadata:             metadata,
	}}, nil
}
