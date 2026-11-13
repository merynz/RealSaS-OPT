package capability

import "context"

// RSSvNextSkinCandidateProvider consumes one pinned RSS shape inference plus
// CanonicalMesh.v1, CanonicalRig.v2 and exact causal FSQ codes. It replaces the
// independent MIRA ModelRelease provider, never runs another neural release.
type RSSvNextSkinCandidateProvider struct{}

const (
	RSSvNextSkinContractV3 = "RealSaS.ProductSkinInference.v3"
	RSSvNextSkinContractV4 = "RealSaS.ProductSkinInference.v4"
)

func (RSSvNextSkinCandidateProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleSkin)
	if err != nil {
		return nil, err
	}
	policy := map[string]any{}
	policy["max_simplex_repair_l1"] = 0.00001
	policy["max_total_correction_l1"] = 0.0001
	policy["negative_tolerance"] = 0.00000001
	policy["max_influences"] = 41
	metadata := map[string]any{}
	metadata["product_role_id"] = string(ProductRoleSkin)
	metadata["implementation_class"] = string(ImplementationCompiler)
	metadata["interface_contract"] = RSSvNextSkinContractV4
	metadata["executor_contract"] = RSSvNextSkinContractV4
	metadata["required_models"] = []any{}
	metadata["input_source_artifact_type"] = "RealSaS.RSSvNextUncalibratedTwoImageSource"
	metadata["alternate_calibrated_source_artifact_type"] = "RealSaS.RSSvNextTwoViewSource"
	metadata["raw_scene_artifact_type"] = "RealSaS.RSSvNextUncalibratedNeuralSceneCarrier"
	metadata["raw_scene_schema_version"] = "v1"
	metadata["raw_model_config_artifact_type"] = "RealSaS.RSSvNextUnifiedModelConfig"
	metadata["raw_embedded_mira_decoder_artifact_type"] = "RealSaS.RSSvNextEmbeddedMIRADecoder"
	metadata["raw_source_static_replay_required"] = true
	metadata["input_source_schema_version"] = "v1"
	metadata["input_fanout_artifact_type"] = "RealSaS.RSSvNextProductEvidenceFanout"
	metadata["input_fanout_schema_version"] = "v1"
	metadata["input_mesh_artifact_type"] = "RealSaS.CanonicalMeshIR"
	metadata["input_mesh_schema_version"] = "v1"
	metadata["input_rig_artifact_type"] = "RealSaS.CanonicalRigIR"
	metadata["input_rig_schema_version"] = "v2"
	metadata["input_skin_code_artifact_type"] = "RealSaS.SkinCodeProposalIR"
	metadata["input_skin_code_schema_version"] = "v1"
	metadata["output_artifact_type"] = "RealSaS.CanonicalSkinIR"
	metadata["output_schema_version"] = "v2"
	metadata["carrier_skin_qualification"] = policy
	metadata["single_checkpoint_inference_reused"] = true
	metadata["exact_embedded_mira_decoder_replayed"] = true
	metadata["independent_mira_model_release_allowed"] = false
	metadata["post_hoc_skin_transfer_allowed"] = false
	metadata["compiler_code_search_allowed"] = false
	metadata["compiler_code_repair_allowed"] = false
	metadata["fit_allowed_during_product_execution"] = false
	metadata["mesh_and_rig_are_canonical_parents"] = true
	metadata["skin_binding_target"] = "DEFORM_STRUCTURE_HASH"
	metadata["pose_deformation_stress_qualification_owned_by_mechanics"] = true
	metadata["can_mint_product_revision"] = false
	metadata["responsibility_ids"] = responsibilities
	return []Descriptor{{
		ID:                   ProductSkinCapabilityID,
		Kind:                 KindModule,
		OwnerModuleID:        "product.skin",
		ExecutorActivity:     "engine.execute_capability.v1",
		Dependencies:         []string{ProductShapeCapabilityID, ProductMeshCapabilityID, ProductRigCapabilityID},
		AllowedModes:         []Mode{ModeProduct, ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"RSS vNext Canonical Skin", "Embedded SkinTokens"},
		Metadata:             metadata,
	}}, nil
}
