package capability

import "context"

// RSSvNextInputCandidateProvider is the proposed REPLACEMENT for product.input.
//
// This is an alternate provider of the same stable capability ID, never a
// concurrent second product.input. It is deliberately NOT in ProductProviders
// until the source upload and downstream Compiler/Studio release courts move
// together. Its executable Python boundary already has a real CAS E2E court.
type RSSvNextInputCandidateProvider struct{}

const (
	RSSvNextInputContractV2 = "RealSaS.ProductInputTwoView.v2"
	RSSvNextShapeContractV3 = "RealSaS.ProductShapeInference.v3"
	RSSvNextModelFamily     = "RSSVNEXT"
	RSSvNextArchitectureID  = "RSS.vNext.UnifiedCharacterCompiler.v1"
)

func (RSSvNextInputCandidateProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleInput)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:                   ProductInputCapabilityID,
		Kind:                 KindModule,
		OwnerModuleID:        "product.input",
		ExecutorActivity:     "engine.execute_capability.v1",
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"RSS vNext Input", "Two Original Views"},
		Metadata: map[string]any{
			"product_role_id":                       string(ProductRoleInput),
			"implementation_class":                  string(ImplementationHybrid),
			"interface_contract":                    RSSvNextInputContractV2,
			"executor_contract":                     RSSvNextInputContractV2,
			"input_artifact_type":                   "RealSaS.RSSvNextTwoViewSource",
			"input_schema_version":                  "v1",
			"output_artifact_type":                  "RealSaS.RSSvNextTwoViewSource",
			"output_schema_version":                 "v1",
			"original_observed_view_count":          2,
			"source_copy_policy":                    "BYTE_IDENTICAL_CAS_REUSE",
			"model_inference_in_input_forbidden":    true,
			"platform_view_count_invariant":         false,
			"source_rights_verified_by_input":       false,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids":                    responsibilities,
		},
	}}, nil
}

// RSSvNextShapeCandidateProvider replaces the historical separate IRIS/TESSA
// provider when the ONE unified RSS release is admitted. The production Go
// ProductProviders list and responsibility ledger still name the historical
// family split; this candidate is not the active release catalog.
type RSSvNextShapeCandidateProvider struct{}

func (RSSvNextShapeCandidateProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleShape)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:                   ProductShapeCapabilityID,
		Kind:                 KindModule,
		OwnerModuleID:        "product.shape",
		ExecutorActivity:     "engine.execute_capability.v1",
		Dependencies:         []string{ProductInputCapabilityID},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"RSS vNext Unified", "Geometry", "Mechanics", "Semantic", "Graphics"},
		Metadata: map[string]any{
			"product_role_id":       string(ProductRoleShape),
			"implementation_class":  string(ImplementationModel),
			"interface_contract":    RSSvNextShapeContractV3,
			"executor_contract":     RSSvNextShapeContractV3,
			"input_artifact_type":   "RealSaS.RSSvNextTwoViewSource",
			"input_schema_version":  "v1",
			"output_artifact_type":  "RealSaS.RSSvNextInferenceEvidence",
			"output_schema_version": "v1",
			"fanout_contract":       "RealSaS.RSSvNext.ProductEvidenceFanout.v1",
			"maximum_new_tokens":    131072,
			"required_models": []any{
				map[string]any{
					"family":          RSSvNextModelFamily,
					"architecture_id": RSSvNextArchitectureID,
				},
			},
			"auxiliary_outputs": []any{
				map[string]any{"role": "original_source", "artifact_type": "RealSaS.RSSvNextOriginalSource", "schema_version": "v1"},
				map[string]any{"role": "normalized_model_source", "artifact_type": "RealSaS.RSSvNextModelSourceTensors", "schema_version": "v1"},
				map[string]any{"role": "neural_scene", "artifact_type": "RealSaS.RSSvNextNeuralSceneCarrier", "schema_version": "v1"},
				map[string]any{"role": "inference_receipt", "artifact_type": "RealSaS.RSSvNextInferenceReceipt", "schema_version": "v2"},
				map[string]any{"role": "rss_evidence_fanout", "artifact_type": "RealSaS.RSSvNextProductEvidenceFanout", "schema_version": "v1"},
			},
			"input_observed_front_back_only":     true,
			"neural_scene_survives_mesh_abstain": true,
			"learned_model_results_not_product":  true,
			"independent_iris_tessa_forbidden":   true,
			"responsibility_ids":                 responsibilities,
		},
	}}, nil
}
