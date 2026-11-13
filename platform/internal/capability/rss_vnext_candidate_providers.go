package capability

import "context"

// RSSvNextInputCandidateProvider is the active one-trigger two-view Input
// boundary. It remains a replaceable provider of the stable product.input
// role, never a second parallel product.input. Source rights and final
// ProductRevision proof are separate downstream admission gates.
type RSSvNextInputCandidateProvider struct{}

const (
	RSSvNextInputContractV2 = "RealSaS.ProductInputTwoView.v2"
	RSSvNextInputContractV3 = "RealSaS.ProductInputTwoView.v3"
	RSSvNextShapeContractV3 = "RealSaS.ProductShapeInference.v3"
	RSSvNextShapeContractV4 = "RealSaS.ProductShapeInference.v4"
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
		AllowedModes:         []Mode{ModeProduct, ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"RSS vNext Input", "Two Original Views"},
		Metadata: map[string]any{
			"product_role_id":                              string(ProductRoleInput),
			"implementation_class":                         string(ImplementationHybrid),
			"interface_contract":                           RSSvNextInputContractV3,
			"executor_contract":                            RSSvNextInputContractV3,
			"input_artifact_type":                          "RealSaS.RSSvNextUncalibratedTwoImageSource",
			"input_schema_version":                         "v1",
			"output_artifact_type":                         "RealSaS.RSSvNextUncalibratedTwoImageSource",
			"output_schema_version":                        "v1",
			"alternate_calibrated_source_artifact_type":    "RealSaS.RSSvNextTwoViewSource",
			"alternate_calibrated_source_schema_version":   "v1",
			"uncalibrated_source_original_camera_required": false,
			"original_observed_view_count":                 2,
			"source_copy_policy":                           "BYTE_IDENTICAL_CAS_REUSE",
			"model_inference_in_input_forbidden":           true,
			"platform_view_count_invariant":                false,
			"source_rights_verified_by_input":              false,
			"stable_product_role_survives_rewiring":        true,
			"responsibility_ids":                 responsibilities,
		},
	}}, nil
}

// RSSvNextShapeCandidateProvider is the active one-checkpoint Shape provider.
// It emits seven independently addressable specialist CAS children. IRIS,
// TESSA, AXIS and MIRA remain expert responsibilities inside RSSVNEXT; they
// are not four ModelReleases or independent downstream inference jobs.
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
		AllowedModes:         []Mode{ModeProduct, ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"RSS vNext Unified", "Geometry", "Mechanics", "Semantic", "Graphics"},
		Metadata: map[string]any{
			"product_role_id":                            string(ProductRoleShape),
			"implementation_class":                       string(ImplementationModel),
			"interface_contract":                         RSSvNextShapeContractV4,
			"executor_contract":                          RSSvNextShapeContractV4,
			"input_artifact_type":                        "RealSaS.RSSvNextUncalibratedTwoImageSource",
			"input_schema_version":                       "v1",
			"output_artifact_type":                       "RealSaS.RSSvNextInferenceEvidence",
			"output_schema_version":                      "v1",
			"fanout_contract":                            "RealSaS.RSSvNext.ProductEvidenceFanout.v1",
			"alternate_calibrated_source_artifact_type":  "RealSaS.RSSvNextTwoViewSource",
			"alternate_calibrated_source_schema_version": "v1",
			"raw_neural_scene_candidate_artifact_type":   "RealSaS.RSSvNextUncalibratedNeuralSceneCarrier",
			"raw_generated_views_artifact_type":          "RealSaS.RSSvNextGeneratedConditioningViews",
			"raw_construction_receipt_artifact_type":     "RealSaS.RSSvNextUncalibratedConstructionReceipt",
			"raw_source_camera_calibration_authority":    false,
			"raw_mesh_proof_is_separate_admission":       true,
			"maximum_new_tokens":                         131072,
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
				map[string]any{"role": "unified_model_config", "artifact_type": "RealSaS.RSSvNextUnifiedModelConfig", "schema_version": "v1"},
				map[string]any{"role": "embedded_mira_decoder", "artifact_type": "RealSaS.RSSvNextEmbeddedMIRADecoder", "schema_version": "v1"},
				map[string]any{"role": "rss_evidence_fanout", "artifact_type": "RealSaS.RSSvNextProductEvidenceFanout", "schema_version": "v1"},
			},
			"input_observed_front_back_only":     false,
			"neural_scene_survives_mesh_abstain": true,
			"learned_model_results_not_product":  true,
			"independent_iris_tessa_forbidden":   true,
			"responsibility_ids":                         responsibilities,
		},
	}}, nil
}

// RSSvNextMeshCandidateProvider consumes the observed front/back source and
// the exact RSS Shape fanout. The active Mesh role requires a static
// CanonicalMesh or fails closed; TESSA candidates alone never count. The
// frozen mechanical policy and provisional two-view source raster policy are
// independently pinned. ProductRevision and 2D appearance remain unqualified.
type RSSvNextMeshCandidateProvider struct{}

const RSSvNextMeshContractV2 = "RealSaS.ProductMeshCompile.v2"

func (RSSvNextMeshCandidateProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleMesh)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:                   ProductMeshCapabilityID,
		Kind:                 KindModule,
		OwnerModuleID:        "product.mesh",
		ExecutorActivity:     "engine.execute_capability.v1",
		Dependencies:         []string{ProductShapeCapabilityID},
		AllowedModes:         []Mode{ModeProduct, ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"RSS vNext Mesh Compiler", "Topology Replay"},
		Metadata: map[string]any{
			"product_role_id":                               string(ProductRoleMesh),
			"implementation_class":                          string(ImplementationCompiler),
			"interface_contract":                            RSSvNextMeshContractV2,
			"executor_contract":                             RSSvNextMeshContractV2,
			"input_shape_artifact_type":                     "RealSaS.RSSvNextInferenceEvidence",
			"input_shape_schema_version":                    "v1",
			"input_source_artifact_type":                    "RealSaS.RSSvNextTwoViewSource",
			"input_source_schema_version":                   "v1",
			"input_fanout_artifact_type":                    "RealSaS.RSSvNextProductEvidenceFanout",
			"input_fanout_schema_version":                   "v1",
			"requires_shape_parent_binding":                 true,
			"requires_full_neural_scene_if_abstain":         false,
			"output_artifact_type":                          "RealSaS.CanonicalMeshIR",
			"output_schema_version":                         "v1",
			"candidate_output_artifact_type":                "RealSaS.RSSvNextComputedCandidateSurface",
			"candidate_output_schema_version":               "v1",
			"canonical_mesh_required_for_product_execution": true,
			"two_view_static_policy_sha256":                 "f7850b67193323f2ca37fb0c5f53111c187352d234c7ef54303e4880a3612bfb",
			"mesh_policy_document_sha256":                   "66791e0e309209d28d16e56a550846de6f2edd87347806139e137e6f12d07c01",
			"two_view_static_policy": map[string]any{
				"schema":                             "RealSaS.RSSvNext.TwoObservedViewMeshPolicy.v1",
				"min_recall":                         0.9,
				"min_precision":                      0.9,
				"max_largest_coherent_hole_fraction": 0.1,
				"max_interior_uncovered_fraction":    0.1,
				"min_component_recall":               0.85,
				"max_silhouette_edge_p95_px":         2.0,
				"source_alpha_threshold":             128,
				"component_min_foreground_fraction":  0.005,
			},
			"two_view_static_policy_stage": "PROVISIONAL_PRETRAINING_NEEDS_UNSEEN_CALIBRATION",
			"auxiliary_outputs": []any{
				map[string]any{"role": "rss_sparse_face_relation_candidates", "artifact_type": "RealSaS.RSSvNextSparseCandidateFaceRelations", "schema_version": "v1"},
				map[string]any{"role": "rss_two_view_raster_diagnostic", "artifact_type": "RealSaS.RSSvNextTwoViewRasterDiagnostic", "schema_version": "v1"},
				map[string]any{"role": "rss_original_source_ray_replay", "artifact_type": "RealSaS.RSSvNextSourceRayReplayCourt", "schema_version": "v1"},
				map[string]any{"role": "rss_relation_epistemic_court", "artifact_type": "RealSaS.RSSvNextRelationEpistemicCourt", "schema_version": "v1"},
				map[string]any{"role": "rss_cross_expert_capability_court", "artifact_type": "RealSaS.RSSvNextCrossExpertCapabilityCourt", "schema_version": "v1"},
				map[string]any{"role": "rss_canonical_mesh_candidate_bridge", "artifact_type": "RealSaS.RSSvNextCanonicalMeshCandidateBridge", "schema_version": "v1"},
				map[string]any{"role": "rss_evidence_continuity", "artifact_type": "RealSaS.RSSvNextEvidenceContinuity", "schema_version": "v1"},
			},
			"canonical_mesh_authority":               false,
			"canonical_candidate_basis":              "TESSA_NEURAL_PROPOSAL_SOURCE_RASTER_QUALIFICATION_PENDING",
			"learned_evidence_retained_by_reference": true,
			"independent_model_inference_forbidden":  true,
			"requires_legacy_eight_view_stages":      false,
			"required_models": []any{},
			"responsibility_ids":                            responsibilities,
		},
	}}, nil
}
