package capability

import "context"

const (
	ProductRigCapabilityID = "product.rig"
	ProductRigContractV1   = "RealSaS.ProductRigInference.v1"
)

// ProductRigProvider owns the stable Rig product boundary. AXIS is the current
// structure/geometry proposal provider; the Compiler validates the hard required
// tree without parent reselection and mints CanonicalRigIR on the exact Mesh.
// Stage15 is current feature evidence for the AXIS V5.4.1 adapter, not the
// canonical Rig parent and may be rewired in a later release. Canonical Rig root
// semantics are a non-empty deform-root set even though this AXIS release emits
// one hard causal tree.
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
			StageCapabilityID("15_RIGGING_SURFACE_QUALIFIED"),
		},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"Rig", "AXIS"},
		Metadata: map[string]any{
			"product_role_id":           string(ProductRoleRig),
			"implementation_class":      string(ImplementationHybrid),
			"interface_contract":        ProductRigContractV1,
			"executor_contract":         ProductRigContractV1,
			"input_mesh_artifact_type":  "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version": "v1",
			"feature_evidence_type":     "RealSaS.RiggingSurfaceIR",
			"feature_evidence_version":  "v1",
			"output_artifact_type":      "RealSaS.CanonicalRigIR",
			"output_schema_version":     "v1",
			"resource_step_limit_max":   256,
			"required_models": []any{
				map[string]any{
					"family":          "AXIS",
					"architecture_id": "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1",
				},
			},
			"model_checkpoint_artifact_type":         "RealSaS.AXISRuntimeCheckpoint",
			"model_checkpoint_schema_version":        "v1",
			"mesh_is_canonical_parent":               true,
			"feature_evidence_is_canonical_parent":   false,
			"canonical_root_semantics":               "NON_EMPTY_DEFORM_ROOT_SET",
			"current_axis_root_semantics":            "SINGLE_HARD_CAUSAL_TREE_IMPLEMENTATION_DETAIL",
			"hard_parent_authority_preserved":        true,
			"compiler_parent_reselection_allowed":    false,
			"frame_observability_claimed":            false,
			"post_bind_frame_observability_required": true,
			"fit_allowed_during_product_execution":   false,
			"implementation_owner_may_change":        true,
			"stable_product_role_survives_rewiring":  true,
			"responsibility_ids":                     responsibilities,
		},
	}}, nil
}
