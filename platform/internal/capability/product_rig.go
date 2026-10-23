package capability

import "context"

const (
	ProductRigCapabilityID = "product.rig"
	ProductRigContractV1   = "RealSaS.ProductRigCompile.v1"

	ProductRigAXISArchitectureID = "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1"
	ProductRigAXISCheckpointV1   = "RealSaS.AXISBFSDiscreteXYZV541.DeployableCheckpoint.v1"
	ProductRigAXISInputAdapterV1 = "RealSaS.GeppettoRiggingSurfaceTensorization.v1"
)

type ProductRigProvider struct{}

func (ProductRigProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleRig)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:               ProductRigCapabilityID,
		Kind:             KindModule,
		OwnerModuleID:    ProductRigCapabilityID,
		ExecutorActivity: "engine.execute_capability.v1",
		Dependencies: []string{
			ProductMeshCapabilityID,
			StageCapabilityID("15_RIGGING_SURFACE_QUALIFIED"),
		},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"Rig", "AXIS"},
		Metadata: map[string]any{
			"product_role_id":                                   string(ProductRoleRig),
			"implementation_class":                              string(ImplementationHybrid),
			"interface_contract":                                ProductRigContractV1,
			"executor_contract":                                 ProductRigContractV1,
			"canonical_parent_role":                             string(ProductRoleMesh),
			"input_mesh_artifact_type":                          "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version":                         "v1",
			"input_surface_artifact_type":                       "RealSaS.RiggingSurfaceIR",
			"input_surface_schema_version":                      "v1",
			"axis_input_adapter_contract":                       ProductRigAXISInputAdapterV1,
			"axis_input_adapter_is_canonical_authority":         false,
			"axis_input_adapter_replaceable":                    true,
			"surface_dependency_is_compatibility_evidence_only": true,
			"output_artifact_type":                              "RealSaS.QualifiedSkeletonIR",
			"output_schema_version":                             "v1",
			"output_authority_class":                            "COMPILER_CANONICAL_RIG",
			"axis_checkpoint_schema":                            ProductRigAXISCheckpointV1,
			"maximum_joint_steps":                               256,
			"compiler_parent_reselection_allowed":               false,
			"proposal_authority_artifact_emitted":               false,
			"canonical_mesh_mutation_allowed":                   false,
			"post_bind_frame_observability_claimed":             false,
			"frame_observability_deferred_to_mechanics":         true,
			"required_models": []any{
				map[string]any{
					"family":          "AXIS",
					"architecture_id": ProductRigAXISArchitectureID,
				},
			},
			"implementation_owner_may_change":       true,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids":                    responsibilities,
		},
	}}, nil
}
