package capability

import "context"

const (
	ProductMechanicsCapabilityID = "product.mechanics"
	ProductMechanicsContractV1   = "RealSaS.ProductMechanicsCompile.v1"
)

// ProductMechanicsProvider owns executable control/constraint semantics after
// Mesh/Rig/Skin are canonical. It introduces no learned model and may not mutate
// the rig program or skin. The compiler court seals rest-pose execution and the
// runtime reuses the same deterministic solver for authored/Motion control poses.
type ProductMechanicsProvider struct{}

func (ProductMechanicsProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleMechanics)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:               ProductMechanicsCapabilityID,
		Kind:             KindModule,
		OwnerModuleID:    "product.mechanics",
		ExecutorActivity: "engine.execute_capability.v1",
		Dependencies: []string{
			ProductMeshCapabilityID,
			ProductRigCapabilityID,
			ProductSkinCapabilityID,
		},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"Mechanics", "ControlSolver"},
		Metadata: map[string]any{
			"product_role_id":                  string(ProductRoleMechanics),
			"implementation_class":             string(ImplementationCompiler),
			"interface_contract":               ProductMechanicsContractV1,
			"executor_contract":                ProductMechanicsContractV1,
			"input_mesh_artifact_type":         "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version":        "v1",
			"input_rig_artifact_type":          "RealSaS.CanonicalRigIR",
			"input_rig_schema_version":         "v2",
			"input_skin_artifact_type":         "RealSaS.CanonicalSkinIR",
			"input_skin_schema_version":        "v2",
			"output_artifact_type":             "RealSaS.CanonicalMechanicsIR",
			"output_schema_version":            "v1",
			"required_models":                  []any{},
			"learned_inference_allowed":        false,
			"rig_program_mutation_allowed":     false,
			"skin_mutation_allowed":            false,
			"constraint_execution_owner":       "MECHANICS",
			"runtime_and_compile_solver_shared": true,
			"rest_qualification": map[string]any{
				"max_position_drift":      1.0e-5,
				"max_constraint_residual": 1.0e-4,
			},
			"implementation_owner_may_change":       true,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids":                    responsibilities,
		},
	}}, nil
}
