package capability

import "context"

const (
	ProductMeshCapabilityID = "product.mesh"
	ProductMeshContractV1   = "RealSaS.ProductMeshCompile.v1"
)

// ProductMeshProvider owns the stable Mesh product boundary. Learned Shape is
// proposal evidence only. The current Mesh implementation derives the qualified
// RiggingSurface evidence, structural partition/carrier policy and static
// source-fidelity qualification from that exact Shape proposal plus the source
// camera/observation/normalization contracts. Legacy IRIS fit stages are not a
// product dependency. Compiler alone mints immutable CanonicalMeshIR.
type ProductMeshProvider struct{}

func (ProductMeshProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleMesh)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:               ProductMeshCapabilityID,
		Kind:             KindModule,
		OwnerModuleID:    "product.mesh",
		ExecutorActivity: "engine.execute_capability.v1",
		Dependencies: []string{
			ProductShapeCapabilityID,
			StageCapabilityID("05_CAMERA_CONTRACT_SOLVED"),
			StageCapabilityID("07_OBSERVATION_CONTRACT_QUALIFIED"),
			StageCapabilityID("08_NORMALIZATION_DOMAIN_QUALIFIED"),
		},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"Mesh", "Compiler"},
		Metadata: map[string]any{
			"product_role_id":             string(ProductRoleMesh),
			"implementation_class":        string(ImplementationCompiler),
			"interface_contract":          ProductMeshContractV1,
			"executor_contract":           ProductMeshContractV1,
			"input_shape_artifact_type":   "RealSaS.ShapeProposal",
			"input_shape_schema_version":  "v2",
			"input_source_artifact_type":  "RealSaS.SourceObservation",
			"input_source_schema_version": "v1",
			"output_artifact_type":        "RealSaS.CanonicalMeshIR",
			"output_schema_version":       "v1",
			"surface_artifact_type":       "RealSaS.RiggingSurfaceIR",
			"surface_schema_version":      "v1",
			"mesh_policy_document_sha256": "66791e0e309209d28d16e56a550846de6f2edd87347806139e137e6f12d07c01",
			"shape_surface_court_policy_sha256": "e182023bd59d079ecb747d0812af4e83de7415593c4bca54c753d0f7abfa6f51",
			"source_fidelity_policy_sha256":     "8edf9934f1d14f45dcb719fcf39691d69c5797f7410c958ae3005ef639a3c214",
			"surface_adequacy_policy_sha256":    "bc98d3aa3cd419ca9ae43010e890fc554ab62e57268933afc630aae20c38139b",
			"support_binding": map[string]any{
				"max_support_nodes":      4,
				"max_graph_hops":         2,
				"inverse_distance_power": 2.0,
			},
			"static_repair": map[string]any{
				"collapse_batch_size":   32,
				"collapse_limit":        256,
				"relaxation_batch_size": 32,
				"relaxation_limit":      512,
			},
			"required_models":                       []any{},
			"fit_allowed_during_product_execution":  false,
			"legacy_ir_fit_stage_dependency":        false,
			"surface_born_inside_mesh_court":        true,
			"surface_is_canonical_mesh_authority":   false,
			"presentation_direction_dependency":     false,
			"graphics_authority_claimed":            false,
			"rig_authority_claimed":                 false,
			"skin_authority_claimed":                false,
			"implementation_owner_may_change":       true,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids":                    responsibilities,
		},
	}}, nil
}
