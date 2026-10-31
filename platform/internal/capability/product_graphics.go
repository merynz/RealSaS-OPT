package capability

import "context"

const (
	ProductGraphicsCapabilityID = "product.graphics"
	ProductGraphicsContractV1   = "RealSaS.ProductGraphicsCompile.v1"
)

// ProductGraphicsProvider owns the complete visual/presentation domain after
// canonical Mesh/Skin/Mechanics exist. Complete Appearance Authority is an
// internal Graphics sublayer; visibility, occlusion, fragmentation, draw order
// and presentation attachment bindings remain Graphics responsibilities too.
//
// Learned relation/appearance proposal evidence is read from the same IRIS V3
// ModelRelease and serialized ShapeProposal shared-scene H used by Shape. The
// scene encoder is not rerun here. Compiler remains the final Graphics/CAA
// authority and may not fabricate missing perceptual evidence.
type ProductGraphicsProvider struct{}

func (ProductGraphicsProvider) Capabilities(context.Context) ([]Descriptor, error) {
	responsibilities, err := responsibilityIDsForRole(ProductRoleGraphics)
	if err != nil {
		return nil, err
	}
	return []Descriptor{{
		ID:               ProductGraphicsCapabilityID,
		Kind:             KindModule,
		OwnerModuleID:    "product.graphics",
		ExecutorActivity: "engine.execute_capability.v1",
		Dependencies: []string{
			ProductShapeCapabilityID,
			ProductMeshCapabilityID,
			ProductSkinCapabilityID,
			ProductMechanicsCapabilityID,
		},
		AllowedModes:         []Mode{ModeDeveloper, ModeProduct},
		ProductPassAuthority: false,
		Aliases:              []string{"Graphics", "Presentation"},
		Metadata: map[string]any{
			"product_role_id":                       string(ProductRoleGraphics),
			"implementation_class":                  string(ImplementationHybrid),
			"interface_contract":                    ProductGraphicsContractV1,
			"executor_contract":                     ProductGraphicsContractV1,
			"input_shape_artifact_type":             "RealSaS.ShapeProposal",
			"input_shape_schema_version":            "v2",
			"input_mesh_artifact_type":              "RealSaS.CanonicalMeshIR",
			"input_mesh_schema_version":             "v1",
			"input_reference_binding_artifact_type": "RealSaS.TESSAReferenceBindingIR",
			"input_reference_binding_schema_version": "v1",
			"input_skin_artifact_type":              "RealSaS.CanonicalSkinIR",
			"input_skin_schema_version":             "v2",
			"input_mechanics_artifact_type":         "RealSaS.CanonicalMechanicsIR",
			"input_mechanics_schema_version":        "v1",
			"learned_relation_evidence_schema":      "RealSaS.IRIS.GraphicsRelationHead.v3",
			"learned_appearance_evidence_schema":    "RealSaS.IRIS.AppearanceEvidenceHead.v3",
			"appearance_bundle_artifact_type":       "RealSaS.GraphicsAppearanceBundle",
			"appearance_bundle_schema_version":      "v1",
			"output_artifact_type":                  "RealSaS.CanonicalGraphicsIR",
			"output_schema_version":                 "v1",
			"required_models": []any{
				map[string]any{"family": "IRIS", "architecture_id": "RealSaS.IRIS.VNextEvidenceModel.v3"},
			},
			"shared_scene_h_reused_without_scene_reencode": true,
			"graphics_owns_all_visual_presentation":       true,
			"caa_is_graphics_appearance_sublayer":         true,
			"visibility_occlusion_owner":                   "GRAPHICS",
			"draw_order_owner":                             "GRAPHICS",
			"fragmentation_owner":                          "GRAPHICS",
			"presentation_attachment_owner":                "GRAPHICS",
			"source_art_wins_where_observed":               true,
			"compiler_perception_synthesis_allowed":        false,
			"neural_confidence_output_allowed":             false,
			"pairwise_visibility_output_allowed":           false,
			"local_interleaving_without_runtime_support":   "FAIL_CLOSED",
			"mechanics_draw_order_authority_allowed":       false,
			"motion_draw_order_authority_allowed":          false,
			"presentation_compile_policy": map[string]any{
				"same_drawable_min":               0.65,
				"cut_max":                         0.35,
				"boundary_max":                    0.50,
				"front_probability_min":           0.65,
				"occlusion_probability_min":       0.50,
				"visibility_probability_min":      0.50,
				"attachment_affinity_min":         0.70,
				"mechanical_contact_min":          0.75,
				"support_band_alpha_boundary_min": 0.65,
				"maximum_support_band_pixels":     2,
			},
			"implementation_owner_may_change":       true,
			"stable_product_role_survives_rewiring": true,
			"responsibility_ids":                    responsibilities,
		},
	}}, nil
}
