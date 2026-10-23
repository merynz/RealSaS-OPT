package capability

import "context"

const (
	ProductMeshCapabilityID = "product.mesh"
	ProductMeshContractV1   = "RealSaS.ProductMeshCompile.v1"
)

// ProductMeshProvider is the stable Mesh product role. The current provider is
// Compiler-owned: learned Shape remains proposal evidence and only this role may
// mint the static canonical carrier consumed by later Rig/Skin/Mechanics roles.
type ProductMeshProvider struct{}

func (ProductMeshProvider) Capabilities(context.Context) ([]Descriptor, error) {
	return []Descriptor{{
		ID:                   ProductMeshCapabilityID,
		Kind:                 KindModule,
		OwnerModuleID:        "compiler.mesh",
		ExecutorActivity:     "engine.execute_capability.v1",
		Dependencies:         []string{ProductShapeCapabilityID},
		AllowedModes:         []Mode{ModeDeveloper},
		ProductPassAuthority: false,
		Aliases:              []string{"Mesh", "Canonical Mesh"},
		Metadata: map[string]any{
			"product_role_id":                       string(ProductRoleMesh),
			"implementation_class":                  string(ImplementationCompiler),
			"interface_contract":                    ProductMeshContractV1,
			"executor_contract":                     ProductMeshContractV1,
			"source_artifact_type":                  "RealSaS.SourceObservation",
			"source_schema_version":                 "v1",
			"shape_artifact_type":                   "RealSaS.ShapeProposal",
			"shape_schema_version":                  "v2",
			"output_artifact_type":                  "RealSaS.CanonicalMesh",
			"output_schema_version":                 "v1",
			"source_fidelity_policy":                "Stage13-P999",
			"dynamic_qualification_required_at_birth": false,
			"implementation_owner_may_change":       true,
			"stable_product_role_survives_rewiring": true,
		},
	}}, nil
}
