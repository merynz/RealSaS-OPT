package capability

import "context"

// Each stable role has exactly one provider. Proof owns the permission to
// decide product PASS, not a grant to manufacture PASS from candidate bytes.
const ProductMechanicsContractV1 = "RealSaS.ProductMechanicsCompile.v1"
const ProductGraphicsContractV1 = "RealSaS.ProductGraphicsCompile.v1"
const ProductMotionContractV2 = "RealSaS.ProductMotionCompile.v2"
const ProductProofContractV1 = "RealSaS.ProductProofCompile.v1"

type ProductMechanicsProvider struct{}
type ProductGraphicsProvider struct{}
type ProductMotionProvider struct{}
type ProductProofProvider struct{}

func downstreamDescriptor(role ProductRoleID, dependencies []string, metadata map[string]any) ([]Descriptor, error) {
	ids, err := responsibilityIDsForRole(role)
	if err != nil {
		return nil, err
	}
	metadata["product_role_id"] = string(role)
	metadata["responsibility_ids"] = ids
	row := Descriptor{}
	row.ID = "product." + string(role)
	row.Kind = KindModule
	row.OwnerModuleID = "product." + string(role)
	row.ExecutorActivity = "engine.execute_capability.v1"
	row.Dependencies = dependencies
	row.AllowedModes = []Mode{ModeDeveloper}
	row.ProductPassAuthority = role == ProductRoleProof
	row.Metadata = metadata
	return []Descriptor{row}, nil
}

func (ProductMechanicsProvider) Capabilities(context.Context) ([]Descriptor, error) {
	metadata := map[string]any{}
	metadata["implementation_class"] = "COMPILER"
	metadata["interface_contract"] = "RealSaS.ProductMechanicsCompile.v1"
	metadata["executor_contract"] = "RealSaS.ProductMechanicsCompile.v1"
	metadata["input_mesh_artifact_type"] = "RealSaS.CanonicalMeshIR"
	metadata["input_mesh_schema_version"] = "v1"
	metadata["input_rig_artifact_type"] = "RealSaS.CanonicalRigIR"
	metadata["input_rig_schema_version"] = "v2"
	metadata["input_skin_artifact_type"] = "RealSaS.CanonicalSkinIR"
	metadata["input_skin_schema_version"] = "v2"
	metadata["output_artifact_type"] = "RealSaS.CanonicalMechanicsIR"
	metadata["output_schema_version"] = "v1"
	metadata["required_models"] = []any{}
	policy := map[string]any{}
	policy["max_position_drift"] = 0.00001
	policy["max_constraint_residual"] = 0.0001
	metadata["rest_qualification"] = policy
	metadata["can_mint_product_revision"] = false
	return downstreamDescriptor(ProductRoleMechanics, []string{"product.mesh", "product.rig", "product.skin"}, metadata)
}

func (ProductGraphicsProvider) Capabilities(context.Context) ([]Descriptor, error) {
	metadata := map[string]any{}
	metadata["implementation_class"] = "COMPILER"
	metadata["interface_contract"] = "RealSaS.ProductGraphicsCompile.v1"
	metadata["executor_contract"] = "RealSaS.ProductGraphicsCompile.v1"
	metadata["input_mesh_artifact_type"] = "RealSaS.CanonicalMeshIR"
	metadata["input_mesh_schema_version"] = "v1"
	metadata["input_rig_artifact_type"] = "RealSaS.CanonicalRigIR"
	metadata["input_rig_schema_version"] = "v2"
	metadata["input_skin_artifact_type"] = "RealSaS.CanonicalSkinIR"
	metadata["input_skin_schema_version"] = "v2"
	metadata["input_mechanics_artifact_type"] = "RealSaS.CanonicalMechanicsIR"
	metadata["input_mechanics_schema_version"] = "v1"
	metadata["input_evidence_artifact_type"] = "RealSaS.GraphicsEvidence"
	metadata["input_evidence_schema_version"] = "v1"
	metadata["output_artifact_type"] = "RealSaS.CompiledGraphics"
	metadata["output_schema_version"] = "v1"
	metadata["required_models"] = []any{}
	metadata["source_appearance_qualified_by_graphics_evidence"] = false
	metadata["can_mint_product_revision"] = false
	return downstreamDescriptor(ProductRoleGraphics, []string{"product.mesh", "product.rig", "product.skin", "product.mechanics"}, metadata)
}

func (ProductMotionProvider) Capabilities(context.Context) ([]Descriptor, error) {
	metadata := map[string]any{}
	metadata["implementation_class"] = "COMPILER"
	metadata["interface_contract"] = "RealSaS.ProductMotionCompile.v2"
	metadata["executor_contract"] = "RealSaS.ProductMotionCompile.v2"
	metadata["input_rig_artifact_type"] = "RealSaS.CanonicalRigIR"
	metadata["input_rig_schema_version"] = "v2"
	metadata["input_mechanics_artifact_type"] = "RealSaS.CanonicalMechanicsIR"
	metadata["input_mechanics_schema_version"] = "v1"
	metadata["input_proposal_artifact_type"] = "RealSaS.ControlMotionClipProposalIR"
	metadata["input_proposal_schema_version"] = "v1"
	metadata["output_artifact_type"] = "RealSaS.CanonicalMotionIR"
	metadata["output_schema_version"] = "v2"
	metadata["required_models"] = []any{}
	metadata["animation_source_input_required"] = true
	metadata["can_mint_product_revision"] = false
	return downstreamDescriptor(ProductRoleMotion, []string{"product.rig", "product.mechanics"}, metadata)
}

func (ProductProofProvider) Capabilities(context.Context) ([]Descriptor, error) {
	metadata := map[string]any{}
	metadata["implementation_class"] = "COMPILER"
	metadata["interface_contract"] = "RealSaS.ProductProofCompile.v1"
	metadata["executor_contract"] = "RealSaS.ProductProofCompile.v1"
	metadata["input_mechanics_artifact_type"] = "RealSaS.CanonicalMechanicsIR"
	metadata["input_mechanics_schema_version"] = "v1"
	metadata["input_graphics_artifact_type"] = "RealSaS.CompiledGraphics"
	metadata["input_graphics_schema_version"] = "v1"
	metadata["input_motion_artifact_type"] = "RealSaS.CanonicalMotionIR"
	metadata["input_motion_schema_version"] = "v2"
	metadata["output_artifact_type"] = "RealSaS.ProductProofDiagnostic"
	metadata["output_schema_version"] = "v1"
	metadata["required_models"] = []any{}
	metadata["product_revision_requires_independent_admission"] = true
	metadata["can_mint_product_revision"] = false
	return downstreamDescriptor(ProductRoleProof, []string{"product.mechanics", "product.graphics", "product.motion"}, metadata)
}
