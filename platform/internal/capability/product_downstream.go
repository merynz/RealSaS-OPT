package capability

import "context"

// Downstream product roles are first-class Compiler capabilities. An admitted
// descriptor is not a grant of source truth or ProductRevision authority.
// Proof and its Platform transaction remain separate from a MODULE PASS.
const (
    ProductMechanicsContractV1 = "RealSaS.ProductMechanicsCompile.v1"
    ProductGraphicsContractV1 = "RealSaS.ProductGraphicsCompile.v1"
    ProductMotionContractV2 = "RealSaS.ProductMotionCompile.v2"
    ProductProofContractV1 = "RealSaS.ProductProofCompile.v1"
)

type ProductMechanicsProvider struct{}
type ProductGraphicsProvider struct{}
type ProductMotionProvider struct{}
type ProductProofProvider struct{}

func downstreamDescriptor(role ProductRoleID, dependencies []string, metadata map[string]any) ([]Descriptor, error) {
    ids, err := responsibilityIDsForRole(role)
    if err != nil { return nil, err }
    metadata["product_role_id"] = string(role)
    metadata["responsibility_ids"] = ids
    return []Descriptor{{
        ID: "product."+string(role), Kind: KindModule,
        OwnerModuleID: "product."+string(role),
        ExecutorActivity: "engine.execute_capability.v1",
        Dependencies: dependencies,
        AllowedModes: []Mode{ModeDeveloper},
        // This declares who may *decide* a product PASS; the worker still
        // returns WAITING until independent product witnesses are admitted.
        ProductPassAuthority: role == ProductRoleProof,
        Metadata: metadata,
    }}, nil
}

func (ProductMechanicsProvider) Capabilities(context.Context) ([]Descriptor,error) {
    return downstreamDescriptor(ProductRoleMechanics,
        []string{"product.mesh","product.rig","product.skin"},
        map[string]any{
            "implementation_class":string(ImplementationCompiler),
            "interface_contract":ProductMechanicsContractV1,
            "executor_contract":ProductMechanicsContractV1,
            "input_mesh_artifact_type":"RealSaS.CanonicalMeshIR",
            "input_mesh_schema_version":"v1",
            "input_rig_artifact_type":"RealSaS.CanonicalRigIR",
            "input_rig_schema_version":"v2",
            "input_skin_artifact_type":"RealSaS.CanonicalSkinIR",
            "input_skin_schema_version":"v2",
            "output_artifact_type":"RealSaS.CanonicalMechanicsIR",
            "output_schema_version":"v1",
            "required_models":[]any{},
            "rest_qualification":map[string]any{"max_position_drift":0.00001,"max_constraint_residual":0.0001},
            "can_mint_product_revision":false,
        })
}

func (ProductGraphicsProvider) Capabilities(context.Context) ([]Descriptor,error) {
    return downstreamDescriptor(ProductRoleGraphics,
        []string{"product.mesh","product.rig","product.skin","product.mechanics"},
        map[string]any{
            "implementation_class":string(ImplementationCompiler),
            "interface_contract":ProductGraphicsContractV1,
            "executor_contract":ProductGraphicsContractV1,
            "input_mesh_artifact_type":"RealSaS.CanonicalMeshIR",
            "input_mesh_schema_version":"v1",
            "input_rig_artifact_type":"RealSaS.CanonicalRigIR",
            "input_rig_schema_version":"v2",
            "input_skin_artifact_type":"RealSaS.CanonicalSkinIR",
            "input_skin_schema_version":"v2",
            "input_mechanics_artifact_type":"RealSaS.CanonicalMechanicsIR",
            "input_mechanics_schema_version":"v1",
            "input_evidence_artifact_type":"RealSaS.GraphicsEvidence",
            "input_evidence_schema_version":"v1",
            "output_artifact_type":"RealSaS.CompiledGraphics",
            "output_schema_version":"v1",
            "required_models":[]any{},
            "source_appearance_qualified_by_graphics_evidence":false,
            "can_mint_product_revision":false,
        })
}

func (ProductMotionProvider) Capabilities(context.Context) ([]Descriptor,error) {
    return downstreamDescriptor(ProductRoleMotion,
        []string{"product.rig","product.mechanics"},
        map[string]any{
            "implementation_class":string(ImplementationCompiler),
            "interface_contract":ProductMotionContractV2,
            "executor_contract":ProductMotionContractV2,
            "input_rig_artifact_type":"RealSaS.CanonicalRigIR",
            "input_rig_schema_version":"v2",
            "input_mechanics_artifact_type":"RealSaS.CanonicalMechanicsIR",
            "input_mechanics_schema_version":"v1",
            "input_proposal_artifact_type":"RealSaS.ControlMotionClipProposalIR",
            "input_proposal_schema_version":"v1",
            "output_artifact_type":"RealSaS.CanonicalMotionIR",
            "output_schema_version":"v2",
            "required_models":[]any{},
            "animation_source_input_required":true,
            "can_mint_product_revision":false,
        })
}

func (ProductProofProvider) Capabilities(context.Context) ([]Descriptor,error) {
    return downstreamDescriptor(ProductRoleProof,
        []string{"product.mechanics","product.graphics","product.motion"},
        map[string]any{
            "implementation_class":string(ImplementationCompiler),
            "interface_contract":ProductProofContractV1,
            "executor_contract":ProductProofContractV1,
            "input_mechanics_artifact_type":"RealSaS.CanonicalMechanicsIR",
            "input_mechanics_schema_version":"v1",
            "input_graphics_artifact_type":"RealSaS.CompiledGraphics",
            "input_graphics_schema_version":"v1",
            "input_motion_artifact_type":"RealSaS.CanonicalMotionIR",
            "input_motion_schema_version":"v2",
            "output_artifact_type":"RealSaS.ProductProofDiagnostic",
            "output_schema_version":"v1",
            "required_models":[]any{},
            "product_revision_requires_independent_admission":true,
            "can_mint_product_revision":false,
        })
}
