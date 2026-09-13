from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any
from realsas_artifacts.bundle import BundleWriter
from .types import (
    ObservationEvidenceIR, RiggingSurfaceIR, SkeletonProposalIR, QualifiedSkeletonIR,
    SkinProposalIR, QualifiedSkinIR, MeshDiscretizationCandidateIR,
    QualifiedEditableMeshIR, QualifiedMeshSkinIR, CanonicalPuppetGraph,
    CanonicalPuppetGraphV2, ProofFrame, RepairDirective, RuntimePackageIR,
    QualificationError,
)
from .v4_types import (
    QualifiedSkeletonIRV2, MechanicalStateIR, AppearanceBindingIR,
    VisualCompletionProposalIR, QualifiedVisualCompletionIR,
    RenderableComponentIR, DirectionalRenderableIR, DirectionalRenderableSetIR,
    CapabilityContractIR, MotionStateIR, JointTransformTrackIR,
    ComponentOrderTrackIR, ComponentVisibilityTrackIR, CanonicalPuppetGraphV3,
    ProofPlanIR, MeasurementReportIR, DomainProofReportIR, ProductProofBundleIR,
    CapabilityQualificationIR,
)
from .directional_binding import DirectionalJointViewBindingSetIR
from .compile_transaction import CompileRequestIR, CompileTransactionIR, CompileResultIR

@dataclass(frozen=True)
class ArtifactRoute:
    authority_class: str
    producer: str
    section: str
    filename: str
    allowed_consumers: tuple[str, ...]

ARTIFACT_ROUTES: dict[type, ArtifactRoute] = {
    ObservationEvidenceIR: ArtifactRoute("EVIDENCE", "IRIS+observation bookkeeping", "ir", "observation_evidence_ir.json", ("GeometricSubstrateAssembler",)),
    RiggingSurfaceIR: ArtifactRoute("QUALIFIED_GEOMETRY", "GeometricSubstrateAssembler", "ir", "rigging_surface_ir.json", ("Geppetto", "Arachne", "Compiler")),
    SkeletonProposalIR: ArtifactRoute("PROPOSAL", "Geppetto", "candidates", "skeleton_proposal_ir.json", ("Compiler.rig_qualification",)),
    QualifiedSkeletonIR: ArtifactRoute("QUALIFIED_LEGACY_V1", "Compiler.rig_qualification", "rig", "qualified_skeleton_ir.json", ("Arachne", "Compiler.product_assembly")),
    QualifiedSkeletonIRV2: ArtifactRoute("QUALIFIED", "Compiler.rig_qualification", "rig", "qualified_skeleton_ir_v2.json", ("Arachne", "Compiler.product_assembly_v3")),
    SkinProposalIR: ArtifactRoute("PROPOSAL", "Arachne", "candidates", "skin_proposal_ir.json", ("Compiler.skin_qualification",)),
    QualifiedSkinIR: ArtifactRoute("QUALIFIED", "Compiler.skin_qualification", "weight", "qualified_skin_ir.json", ("Compiler.mesh_weight_binding", "Compiler.product_assembly")),
    MeshDiscretizationCandidateIR: ArtifactRoute("DERIVED_CANDIDATE", "Compiler.mesh_discretization", "candidates/mesh", "mesh_discretization_candidate_ir.json", ("Compiler.geometry_qualification",)),
    QualifiedEditableMeshIR: ArtifactRoute("QUALIFIED_PRODUCT_COMPONENT", "Compiler.geometry_qualification", "mesh", "qualified_editable_mesh_ir.json", ("Compiler.mesh_weight_binding", "Compiler.product_assembly")),
    QualifiedMeshSkinIR: ArtifactRoute("QUALIFIED_PRODUCT_COMPONENT", "Compiler.mesh_weight_binding", "weight", "qualified_mesh_skin_ir.json", ("Compiler.product_assembly", "MotionProof")),
    CanonicalPuppetGraph: ArtifactRoute("CANONICAL_LEGACY_V1", "Compiler.product_assembly", "puppet", "canonical_puppet_graph.json", ("MotionProof", "Repair", "Export")),
    CanonicalPuppetGraphV2: ArtifactRoute("CANONICAL_LEGACY_V2", "Compiler.product_assembly", "puppet", "canonical_puppet_graph_v2.json", ("MotionProof", "Repair", "Export")),
    MechanicalStateIR: ArtifactRoute("CANONICAL_COMPONENT", "Compiler.product_assembly_v3", "puppet", "mechanical_state_ir.json", ("DirectionalRenderer", "Proof", "Export")),
    DirectionalRenderableSetIR: ArtifactRoute("CANONICAL_COMPONENT", "Compiler.product_assembly_v3", "renderables", "directional_renderable_set_ir.json", ("Proof", "Export")),
    DirectionalJointViewBindingSetIR: ArtifactRoute("DERIVED_QUALIFIED_BINDING", "Compiler.directional_binding", "renderables", "directional_joint_view_binding_set_ir.json", ("MotionProof", "Export", "LivingCompile")),
    CapabilityContractIR: ArtifactRoute("CANONICAL_POLICY", "Compiler.product_assembly_v3", "puppet", "capability_contract_ir.json", ("Proof", "Export")),
    MotionStateIR: ArtifactRoute("CANONICAL_COMPONENT", "MotionCompiler", "motion", "motion_state_ir.json", ("Proof", "Runtime")),
    CanonicalPuppetGraphV3: ArtifactRoute("CANONICAL", "Compiler.product_assembly_v3", "puppet", "canonical_puppet_graph_v3.json", ("Proof", "Repair", "Export")),
    ProofFrame: ArtifactRoute("DERIVED_LEGACY", "MotionProof", "proof", "proof_frame.json", ("Repair", "Export")),
    ProofPlanIR: ArtifactRoute("DERIVED_PLAN", "ProofPlanner", "proof", "proof_plan_ir.json", ("Measurement",)),
    MeasurementReportIR: ArtifactRoute("DERIVED_MEASUREMENT", "ProofMeasurement", "proof", "measurement_report_ir.json", ("Proof",)),
    DomainProofReportIR: ArtifactRoute("DERIVED_PROOF", "DomainProof", "proof", "domain_proof_report_ir.json", ("ProductProof", "Repair")),
    ProductProofBundleIR: ArtifactRoute("DERIVED_PROOF_BUNDLE", "ProductProof", "proof", "product_proof_bundle_ir.json", ("Repair", "Export")),
    RepairDirective: ArtifactRoute("DERIVED_COMMAND", "Repair", "orchestrator", "repair_directive.json", ("Compiler.repair_application",)),
    RuntimePackageIR: ArtifactRoute("RUNTIME_PROJECTION", "Export", "exports", "runtime_package_ir.json", ("runtime/realsas_cpp",)),
    CompileRequestIR: ArtifactRoute("EXECUTION_CONTRACT", "Compiler.compile_transaction", "orchestrator", "compile_request_ir.json", ("Compiler", "ProductShell")),
    CompileTransactionIR: ArtifactRoute("EXECUTION_LEDGER", "Compiler.compile_transaction", "orchestrator", "compile_transaction_ir.json", ("Compiler", "Proof", "Export", "ProductShell")),
    CompileResultIR: ArtifactRoute("EXECUTION_RESULT", "Compiler.compile_transaction", "orchestrator", "compile_result_ir.json", ("Export", "Runtime", "ProductShell")),
}

def _slug(value: str) -> str:
    text=re.sub(r"[^A-Za-z0-9_.-]+","_",str(value)).strip("._")
    return text or "unnamed"

def route_for(value: Any) -> ArtifactRoute:
    if isinstance(value, DirectionalRenderableIR):
        return ArtifactRoute("CANONICAL_DIRECTION", "Compiler.product_assembly_v3", "renderables", f"view_{value.view_index:02d}__directional_renderable_ir.json", ("Proof", "Runtime"))
    if isinstance(value, RenderableComponentIR):
        return ArtifactRoute("CANONICAL_RENDERABLE_COMPONENT", "Compiler.product_assembly_v3", "renderables", f"view_{value.view_index:02d}__component_{_slug(value.component_id)}.json", ("Proof", "Runtime"))
    if isinstance(value, AppearanceBindingIR):
        return ArtifactRoute("CANONICAL_APPEARANCE_BINDING", "AppearanceCompiler", "appearance", f"view_{value.target_view_index:02d}__appearance_{value.appearance_lineage_hash[:12]}.json", ("RenderableComponent", "Proof"))
    if isinstance(value, VisualCompletionProposalIR):
        return ArtifactRoute("PROPOSAL", "VisualCompletion", "candidates/completion", f"completion_{_slug(value.completion_id)}.json", ("Compiler.visual_completion_qualification",))
    if isinstance(value, QualifiedVisualCompletionIR):
        return ArtifactRoute("QUALIFIED_VISUAL_COMPLETION", "Compiler.visual_completion_qualification", "appearance", f"view_{value.target_view_index:02d}__completion_{_slug(value.completion_id)}.json", ("RenderableComponent", "Proof"))
    if isinstance(value, JointTransformTrackIR):
        return ArtifactRoute("CANONICAL_MOTION_TRACK", "MotionCompiler", "motion/tracks", f"joint_{_slug(value.canonical_joint_id)}__{_slug(value.track_id)}.json", ("MotionState", "Runtime", "Proof"))
    if isinstance(value, ComponentOrderTrackIR):
        return ArtifactRoute("CANONICAL_VISUAL_MOTION_TRACK", "MotionCompiler", "motion/tracks", f"view_{value.view_index:02d}__order_{_slug(value.track_id)}.json", ("MotionState", "Runtime", "Proof"))
    if isinstance(value, ComponentVisibilityTrackIR):
        return ArtifactRoute("CANONICAL_VISUAL_MOTION_TRACK", "MotionCompiler", "motion/tracks", f"view_{value.view_index:02d}__visibility_{_slug(value.track_id)}.json", ("MotionState", "Runtime", "Proof"))
    if isinstance(value, CapabilityQualificationIR):
        return ArtifactRoute("DERIVED_CAPABILITY_QUALIFICATION", "ProductProof", "proof", f"capability_{_slug(value.capability_id)}.json", ("Export",))
    try:
        return ARTIFACT_ROUTES[type(value)]
    except KeyError as exc:
        raise QualificationError(f"unregistered compiler artifact type:{type(value).__name__}") from exc

def write_typed_artifact(writer: BundleWriter, value: Any) -> str:
    route=route_for(value)
    writer.write_json(route.section, route.filename, value)
    return f"{route.section}/{route.filename}"
