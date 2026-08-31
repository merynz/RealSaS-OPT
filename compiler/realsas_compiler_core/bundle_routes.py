from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from realsas_artifacts.bundle import BundleWriter
from .types import (
    ObservationEvidenceIR, RiggingSurfaceIR, SkeletonProposalIR, QualifiedSkeletonIR,
    SkinProposalIR, QualifiedSkinIR, MeshDiscretizationCandidateIR,
    QualifiedEditableMeshIR, QualifiedMeshSkinIR, CanonicalPuppetGraph,
    CanonicalPuppetGraphV2, ProofFrame, RepairDirective, RuntimePackageIR,
    QualificationError,
)

@dataclass(frozen=True)
class ArtifactRoute:
    authority_class: str
    producer: str
    section: str
    filename: str
    allowed_consumers: tuple[str, ...]

# Physical bundle section mirrors semantic authority. Proposal artifacts never land in
# canonical rig/weight/puppet sections; runtime never becomes a second product truth.
ARTIFACT_ROUTES: dict[type, ArtifactRoute] = {
    ObservationEvidenceIR: ArtifactRoute("EVIDENCE", "IRIS+observation bookkeeping", "ir", "observation_evidence_ir.json", ("GeometricSubstrateAssembler",)),
    RiggingSurfaceIR: ArtifactRoute("QUALIFIED_GEOMETRY", "GeometricSubstrateAssembler", "ir", "rigging_surface_ir.json", ("Geppetto", "Arachne", "Compiler")),
    SkeletonProposalIR: ArtifactRoute("PROPOSAL", "Geppetto", "candidates", "skeleton_proposal_ir.json", ("Compiler.rig_qualification",)),
    QualifiedSkeletonIR: ArtifactRoute("QUALIFIED", "Compiler.rig_qualification", "rig", "qualified_skeleton_ir.json", ("Arachne", "Compiler.product_assembly")),
    SkinProposalIR: ArtifactRoute("PROPOSAL", "Arachne", "candidates", "skin_proposal_ir.json", ("Compiler.skin_qualification",)),
    QualifiedSkinIR: ArtifactRoute("QUALIFIED", "Compiler.skin_qualification", "weight", "qualified_skin_ir.json", ("Compiler.mesh_weight_binding", "Compiler.product_assembly")),
    MeshDiscretizationCandidateIR: ArtifactRoute("DERIVED_CANDIDATE", "Compiler.mesh_discretization", "candidates/mesh", "mesh_discretization_candidate_ir.json", ("Compiler.geometry_qualification",)),
    QualifiedEditableMeshIR: ArtifactRoute("QUALIFIED_PRODUCT_COMPONENT", "Compiler.geometry_qualification", "mesh", "qualified_editable_mesh_ir.json", ("Compiler.mesh_weight_binding", "Compiler.product_assembly")),
    QualifiedMeshSkinIR: ArtifactRoute("QUALIFIED_PRODUCT_COMPONENT", "Compiler.mesh_weight_binding", "weight", "qualified_mesh_skin_ir.json", ("Compiler.product_assembly", "MotionProof")),
    CanonicalPuppetGraph: ArtifactRoute("CANONICAL_LEGACY_V1", "Compiler.product_assembly", "puppet", "canonical_puppet_graph.json", ("MotionProof", "Repair", "Export")),
    CanonicalPuppetGraphV2: ArtifactRoute("CANONICAL", "Compiler.product_assembly", "puppet", "canonical_puppet_graph_v2.json", ("MotionProof", "Repair", "Export")),
    ProofFrame: ArtifactRoute("DERIVED", "MotionProof", "proof", "proof_frame.json", ("Repair", "Export")),
    RepairDirective: ArtifactRoute("DERIVED_COMMAND", "Repair", "orchestrator", "repair_directive.json", ("Compiler.repair_application",)),
    RuntimePackageIR: ArtifactRoute("RUNTIME_PROJECTION", "Export", "exports", "runtime_package_ir.json", ("runtime/realsas_cpp",)),
}

def route_for(value: Any) -> ArtifactRoute:
    try:
        return ARTIFACT_ROUTES[type(value)]
    except KeyError as exc:
        raise QualificationError(f"unregistered compiler artifact type:{type(value).__name__}") from exc

def write_typed_artifact(writer: BundleWriter, value: Any) -> str:
    route=route_for(value)
    writer.write_json(route.section, route.filename, value)
    return f"{route.section}/{route.filename}"
