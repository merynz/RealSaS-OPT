from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from realsas_artifacts.bundle import BundleWriter
from .types import (
    ObservationEvidenceIR, RiggingSurfaceIR, SkeletonProposalIR, QualifiedSkeletonIR,
    SkinProposalIR, QualifiedSkinIR, CanonicalPuppetGraph, ProofFrame, RepairDirective,
    RuntimePackageIR, QualificationError,
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
    ObservationEvidenceIR: ArtifactRoute("EVIDENCE", "IRIS+observation bookkeeping", "ir", "observation_evidence_ir.json", ("SurfaceBuilder",)),
    RiggingSurfaceIR: ArtifactRoute("QUALIFIED_GEOMETRY", "SurfaceBuilder", "ir", "rigging_surface_ir.json", ("Geppetto", "Arachne", "Compiler")),
    SkeletonProposalIR: ArtifactRoute("PROPOSAL", "Geppetto", "candidates", "skeleton_proposal_ir.json", ("Compiler.rig_qualification",)),
    QualifiedSkeletonIR: ArtifactRoute("QUALIFIED", "Compiler.rig_qualification", "rig", "qualified_skeleton_ir.json", ("Arachne", "Compiler.product_assembly")),
    SkinProposalIR: ArtifactRoute("PROPOSAL", "Arachne", "candidates", "skin_proposal_ir.json", ("Compiler.skin_qualification",)),
    QualifiedSkinIR: ArtifactRoute("QUALIFIED", "Compiler.skin_qualification", "weight", "qualified_skin_ir.json", ("Compiler.product_assembly",)),
    CanonicalPuppetGraph: ArtifactRoute("CANONICAL", "Compiler.product_assembly", "puppet", "canonical_puppet_graph.json", ("MotionProof", "Repair", "Export")),
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
