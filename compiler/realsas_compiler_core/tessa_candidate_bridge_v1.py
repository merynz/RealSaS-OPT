from __future__ import annotations

"""Typed bridge from a TESSA learned mesh proposal to a Compiler mesh candidate.

This module deliberately preserves the authority split discovered by the Knight
mechanical-coherence work:

- TESSA owns a learned production-geometry *proposal* (XYZ + topology).
- ``TESSAMaterialSupportFieldIR`` owns component-safe continuous material/
  mechanical support over ``RiggingSurfaceIR``.
- Material support is not geometric support and may not be used to pretend that
  learned XYZ is a convex lift of GSA nodes.
- Final static carrier admission is the existing V2 Stage19 exact-candidate
  qualification, not a new parallel geometry court.

The bridge therefore creates a ``CanonicalMeshCandidateIR`` whose vertex XYZ is
exactly the learned proposal XYZ while its ``SurfaceSupportBinding`` is explicitly
marked material-only.  Product authority remains Compiler-owned downstream.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from models.tessa.v1.contracts_v1 import (
    TESSAMeshProposalV1,
    TESSAScalingPolicyV1,
    TESSAVertexProposalV1,
)

from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    ComponentCarrierPolicyIR,
    MechanicalPartitionIR,
    canonical_mesh_candidate_lineage_hash,
    validate_canonical_mesh_candidate,
)
from .tessa_surface_field_v1 import (
    TESSAMaterialSupportFieldIR,
    material_support_binding_v1,
    validate_tessa_material_support_field_v1,
)
from .types import QualificationError, RiggingSurfaceIR

Json = dict[str, Any]
_LEARNED_GEOMETRY_AUTHORITY_CLASS = "TESSA_LEARNED_GEOMETRY_PROPOSAL_V1"


@dataclass(frozen=True)
class TESSACandidateBridgeEvidenceIR:
    proposal_geometry_hash: str
    proposal_topology_sequence_hash: str
    proposal_source_geometry_lineage_hash: str
    material_support_field_hash: str
    surface_binding_hash: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    vertex_count: int
    face_count: int
    nearest_surface_distance_min: float
    nearest_surface_distance_median: float
    nearest_surface_distance_p95: float
    nearest_surface_distance_max: float
    evidence_hash: str
    schema_version: str = "RealSaS.TESSACandidateBridgeEvidenceIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def tessa_mesh_proposal_from_dict_v1(payload: Mapping[str, Any]) -> TESSAMeshProposalV1:
    if str(payload.get("schema_version") or payload.get("schema") or "") != "RealSaS.TESSAMeshProposal.v1":
        raise QualificationError("TESSA_PROPOSAL_SCHEMA_INVALID")
    raw_vertices = tuple(payload.get("vertices") or ())
    raw_faces = tuple(payload.get("faces") or ())
    vertices = tuple(
        TESSAVertexProposalV1(
            proposal_vertex_id=str(row["proposal_vertex_id"]),
            P=tuple(map(float, row["P"])),
            primary_surface_id=str(row["primary_surface_id"]),
            component_id=str(row["component_id"]),
            confidence=float(row.get("confidence", 1.0)),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in raw_vertices
    )
    proposal = TESSAMeshProposalV1(
        vertices=vertices,
        faces=tuple(tuple(map(str, row)) for row in raw_faces),
        source_geometry_lineage_hash=str(payload.get("source_geometry_lineage_hash") or ""),
        model_provenance=str(payload.get("model_provenance") or ""),
        topology_sequence_hash=str(payload.get("topology_sequence_hash") or ""),
        metadata=dict(payload.get("metadata") or {}),
        schema_version="RealSaS.TESSAMeshProposal.v1",
    )
    try:
        proposal.validate(TESSAScalingPolicyV1())
    except (TypeError, ValueError, KeyError) as exc:
        raise QualificationError(f"TESSA_PROPOSAL_PAYLOAD_INVALID:{type(exc).__name__}") from exc
    if not proposal.model_provenance:
        raise QualificationError("TESSA_PROPOSAL_MODEL_PROVENANCE_MISSING")
    if dict(proposal.metadata or {}).get("product_authority_claimed") is True:
        raise QualificationError("TESSA_PROPOSAL_PREMATURE_PRODUCT_AUTHORITY")
    return proposal


def tessa_candidate_bridge_evidence_from_dict_v1(
    payload: Mapping[str, Any],
) -> TESSACandidateBridgeEvidenceIR:
    if str(payload.get("schema_version") or payload.get("schema") or "") != "RealSaS.TESSACandidateBridgeEvidenceIR.v1":
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_SCHEMA_INVALID")
    return TESSACandidateBridgeEvidenceIR(
        proposal_geometry_hash=str(payload.get("proposal_geometry_hash") or ""),
        proposal_topology_sequence_hash=str(payload.get("proposal_topology_sequence_hash") or ""),
        proposal_source_geometry_lineage_hash=str(payload.get("proposal_source_geometry_lineage_hash") or ""),
        material_support_field_hash=str(payload.get("material_support_field_hash") or ""),
        surface_binding_hash=str(payload.get("surface_binding_hash") or ""),
        partition_binding_hash=str(payload.get("partition_binding_hash") or ""),
        carrier_policy_binding_hash=str(payload.get("carrier_policy_binding_hash") or ""),
        vertex_count=int(payload.get("vertex_count", 0)),
        face_count=int(payload.get("face_count", 0)),
        nearest_surface_distance_min=float(payload.get("nearest_surface_distance_min", float("nan"))),
        nearest_surface_distance_median=float(payload.get("nearest_surface_distance_median", float("nan"))),
        nearest_surface_distance_p95=float(payload.get("nearest_surface_distance_p95", float("nan"))),
        nearest_surface_distance_max=float(payload.get("nearest_surface_distance_max", float("nan"))),
        evidence_hash=str(payload.get("evidence_hash") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )


def tessa_proposal_geometry_hash_v1(proposal: TESSAMeshProposalV1) -> str:
    return content_sha256({
        "schema": "RealSaS.TESSAProposalGeometryBinding.v1",
        "source_geometry_lineage_hash": proposal.source_geometry_lineage_hash,
        "topology_sequence_hash": proposal.topology_sequence_hash,
        "model_provenance": proposal.model_provenance,
        "vertices": tuple(
            {
                "proposal_vertex_id": str(vertex.proposal_vertex_id),
                "P": tuple(map(float, vertex.P)),
                "primary_surface_id": str(vertex.primary_surface_id),
                "component_id": str(vertex.component_id),
            }
            for vertex in proposal.vertices
        ),
        "faces": tuple(tuple(map(str, face)) for face in proposal.faces),
    })


def tessa_candidate_bridge_evidence_hash_v1(
    value: TESSACandidateBridgeEvidenceIR,
) -> str:
    payload = value.to_dict()
    payload.pop("evidence_hash", None)
    return content_sha256(payload)


def validate_tessa_candidate_bridge_evidence_v1(
    value: TESSACandidateBridgeEvidenceIR,
    *,
    proposal: TESSAMeshProposalV1,
    support_field: TESSAMaterialSupportFieldIR,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
) -> None:
    proposal.validate(TESSAScalingPolicyV1())
    validate_tessa_material_support_field_v1(
        support_field,
        surface=surface,
        partition=partition,
        expected_vertex_ids=tuple(v.proposal_vertex_id for v in proposal.vertices),
    )
    if proposal.source_geometry_lineage_hash != surface.geometry_lineage_hash:
        raise QualificationError("TESSA_CANDIDATE_SOURCE_GEOMETRY_LINEAGE_MISMATCH")
    if support_field.topology_sequence_hash != proposal.topology_sequence_hash:
        raise QualificationError("TESSA_CANDIDATE_TOPOLOGY_SEQUENCE_MISMATCH")
    if value.proposal_geometry_hash != tessa_proposal_geometry_hash_v1(proposal):
        raise QualificationError("TESSA_CANDIDATE_PROPOSAL_GEOMETRY_HASH_MISMATCH")
    if value.proposal_topology_sequence_hash != proposal.topology_sequence_hash:
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_TOPOLOGY_MISMATCH")
    if value.proposal_source_geometry_lineage_hash != surface.geometry_lineage_hash:
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_SURFACE_MISMATCH")
    if value.material_support_field_hash != support_field.field_lineage_hash:
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_SUPPORT_FIELD_MISMATCH")
    if value.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_SURFACE_BINDING_MISMATCH")
    if value.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_PARTITION_BINDING_MISMATCH")
    if value.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_CARRIER_POLICY_MISMATCH")
    if value.vertex_count != len(proposal.vertices) or value.face_count != len(proposal.faces):
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_COUNT_MISMATCH")
    distances = (
        value.nearest_surface_distance_min,
        value.nearest_surface_distance_median,
        value.nearest_surface_distance_p95,
        value.nearest_surface_distance_max,
    )
    if any(not math.isfinite(float(x)) or float(x) < 0.0 for x in distances):
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_DISTANCE_INVALID")
    if not (
        value.nearest_surface_distance_min
        <= value.nearest_surface_distance_median
        <= value.nearest_surface_distance_p95
        <= value.nearest_surface_distance_max
    ):
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_DISTANCE_ORDER_INVALID")
    md = dict(value.metadata or {})
    if md.get("authority_class") != "LEARNED_GEOMETRY_PROPOSAL_EVIDENCE":
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_AUTHORITY_CLASS_INVALID")
    if md.get("geometry_position_derived_from_material_support") is not False:
        raise QualificationError("TESSA_CANDIDATE_GEOMETRY_SUPPORT_CONFLATION")
    if md.get("product_geometry_authority_claimed") is not False:
        raise QualificationError("TESSA_CANDIDATE_PREMATURE_PRODUCT_AUTHORITY")
    if md.get("teacher_vertex_index_used") is not False:
        raise QualificationError("TESSA_CANDIDATE_TEACHER_VERTEX_INDEX_FORBIDDEN")
    if value.evidence_hash != tessa_candidate_bridge_evidence_hash_v1(value):
        raise QualificationError("TESSA_CANDIDATE_EVIDENCE_HASH_MISMATCH")


def build_tessa_candidate_bridge_v1(
    *,
    proposal: TESSAMeshProposalV1,
    support_field: TESSAMaterialSupportFieldIR,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
    producer_policy_hash: str,
) -> tuple[CanonicalMeshCandidateIR, TESSACandidateBridgeEvidenceIR]:
    """Build a non-authoritative Compiler candidate from exact TESSA evidence."""
    proposal.validate(TESSAScalingPolicyV1())
    if proposal.source_geometry_lineage_hash != surface.geometry_lineage_hash:
        raise QualificationError("TESSA_CANDIDATE_SOURCE_GEOMETRY_LINEAGE_MISMATCH")
    if not producer_policy_hash:
        raise QualificationError("TESSA_CANDIDATE_PRODUCER_POLICY_HASH_MISSING")

    proposal_ids = tuple(str(v.proposal_vertex_id) for v in proposal.vertices)
    validate_tessa_material_support_field_v1(
        support_field,
        surface=surface,
        partition=partition,
        expected_vertex_ids=proposal_ids,
    )
    if support_field.topology_sequence_hash != proposal.topology_sequence_hash:
        raise QualificationError("TESSA_CANDIDATE_TOPOLOGY_SEQUENCE_MISMATCH")

    row_by_id = {str(row.proposal_vertex_id): row for row in support_field.rows}
    if set(row_by_id) != set(proposal_ids):
        raise QualificationError("TESSA_CANDIDATE_SUPPORT_VERTEX_ACCOUNTING_MISMATCH")

    proposal_geometry_hash = tessa_proposal_geometry_hash_v1(proposal)
    candidate_id = {
        proposal_id: "TCV:" + content_sha256({
            "proposal_geometry_hash": proposal_geometry_hash,
            "proposal_vertex_id": proposal_id,
        })[:24]
        for proposal_id in proposal_ids
    }

    vertices = []
    for vertex in proposal.vertices:
        if len(vertex.P) != 3 or any(not math.isfinite(float(x)) for x in vertex.P):
            raise QualificationError("TESSA_CANDIDATE_VERTEX_POSITION_INVALID")
        row = row_by_id[str(vertex.proposal_vertex_id)]
        vertices.append(
            CanonicalMeshVertexCandidateIR(
                candidate_vertex_id=candidate_id[str(vertex.proposal_vertex_id)],
                support_binding=material_support_binding_v1(row),
                component_id=str(row.mechanical_component_id),
                P=tuple(map(float, vertex.P)),
                refinement=None,
                metadata={
                    "geometry_authority_class": _LEARNED_GEOMETRY_AUTHORITY_CLASS,
                    "proposal_vertex_id": str(vertex.proposal_vertex_id),
                    "proposal_primary_surface_id": str(vertex.primary_surface_id),
                    "proposal_component_id": str(vertex.component_id),
                    "proposal_geometry_hash": proposal_geometry_hash,
                    "material_support_field_hash": support_field.field_lineage_hash,
                    "material_support_is_not_geometry_support": True,
                    "geometry_position_derived_from_material_support": False,
                    "teacher_vertex_index_used": False,
                },
            )
        )

    faces = tuple(
        tuple(candidate_id[str(vertex_id)] for vertex_id in face)
        for face in proposal.faces
    )
    edges = tuple(sorted({
        tuple(sorted((face[i], face[j])))
        for face in faces
        for i, j in ((0, 1), (1, 2), (2, 0))
    }))

    distances = sorted(float(row.nearest_surface_distance) for row in support_field.rows)
    n = len(distances)
    if n == 0:
        raise QualificationError("TESSA_CANDIDATE_SUPPORT_FIELD_EMPTY")

    def percentile_nearest(q: float) -> float:
        if n == 1:
            return distances[0]
        position = q * (n - 1)
        lo = int(math.floor(position))
        hi = int(math.ceil(position))
        if lo == hi:
            return distances[lo]
        alpha = position - lo
        return distances[lo] * (1.0 - alpha) + distances[hi] * alpha

    provisional_evidence = TESSACandidateBridgeEvidenceIR(
        proposal_geometry_hash=proposal_geometry_hash,
        proposal_topology_sequence_hash=str(proposal.topology_sequence_hash),
        proposal_source_geometry_lineage_hash=str(proposal.source_geometry_lineage_hash),
        material_support_field_hash=str(support_field.field_lineage_hash),
        surface_binding_hash=str(surface.geometry_lineage_hash),
        partition_binding_hash=str(partition.partition_lineage_hash),
        carrier_policy_binding_hash=str(carrier_policy.carrier_policy_lineage_hash),
        vertex_count=len(vertices),
        face_count=len(faces),
        nearest_surface_distance_min=float(distances[0]),
        nearest_surface_distance_median=float(percentile_nearest(0.5)),
        nearest_surface_distance_p95=float(percentile_nearest(0.95)),
        nearest_surface_distance_max=float(distances[-1]),
        evidence_hash="",
        metadata={
            "authority_class": "LEARNED_GEOMETRY_PROPOSAL_EVIDENCE",
            "geometry_position_derived_from_material_support": False,
            "product_geometry_authority_claimed": False,
            "teacher_vertex_index_used": False,
            "distance_metrics_are_diagnostic_not_acceptance_thresholds": True,
            "requires_stage19_static_qualification": True,
            "requires_independent_multiview_source_fidelity": True,
        },
    )
    evidence = replace(
        provisional_evidence,
        evidence_hash=tessa_candidate_bridge_evidence_hash_v1(provisional_evidence),
    )
    validate_tessa_candidate_bridge_evidence_v1(
        evidence,
        proposal=proposal,
        support_field=support_field,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
    )

    provisional_candidate = CanonicalMeshCandidateIR(
        vertices=tuple(vertices),
        faces=faces,
        edges=edges,
        surface_binding_hash=str(surface.geometry_lineage_hash),
        partition_binding_hash=str(partition.partition_lineage_hash),
        carrier_policy_binding_hash=str(carrier_policy.carrier_policy_lineage_hash),
        producer_id="RealSaS.TESSALearnedMechanicalCarrierProposal.v1",
        producer_policy_hash=str(producer_policy_hash),
        candidate_lineage_hash="",
        metadata={
            "geometry_authority_class": _LEARNED_GEOMETRY_AUTHORITY_CLASS,
            "proposal_geometry_hash": proposal_geometry_hash,
            "proposal_topology_sequence_hash": str(proposal.topology_sequence_hash),
            "material_support_field_hash": str(support_field.field_lineage_hash),
            "candidate_bridge_evidence_hash": evidence.evidence_hash,
            "material_support_is_not_geometry_support": True,
            "learned_xyz_preserved_exactly": True,
            "legacy_g1_convex_lift_claimed": False,
            "product_geometry_authority_claimed": False,
            "requires_stage19_static_qualification": True,
        },
    )
    candidate = replace(
        provisional_candidate,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional_candidate),
    )
    validate_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
    )
    return candidate, evidence
