from __future__ import annotations

"""Presentation-independent static Mesh authority for the product capability.

V1 static qualification remains immutable for historical DAG replay and binds a
legacy AppearanceDomain. V2 removes that accidental presentation dependency:
Mesh admission depends only on candidate geometry/topology, stable mechanical
surface addressing, geometry/source-fidelity policy, and mechanical partition.
Graphics may later bind its own presentation domain to the admitted Mesh.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from .hashing import content_sha256
from .mechanical_carrier_evidence_v1 import (
    MechanicalCarrierEvidenceIR,
    build_mechanical_carrier_evidence_v1,
)
from .static_mesh_qualification_v1 import (
    evaluate_candidate_source_fidelity_v1,
    relation_parent_quality_report_v1,
)
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class StaticMeshQualificationIRV2:
    candidate_mesh_binding_hash: str
    surface_addressing_binding_hash: str
    geometry_gate_binding_hash: str
    partition_binding_hash: str
    qualification_report: Json
    qualification_hash: str
    schema_version: str = "RealSaS.StaticMeshQualificationIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StaticMeshQualificationResultV2:
    status: str
    qualification: StaticMeshQualificationIRV2 | None
    carrier_evidence: MechanicalCarrierEvidenceIR | None
    blockers: tuple[str, ...]
    diagnostics: Json


def static_mesh_qualification_hash_v2(value: StaticMeshQualificationIRV2) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def qualify_static_mesh_v2(
    *,
    candidate,
    surface_addressing,
    geometry,
    mesh_policy,
    partition,
    cameras,
    observation,
    source_foreground: Mapping[int, bytes],
) -> StaticMeshQualificationResultV2:
    if surface_addressing.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise QualificationError("STATIC_MESH_V2_ADDRESSING_CANDIDATE_DRIFT")
    if surface_addressing.surface_binding_hash != candidate.surface_binding_hash:
        raise QualificationError("STATIC_MESH_V2_ADDRESSING_SOURCE_DRIFT")
    if partition.partition_lineage_hash != candidate.partition_binding_hash:
        raise QualificationError("STATIC_MESH_V2_PARTITION_BINDING_DRIFT")

    vertices = {
        str(vertex.candidate_vertex_id): tuple(map(float, vertex.P))
        for vertex in candidate.vertices
    }
    if not vertices or not candidate.faces:
        raise QualificationError("STATIC_MESH_V2_EMPTY")
    if any(
        len(point) != 3 or not all(math.isfinite(value) for value in point)
        for point in vertices.values()
    ):
        raise QualificationError("STATIC_MESH_V2_NONFINITE")
    if len(surface_addressing.vertex_addresses) != len(candidate.vertices):
        raise QualificationError("STATIC_MESH_V2_VERTEX_ADDRESSABILITY_INCOMPLETE")
    if len(surface_addressing.face_address_ids) != len(candidate.faces):
        raise QualificationError("STATIC_MESH_V2_FACE_ADDRESSABILITY_INCOMPLETE")
    if not all(view.passed for view in geometry.views):
        raise QualificationError("STATIC_MESH_V2_GEOMETRY_SUBSTRATE_NOT_PASS")

    static_quality = relation_parent_quality_report_v1(candidate, mesh_policy)
    if int(static_quality["degenerate_face_count"]) != 0:
        raise QualificationError("STATIC_MESH_V2_DEGENERATE_FACE")
    static_quality_passed = int(static_quality["policy_violating_face_count"]) == 0
    if not static_quality_passed:
        return StaticMeshQualificationResultV2(
            status="FAIL",
            qualification=None,
            carrier_evidence=None,
            blockers=("STATIC_MESH_QUALITY_POLICY_FAILED",),
            diagnostics={
                "candidate_mesh_binding_hash": candidate.candidate_lineage_hash,
                "mesh_policy_hash": mesh_policy.qualification_policy_lineage_hash,
                "static_quality": static_quality,
            },
        )

    source_fidelity_passed, source_fidelity_rows = evaluate_candidate_source_fidelity_v1(
        candidate=candidate,
        geometry=geometry,
        cameras=cameras,
        observation=observation,
        source_foreground=source_foreground,
    )
    if not source_fidelity_passed:
        return StaticMeshQualificationResultV2(
            status="FAIL",
            qualification=None,
            carrier_evidence=None,
            blockers=("STATIC_MESH_SOURCE_FIDELITY_FAILED",),
            diagnostics={
                "candidate_mesh_binding_hash": candidate.candidate_lineage_hash,
                "geometry_substrate_hash": geometry.substrate_hash,
                "per_view": source_fidelity_rows,
                "policy": dict(geometry.policy),
            },
        )

    report = {
        "status": "PASS_STATIC_CANONICAL_CARRIER",
        "vertex_count": len(candidate.vertices),
        "face_count": len(candidate.faces),
        "static_quality_policy_passed": True,
        "static_quality_min_angle_deg": float(static_quality["min_angle_deg"]),
        "static_quality_max_aspect": float(
            static_quality["max_aspect_longest_over_min_altitude"]
        ),
        "static_quality_policy_violating_face_count": 0,
        "mesh_policy_hash": mesh_policy.qualification_policy_lineage_hash,
        "surface_addressability_fraction": 1.0,
        "geometry_substrate_policy_replayed_on_actual_candidate_mesh": True,
        "actual_candidate_source_fidelity_passed": True,
        "actual_candidate_source_fidelity_views": source_fidelity_rows,
        "source_fidelity_qualification_passed": True,
        "demo_geometry_lineage": False,
        "product_authority_claimed": False,
        "presentation_domain_required_for_mesh_admission": False,
        "silhouette_is_geometry_authority_not_graphics": True,
    }
    value = StaticMeshQualificationIRV2(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash=surface_addressing.addressing_hash,
        geometry_gate_binding_hash=geometry.substrate_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        qualification_report=report,
        qualification_hash="",
        metadata={
            "mesh_frozen_for_branching": True,
            "repair_failure_requires_new_mesh_lineage": True,
            "source_fidelity_policy_owner": "GEOMETRY_SUBSTRATE_POLICY",
            "source_fidelity_measurement_target": "ACTUAL_CANDIDATE_MESH",
            "static_quality_measurement_target": "ACTUAL_CANDIDATE_MESH",
            "presentation_direction_binding": "FORBIDDEN_AT_MESH_BIRTH",
            "graphics_authority_claimed": False,
        },
    )
    value = replace(value, qualification_hash=static_mesh_qualification_hash_v2(value))
    carrier = build_mechanical_carrier_evidence_v1(
        candidate,
        static_qualification=value,
        surface_addressing=surface_addressing,
    )
    return StaticMeshQualificationResultV2(
        status="PASS",
        qualification=value,
        carrier_evidence=carrier,
        blockers=(),
        diagnostics={
            "qualification_hash": value.qualification_hash,
            "mechanical_carrier_evidence_hash": carrier.carrier_evidence_hash,
            "mechanical_carrier_topology_hash": carrier.topology_hash,
            "mechanical_carrier_geometry_hash": carrier.geometry_hash,
            "mechanical_carrier_invalid_normal_count": int(
                sum(not flag for flag in carrier.normal_valid)
            ),
            **report,
        },
    )


def static_mesh_qualification_v2_from_dict(payload: Mapping[str, Any]) -> StaticMeshQualificationIRV2:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.StaticMeshQualificationIR.v2":
        raise QualificationError("STATIC_MESH_V2_SCHEMA_DRIFT")
    value = StaticMeshQualificationIRV2(
        candidate_mesh_binding_hash=str(payload["candidate_mesh_binding_hash"]),
        surface_addressing_binding_hash=str(payload["surface_addressing_binding_hash"]),
        geometry_gate_binding_hash=str(payload["geometry_gate_binding_hash"]),
        partition_binding_hash=str(payload["partition_binding_hash"]),
        qualification_report=dict(payload.get("qualification_report") or {}),
        qualification_hash=str(payload["qualification_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.qualification_hash != static_mesh_qualification_hash_v2(value):
        raise QualificationError("STATIC_MESH_V2_QUALIFICATION_HASH_DRIFT")
    if value.qualification_report.get("status") != "PASS_STATIC_CANONICAL_CARRIER":
        raise QualificationError("STATIC_MESH_V2_NOT_PASS")
    if value.qualification_report.get("demo_geometry_lineage") is not False:
        raise QualificationError("STATIC_MESH_V2_DEMO_LINEAGE_FORBIDDEN")
    if value.qualification_report.get("presentation_domain_required_for_mesh_admission") is not False:
        raise QualificationError("STATIC_MESH_V2_PRESENTATION_DEPENDENCY_FORBIDDEN")
    return value


__all__ = [
    "StaticMeshQualificationIRV2",
    "StaticMeshQualificationResultV2",
    "qualify_static_mesh_v2",
    "static_mesh_qualification_hash_v2",
    "static_mesh_qualification_v2_from_dict",
]
