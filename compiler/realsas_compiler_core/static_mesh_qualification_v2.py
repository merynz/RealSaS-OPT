from __future__ import annotations

"""Presentation-independent static Mesh authority for the product capability.

Historical V2 callers may bind source-fidelity through a qualified geometry
substrate. The current product path may instead bind the exact frozen Stage13
source-fidelity policy directly to the qualified camera/observation contract.
That direct path avoids fabricating a legacy zero-surface GeometrySubstrateIR
merely to replay the same source-fidelity court on the actual Mesh candidate.
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
class SourceFidelityPolicyBindingV1:
    camera_set_binding_hash: str
    observation_set_binding_hash: str
    policy_document_sha256: str
    policy: Json
    binding_hash: str
    schema_version: str = "RealSaS.SourceFidelityPolicyBindingIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


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


def source_fidelity_policy_binding_hash_v1(value: SourceFidelityPolicyBindingV1) -> str:
    payload = value.to_dict()
    payload.pop("binding_hash", None)
    return content_sha256(payload)


def build_source_fidelity_policy_binding_v1(
    *,
    cameras,
    observation,
    policy_document: Mapping[str, Any],
    policy_document_sha256: str,
) -> SourceFidelityPolicyBindingV1:
    document = dict(policy_document)
    if document.get("schema") != "RealSaS.Stage13RestReprojectionPolicy.v2":
        raise QualificationError("SOURCE_FIDELITY_POLICY_SCHEMA_DRIFT")
    if len(str(policy_document_sha256)) != 64:
        raise QualificationError("SOURCE_FIDELITY_POLICY_SHA_REQUIRED")
    if str(document.get("selected_profile") or "") != "P999":
        raise QualificationError("SOURCE_FIDELITY_POLICY_PROFILE_DRIFT")
    thresholds = dict(document.get("thresholds") or {})
    required = {
        "min_recall",
        "min_precision",
        "max_largest_coherent_hole_fraction",
        "max_interior_uncovered_fraction",
        "min_component_recall",
        "component_min_foreground_fraction",
        "max_silhouette_edge_p95_px",
    }
    if set(thresholds) != required:
        raise QualificationError("SOURCE_FIDELITY_POLICY_THRESHOLD_SET_DRIFT")
    aggregation = dict(document.get("aggregation") or {})
    if (
        aggregation.get("exact_eight_views_required") is not True
        or aggregation.get("every_view_must_pass_every_rule") is not True
        or aggregation.get("aggregate_compensation_forbidden") is not True
    ):
        raise QualificationError("SOURCE_FIDELITY_POLICY_AGGREGATION_DRIFT")
    if len(cameras.cameras) != 8 or len(observation.views) != 8:
        raise QualificationError("SOURCE_FIDELITY_POLICY_REQUIRES_EXACT_8_VIEWS")
    value = SourceFidelityPolicyBindingV1(
        camera_set_binding_hash=str(cameras.camera_set_hash),
        observation_set_binding_hash=str(observation.observation_set_hash),
        policy_document_sha256=str(policy_document_sha256),
        policy=thresholds,
        binding_hash="",
        metadata={
            "policy_schema": str(document["schema"]),
            "selected_profile": "P999",
            "policy_status": str(document.get("status") or ""),
            "policy_authority": str(document.get("authority") or ""),
            "direct_product_binding": True,
            "legacy_geometry_substrate_fabricated": False,
        },
    )
    return replace(value, binding_hash=source_fidelity_policy_binding_hash_v1(value))


def static_mesh_qualification_hash_v2(value: StaticMeshQualificationIRV2) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def qualify_static_mesh_v2(
    *,
    candidate,
    surface_addressing,
    mesh_policy,
    partition,
    cameras,
    observation,
    source_foreground: Mapping[int, bytes],
    geometry=None,
    source_fidelity_binding: SourceFidelityPolicyBindingV1 | None = None,
) -> StaticMeshQualificationResultV2:
    if (geometry is None) == (source_fidelity_binding is None):
        raise QualificationError("STATIC_MESH_V2_EXACTLY_ONE_SOURCE_FIDELITY_BINDING_REQUIRED")
    if surface_addressing.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise QualificationError("STATIC_MESH_V2_ADDRESSING_CANDIDATE_DRIFT")
    if surface_addressing.surface_binding_hash != candidate.surface_binding_hash:
        raise QualificationError("STATIC_MESH_V2_ADDRESSING_SOURCE_DRIFT")
    if partition.partition_lineage_hash != candidate.partition_binding_hash:
        raise QualificationError("STATIC_MESH_V2_PARTITION_BINDING_DRIFT")

    if geometry is not None:
        if not all(view.passed for view in geometry.views):
            raise QualificationError("STATIC_MESH_V2_GEOMETRY_SUBSTRATE_NOT_PASS")
        fidelity_binding = geometry
        gate_hash = str(geometry.substrate_hash)
        fidelity_owner = "GEOMETRY_SUBSTRATE_POLICY"
        direct_policy_binding = False
    else:
        fidelity_binding = source_fidelity_binding
        assert fidelity_binding is not None
        if fidelity_binding.binding_hash != source_fidelity_policy_binding_hash_v1(fidelity_binding):
            raise QualificationError("STATIC_MESH_V2_SOURCE_POLICY_BINDING_HASH_DRIFT")
        gate_hash = str(fidelity_binding.binding_hash)
        fidelity_owner = "DIRECT_FROZEN_SOURCE_FIDELITY_POLICY"
        direct_policy_binding = True

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
        geometry=fidelity_binding,
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
                "source_fidelity_binding_hash": gate_hash,
                "source_fidelity_binding_owner": fidelity_owner,
                "per_view": source_fidelity_rows,
                "policy": dict(fidelity_binding.policy),
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
        "source_fidelity_policy_replayed_on_actual_candidate_mesh": True,
        "legacy_geometry_substrate_required_for_source_fidelity": False if direct_policy_binding else True,
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
        geometry_gate_binding_hash=gate_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        qualification_report=report,
        qualification_hash="",
        metadata={
            "mesh_frozen_for_branching": True,
            "repair_failure_requires_new_mesh_lineage": True,
            "source_fidelity_policy_owner": fidelity_owner,
            "source_fidelity_measurement_target": "ACTUAL_CANDIDATE_MESH",
            "direct_source_fidelity_policy_binding": direct_policy_binding,
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
            "source_fidelity_binding_hash": gate_hash,
            "source_fidelity_binding_owner": fidelity_owner,
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
    "SourceFidelityPolicyBindingV1",
    "StaticMeshQualificationIRV2",
    "StaticMeshQualificationResultV2",
    "build_source_fidelity_policy_binding_v1",
    "qualify_static_mesh_v2",
    "source_fidelity_policy_binding_hash_v1",
    "static_mesh_qualification_hash_v2",
    "static_mesh_qualification_v2_from_dict",
]
