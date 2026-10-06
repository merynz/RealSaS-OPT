from __future__ import annotations

"""V2 product-mesh authority dispatcher.

Deterministic candidates retain the exact V1 authority path.  TESSA learned
candidates use the existing Stage19 static carrier qualification as geometry
position authority while retaining component-safe GSA material support for skin
and field sampling.  No legacy G1 threshold is weakened or reinterpreted.
"""

from dataclasses import replace
import math
from typing import Any

from .hashing import content_sha256
from .mesh.conditioning_v1 import triangle_rest_metric
from .product_authority_v1 import (
    G3_NUMERICAL_MAX_ASPECT,
    G3_NUMERICAL_MIN_ANGLE_DEG,
    CanonicalMeshCandidateIR,
    ComponentCarrierPolicyIR,
    DeformationCapabilityEnvelopeIR,
    MechanicalPartitionIR,
    MeshQualificationPolicyIR,
    QualifiedMeshIR,
    QualifiedMeshVertexIR,
    _edge_key,
    _geometric_topology_crack_audit,
    _mesh_connected_labels,
    _validate_g5_matrix,
    qualify_canonical_mesh_candidate as qualify_canonical_mesh_candidate_v1,
    qualified_mesh_lineage_hash,
    validate_canonical_mesh_candidate,
    validate_component_carrier_policy,
    validate_deformation_capability_envelope,
    validate_mechanical_partition,
    validate_mesh_qualification_policy,
    validate_qualified_mesh as validate_qualified_mesh_v1,
)
from .surface_addressing_v1 import StaticCanonicalMeshQualificationIR
from .tessa_candidate_bridge_v1 import TESSACandidateBridgeEvidenceIR
from .tessa_geometry_qualification_v1 import (
    TESSAStaticCarrierBindingIR,
    validate_tessa_static_carrier_binding_v1,
)
from .types import QualificationError, RiggingSurfaceIR

Json = dict[str, Any]
_TESSA_PRODUCER = "RealSaS.TESSALearnedMechanicalCarrierProposal.v1"
_TESSA_GEOMETRY_CLASS = "TESSA_LEARNED_GEOMETRY_PROPOSAL_V1"
_SIMPLEX_TOL = 1.0e-9


def _vec_finite(value) -> bool:
    return len(value) == 3 and all(math.isfinite(float(x)) for x in value)


def _is_tessa_candidate(candidate: CanonicalMeshCandidateIR) -> bool:
    return str(candidate.producer_id) == _TESSA_PRODUCER


def _is_tessa_mesh(mesh: QualifiedMeshIR) -> bool:
    return str(dict(mesh.metadata or {}).get("geometry_authority_class") or "") == _TESSA_GEOMETRY_CLASS


def _learned_support_audit(binding, *, surface_nodes, owner, component_id: str) -> set[str]:
    if binding.mode not in {"IDENTITY_SURFACE_NODE", "LOCAL_CONVEX_INTERPOLATION"}:
        raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_MODE_INVALID")
    if not binding.coefficients:
        raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_EMPTY")
    md = dict(binding.metadata or {})
    if md.get("authority_class") != "MATERIAL_SUPPORT_ONLY":
        raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_AUTHORITY_INVALID")
    if md.get("geometry_position_derived_from_support") is not False:
        raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_GEOMETRY_CONFLATION")
    if md.get("teacher_vertex_index_used") is not False:
        raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_TEACHER_INDEX_FORBIDDEN")
    if str(md.get("mechanical_component_id") or "") != str(component_id):
        raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_COMPONENT_METADATA_DRIFT")

    total = 0.0
    seen: set[str] = set()
    for sid, coefficient in binding.coefficients:
        sid = str(sid)
        if sid in seen or sid not in surface_nodes:
            raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_ID_INVALID")
        if owner.get(sid) != component_id:
            raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_CROSS_COMPONENT")
        value = float(coefficient)
        if not math.isfinite(value) or value < 0.0:
            raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_COEFFICIENT_INVALID")
        seen.add(sid)
        total += value
    if abs(total - 1.0) > _SIMPLEX_TOL:
        raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_SIMPLEX_INVALID")
    if binding.mode == "IDENTITY_SURFACE_NODE":
        if len(binding.coefficients) != 1 or abs(float(binding.coefficients[0][1]) - 1.0) > _SIMPLEX_TOL:
            raise QualificationError("QUALIFIED_TESSA_MESH_IDENTITY_SUPPORT_INVALID")
    return seen


def learned_mesh_intrinsic_audit_v2(
    mesh: QualifiedMeshIR,
    *,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
) -> Json:
    """Recompute TESSA G1-material/G2/G4 and rest conditioning.

    Learned XYZ is intentionally *not* reconstructed from material support.  Its
    position authority is the hash-bound Stage19 static carrier qualification.
    """
    validate_mechanical_partition(partition, surface)
    surface_nodes = {str(node.surface_id): node for node in surface.surface_nodes}
    owner = {
        str(sid): str(component.component_id)
        for component in partition.components
        for sid in component.surface_ids
    }
    component_ids = {str(component.component_id) for component in partition.components}
    if len(mesh.vertices) < 3 or not mesh.faces:
        raise QualificationError("QUALIFIED_TESSA_MESH_EMPTY")

    vertex_by_id = {}
    support_ids_by_vertex: dict[str, set[str]] = {}
    component_vertex_ids = {cid: set() for cid in component_ids}
    for vertex in mesh.vertices:
        vid = str(vertex.canonical_mesh_vertex_id)
        component_id = str(vertex.component_id)
        if not vid or vid in vertex_by_id or not _vec_finite(vertex.P):
            raise QualificationError("QUALIFIED_TESSA_MESH_VERTEX_INVALID")
        if component_id not in component_ids:
            raise QualificationError("QUALIFIED_TESSA_MESH_COMPONENT_INVALID")
        if vertex.refinement is not None:
            raise QualificationError("QUALIFIED_TESSA_MESH_LEGACY_REFINEMENT_FORBIDDEN")
        vmd = dict(vertex.metadata or {})
        if vmd.get("geometry_authority_class") != _TESSA_GEOMETRY_CLASS:
            raise QualificationError("QUALIFIED_TESSA_MESH_VERTEX_GEOMETRY_AUTHORITY_INVALID")
        if vmd.get("material_support_is_not_geometry_support") is not True:
            raise QualificationError("QUALIFIED_TESSA_MESH_VERTEX_SUPPORT_SEPARATION_MISSING")
        if vmd.get("geometry_position_derived_from_material_support") is not False:
            raise QualificationError("QUALIFIED_TESSA_MESH_VERTEX_GEOMETRY_SUPPORT_CONFLATION")
        support_ids = _learned_support_audit(
            vertex.support_binding,
            surface_nodes=surface_nodes,
            owner=owner,
            component_id=component_id,
        )
        support_ids_by_vertex[vid] = support_ids
        component_vertex_ids[component_id].add(vid)
        vertex_by_id[vid] = vertex

    face_keys = set()
    derived_edges: set[tuple[str, str]] = set()
    used_vertex_ids: set[str] = set()
    face_count_by_component = {cid: 0 for cid in component_ids}
    min_angle = float("inf")
    max_aspect = 0.0
    for face in mesh.faces:
        if len(face) != 3 or len(set(face)) != 3 or any(str(vid) not in vertex_by_id for vid in face):
            raise QualificationError("QUALIFIED_TESSA_MESH_FACE_INVALID")
        face = tuple(map(str, face))
        key = tuple(sorted(face))
        if key in face_keys:
            raise QualificationError("QUALIFIED_TESSA_MESH_DUPLICATE_FACE")
        face_keys.add(key)
        used_vertex_ids.update(face)
        components = {str(vertex_by_id[vid].component_id) for vid in face}
        if len(components) != 1:
            raise QualificationError("QUALIFIED_TESSA_MESH_FACE_CROSSES_COMPONENT")
        component_id = next(iter(components))
        face_count_by_component[component_id] += 1
        points = tuple(vertex_by_id[vid].P for vid in face)
        metric = triangle_rest_metric(points)
        if metric["degenerate"]:
            raise QualificationError("QUALIFIED_TESSA_MESH_DEGENERATE_FACE")
        min_angle = min(min_angle, float(metric["min_angle_deg"]))
        max_aspect = max(max_aspect, float(metric["aspect_longest_over_min_altitude"]))
        derived_edges.update(
            (
                _edge_key(face[0], face[1]),
                _edge_key(face[1], face[2]),
                _edge_key(face[2], face[0]),
            )
        )

    if any(count <= 0 for count in face_count_by_component.values()):
        raise QualificationError("QUALIFIED_TESSA_MESH_COMPONENT_WITHOUT_FACE")
    if used_vertex_ids != set(vertex_by_id):
        raise QualificationError("QUALIFIED_TESSA_MESH_UNUSED_VERTEX")

    declared_edges = set()
    for edge in mesh.edges:
        if len(edge) != 2 or any(str(vid) not in vertex_by_id for vid in edge):
            raise QualificationError("QUALIFIED_TESSA_MESH_EDGE_INVALID")
        key = _edge_key(str(edge[0]), str(edge[1]))
        if key in declared_edges:
            raise QualificationError("QUALIFIED_TESSA_MESH_DUPLICATE_EDGE")
        declared_edges.add(key)
    if declared_edges != derived_edges:
        raise QualificationError("QUALIFIED_TESSA_MESH_EDGE_FACE_TOPOLOGY_MISMATCH")

    geometric_topology = _geometric_topology_crack_audit(vertex_by_id, mesh.faces)
    if min_angle + 1.0e-9 < G3_NUMERICAL_MIN_ANGLE_DEG:
        raise QualificationError("QUALIFIED_TESSA_MESH_NUMERICAL_MIN_ANGLE_FAIL")
    if max_aspect - 1.0e-9 > G3_NUMERICAL_MAX_ASPECT:
        raise QualificationError("QUALIFIED_TESSA_MESH_NUMERICAL_ASPECT_FAIL")

    labels = _mesh_connected_labels(set(vertex_by_id), declared_edges)
    preserve_checked = 0
    for constraint in partition.boundary_constraints:
        if constraint.decision != "PRESERVE_CONTINUITY":
            continue
        left = [vid for vid, sids in support_ids_by_vertex.items() if constraint.a_surface_id in sids]
        right = [vid for vid, sids in support_ids_by_vertex.items() if constraint.b_surface_id in sids]
        shared = [
            vid for vid, sids in support_ids_by_vertex.items()
            if constraint.a_surface_id in sids and constraint.b_surface_id in sids
        ]
        if not left or not right:
            raise QualificationError("QUALIFIED_TESSA_MESH_PRESERVE_SUPPORT_NOT_REPRESENTED")
        if not any(labels[a] == labels[b] for a in left for b in right):
            raise QualificationError("QUALIFIED_TESSA_MESH_PRESERVE_CONTINUITY_BROKEN")
        direct_edge = any(
            (a in left and b in right) or (a in right and b in left)
            for a, b in declared_edges
        )
        if not shared and not direct_edge:
            raise QualificationError("QUALIFIED_TESSA_MESH_PRESERVE_LOCAL_ADJACENCY_MISSING")
        preserve_checked += 1

    return {
        "geometry_authority_semantics": "STAGE19_EXACT_CANDIDATE_STATIC_SOURCE_FIDELITY",
        "material_support_semantics": "GSA_COMPONENT_SAFE_FIELD_SAMPLING_ONLY",
        "legacy_g1_convex_lift_applicable": False,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "edge_count": len(declared_edges),
        "component_count": len(component_ids),
        "component_face_counts": {key: face_count_by_component[key] for key in sorted(face_count_by_component)},
        "rest_min_angle_deg": min_angle,
        "rest_max_aspect_longest_over_min_altitude": max_aspect,
        "g3_numerical_min_angle_floor_deg": G3_NUMERICAL_MIN_ANGLE_DEG,
        "g3_numerical_max_aspect_ceiling": G3_NUMERICAL_MAX_ASPECT,
        "g4_preserve_constraint_count": preserve_checked,
        "g2_geometric_topology": geometric_topology,
    }


def _validate_common_report(
    mesh: QualifiedMeshIR,
    *,
    carrier_policy: ComponentCarrierPolicyIR,
    envelope: DeformationCapabilityEnvelopeIR,
    policy: MeshQualificationPolicyIR,
    intrinsic: Json,
) -> None:
    report = dict(mesh.qualification_report or {})
    if report.get("intrinsic_audit_hash") != content_sha256(intrinsic):
        raise QualificationError("QUALIFIED_TESSA_MESH_INTRINSIC_AUDIT_BINDING_MISMATCH")
    if intrinsic["rest_min_angle_deg"] + 1.0e-9 < policy.g3_min_angle_deg:
        raise QualificationError("QUALIFIED_TESSA_MESH_POLICY_MIN_ANGLE_FAIL")
    if intrinsic["rest_max_aspect_longest_over_min_altitude"] - 1.0e-9 > policy.g3_max_aspect_longest_over_min_altitude:
        raise QualificationError("QUALIFIED_TESSA_MESH_POLICY_ASPECT_FAIL")
    required = {
        "G1_SUPPORT_LINEAGE",
        "G2_TOPOLOGY",
        "G3_DEFORMATION",
        "G3B_SKIN_TOPOLOGY_COMPATIBILITY",
        "G4_COMPONENT_BOUNDARY",
        "G5_MULTIVIEW_COVERAGE",
    }
    gates = dict(report.get("gates") or {})
    if set(gates) != required or any(gates[key] != "PASS" for key in required):
        raise QualificationError("QUALIFIED_TESSA_MESH_REQUIRES_ALL_SIX_GATES_PASS")
    if report.get("g1_support_semantics") != "MATERIAL_SUPPORT_LINEAGE_ONLY":
        raise QualificationError("QUALIFIED_TESSA_MESH_G1_SUPPORT_SEMANTICS_INVALID")
    if report.get("geometry_authority_semantics") != "STAGE19_STATIC_CANONICAL_CARRIER":
        raise QualificationError("QUALIFIED_TESSA_MESH_GEOMETRY_AUTHORITY_SEMANTICS_INVALID")
    if report.get("skin_topology_compatibility_status") != "PASS":
        raise QualificationError("QUALIFIED_TESSA_MESH_G3B_NOT_PASS")
    if report.get("skin_topology_weight_mutation") is not False:
        raise QualificationError("QUALIFIED_TESSA_MESH_G3B_WEIGHT_MUTATION_FORBIDDEN")
    if report.get("single_aggregate_score_authority") is not False:
        raise QualificationError("QUALIFIED_TESSA_MESH_AGGREGATE_SCORE_FORBIDDEN")
    if report.get("g3_envelope_binding_hash") != envelope.envelope_lineage_hash:
        raise QualificationError("QUALIFIED_TESSA_MESH_G3_ENVELOPE_BINDING_MISMATCH")
    if not report.get("g3_stress_probe_hash") or report.get("g3_stress_probe_status") != "PASS":
        raise QualificationError("QUALIFIED_TESSA_MESH_G3_STRESS_NOT_PASS")
    if int(report.get("consequential_unknown_boundary_count", -1)) != 0:
        raise QualificationError("QUALIFIED_TESSA_MESH_UNKNOWN_BOUNDARY")
    if report.get("carrier_policy_hash") != carrier_policy.carrier_policy_lineage_hash:
        raise QualificationError("QUALIFIED_TESSA_MESH_CARRIER_POLICY_BINDING_MISMATCH")
    component_ids = {str(component.component_id) for component in carrier_policy.decisions}
    _validate_g5_matrix(
        report,
        component_ids=component_ids,
        carrier_policy=carrier_policy,
        policy=policy,
    )
    if report.get("view_component_coverage_matrix_complete") is not True:
        raise QualificationError("QUALIFIED_TESSA_MESH_G5_MATRIX_REQUIRED")


def validate_qualified_mesh_v2(
    mesh: QualifiedMeshIR,
    *,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
    envelope: DeformationCapabilityEnvelopeIR,
    policy: MeshQualificationPolicyIR,
) -> None:
    if not _is_tessa_mesh(mesh):
        validate_qualified_mesh_v1(
            mesh,
            surface=surface,
            partition=partition,
            carrier_policy=carrier_policy,
            envelope=envelope,
            policy=policy,
        )
        return

    validate_mechanical_partition(partition, surface)
    validate_component_carrier_policy(carrier_policy, partition)
    validate_deformation_capability_envelope(envelope)
    validate_mesh_qualification_policy(policy)
    if mesh.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("QUALIFIED_TESSA_MESH_SURFACE_LINEAGE_MISMATCH")
    if mesh.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("QUALIFIED_TESSA_MESH_PARTITION_LINEAGE_MISMATCH")
    if mesh.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
        raise QualificationError("QUALIFIED_TESSA_MESH_CARRIER_POLICY_LINEAGE_MISMATCH")
    if mesh.envelope_binding_hash != envelope.envelope_lineage_hash:
        raise QualificationError("QUALIFIED_TESSA_MESH_ENVELOPE_LINEAGE_MISMATCH")
    if mesh.qualification_policy_hash != policy.qualification_policy_lineage_hash:
        raise QualificationError("QUALIFIED_TESSA_MESH_POLICY_BINDING_MISMATCH")

    md = dict(mesh.metadata or {})
    required_hashes = (
        "source_candidate_lineage_hash",
        "tessa_static_carrier_binding_hash",
        "stage19_static_qualification_hash",
        "proposal_geometry_hash",
        "material_support_field_hash",
    )
    if any(len(str(md.get(key) or "")) != 64 for key in required_hashes):
        raise QualificationError("QUALIFIED_TESSA_MESH_AUTHORITY_HASH_MISSING")
    if md.get("legacy_g1_convex_lift_applicable") is not False:
        raise QualificationError("QUALIFIED_TESSA_MESH_LEGACY_G1_FLAG_INVALID")
    if md.get("material_support_is_geometry_authority") is not False:
        raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_GEOMETRY_AUTHORITY_FORBIDDEN")

    intrinsic = learned_mesh_intrinsic_audit_v2(mesh, surface=surface, partition=partition)
    _validate_common_report(
        mesh,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
        intrinsic=intrinsic,
    )
    report = dict(mesh.qualification_report or {})
    if report.get("tessa_static_carrier_binding_hash") != md["tessa_static_carrier_binding_hash"]:
        raise QualificationError("QUALIFIED_TESSA_MESH_STATIC_BINDING_HASH_DRIFT")
    if report.get("stage19_static_qualification_hash") != md["stage19_static_qualification_hash"]:
        raise QualificationError("QUALIFIED_TESSA_MESH_STAGE19_HASH_DRIFT")
    if report.get("proposal_geometry_hash") != md["proposal_geometry_hash"]:
        raise QualificationError("QUALIFIED_TESSA_MESH_PROPOSAL_HASH_DRIFT")
    if report.get("material_support_field_hash") != md["material_support_field_hash"]:
        raise QualificationError("QUALIFIED_TESSA_MESH_SUPPORT_FIELD_HASH_DRIFT")
    if mesh.mesh_lineage_hash != qualified_mesh_lineage_hash(mesh):
        raise QualificationError("QUALIFIED_TESSA_MESH_LINEAGE_HASH_MISMATCH")


def qualify_canonical_mesh_candidate_v2(
    candidate: CanonicalMeshCandidateIR,
    *,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
    envelope: DeformationCapabilityEnvelopeIR,
    policy: MeshQualificationPolicyIR,
    qualification_report: Json,
    static_qualification: StaticCanonicalMeshQualificationIR | None = None,
    tessa_bridge_evidence: TESSACandidateBridgeEvidenceIR | None = None,
    tessa_static_binding: TESSAStaticCarrierBindingIR | None = None,
) -> QualifiedMeshIR:
    if not _is_tessa_candidate(candidate):
        return qualify_canonical_mesh_candidate_v1(
            candidate,
            surface=surface,
            partition=partition,
            carrier_policy=carrier_policy,
            envelope=envelope,
            policy=policy,
            qualification_report=qualification_report,
        )

    if static_qualification is None or tessa_bridge_evidence is None or tessa_static_binding is None:
        raise QualificationError("TESSA_PRODUCT_MESH_STAGE19_AUTHORITY_REQUIRED")
    validate_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
    )
    validate_deformation_capability_envelope(envelope)
    validate_mesh_qualification_policy(policy)
    validate_tessa_static_carrier_binding_v1(
        tessa_static_binding,
        candidate=candidate,
        bridge_evidence=tessa_bridge_evidence,
        static_mesh=static_qualification,
    )

    id_map = {
        vertex.candidate_vertex_id: "MV:" + content_sha256({
            "candidate_lineage_hash": candidate.candidate_lineage_hash,
            "candidate_vertex_id": vertex.candidate_vertex_id,
        })[:24]
        for vertex in candidate.vertices
    }
    vertices = tuple(
        QualifiedMeshVertexIR(
            canonical_mesh_vertex_id=id_map[vertex.candidate_vertex_id],
            support_binding=vertex.support_binding,
            component_id=vertex.component_id,
            P=vertex.P,
            source_candidate_vertex_id=vertex.candidate_vertex_id,
            refinement=None,
            metadata=dict(vertex.metadata or {}),
        )
        for vertex in candidate.vertices
    )
    faces = tuple(tuple(id_map[vid] for vid in face) for face in candidate.faces)
    edges = tuple(tuple(id_map[vid] for vid in edge) for edge in candidate.edges)

    report = dict(qualification_report or {})
    report.pop("intrinsic_audit_hash", None)
    report.update(
        {
            "g1_support_semantics": "MATERIAL_SUPPORT_LINEAGE_ONLY",
            "geometry_authority_semantics": "STAGE19_STATIC_CANONICAL_CARRIER",
            "tessa_static_carrier_binding_hash": tessa_static_binding.binding_hash,
            "stage19_static_qualification_hash": static_qualification.qualification_hash,
            "proposal_geometry_hash": tessa_static_binding.proposal_geometry_hash,
            "material_support_field_hash": tessa_static_binding.material_support_field_hash,
        }
    )
    mesh = QualifiedMeshIR(
        vertices=vertices,
        faces=faces,
        edges=edges,
        surface_binding_hash=surface.geometry_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        envelope_binding_hash=envelope.envelope_lineage_hash,
        qualification_policy_hash=policy.qualification_policy_lineage_hash,
        qualification_report=report,
        mesh_lineage_hash="",
        metadata={
            "source_candidate_lineage_hash": candidate.candidate_lineage_hash,
            "source_producer_id": candidate.producer_id,
            "source_producer_policy_hash": candidate.producer_policy_hash,
            "geometry_authority_class": _TESSA_GEOMETRY_CLASS,
            "tessa_static_carrier_binding_hash": tessa_static_binding.binding_hash,
            "stage19_static_qualification_hash": static_qualification.qualification_hash,
            "proposal_geometry_hash": tessa_static_binding.proposal_geometry_hash,
            "material_support_field_hash": tessa_static_binding.material_support_field_hash,
            "legacy_g1_convex_lift_applicable": False,
            "material_support_is_geometry_authority": False,
        },
    )
    intrinsic = learned_mesh_intrinsic_audit_v2(mesh, surface=surface, partition=partition)
    mesh = replace(
        mesh,
        qualification_report={**report, "intrinsic_audit_hash": content_sha256(intrinsic)},
    )
    mesh = replace(mesh, mesh_lineage_hash=qualified_mesh_lineage_hash(mesh))
    validate_qualified_mesh_v2(
        mesh,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
    )
    return mesh
