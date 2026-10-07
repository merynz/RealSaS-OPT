from __future__ import annotations

"""V2 QualifiedMesh minting with an explicit TESSA learned-geometry authority lane.

Deterministic candidates continue through ``product_authority_v1`` unchanged.
TESSA candidates preserve learned XYZ/topology and are admitted only when the
exact candidate is already bound to the existing Stage19 static carrier PASS.
Material support remains a field-sampling / mechanical-support authority; it is
never reinterpreted as the source of learned geometry positions.

The historical Stage19 TESSA receipt remains static-only. Downstream Stage35 may
satisfy the old dynamic-field interlock only with exact carrier-native evidence:
the same Stage19 M, a qualified W_M bound to that M and skeleton, and a qualified
post-bind frame/skin-topology proof. No semantic skin transfer is accepted.
"""

from dataclasses import replace
import math
from typing import Any

from .carrier_skin_v1 import QualifiedCarrierSkinIR
from .hashing import content_sha256
from .mechanical_carrier_evidence_v1 import MechanicalCarrierEvidenceIR
from .mesh.conditioning_v1 import triangle_rest_metric
from .product_authority_v1 import (
    G3_NUMERICAL_MAX_ASPECT,
    G3_NUMERICAL_MIN_ANGLE_DEG,
    CanonicalMeshCandidateIR,
    QualifiedMeshIR,
    QualifiedMeshVertexIR,
    _edge_key,
    _geometric_topology_crack_audit,
    _mesh_connected_labels,
    _validate_g5_matrix,
    qualify_canonical_mesh_candidate,
    qualified_mesh_lineage_hash,
    validate_canonical_mesh_candidate,
    validate_component_carrier_policy,
    validate_deformation_capability_envelope,
    validate_mechanical_partition,
    validate_mesh_qualification_policy,
)
from .surface_addressing_v1 import StaticCanonicalMeshQualificationIR
from .tessa_candidate_bridge_v1 import TESSACandidateBridgeEvidenceIR
from .tessa_geometry_qualification_v1 import (
    TESSAStaticCarrierBindingIR,
    validate_tessa_static_carrier_binding_v1,
)
from .types import QualificationError

Json = dict[str, Any]
_TESSA_PRODUCER = "RealSaS.TESSALearnedMechanicalCarrierProposal.v1"
_DYNAMIC_FIELD_BLOCKER = "TESSA_MIRA_CARRIER_FIELD_BINDING_NOT_PRODUCT_QUALIFIED"
_TOL = 1.0e-9


def carrier_native_dynamic_binding_hash_v1(
    *,
    candidate: CanonicalMeshCandidateIR,
    carrier_evidence: MechanicalCarrierEvidenceIR,
    carrier_skin: QualifiedCarrierSkinIR,
    envelope,
    qualification_report: Json,
) -> str:
    """Validate and hash the generic exact-M dynamic binding proof.

    This proof replaces the *missing transport proof* that the old TESSA lane
    required. It does not waive G3; QualifiedMesh minting still requires the
    normal six-gate dynamic report separately.
    """
    if carrier_evidence.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise QualificationError("TESSA_CARRIER_NATIVE_CANDIDATE_BINDING_DRIFT")
    if carrier_skin.carrier_evidence_hash != carrier_evidence.carrier_evidence_hash:
        raise QualificationError("TESSA_CARRIER_NATIVE_SKIN_EVIDENCE_DRIFT")
    if carrier_skin.carrier_topology_hash != carrier_evidence.topology_hash:
        raise QualificationError("TESSA_CARRIER_NATIVE_SKIN_TOPOLOGY_DRIFT")
    if carrier_skin.carrier_geometry_hash != carrier_evidence.geometry_hash:
        raise QualificationError("TESSA_CARRIER_NATIVE_SKIN_GEOMETRY_DRIFT")
    if carrier_skin.skeleton_binding_hash != envelope.skeleton_lineage_hash:
        raise QualificationError("TESSA_CARRIER_NATIVE_SKIN_SKELETON_DRIFT")

    skin_report = dict(carrier_skin.qualification_report or {})
    if skin_report.get("status") != "PASS_DIRECT_CARRIER_SKIN":
        raise QualificationError("TESSA_CARRIER_NATIVE_SKIN_NOT_QUALIFIED")
    if skin_report.get("product_skin_evidence_complete") is not True:
        raise QualificationError("TESSA_CARRIER_NATIVE_SKIN_EVIDENCE_INCOMPLETE")
    if skin_report.get("semantic_skin_transfer_performed") is not False:
        raise QualificationError("TESSA_CARRIER_NATIVE_SEMANTIC_SKIN_TRANSFER_FORBIDDEN")

    report = dict(qualification_report or {})
    if report.get("mechanical_skin_domain") != "EXACT_STAGE19_CARRIER_M":
        raise QualificationError("TESSA_CARRIER_NATIVE_MECHANICAL_DOMAIN_DRIFT")
    if report.get("semantic_skin_transfer_performed") is not False:
        raise QualificationError("TESSA_CARRIER_NATIVE_STAGE35_TRANSFER_FORBIDDEN")
    if report.get("skin_topology_compatibility_status") != "PASS":
        raise QualificationError("TESSA_CARRIER_NATIVE_G3B_NOT_PASS")
    if report.get("skin_topology_weight_mutation") is not False:
        raise QualificationError("TESSA_CARRIER_NATIVE_WEIGHT_MUTATION_FORBIDDEN")

    compatibility_hash = str(report.get("skin_topology_compatibility_report_hash") or "")
    frame_hash = str(report.get("post_bind_joint_frame_qualification_hash") or "")
    if len(compatibility_hash) != 64:
        raise QualificationError("TESSA_CARRIER_NATIVE_G3B_HASH_MISSING")
    if len(frame_hash) != 64:
        raise QualificationError("TESSA_CARRIER_NATIVE_POST_BIND_FRAME_HASH_MISSING")

    return content_sha256(
        {
            "schema": "RealSaS.CarrierNativeDynamicBindingProof.v1",
            "candidate_mesh_binding_hash": candidate.candidate_lineage_hash,
            "carrier_evidence_hash": carrier_evidence.carrier_evidence_hash,
            "carrier_topology_hash": carrier_evidence.topology_hash,
            "carrier_geometry_hash": carrier_evidence.geometry_hash,
            "skeleton_binding_hash": carrier_skin.skeleton_binding_hash,
            "carrier_skin_lineage_hash": carrier_skin.skin_lineage_hash,
            "skin_topology_compatibility_report_hash": compatibility_hash,
            "post_bind_joint_frame_qualification_hash": frame_hash,
            "mechanical_skin_domain": "EXACT_STAGE19_CARRIER_M",
            "semantic_skin_transfer_performed": False,
            "weight_mutation_performed": False,
        }
    )


def _validate_tessa_material_support(vertex, *, surface_nodes, owner) -> set[str]:
    binding = vertex.support_binding
    if str(binding.mode) not in {"IDENTITY_SURFACE_NODE", "LOCAL_CONVEX_INTERPOLATION"}:
        raise QualificationError("TESSA_QUALIFIED_MESH_SUPPORT_MODE_INVALID")
    if not binding.coefficients:
        raise QualificationError("TESSA_QUALIFIED_MESH_SUPPORT_EMPTY")
    md = dict(binding.metadata or {})
    if md.get("authority_class") != "MATERIAL_SUPPORT_ONLY":
        raise QualificationError("TESSA_QUALIFIED_MESH_SUPPORT_AUTHORITY_INVALID")
    if md.get("geometry_position_derived_from_support") is not False:
        raise QualificationError("TESSA_QUALIFIED_MESH_SUPPORT_GEOMETRY_CONFLATION")
    if md.get("teacher_vertex_index_used") is not False:
        raise QualificationError("TESSA_QUALIFIED_MESH_TEACHER_VERTEX_INDEX_FORBIDDEN")

    seen: set[str] = set()
    total = 0.0
    for sid, raw_weight in binding.coefficients:
        sid = str(sid)
        weight = float(raw_weight)
        if sid in seen or sid not in surface_nodes:
            raise QualificationError("TESSA_QUALIFIED_MESH_SUPPORT_ID_INVALID")
        if owner.get(sid) != vertex.component_id:
            raise QualificationError("TESSA_QUALIFIED_MESH_SUPPORT_CROSS_COMPONENT")
        if not math.isfinite(weight) or weight < 0.0:
            raise QualificationError("TESSA_QUALIFIED_MESH_SUPPORT_WEIGHT_INVALID")
        seen.add(sid)
        total += weight
    if abs(total - 1.0) > _TOL:
        raise QualificationError("TESSA_QUALIFIED_MESH_SUPPORT_SIMPLEX_INVALID")
    if binding.mode == "IDENTITY_SURFACE_NODE" and (
        len(binding.coefficients) != 1
        or abs(float(binding.coefficients[0][1]) - 1.0) > _TOL
    ):
        raise QualificationError("TESSA_QUALIFIED_MESH_IDENTITY_SUPPORT_INVALID")
    return seen


def tessa_qualified_mesh_intrinsic_audit_v1(
    value: QualifiedMeshIR,
    *,
    surface,
    partition,
) -> Json:
    """Recompute exact-geometry G2/G4/rest checks without legacy convex-lift G1.

    The omitted statement is only ``P == convex(surface support)``. Support
    simplex/component provenance, topology, geometric crack checks, numerical rest
    conditioning and preserve-continuity checks remain fail-closed.
    """
    surface_nodes = {str(node.surface_id): node for node in surface.surface_nodes}
    owner = {
        str(sid): str(component.component_id)
        for component in partition.components
        for sid in component.surface_ids
    }
    component_ids = {str(component.component_id) for component in partition.components}
    if len(value.vertices) < 3 or not value.faces:
        raise QualificationError("TESSA_QUALIFIED_MESH_EMPTY_PRODUCT_GEOMETRY")

    vertex_by_id = {}
    support_ids_by_vertex: dict[str, set[str]] = {}
    for vertex in value.vertices:
        vid = str(vertex.canonical_mesh_vertex_id)
        p = tuple(map(float, vertex.P))
        if not vid or vid in vertex_by_id or len(p) != 3 or any(not math.isfinite(x) for x in p):
            raise QualificationError("TESSA_QUALIFIED_MESH_VERTEX_INVALID")
        if str(vertex.component_id) not in component_ids:
            raise QualificationError("TESSA_QUALIFIED_MESH_COMPONENT_INVALID")
        if vertex.refinement is not None:
            raise QualificationError("TESSA_QUALIFIED_MESH_LEGACY_REFINEMENT_FORBIDDEN")
        vmd = dict(vertex.metadata or {})
        if vmd.get("material_support_is_not_geometry_support") is not True:
            raise QualificationError("TESSA_QUALIFIED_MESH_VERTEX_AUTHORITY_SPLIT_MISSING")
        if vmd.get("geometry_position_derived_from_material_support") is not False:
            raise QualificationError("TESSA_QUALIFIED_MESH_VERTEX_GEOMETRY_CONFLATION")
        support_ids_by_vertex[vid] = _validate_tessa_material_support(
            vertex, surface_nodes=surface_nodes, owner=owner
        )
        vertex_by_id[vid] = vertex

    face_keys = set()
    used_vertex_ids: set[str] = set()
    derived_edges: set[tuple[str, str]] = set()
    face_count_by_component = {cid: 0 for cid in component_ids}
    min_angle = float("inf")
    max_aspect = 0.0
    for face in value.faces:
        ids = tuple(map(str, face))
        if len(ids) != 3 or len(set(ids)) != 3 or any(vid not in vertex_by_id for vid in ids):
            raise QualificationError("TESSA_QUALIFIED_MESH_FACE_INVALID")
        key = tuple(sorted(ids))
        if key in face_keys:
            raise QualificationError("TESSA_QUALIFIED_MESH_DUPLICATE_FACE")
        face_keys.add(key)
        used_vertex_ids.update(ids)
        components = {str(vertex_by_id[vid].component_id) for vid in ids}
        if len(components) != 1:
            raise QualificationError("TESSA_QUALIFIED_MESH_FACE_CROSSES_COMPONENT")
        component_id = next(iter(components))
        face_count_by_component[component_id] += 1
        metric = triangle_rest_metric(tuple(vertex_by_id[vid].P for vid in ids))
        if bool(metric["degenerate"]):
            raise QualificationError("TESSA_QUALIFIED_MESH_DEGENERATE_FACE")
        min_angle = min(min_angle, float(metric["min_angle_deg"]))
        max_aspect = max(max_aspect, float(metric["aspect_longest_over_min_altitude"]))
        derived_edges.update(
            (_edge_key(ids[0], ids[1]), _edge_key(ids[1], ids[2]), _edge_key(ids[2], ids[0]))
        )

    if any(count <= 0 for count in face_count_by_component.values()):
        raise QualificationError("TESSA_QUALIFIED_MESH_COMPONENT_WITHOUT_FACE")
    if used_vertex_ids != set(vertex_by_id):
        raise QualificationError("TESSA_QUALIFIED_MESH_UNUSED_VERTEX")

    declared_edges: set[tuple[str, str]] = set()
    for edge in value.edges:
        ids = tuple(map(str, edge))
        if len(ids) != 2 or any(vid not in vertex_by_id for vid in ids):
            raise QualificationError("TESSA_QUALIFIED_MESH_EDGE_INVALID")
        key = _edge_key(ids[0], ids[1])
        if key in declared_edges:
            raise QualificationError("TESSA_QUALIFIED_MESH_DUPLICATE_EDGE")
        declared_edges.add(key)
    if declared_edges != derived_edges:
        raise QualificationError("TESSA_QUALIFIED_MESH_EDGE_FACE_TOPOLOGY_MISMATCH")

    geometric_topology = _geometric_topology_crack_audit(vertex_by_id, value.faces)
    if min_angle + 1.0e-9 < G3_NUMERICAL_MIN_ANGLE_DEG:
        raise QualificationError("TESSA_QUALIFIED_MESH_NUMERICAL_MIN_ANGLE_FAIL")
    if max_aspect - 1.0e-9 > G3_NUMERICAL_MAX_ASPECT:
        raise QualificationError("TESSA_QUALIFIED_MESH_NUMERICAL_ASPECT_FAIL")

    labels = _mesh_connected_labels(set(vertex_by_id), declared_edges)
    preserve_checked = 0
    for constraint in partition.boundary_constraints:
        if constraint.decision != "PRESERVE_CONTINUITY":
            continue
        left = [
            vid for vid, sids in support_ids_by_vertex.items()
            if str(constraint.a_surface_id) in sids
        ]
        right = [
            vid for vid, sids in support_ids_by_vertex.items()
            if str(constraint.b_surface_id) in sids
        ]
        shared = [
            vid for vid, sids in support_ids_by_vertex.items()
            if str(constraint.a_surface_id) in sids
            and str(constraint.b_surface_id) in sids
        ]
        if not left or not right:
            raise QualificationError("TESSA_QUALIFIED_MESH_PRESERVE_SUPPORT_NOT_REPRESENTED")
        if not any(labels[a] == labels[b] for a in left for b in right):
            raise QualificationError("TESSA_QUALIFIED_MESH_PRESERVE_CONTINUITY_BROKEN")
        direct_edge = any(
            (a in left and b in right) or (a in right and b in left)
            for a, b in declared_edges
        )
        if not shared and not direct_edge:
            raise QualificationError("TESSA_QUALIFIED_MESH_PRESERVE_LOCAL_ADJACENCY_MISSING")
        preserve_checked += 1

    return {
        "audit_kind": "TESSA_STAGE19_BOUND_INTRINSIC_V1",
        "vertex_count": len(value.vertices),
        "face_count": len(value.faces),
        "edge_count": len(declared_edges),
        "component_count": len(component_ids),
        "component_face_counts": {
            key: face_count_by_component[key] for key in sorted(face_count_by_component)
        },
        "rest_min_angle_deg": float(min_angle),
        "rest_max_aspect_longest_over_min_altitude": float(max_aspect),
        "g3_numerical_min_angle_floor_deg": G3_NUMERICAL_MIN_ANGLE_DEG,
        "g3_numerical_max_aspect_ceiling": G3_NUMERICAL_MAX_ASPECT,
        "g4_preserve_constraint_count": preserve_checked,
        "g2_geometric_topology": geometric_topology,
        "material_support_simplex_and_component_checked": True,
        "material_support_used_as_geometry_authority": False,
        "legacy_convex_lift_position_equality_checked": False,
        "geometry_authority": "EXISTING_STAGE19_STATIC_CANONICAL_CARRIER",
    }


def _validate_dynamic_qualification_report(
    value: QualifiedMeshIR,
    *,
    partition,
    carrier_policy,
    envelope,
    policy,
) -> None:
    required = {
        "G1_SUPPORT_LINEAGE",
        "G2_TOPOLOGY",
        "G3_DEFORMATION",
        "G3B_SKIN_TOPOLOGY_COMPATIBILITY",
        "G4_COMPONENT_BOUNDARY",
        "G5_MULTIVIEW_COVERAGE",
    }
    report = dict(value.qualification_report or {})
    gates = dict(report.get("gates") or {})
    if set(gates) != required or any(gates[key] != "PASS" for key in required):
        raise QualificationError("TESSA_QUALIFIED_MESH_REQUIRES_ALL_SIX_GATES_PASS")
    if not str(report.get("skin_topology_compatibility_report_hash") or ""):
        raise QualificationError("TESSA_QUALIFIED_MESH_G3B_REPORT_HASH_MISSING")
    if report.get("skin_topology_compatibility_status") != "PASS":
        raise QualificationError("TESSA_QUALIFIED_MESH_G3B_NOT_PASS")
    if report.get("skin_topology_weight_mutation") is not False:
        raise QualificationError("TESSA_QUALIFIED_MESH_G3B_WEIGHT_MUTATION_FORBIDDEN")
    if report.get("single_aggregate_score_authority") is not False:
        raise QualificationError("TESSA_QUALIFIED_MESH_AGGREGATE_SCORE_FORBIDDEN")
    if report.get("g3_envelope_binding_hash") != envelope.envelope_lineage_hash:
        raise QualificationError("TESSA_QUALIFIED_MESH_G3_ENVELOPE_BINDING_MISMATCH")
    if not report.get("g3_stress_probe_hash") or report.get("g3_stress_probe_status") != "PASS":
        raise QualificationError("TESSA_QUALIFIED_MESH_G3_STRESS_NOT_PASS")
    if report.get("g3_motion_capability_claimed") is not False:
        raise QualificationError("TESSA_QUALIFIED_MESH_PREMATURE_MOTION_AUTHORITY")
    if int(report.get("consequential_unknown_boundary_count", -1)) != 0:
        raise QualificationError("TESSA_QUALIFIED_MESH_CONSEQUENTIAL_UNKNOWN_BOUNDARY")
    if any(row.decision == "UNKNOWN" for row in partition.boundary_constraints):
        if not report.get("unknown_boundary_analysis_hash"):
            raise QualificationError("TESSA_QUALIFIED_MESH_UNKNOWN_ANALYSIS_MISSING")
    component_ids = {str(component.component_id) for component in partition.components}
    _validate_g5_matrix(
        report,
        component_ids=component_ids,
        carrier_policy=carrier_policy,
        policy=policy,
    )
    if report.get("view_component_coverage_matrix_complete") is not True:
        raise QualificationError("TESSA_QUALIFIED_MESH_VIEW_COMPONENT_COVERAGE_REQUIRED")


def _validate_exact_candidate_promotion(
    value: QualifiedMeshIR,
    *,
    candidate: CanonicalMeshCandidateIR,
) -> None:
    id_map = {
        vertex.candidate_vertex_id: "MV:" + content_sha256({
            "candidate_lineage_hash": candidate.candidate_lineage_hash,
            "candidate_vertex_id": vertex.candidate_vertex_id,
        })[:24]
        for vertex in candidate.vertices
    }
    expected_vertices = {
        id_map[vertex.candidate_vertex_id]: vertex for vertex in candidate.vertices
    }
    actual_vertices = {str(vertex.canonical_mesh_vertex_id): vertex for vertex in value.vertices}
    if set(actual_vertices) != set(expected_vertices):
        raise QualificationError("TESSA_QUALIFIED_MESH_VERTEX_IDENTITY_DRIFT")
    for mesh_id, candidate_vertex in expected_vertices.items():
        vertex = actual_vertices[mesh_id]
        if (
            str(vertex.source_candidate_vertex_id) != str(candidate_vertex.candidate_vertex_id)
            or tuple(vertex.P) != tuple(candidate_vertex.P)
            or str(vertex.component_id) != str(candidate_vertex.component_id)
            or vertex.support_binding != candidate_vertex.support_binding
            or vertex.refinement != candidate_vertex.refinement
        ):
            raise QualificationError("TESSA_QUALIFIED_MESH_VERTEX_PROMOTION_DRIFT")
    expected_faces = tuple(
        tuple(id_map[str(vid)] for vid in face) for face in candidate.faces
    )
    expected_edges = tuple(
        tuple(id_map[str(vid)] for vid in edge) for edge in candidate.edges
    )
    if tuple(value.faces) != expected_faces or tuple(value.edges) != expected_edges:
        raise QualificationError("TESSA_QUALIFIED_MESH_TOPOLOGY_PROMOTION_DRIFT")


def validate_tessa_qualified_mesh_v2(
    value: QualifiedMeshIR,
    *,
    candidate: CanonicalMeshCandidateIR,
    bridge_evidence: TESSACandidateBridgeEvidenceIR,
    static_mesh: StaticCanonicalMeshQualificationIR,
    static_binding: TESSAStaticCarrierBindingIR,
    carrier_evidence: MechanicalCarrierEvidenceIR,
    carrier_skin: QualifiedCarrierSkinIR,
    surface,
    partition,
    carrier_policy,
    envelope,
    policy,
) -> None:
    validate_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
    )
    if candidate.producer_id != _TESSA_PRODUCER:
        raise QualificationError("TESSA_QUALIFIED_MESH_PRODUCER_INVALID")

    # Stage19 receipt remains static-only. The downstream carrier-native proof is
    # validated independently before the static receipt's dynamic requirement is
    # deliberately disabled for this one qualified path.
    expected_dynamic_hash = carrier_native_dynamic_binding_hash_v1(
        candidate=candidate,
        carrier_evidence=carrier_evidence,
        carrier_skin=carrier_skin,
        envelope=envelope,
        qualification_report=dict(value.qualification_report or {}),
    )
    validate_tessa_static_carrier_binding_v1(
        static_binding,
        candidate=candidate,
        bridge_evidence=bridge_evidence,
        static_mesh=static_mesh,
        require_dynamic_carrier_field_binding=False,
    )
    validate_mechanical_partition(partition, surface)
    validate_component_carrier_policy(carrier_policy, partition)
    validate_deformation_capability_envelope(envelope)
    validate_mesh_qualification_policy(policy)

    if value.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("TESSA_QUALIFIED_MESH_SURFACE_LINEAGE_MISMATCH")
    if value.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("TESSA_QUALIFIED_MESH_PARTITION_LINEAGE_MISMATCH")
    if value.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
        raise QualificationError("TESSA_QUALIFIED_MESH_CARRIER_POLICY_LINEAGE_MISMATCH")
    if value.envelope_binding_hash != envelope.envelope_lineage_hash:
        raise QualificationError("TESSA_QUALIFIED_MESH_ENVELOPE_LINEAGE_MISMATCH")
    if value.qualification_policy_hash != policy.qualification_policy_lineage_hash:
        raise QualificationError("TESSA_QUALIFIED_MESH_POLICY_LINEAGE_MISMATCH")

    _validate_exact_candidate_promotion(value, candidate=candidate)
    intrinsic = tessa_qualified_mesh_intrinsic_audit_v1(
        value, surface=surface, partition=partition
    )
    if intrinsic["rest_min_angle_deg"] + 1.0e-9 < float(policy.g3_min_angle_deg):
        raise QualificationError("TESSA_QUALIFIED_MESH_POLICY_MIN_ANGLE_FAIL")
    if intrinsic["rest_max_aspect_longest_over_min_altitude"] - 1.0e-9 > float(
        policy.g3_max_aspect_longest_over_min_altitude
    ):
        raise QualificationError("TESSA_QUALIFIED_MESH_POLICY_ASPECT_FAIL")
    report = dict(value.qualification_report or {})
    if report.get("intrinsic_audit_kind") != "TESSA_STAGE19_BOUND_INTRINSIC_V1":
        raise QualificationError("TESSA_QUALIFIED_MESH_INTRINSIC_AUDIT_KIND_INVALID")
    if report.get("intrinsic_audit_hash") != content_sha256(intrinsic):
        raise QualificationError("TESSA_QUALIFIED_MESH_INTRINSIC_AUDIT_HASH_MISMATCH")
    if report.get("stage19_static_binding_hash") != static_binding.binding_hash:
        raise QualificationError("TESSA_QUALIFIED_MESH_STAGE19_BINDING_HASH_MISMATCH")
    if report.get("carrier_native_dynamic_binding_hash") != expected_dynamic_hash:
        raise QualificationError("TESSA_QUALIFIED_MESH_DYNAMIC_BINDING_HASH_MISMATCH")
    if report.get("dynamic_interlock_satisfied_by") != "CARRIER_NATIVE_MIRA_W_M_V1":
        raise QualificationError("TESSA_QUALIFIED_MESH_DYNAMIC_BINDING_AUTHORITY_DRIFT")
    _validate_dynamic_qualification_report(
        value,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
    )

    md = dict(value.metadata or {})
    expected = {
        "source_candidate_lineage_hash": candidate.candidate_lineage_hash,
        "source_producer_id": candidate.producer_id,
        "source_producer_policy_hash": candidate.producer_policy_hash,
        "geometry_authority": "EXISTING_STAGE19_STATIC_CANONICAL_CARRIER",
        "tessa_static_carrier_binding_hash": static_binding.binding_hash,
        "tessa_proposal_geometry_hash": static_binding.proposal_geometry_hash,
        "material_support_field_hash": static_binding.material_support_field_hash,
        "carrier_native_dynamic_binding_hash": expected_dynamic_hash,
        "carrier_evidence_hash": carrier_evidence.carrier_evidence_hash,
        "carrier_skin_lineage_hash": carrier_skin.skin_lineage_hash,
    }
    for key, expected_value in expected.items():
        if md.get(key) != expected_value:
            raise QualificationError(f"TESSA_QUALIFIED_MESH_METADATA_BINDING_DRIFT:{key}")
    if md.get("legacy_g1_convex_lift_applicable") is not False:
        raise QualificationError("TESSA_QUALIFIED_MESH_LEGACY_G1_APPLICABILITY_DRIFT")
    if md.get("material_support_is_geometry_authority") is not False:
        raise QualificationError("TESSA_QUALIFIED_MESH_SUPPORT_GEOMETRY_AUTHORITY_FORBIDDEN")
    if md.get("learned_xyz_preserved_exactly") is not True:
        raise QualificationError("TESSA_QUALIFIED_MESH_LEARNED_XYZ_PRESERVATION_MISSING")
    if md.get("semantic_skin_transfer_performed") is not False:
        raise QualificationError("TESSA_QUALIFIED_MESH_SEMANTIC_SKIN_TRANSFER_FORBIDDEN")
    if md.get("motion_capability_claimed") is not False:
        raise QualificationError("TESSA_QUALIFIED_MESH_PREMATURE_MOTION_METADATA")
    if value.mesh_lineage_hash != qualified_mesh_lineage_hash(value):
        raise QualificationError("TESSA_QUALIFIED_MESH_LINEAGE_HASH_MISMATCH")


def qualify_tessa_canonical_mesh_candidate_v2(
    candidate: CanonicalMeshCandidateIR,
    *,
    bridge_evidence: TESSACandidateBridgeEvidenceIR,
    static_mesh: StaticCanonicalMeshQualificationIR,
    static_binding: TESSAStaticCarrierBindingIR,
    carrier_evidence: MechanicalCarrierEvidenceIR,
    carrier_skin: QualifiedCarrierSkinIR,
    surface,
    partition,
    carrier_policy,
    envelope,
    policy,
    qualification_report: Json,
) -> QualifiedMeshIR:
    validate_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
    )
    if candidate.producer_id != _TESSA_PRODUCER:
        raise QualificationError("TESSA_QUALIFIED_MESH_PRODUCER_INVALID")

    dynamic_binding_hash = carrier_native_dynamic_binding_hash_v1(
        candidate=candidate,
        carrier_evidence=carrier_evidence,
        carrier_skin=carrier_skin,
        envelope=envelope,
        qualification_report=qualification_report,
    )
    validate_tessa_static_carrier_binding_v1(
        static_binding,
        candidate=candidate,
        bridge_evidence=bridge_evidence,
        static_mesh=static_mesh,
        require_dynamic_carrier_field_binding=False,
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
            refinement=vertex.refinement,
            metadata=dict(vertex.metadata or {}),
        )
        for vertex in candidate.vertices
    )
    faces = tuple(tuple(id_map[str(vid)] for vid in face) for face in candidate.faces)
    edges = tuple(tuple(id_map[str(vid)] for vid in edge) for edge in candidate.edges)
    report = dict(qualification_report or {})
    report.pop("intrinsic_audit_hash", None)
    report.pop("intrinsic_audit_kind", None)
    report["stage19_static_binding_hash"] = static_binding.binding_hash
    report["carrier_native_dynamic_binding_hash"] = dynamic_binding_hash
    report["dynamic_interlock_satisfied_by"] = "CARRIER_NATIVE_MIRA_W_M_V1"

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
            "geometry_authority": "EXISTING_STAGE19_STATIC_CANONICAL_CARRIER",
            "tessa_static_carrier_binding_hash": static_binding.binding_hash,
            "tessa_proposal_geometry_hash": static_binding.proposal_geometry_hash,
            "material_support_field_hash": static_binding.material_support_field_hash,
            "carrier_native_dynamic_binding_hash": dynamic_binding_hash,
            "carrier_evidence_hash": carrier_evidence.carrier_evidence_hash,
            "carrier_skin_lineage_hash": carrier_skin.skin_lineage_hash,
            "legacy_g1_convex_lift_applicable": False,
            "material_support_is_geometry_authority": False,
            "learned_xyz_preserved_exactly": True,
            "semantic_skin_transfer_performed": False,
            "motion_capability_claimed": False,
        },
    )
    intrinsic = tessa_qualified_mesh_intrinsic_audit_v1(
        mesh, surface=surface, partition=partition
    )
    mesh = replace(
        mesh,
        qualification_report={
            **report,
            "intrinsic_audit_kind": "TESSA_STAGE19_BOUND_INTRINSIC_V1",
            "intrinsic_audit_hash": content_sha256(intrinsic),
        },
    )
    mesh = replace(mesh, mesh_lineage_hash=qualified_mesh_lineage_hash(mesh))
    validate_tessa_qualified_mesh_v2(
        mesh,
        candidate=candidate,
        bridge_evidence=bridge_evidence,
        static_mesh=static_mesh,
        static_binding=static_binding,
        carrier_evidence=carrier_evidence,
        carrier_skin=carrier_skin,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
    )
    return mesh


def qualify_canonical_mesh_candidate_v2(
    candidate: CanonicalMeshCandidateIR,
    *,
    surface,
    partition,
    carrier_policy,
    envelope,
    policy,
    qualification_report: Json,
    bridge_evidence: TESSACandidateBridgeEvidenceIR | None = None,
    static_mesh: StaticCanonicalMeshQualificationIR | None = None,
    static_binding: TESSAStaticCarrierBindingIR | None = None,
    carrier_evidence: MechanicalCarrierEvidenceIR | None = None,
    carrier_skin: QualifiedCarrierSkinIR | None = None,
) -> QualifiedMeshIR:
    """V2 dispatcher; deterministic candidates remain exactly on legacy G1."""
    if candidate.producer_id != _TESSA_PRODUCER:
        return qualify_canonical_mesh_candidate(
            candidate,
            surface=surface,
            partition=partition,
            carrier_policy=carrier_policy,
            envelope=envelope,
            policy=policy,
            qualification_report=qualification_report,
        )
    if bridge_evidence is None or static_mesh is None or static_binding is None:
        raise QualificationError("TESSA_QUALIFIED_MESH_STAGE19_EVIDENCE_REQUIRED")
    if carrier_evidence is None or carrier_skin is None:
        raise QualificationError(_DYNAMIC_FIELD_BLOCKER)
    return qualify_tessa_canonical_mesh_candidate_v2(
        candidate,
        bridge_evidence=bridge_evidence,
        static_mesh=static_mesh,
        static_binding=static_binding,
        carrier_evidence=carrier_evidence,
        carrier_skin=carrier_skin,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
        qualification_report=qualification_report,
    )
