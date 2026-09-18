from __future__ import annotations

"""G3 deformation-conditioning stress probe for canonical mesh candidates.

This is a numerical/mechanical gate, not a motion-aesthetics score. It exercises the
exact candidate support under the exact QualifiedSkinIR and frozen rotation envelope.
No mesh-skin product artifact is minted here; the W->candidate transfer is an internal
measurement witness and stage 28 remains the sole mesh-skin authority seal.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

import numpy as np

from ..hashing import content_sha256
from ..motion_3d_adapter_v1 import build_joint_specs_v1
from ..motion_3d_v1 import apply_lbs_matrix_v1, solve_fk_v1
from ..product_authority_v1 import (
    CanonicalMeshCandidateIR,
    DeformationCapabilityEnvelopeIR,
    MeshQualificationPolicyIR,
    validate_deformation_capability_envelope,
    validate_mesh_qualification_policy,
)
from ..types import QualificationError, QualifiedSkeletonIR, QualifiedSkinIR, RiggingSurfaceIR

Json = dict[str, Any]
G3_STRESS_SCHEMA = "RealSaS.G3DeformationStressReport.v1"
G3_PROBE_PLAN_SCHEMA = "RealSaS.G3RotationEnvelopeProbePlan.v1"


@dataclass(frozen=True)
class G3DeformationStressReportIR:
    candidate_lineage_hash: str
    surface_lineage_hash: str
    skeleton_lineage_hash: str
    skin_lineage_hash: str
    envelope_lineage_hash: str
    axis_contract_hash: str
    qualification_policy_hash: str
    probe_plan_hash: str
    probe_count: int
    face_count: int
    minimum_area_ratio: float
    maximum_area_ratio: float
    maximum_condition_number: float
    minimum_edge_ratio: float
    maximum_edge_ratio: float
    failure_invariants: tuple[str, ...]
    passed: bool
    per_probe: tuple[Json, ...]
    report_hash: str
    schema_version: str = G3_STRESS_SCHEMA
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


def expected_g3_probe_plan_hash(
    *,
    skeleton_lineage_hash: str,
    axis_contract_hash: str,
    joint_ranges,
    camera_binding_hashes,
    allowed_attachment_state_hashes,
) -> str:
    return content_sha256({
        "schema": G3_PROBE_PLAN_SCHEMA,
        "skeleton_lineage_hash": str(skeleton_lineage_hash),
        "axis_contract_hash": str(axis_contract_hash),
        "joint_ranges": tuple(
            {
                "canonical_joint_id": row.canonical_joint_id,
                "min_rotation_deg": float(row.min_rotation_deg),
                "max_rotation_deg": float(row.max_rotation_deg),
                "translation_radius": float(row.translation_radius),
                "min_scale": float(row.min_scale),
                "max_scale": float(row.max_scale),
            }
            for row in sorted(joint_ranges, key=lambda x: x.canonical_joint_id)
        ),
        "camera_binding_hashes": tuple(map(str, camera_binding_hashes)),
        "allowed_attachment_state_hashes": tuple(map(str, allowed_attachment_state_hashes)),
        "probe_recipe": (
            "REST",
            "EACH_JOINT_MIN",
            "EACH_JOINT_MAX",
            "ALL_MIN",
            "ALL_MAX",
            "ALTERNATING_EXTREMES_A",
            "ALTERNATING_EXTREMES_B",
        ),
        "translation_scale_semantics": "V1_REQUIRES_DEFAULT_IDENTITY",
    })


def _probe_rotations(envelope: DeformationCapabilityEnvelopeIR) -> tuple[tuple[str, dict[str,float]], ...]:
    ordered = tuple(sorted(envelope.joint_ranges, key=lambda row: row.canonical_joint_id))
    rows: list[tuple[str, dict[str,float]]] = [("REST", {})]
    for row in ordered:
        if abs(float(row.min_rotation_deg)) > 1e-12:
            rows.append((f"JOINT_MIN:{row.canonical_joint_id}", {row.canonical_joint_id: float(row.min_rotation_deg)}))
        if abs(float(row.max_rotation_deg)) > 1e-12:
            rows.append((f"JOINT_MAX:{row.canonical_joint_id}", {row.canonical_joint_id: float(row.max_rotation_deg)}))
    if ordered:
        rows.append(("ALL_MIN", {row.canonical_joint_id: float(row.min_rotation_deg) for row in ordered}))
        rows.append(("ALL_MAX", {row.canonical_joint_id: float(row.max_rotation_deg) for row in ordered}))
        rows.append(("ALTERNATING_A", {
            row.canonical_joint_id: float(row.min_rotation_deg if i % 2 == 0 else row.max_rotation_deg)
            for i, row in enumerate(ordered)
        }))
        rows.append(("ALTERNATING_B", {
            row.canonical_joint_id: float(row.max_rotation_deg if i % 2 == 0 else row.min_rotation_deg)
            for i, row in enumerate(ordered)
        }))
    dedup = {}
    for probe_id, rotations in rows:
        key = tuple(sorted((jid, round(float(v), 12)) for jid, v in rotations.items() if abs(float(v)) > 1e-12))
        dedup.setdefault(key, (probe_id, {jid: v for jid, v in rotations.items() if abs(float(v)) > 1e-12}))
    return tuple(dedup.values())


def _candidate_skin_matrix(candidate, *, surface, skeleton, skin):
    if candidate.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("G3_CANDIDATE_SURFACE_LINEAGE_MISMATCH")
    if skin.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("G3_SKIN_SURFACE_LINEAGE_MISMATCH")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("G3_SKIN_SKELETON_LINEAGE_MISMATCH")

    joint_ids = tuple(j.canonical_joint_id for j in skeleton.joints)
    joint_index = {jid: i for i, jid in enumerate(joint_ids)}
    if len(joint_index) != len(joint_ids):
        raise QualificationError("G3_SKELETON_DUPLICATE_JOINT")
    source = {row.surface_id: row for row in skin.rows}
    if set(source) != {node.surface_id for node in surface.surface_nodes}:
        raise QualificationError("G3_SKIN_INCOMPLETE_SURFACE_ACCOUNTING")

    weights = np.zeros((len(candidate.vertices), len(joint_ids)), dtype=np.float64)
    positions = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)
    vertex_index = {v.candidate_vertex_id: i for i, v in enumerate(candidate.vertices)}
    if len(vertex_index) != len(candidate.vertices) or not np.isfinite(positions).all():
        raise QualificationError("G3_CANDIDATE_VERTEX_PAYLOAD_INVALID")

    for vi, vertex in enumerate(candidate.vertices):
        total_support = 0.0
        for sid, coeff in vertex.support_binding.coefficients:
            if sid not in source:
                raise QualificationError("G3_CANDIDATE_SUPPORT_NOT_IN_SKIN")
            c = float(coeff)
            if not math.isfinite(c) or c < 0.0:
                raise QualificationError("G3_CANDIDATE_SUPPORT_INVALID")
            total_support += c
            for jid, weight in source[sid].influences:
                if jid not in joint_index:
                    raise QualificationError("G3_SKIN_REFERENCES_UNKNOWN_JOINT")
                w = float(weight)
                if not math.isfinite(w) or w < 0.0:
                    raise QualificationError("G3_SKIN_WEIGHT_INVALID")
                weights[vi, joint_index[jid]] += c * w
        if abs(total_support - 1.0) > 1e-9:
            raise QualificationError("G3_CANDIDATE_SUPPORT_SIMPLEX_INVALID")
        row_sum = float(weights[vi].sum())
        if abs(row_sum - 1.0) > 1e-9:
            raise QualificationError("G3_TRANSFERRED_SKIN_SIMPLEX_INVALID")

    faces = []
    for face in candidate.faces:
        if len(face) != 3 or any(vid not in vertex_index for vid in face):
            raise QualificationError("G3_CANDIDATE_FACE_INVALID")
        faces.append(tuple(vertex_index[vid] for vid in face))
    if not faces:
        raise QualificationError("G3_CANDIDATE_FACE_SET_EMPTY")
    return positions, weights, tuple(faces)


def _triangle_metrics(rest: np.ndarray, posed: np.ndarray) -> tuple[float,float,float,float,float]:
    r1 = rest[1] - rest[0]
    r2 = rest[2] - rest[0]
    l1 = float(np.linalg.norm(r1))
    if l1 <= 1e-12:
        raise QualificationError("G3_REST_TRIANGLE_DEGENERATE")
    u = r1 / l1
    x2 = float(np.dot(r2, u))
    perp = r2 - x2 * u
    y2 = float(np.linalg.norm(perp))
    if y2 <= 1e-12:
        raise QualificationError("G3_REST_TRIANGLE_DEGENERATE")
    rest_basis = np.asarray([[l1, x2], [0.0, y2]], dtype=np.float64)
    p1 = posed[1] - posed[0]
    p2 = posed[2] - posed[0]
    posed_edges = np.column_stack((p1, p2))
    F = posed_edges @ np.linalg.inv(rest_basis)
    singular = np.linalg.svd(F, compute_uv=False)
    smax = float(max(singular))
    smin = float(min(singular))
    condition = float("inf") if smin <= 1e-15 else smax / smin
    area_ratio = smax * smin

    rest_edges = (
        float(np.linalg.norm(rest[1]-rest[0])),
        float(np.linalg.norm(rest[2]-rest[1])),
        float(np.linalg.norm(rest[0]-rest[2])),
    )
    posed_edges_len = (
        float(np.linalg.norm(posed[1]-posed[0])),
        float(np.linalg.norm(posed[2]-posed[1])),
        float(np.linalg.norm(posed[0]-posed[2])),
    )
    ratios = [b/a for a,b in zip(rest_edges, posed_edges_len) if a > 1e-12]
    return area_ratio, condition, min(ratios), max(ratios), smin


def run_g3_deformation_stress(
    candidate: CanonicalMeshCandidateIR,
    *,
    surface: RiggingSurfaceIR,
    skeleton: QualifiedSkeletonIR,
    skin: QualifiedSkinIR,
    envelope: DeformationCapabilityEnvelopeIR,
    axis_contract: Mapping,
    policy: MeshQualificationPolicyIR,
) -> G3DeformationStressReportIR:
    validate_mesh_qualification_policy(policy)
    known_joint_ids = {j.canonical_joint_id for j in skeleton.joints}
    validate_deformation_capability_envelope(envelope, known_joint_ids=known_joint_ids)
    if envelope.skeleton_lineage_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("G3_ENVELOPE_SKELETON_LINEAGE_MISMATCH")
    if any(
        abs(float(row.translation_radius)) > 1e-12
        or abs(float(row.min_scale) - 1.0) > 1e-12
        or abs(float(row.max_scale) - 1.0) > 1e-12
        for row in envelope.joint_ranges
    ):
        raise QualificationError("G3_TRANSLATION_SCALE_STRESS_NOT_IMPLEMENTED")

    specs, bindings, axis_hash = build_joint_specs_v1(skeleton, axis_contract)
    if axis_hash != envelope.axis_contract_hash:
        raise QualificationError("G3_AXIS_CONTRACT_HASH_MISMATCH")
    for row in envelope.joint_ranges:
        if (abs(float(row.min_rotation_deg)) > 1e-12 or abs(float(row.max_rotation_deg)) > 1e-12) and row.canonical_joint_id not in bindings:
            raise QualificationError("G3_ROTATING_JOINT_WITHOUT_AXIS")

    expected_plan = expected_g3_probe_plan_hash(
        skeleton_lineage_hash=envelope.skeleton_lineage_hash,
        axis_contract_hash=envelope.axis_contract_hash,
        joint_ranges=envelope.joint_ranges,
        camera_binding_hashes=envelope.camera_binding_hashes,
        allowed_attachment_state_hashes=envelope.allowed_attachment_state_hashes,
    )
    if envelope.probe_plan_hash != expected_plan:
        raise QualificationError("G3_PROBE_PLAN_HASH_MISMATCH")

    rest_positions, weights, faces = _candidate_skin_matrix(
        candidate, surface=surface, skeleton=skeleton, skin=skin
    )
    per_probe = []
    global_min_area = float("inf")
    global_max_area = 0.0
    global_max_condition = 0.0
    global_min_edge = float("inf")
    global_max_edge = 0.0
    failures = set()

    for probe_id, rotations in _probe_rotations(envelope):
        pose = solve_fk_v1(specs, rotations)
        posed = apply_lbs_matrix_v1(rest_positions, weights, pose.skin_matrices)
        min_area = float("inf")
        max_area = 0.0
        max_condition = 0.0
        min_edge = float("inf")
        max_edge = 0.0
        for face in faces:
            rest_tri = rest_positions[list(face)]
            posed_tri = posed[list(face)]
            area_ratio, condition, edge_min, edge_max, smin = _triangle_metrics(rest_tri, posed_tri)
            if not all(math.isfinite(x) for x in (area_ratio, condition, edge_min, edge_max, smin)):
                failures.add("NONFINITE_DEFORMATION_METRIC")
                continue
            min_area = min(min_area, area_ratio)
            max_area = max(max_area, area_ratio)
            max_condition = max(max_condition, condition)
            min_edge = min(min_edge, edge_min)
            max_edge = max(max_edge, edge_max)
            if area_ratio < policy.g3_min_dynamic_area_ratio:
                failures.add("DYNAMIC_AREA_RATIO_BELOW_MIN")
            if area_ratio > policy.g3_max_dynamic_area_ratio:
                failures.add("DYNAMIC_AREA_RATIO_ABOVE_MAX")
            if condition > policy.g3_max_dynamic_condition_number:
                failures.add("DYNAMIC_CONDITION_NUMBER_ABOVE_MAX")
        global_min_area = min(global_min_area, min_area)
        global_max_area = max(global_max_area, max_area)
        global_max_condition = max(global_max_condition, max_condition)
        global_min_edge = min(global_min_edge, min_edge)
        global_max_edge = max(global_max_edge, max_edge)
        per_probe.append({
            "probe_id": probe_id,
            "rotation_degrees_by_joint": dict(sorted(rotations.items())),
            "pose_hash": pose.pose_hash,
            "minimum_area_ratio": min_area,
            "maximum_area_ratio": max_area,
            "maximum_condition_number": max_condition,
            "minimum_edge_ratio": min_edge,
            "maximum_edge_ratio": max_edge,
        })

    passed = not failures
    value = G3DeformationStressReportIR(
        candidate_lineage_hash=candidate.candidate_lineage_hash,
        surface_lineage_hash=surface.geometry_lineage_hash,
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        envelope_lineage_hash=envelope.envelope_lineage_hash,
        axis_contract_hash=envelope.axis_contract_hash,
        qualification_policy_hash=policy.qualification_policy_lineage_hash,
        probe_plan_hash=envelope.probe_plan_hash,
        probe_count=len(per_probe),
        face_count=len(faces),
        minimum_area_ratio=float(global_min_area),
        maximum_area_ratio=float(global_max_area),
        maximum_condition_number=float(global_max_condition),
        minimum_edge_ratio=float(global_min_edge),
        maximum_edge_ratio=float(global_max_edge),
        failure_invariants=tuple(sorted(failures)),
        passed=passed,
        per_probe=tuple(per_probe),
        report_hash="",
        metadata={
            "measurement_role": "STAGE27_NUMERICAL_CONDITIONING_ONLY",
            "mesh_skin_product_authority_minted": False,
            "translation_scale_probe_support": False,
        },
    )
    payload=value.to_dict()
    payload.pop("report_hash",None)
    return replace(value, report_hash=content_sha256(payload))
