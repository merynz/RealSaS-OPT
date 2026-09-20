from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .hashing import content_sha256
from .mesh.conditioning_v1 import triangle_rest_metric
from .types import QualificationError, RiggingSurfaceIR, SurfaceSupportBinding, Vec3

Json = dict[str, Any]
BOUNDARY_DECISIONS = {"SEPARATE", "PRESERVE_CONTINUITY", "UNKNOWN"}
MECHANICAL_CLASSES = {"RIGID", "DEFORMABLE"}
PRODUCT_GEOMETRY_CARRIER_CLASSES = {"MESH", "PLANAR"}
PRESENTATION_CARRIER_CLASSES = {"MESH", "PLANAR", "CLIP"}
PRESENTATION_EVIDENCE_CLASSES = {
    "MECHANICAL", "OBSERVATION_REST", "POSED_GEOMETRY",
    "MOTION_OVERRIDE", "SOURCE_APPEARANCE",
}
KEYABLE_CHANNELS = {"ATTACHMENT", "TINT", "CLIPPING", "ORDER", "VISIBILITY"}

# Frozen subject-free numerical conditioning floor from
# ANIMATION_GRADE_MESH_CONDITIONING_CALIBRATION_PREREG_20260918.md.
# A later visual/raster study may make these stricter, never weaker.
G3_NUMERICAL_MIN_ANGLE_DEG = 7.5
G3_NUMERICAL_MAX_ASPECT = 16.0
_SUPPORT_SIMPLEX_TOL = 1e-9
_POSITION_REL_TOL = 1e-9
_POSITION_ABS_TOL = 1e-12


@dataclass(frozen=True)
class ComponentRegionIR:
    component_id: str
    surface_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class ComponentBoundaryConstraintIR:
    constraint_id: str
    a_surface_id: str
    b_surface_id: str
    decision: str
    evidence_refs: tuple[str, ...]
    confidence: float = 1.0
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.ComponentBoundaryConstraintIR.v1"
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class MechanicalPartitionIR:
    components: tuple[ComponentRegionIR, ...]
    boundary_constraints: tuple[ComponentBoundaryConstraintIR, ...]
    surface_lineage_hash: str
    partition_lineage_hash: str
    schema_version: str = "RealSaS.MechanicalPartitionIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class ComponentCarrierDecisionIR:
    component_id: str
    carrier_class: str
    evidence_refs: tuple[str, ...]
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class ComponentCarrierPolicyIR:
    partition_binding_hash: str
    decisions: tuple[ComponentCarrierDecisionIR, ...]
    carrier_policy_lineage_hash: str
    schema_version: str = "RealSaS.ComponentCarrierPolicyIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class CarrierCoverageThresholdIR:
    carrier_class: str
    min_recall: float
    min_precision: float
    max_largest_coherent_hole_fraction: float
    max_interior_uncovered_fraction: float
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class MeshQualificationPolicyIR:
    g1_max_normal_refinement_ratio: float
    g1_max_tangential_to_normal_ratio: float
    g3_min_angle_deg: float
    g3_max_aspect_longest_over_min_altitude: float
    coverage_thresholds: tuple[CarrierCoverageThresholdIR, ...]
    qualification_policy_lineage_hash: str
    g3_min_dynamic_area_ratio: float = 0.05
    g3_max_dynamic_area_ratio: float = 20.0
    g3_max_dynamic_condition_number: float = 16.0
    schema_version: str = "RealSaS.MeshQualificationPolicyIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class JointCapabilityRangeIR:
    canonical_joint_id: str
    min_rotation_deg: float
    max_rotation_deg: float
    translation_radius: float = 0.0
    min_scale: float = 1.0
    max_scale: float = 1.0
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class DeformationCapabilityEnvelopeIR:
    skeleton_lineage_hash: str
    joint_ranges: tuple[JointCapabilityRangeIR, ...]
    camera_binding_hashes: tuple[str, ...]
    allowed_attachment_state_hashes: tuple[str, ...]
    axis_contract_hash: str
    probe_plan_hash: str
    envelope_lineage_hash: str
    schema_version: str = "RealSaS.DeformationCapabilityEnvelopeIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class GeometricRefinementIR:
    dense_lineage_hash: str
    base_position: Vec3
    refined_position: Vec3
    local_scale: float
    normal_component: float
    tangential_component: float
    method: str
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class CanonicalMeshVertexCandidateIR:
    candidate_vertex_id: str
    support_binding: SurfaceSupportBinding
    component_id: str
    P: Vec3
    refinement: GeometricRefinementIR | None = None
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class CanonicalMeshCandidateIR:
    vertices: tuple[CanonicalMeshVertexCandidateIR, ...]
    faces: tuple[tuple[str, str, str], ...]
    edges: tuple[tuple[str, str], ...]
    surface_binding_hash: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    producer_id: str
    producer_policy_hash: str
    candidate_lineage_hash: str
    schema_version: str = "RealSaS.CanonicalMeshCandidateIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedMeshVertexIR:
    canonical_mesh_vertex_id: str
    support_binding: SurfaceSupportBinding
    component_id: str
    P: Vec3
    source_candidate_vertex_id: str
    refinement: GeometricRefinementIR | None = None
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedMeshIR:
    vertices: tuple[QualifiedMeshVertexIR, ...]
    faces: tuple[tuple[str, str, str], ...]
    edges: tuple[tuple[str, str], ...]
    surface_binding_hash: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    envelope_binding_hash: str
    qualification_policy_hash: str
    qualification_report: Json
    mesh_lineage_hash: str
    schema_version: str = "RealSaS.QualifiedMeshIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class PresentationDecisionEvidenceIR:
    decision_id: str
    decision_kind: str
    authority_class: str
    evidence_refs: tuple[str, ...]
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class PresentationSlotIR:
    slot_id: str
    bone_id: str
    setup_order: int
    default_attachment_id: str | None
    keyable_channels: tuple[str, ...]
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class PresentationAttachmentIR:
    attachment_id: str
    slot_id: str
    mechanical_component_ids: tuple[str, ...]
    mechanical_class: str
    carrier_class: str
    carrier_binding_hash: str
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class PresentationViewOverlayIR:
    view_index: int
    camera_binding_hash: str
    appearance_binding_hash: str
    composition_binding_hash: str
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedPresentationGraphIR:
    slots: tuple[PresentationSlotIR, ...]
    attachments: tuple[PresentationAttachmentIR, ...]
    view_overlays: tuple[PresentationViewOverlayIR, ...]
    decisions: tuple[PresentationDecisionEvidenceIR, ...]
    skeleton_binding_hash: str
    mesh_binding_hash: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    product_state_binding_hash: str
    presentation_structure_binding_hash: str
    appearance_set_binding_hash: str
    composition_set_binding_hash: str
    qualification_report: Json
    presentation_lineage_hash: str
    schema_version: str = "RealSaS.QualifiedPresentationGraphIR.v1"
    metadata: Json = field(default_factory=dict)
    def to_dict(self): return asdict(self)


def _hash_without(value, field_name: str) -> str:
    payload = value.to_dict()
    payload.pop(field_name, None)
    return content_sha256(payload)


def mechanical_partition_lineage_hash(value): return _hash_without(value, "partition_lineage_hash")
def component_carrier_policy_lineage_hash(value): return _hash_without(value, "carrier_policy_lineage_hash")
def mesh_qualification_policy_lineage_hash(value): return _hash_without(value, "qualification_policy_lineage_hash")
def deformation_envelope_lineage_hash(value): return _hash_without(value, "envelope_lineage_hash")
def canonical_mesh_candidate_lineage_hash(value): return _hash_without(value, "candidate_lineage_hash")
def qualified_mesh_lineage_hash(value): return _hash_without(value, "mesh_lineage_hash")
def qualified_presentation_lineage_hash(value): return _hash_without(value, "presentation_lineage_hash")


def validate_mechanical_partition(value: MechanicalPartitionIR, surface: RiggingSurfaceIR) -> None:
    if value.surface_lineage_hash != surface.geometry_lineage_hash:
        raise QualificationError("PARTITION_SURFACE_LINEAGE_MISMATCH")
    known = {n.surface_id for n in surface.surface_nodes}
    owner = {}
    ids = set()
    for component in value.components:
        if not component.component_id or component.component_id in ids or not component.surface_ids:
            raise QualificationError("PARTITION_COMPONENT_INVALID")
        ids.add(component.component_id)
        for sid in component.surface_ids:
            if sid not in known: raise QualificationError("PARTITION_UNKNOWN_SURFACE")
            if sid in owner: raise QualificationError("PARTITION_OVERLAPPING_SURFACE")
            owner[sid] = component.component_id
    if set(owner) != known:
        raise QualificationError("PARTITION_INCOMPLETE_SURFACE_ACCOUNTING")
    constraint_ids = set()
    for row in value.boundary_constraints:
        if not row.constraint_id or row.constraint_id in constraint_ids:
            raise QualificationError("PARTITION_CONSTRAINT_ID_INVALID")
        constraint_ids.add(row.constraint_id)
        if row.a_surface_id not in known or row.b_surface_id not in known or row.a_surface_id == row.b_surface_id:
            raise QualificationError("PARTITION_CONSTRAINT_SURFACE_INVALID")
        if row.decision not in BOUNDARY_DECISIONS:
            raise QualificationError("PARTITION_BOUNDARY_DECISION_INVALID")
        if not math.isfinite(row.confidence) or not 0.0 <= row.confidence <= 1.0:
            raise QualificationError("PARTITION_CONFIDENCE_INVALID")
        same = owner[row.a_surface_id] == owner[row.b_surface_id]
        if row.decision == "SEPARATE" and same:
            raise QualificationError("PARTITION_SEPARATE_CONSTRAINT_VIOLATED")
        if row.decision == "PRESERVE_CONTINUITY" and not same:
            raise QualificationError("PARTITION_CONTINUITY_CONSTRAINT_VIOLATED")
    if value.partition_lineage_hash != mechanical_partition_lineage_hash(value):
        raise QualificationError("PARTITION_LINEAGE_HASH_MISMATCH")


def build_mechanical_partition(*, surface, components, boundary_constraints, metadata=None):
    value = MechanicalPartitionIR(
        tuple(components), tuple(boundary_constraints), surface.geometry_lineage_hash, "",
        metadata=dict(metadata or {}),
    )
    value = replace(value, partition_lineage_hash=mechanical_partition_lineage_hash(value))
    validate_mechanical_partition(value, surface)
    return value


def validate_component_carrier_policy(value: ComponentCarrierPolicyIR, partition: MechanicalPartitionIR) -> None:
    if value.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("CARRIER_POLICY_PARTITION_LINEAGE_MISMATCH")
    component_ids = {component.component_id for component in partition.components}
    seen = set()
    for row in value.decisions:
        if row.component_id not in component_ids or row.component_id in seen:
            raise QualificationError("CARRIER_POLICY_COMPONENT_INVALID")
        seen.add(row.component_id)
        if row.carrier_class not in PRODUCT_GEOMETRY_CARRIER_CLASSES or not row.evidence_refs:
            raise QualificationError("CARRIER_POLICY_DECISION_INVALID")
    if seen != component_ids:
        raise QualificationError("CARRIER_POLICY_INCOMPLETE_COMPONENT_ACCOUNTING")
    if value.carrier_policy_lineage_hash != component_carrier_policy_lineage_hash(value):
        raise QualificationError("CARRIER_POLICY_LINEAGE_HASH_MISMATCH")


def build_component_carrier_policy(*, partition, decisions, metadata=None) -> ComponentCarrierPolicyIR:
    value = ComponentCarrierPolicyIR(
        partition.partition_lineage_hash,
        tuple(decisions),
        "",
        metadata=dict(metadata or {}),
    )
    value = replace(value, carrier_policy_lineage_hash=component_carrier_policy_lineage_hash(value))
    validate_component_carrier_policy(value, partition)
    return value


def validate_mesh_qualification_policy(value: MeshQualificationPolicyIR) -> None:
    scalars = (
        value.g1_max_normal_refinement_ratio,
        value.g1_max_tangential_to_normal_ratio,
        value.g3_min_angle_deg,
        value.g3_max_aspect_longest_over_min_altitude,
        value.g3_min_dynamic_area_ratio,
        value.g3_max_dynamic_area_ratio,
        value.g3_max_dynamic_condition_number,
    )
    if any(not math.isfinite(float(x)) for x in scalars):
        raise QualificationError("MESH_QUALIFICATION_POLICY_NONFINITE")
    if value.g1_max_normal_refinement_ratio < 0.0 or value.g1_max_tangential_to_normal_ratio < 0.0:
        raise QualificationError("MESH_QUALIFICATION_POLICY_G1_INVALID")
    if value.g3_min_angle_deg + 1e-9 < G3_NUMERICAL_MIN_ANGLE_DEG:
        raise QualificationError("MESH_QUALIFICATION_POLICY_WEAKER_THAN_G3_ANGLE_FLOOR")
    if value.g3_max_aspect_longest_over_min_altitude - 1e-9 > G3_NUMERICAL_MAX_ASPECT:
        raise QualificationError("MESH_QUALIFICATION_POLICY_WEAKER_THAN_G3_ASPECT_FLOOR")
    if value.g3_max_aspect_longest_over_min_altitude <= 0.0:
        raise QualificationError("MESH_QUALIFICATION_POLICY_G3_INVALID")
    if value.g3_min_dynamic_area_ratio <= 0.0 or value.g3_min_dynamic_area_ratio > 1.0:
        raise QualificationError("MESH_QUALIFICATION_POLICY_G3_DYNAMIC_AREA_MIN_INVALID")
    if value.g3_max_dynamic_area_ratio < 1.0 or value.g3_max_dynamic_area_ratio < value.g3_min_dynamic_area_ratio:
        raise QualificationError("MESH_QUALIFICATION_POLICY_G3_DYNAMIC_AREA_MAX_INVALID")
    if value.g3_max_dynamic_condition_number < 1.0:
        raise QualificationError("MESH_QUALIFICATION_POLICY_G3_DYNAMIC_CONDITION_INVALID")
    thresholds = {}
    for row in value.coverage_thresholds:
        if row.carrier_class not in PRODUCT_GEOMETRY_CARRIER_CLASSES or row.carrier_class in thresholds:
            raise QualificationError("MESH_QUALIFICATION_POLICY_G5_CARRIER_INVALID")
        vals = (
            row.min_recall,
            row.min_precision,
            row.max_largest_coherent_hole_fraction,
            row.max_interior_uncovered_fraction,
        )
        if any(not math.isfinite(float(x)) or float(x) < 0.0 or float(x) > 1.0 for x in vals):
            raise QualificationError("MESH_QUALIFICATION_POLICY_G5_THRESHOLD_INVALID")
        thresholds[row.carrier_class] = row
    if set(thresholds) != PRODUCT_GEOMETRY_CARRIER_CLASSES:
        raise QualificationError("MESH_QUALIFICATION_POLICY_REQUIRES_ALL_CARRIER_CLASSES")
    if value.qualification_policy_lineage_hash != mesh_qualification_policy_lineage_hash(value):
        raise QualificationError("MESH_QUALIFICATION_POLICY_LINEAGE_HASH_MISMATCH")


def build_mesh_qualification_policy(*, g1_max_normal_refinement_ratio, g1_max_tangential_to_normal_ratio, g3_min_angle_deg, g3_max_aspect_longest_over_min_altitude, coverage_thresholds, g3_min_dynamic_area_ratio=0.05, g3_max_dynamic_area_ratio=20.0, g3_max_dynamic_condition_number=16.0, metadata=None) -> MeshQualificationPolicyIR:
    value = MeshQualificationPolicyIR(
        float(g1_max_normal_refinement_ratio),
        float(g1_max_tangential_to_normal_ratio),
        float(g3_min_angle_deg),
        float(g3_max_aspect_longest_over_min_altitude),
        tuple(coverage_thresholds),
        "",
        float(g3_min_dynamic_area_ratio),
        float(g3_max_dynamic_area_ratio),
        float(g3_max_dynamic_condition_number),
        metadata=dict(metadata or {}),
    )
    value = replace(value, qualification_policy_lineage_hash=mesh_qualification_policy_lineage_hash(value))
    validate_mesh_qualification_policy(value)
    return value


def validate_deformation_capability_envelope(value: DeformationCapabilityEnvelopeIR, *, known_joint_ids=None) -> None:
    if not value.skeleton_lineage_hash or not value.axis_contract_hash or not value.probe_plan_hash:
        raise QualificationError("DEFORMATION_ENVELOPE_AUTHORITY_MISSING")
    if len(value.camera_binding_hashes) != 8 or len(set(value.camera_binding_hashes)) != 8:
        raise QualificationError("DEFORMATION_ENVELOPE_REQUIRES_EXACT_8_CAMERAS")
    seen = set()
    for row in value.joint_ranges:
        if not row.canonical_joint_id or row.canonical_joint_id in seen:
            raise QualificationError("DEFORMATION_ENVELOPE_JOINT_INVALID")
        seen.add(row.canonical_joint_id)
        if known_joint_ids is not None and row.canonical_joint_id not in known_joint_ids:
            raise QualificationError("DEFORMATION_ENVELOPE_UNKNOWN_JOINT")
        vals = (row.min_rotation_deg, row.max_rotation_deg, row.translation_radius, row.min_scale, row.max_scale)
        if any(not math.isfinite(float(v)) for v in vals):
            raise QualificationError("DEFORMATION_ENVELOPE_NONFINITE")
        if row.min_rotation_deg > row.max_rotation_deg or row.translation_radius < 0 or row.min_scale <= 0 or row.min_scale > row.max_scale:
            raise QualificationError("DEFORMATION_ENVELOPE_RANGE_INVALID")
    if known_joint_ids is not None and seen != set(known_joint_ids):
        raise QualificationError("DEFORMATION_ENVELOPE_INCOMPLETE_JOINT_ACCOUNTING")
    if value.envelope_lineage_hash != deformation_envelope_lineage_hash(value):
        raise QualificationError("DEFORMATION_ENVELOPE_LINEAGE_HASH_MISMATCH")


def _vec_finite(p: Vec3) -> bool:
    return len(p) == 3 and all(math.isfinite(float(x)) for x in p)


def _vec_close(a: Vec3, b: Vec3) -> bool:
    return all(math.isclose(float(x), float(y), rel_tol=_POSITION_REL_TOL, abs_tol=_POSITION_ABS_TOL) for x, y in zip(a, b))


def _support_position(surface_nodes: dict[str, Any], binding: SurfaceSupportBinding) -> Vec3:
    if binding.mode not in {"IDENTITY_SURFACE_NODE", "LOCAL_CONVEX_INTERPOLATION"} or not binding.coefficients:
        raise QualificationError("QUALIFIED_MESH_G1_SUPPORT_MODE_INVALID")
    seen = set()
    total = 0.0
    xyz = [0.0, 0.0, 0.0]
    for sid, coeff in binding.coefficients:
        if sid in seen or sid not in surface_nodes:
            raise QualificationError("QUALIFIED_MESH_G1_SUPPORT_ID_INVALID")
        seen.add(sid)
        c = float(coeff)
        if not math.isfinite(c) or c < 0.0:
            raise QualificationError("QUALIFIED_MESH_G1_SUPPORT_COEFFICIENT_INVALID")
        total += c
        p = surface_nodes[sid].P
        for axis in range(3):
            xyz[axis] += c * float(p[axis])
    if abs(total - 1.0) > _SUPPORT_SIMPLEX_TOL:
        raise QualificationError("QUALIFIED_MESH_G1_SUPPORT_SIMPLEX_INVALID")
    if binding.mode == "IDENTITY_SURFACE_NODE":
        if len(binding.coefficients) != 1 or abs(float(binding.coefficients[0][1]) - 1.0) > _SUPPORT_SIMPLEX_TOL:
            raise QualificationError("QUALIFIED_MESH_G1_IDENTITY_SUPPORT_INVALID")
    return (xyz[0], xyz[1], xyz[2])


def _edge_key(a: str, b: str) -> tuple[str, str]:
    if not a or not b or a == b:
        raise QualificationError("QUALIFIED_MESH_G2_EDGE_INVALID")
    return (a, b) if a < b else (b, a)


def _mesh_connected_labels(vertex_ids: set[str], edges: set[tuple[str, str]]) -> dict[str, str]:
    adjacency = {vid: set() for vid in vertex_ids}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    labels: dict[str, str] = {}
    for seed in sorted(vertex_ids):
        if seed in labels:
            continue
        label = seed
        stack = [seed]
        while stack:
            current = stack.pop()
            if current in labels:
                continue
            labels[current] = label
            stack.extend(sorted(adjacency[current] - labels.keys()))
    return labels


def _geometric_topology_crack_audit(
    vertex_by_id: dict[str, Any],
    faces: tuple[tuple[str, str, str], ...] | list,
) -> Json:
    """Detect undeclared same-component geometric discontinuities.

    Global manifoldness is intentionally not required. This audit targets two
    crack-producing cases that topology-by-ID alone cannot see:
    coincident duplicate vertices and vertex-on-edge T-junctions.
    """
    if not vertex_by_id:
        raise QualificationError("QUALIFIED_MESH_G2_VERTEX_SET_EMPTY")
    xyz_all = np.asarray([vertex.P for vertex in vertex_by_id.values()], dtype=np.float64)
    lo = np.min(xyz_all, axis=0)
    hi = np.max(xyz_all, axis=0)
    scale = max(float(np.linalg.norm(hi - lo)), 1.0)
    tolerance = max(scale * 1.0e-9, 1.0e-12)

    face_incidence: dict[tuple[str, str], int] = {}
    for face in faces:
        for a, b in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            key = _edge_key(str(a), str(b))
            face_incidence[key] = face_incidence.get(key, 0) + 1

    coincident_pairs = []
    t_junctions = []
    component_ids = sorted(
        {str(vertex.component_id) for vertex in vertex_by_id.values()}
    )
    for component_id in component_ids:
        ids = sorted(
            vid
            for vid, vertex in vertex_by_id.items()
            if str(vertex.component_id) == component_id
        )
        if len(ids) < 2:
            continue
        points = np.asarray(
            [vertex_by_id[vid].P for vid in ids],
            dtype=np.float64,
        )
        tree = cKDTree(points)
        for a_local, b_local in sorted(tree.query_pairs(r=tolerance)):
            coincident_pairs.append((ids[int(a_local)], ids[int(b_local)]))

        index_by_id = {vid: index for index, vid in enumerate(ids)}
        component_edges = [
            edge
            for edge in face_incidence
            if edge[0] in index_by_id and edge[1] in index_by_id
        ]
        for a_id, b_id in component_edges:
            a = points[index_by_id[a_id]]
            b = points[index_by_id[b_id]]
            ab = b - a
            length2 = float(np.dot(ab, ab))
            if length2 <= tolerance * tolerance:
                continue
            length = math.sqrt(length2)
            midpoint = 0.5 * (a + b)
            candidates = tree.query_ball_point(
                midpoint,
                r=0.5 * length + tolerance,
            )
            for local_index in candidates:
                vid = ids[int(local_index)]
                if vid == a_id or vid == b_id:
                    continue
                p = points[int(local_index)]
                t = float(np.dot(p - a, ab) / length2)
                if t <= 1.0e-9 or t >= 1.0 - 1.0e-9:
                    continue
                closest = a + t * ab
                distance = float(np.linalg.norm(p - closest))
                if distance <= tolerance:
                    t_junctions.append(
                        {
                            "vertex_id": vid,
                            "edge": [a_id, b_id],
                            "edge_parameter": t,
                            "distance": distance,
                        }
                    )

    if coincident_pairs:
        raise QualificationError(
            "QUALIFIED_MESH_G2_COINCIDENT_DUPLICATE_VERTEX"
        )
    if t_junctions:
        raise QualificationError("QUALIFIED_MESH_G2_T_JUNCTION")

    return {
        "geometric_tolerance": tolerance,
        "coincident_duplicate_vertex_pair_count": 0,
        "t_junction_count": 0,
        "boundary_edge_count": int(
            sum(1 for count in face_incidence.values() if count == 1)
        ),
        "nonmanifold_edge_count_diagnostic": int(
            sum(1 for count in face_incidence.values() if count > 2)
        ),
        "global_two_manifold_required": False,
    }


def qualified_mesh_intrinsic_audit(value: QualifiedMeshIR, *, surface, partition) -> Json:
    """Recompute G1/G2/G4 invariants from the exact product mesh.

    This is deliberately independent of candidate-reported PASS flags. G3 dynamic
    stress and G5 raster measurements remain separate evidence, but rest
    conditioning and matrix structure are checked again here.
    """
    surface_nodes = {node.surface_id: node for node in surface.surface_nodes}
    owner = {}
    for component in partition.components:
        for sid in component.surface_ids:
            owner[sid] = component.component_id
    component_ids = {component.component_id for component in partition.components}

    if len(value.vertices) < 3 or not value.faces:
        raise QualificationError("QUALIFIED_MESH_EMPTY_PRODUCT_GEOMETRY")

    vertex_by_id = {}
    support_ids_by_vertex: dict[str, set[str]] = {}
    component_vertex_ids = {cid: set() for cid in component_ids}
    refinement_count = 0
    max_refinement_normal_ratio = 0.0
    max_refinement_tangent_to_normal = 0.0

    for vertex in value.vertices:
        vid = vertex.canonical_mesh_vertex_id
        if not vid or vid in vertex_by_id or not _vec_finite(vertex.P):
            raise QualificationError("QUALIFIED_MESH_G1_VERTEX_INVALID")
        if vertex.component_id not in component_ids:
            raise QualificationError("QUALIFIED_MESH_G1_COMPONENT_INVALID")
        base = _support_position(surface_nodes, vertex.support_binding)
        sids = {sid for sid, _ in vertex.support_binding.coefficients}
        if any(owner.get(sid) != vertex.component_id for sid in sids):
            raise QualificationError("QUALIFIED_MESH_G1_CROSS_COMPONENT_SUPPORT")
        support_ids_by_vertex[vid] = sids
        component_vertex_ids[vertex.component_id].add(vid)

        if vertex.refinement is None:
            if not _vec_close(vertex.P, base):
                raise QualificationError("QUALIFIED_MESH_G1_UNDECLARED_GEOMETRIC_REFINEMENT")
        else:
            refinement_count += 1
            r = vertex.refinement
            if not r.dense_lineage_hash or not r.method:
                raise QualificationError("QUALIFIED_MESH_G1_REFINEMENT_LINEAGE_MISSING")
            if not _vec_finite(r.base_position) or not _vec_finite(r.refined_position):
                raise QualificationError("QUALIFIED_MESH_G1_REFINEMENT_POSITION_INVALID")
            if not _vec_close(r.base_position, base) or not _vec_close(r.refined_position, vertex.P):
                raise QualificationError("QUALIFIED_MESH_G1_REFINEMENT_BINDING_MISMATCH")
            if not math.isfinite(float(r.local_scale)) or float(r.local_scale) <= 0.0:
                raise QualificationError("QUALIFIED_MESH_G1_REFINEMENT_LOCAL_SCALE_INVALID")
            normal = float(r.normal_component)
            tangent = float(r.tangential_component)
            if not math.isfinite(normal) or not math.isfinite(tangent) or tangent < 0.0:
                raise QualificationError("QUALIFIED_MESH_G1_REFINEMENT_COMPONENT_INVALID")
            displacement = math.dist(tuple(map(float, base)), tuple(map(float, vertex.P)))
            declared = math.hypot(normal, tangent)
            if not math.isclose(displacement, declared, rel_tol=1e-6, abs_tol=float(r.local_scale) * 1e-9):
                raise QualificationError("QUALIFIED_MESH_G1_REFINEMENT_DECOMPOSITION_MISMATCH")
            max_refinement_normal_ratio = max(max_refinement_normal_ratio, abs(normal) / float(r.local_scale))
            if abs(normal) > 1e-15:
                max_refinement_tangent_to_normal = max(max_refinement_tangent_to_normal, tangent / abs(normal))
            elif tangent > 1e-15:
                max_refinement_tangent_to_normal = float("inf")
        vertex_by_id[vid] = vertex

    face_keys = set()
    derived_edges: set[tuple[str, str]] = set()
    face_count_by_component = {cid: 0 for cid in component_ids}
    min_angle = float("inf")
    max_aspect = 0.0
    for face in value.faces:
        if len(face) != 3 or len(set(face)) != 3 or any(vid not in vertex_by_id for vid in face):
            raise QualificationError("QUALIFIED_MESH_G2_FACE_INVALID")
        key = tuple(sorted(face))
        if key in face_keys:
            raise QualificationError("QUALIFIED_MESH_G2_DUPLICATE_FACE")
        face_keys.add(key)
        components = {vertex_by_id[vid].component_id for vid in face}
        if len(components) != 1:
            raise QualificationError("QUALIFIED_MESH_G4_FACE_CROSSES_COMPONENT_BOUNDARY")
        component_id = next(iter(components))
        face_count_by_component[component_id] += 1
        points = tuple(vertex_by_id[vid].P for vid in face)
        metric = triangle_rest_metric(points)
        if metric["degenerate"]:
            raise QualificationError("QUALIFIED_MESH_G2_DEGENERATE_FACE")
        min_angle = min(min_angle, float(metric["min_angle_deg"]))
        max_aspect = max(max_aspect, float(metric["aspect_longest_over_min_altitude"]))
        derived_edges.update((_edge_key(face[0], face[1]), _edge_key(face[1], face[2]), _edge_key(face[2], face[0])))

    if any(count <= 0 for count in face_count_by_component.values()):
        raise QualificationError("QUALIFIED_MESH_G4_COMPONENT_WITHOUT_PRODUCT_FACE")

    declared_edges = set()
    for edge in value.edges:
        if len(edge) != 2 or edge[0] not in vertex_by_id or edge[1] not in vertex_by_id:
            raise QualificationError("QUALIFIED_MESH_G2_EDGE_INVALID")
        key = _edge_key(edge[0], edge[1])
        if key in declared_edges:
            raise QualificationError("QUALIFIED_MESH_G2_DUPLICATE_EDGE")
        declared_edges.add(key)
    if declared_edges != derived_edges:
        raise QualificationError("QUALIFIED_MESH_G2_EDGE_FACE_TOPOLOGY_MISMATCH")

    geometric_topology = _geometric_topology_crack_audit(
        vertex_by_id,
        value.faces,
    )

    if min_angle + 1e-9 < G3_NUMERICAL_MIN_ANGLE_DEG:
        raise QualificationError("QUALIFIED_MESH_G3_NUMERICAL_MIN_ANGLE_FAIL")
    if max_aspect - 1e-9 > G3_NUMERICAL_MAX_ASPECT:
        raise QualificationError("QUALIFIED_MESH_G3_NUMERICAL_ASPECT_FAIL")

    labels = _mesh_connected_labels(set(vertex_by_id), declared_edges)
    preserve_checked = 0
    for constraint in partition.boundary_constraints:
        if constraint.decision != "PRESERVE_CONTINUITY":
            continue
        left = [vid for vid, sids in support_ids_by_vertex.items() if constraint.a_surface_id in sids]
        right = [vid for vid, sids in support_ids_by_vertex.items() if constraint.b_surface_id in sids]
        if not left or not right:
            raise QualificationError("QUALIFIED_MESH_G4_PRESERVE_SUPPORT_NOT_REPRESENTED")
        if not any(labels[a] == labels[b] for a in left for b in right):
            raise QualificationError("QUALIFIED_MESH_G4_PRESERVE_CONTINUITY_BROKEN")
        preserve_checked += 1

    return {
        "vertex_count": len(value.vertices),
        "face_count": len(value.faces),
        "edge_count": len(declared_edges),
        "component_count": len(component_ids),
        "component_face_counts": {k: face_count_by_component[k] for k in sorted(face_count_by_component)},
        "refinement_vertex_count": refinement_count,
        "max_refinement_normal_ratio": max_refinement_normal_ratio,
        "max_refinement_tangent_to_normal": max_refinement_tangent_to_normal,
        "rest_min_angle_deg": min_angle,
        "rest_max_aspect_longest_over_min_altitude": max_aspect,
        "g3_numerical_min_angle_floor_deg": G3_NUMERICAL_MIN_ANGLE_DEG,
        "g3_numerical_max_aspect_ceiling": G3_NUMERICAL_MAX_ASPECT,
        "g2_geometric_topology": geometric_topology,
        "preserve_continuity_constraints_checked": preserve_checked,
    }


def _validate_g5_matrix(report: Json, *, component_ids: set[str], carrier_policy: ComponentCarrierPolicyIR, policy: MeshQualificationPolicyIR) -> None:
    rows = tuple(report.get("view_component_coverage") or ())
    expected = {(view, component_id) for view in range(8) for component_id in component_ids}
    actual = set()
    carrier_by_component = {row.component_id: row.carrier_class for row in carrier_policy.decisions}
    thresholds = {row.carrier_class: row for row in policy.coverage_thresholds}
    for row in rows:
        try:
            view = int(row["view_index"])
            component_id = str(row["component_id"])
            carrier = str(row["carrier_class"])
        except Exception as exc:
            raise QualificationError("QUALIFIED_MESH_G5_MATRIX_ROW_INVALID") from exc
        key = (view, component_id)
        if key in actual or key not in expected or carrier not in PRODUCT_GEOMETRY_CARRIER_CLASSES:
            raise QualificationError("QUALIFIED_MESH_G5_MATRIX_ROW_INVALID")
        actual.add(key)
        if carrier_by_component.get(component_id) != carrier:
            raise QualificationError("QUALIFIED_MESH_G5_CARRIER_CLASS_DRIFT")
        recall = float(row.get("recall", float("nan")))
        precision = float(row.get("precision", float("nan")))
        hole = float(row.get("largest_coherent_hole_fraction", float("nan")))
        interior = float(row.get("interior_uncovered_fraction", float("nan")))
        if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in (recall, precision, hole, interior)):
            raise QualificationError("QUALIFIED_MESH_G5_METRIC_INVALID")
        threshold = thresholds[carrier]
        if recall + 1e-12 < threshold.min_recall:
            raise QualificationError("QUALIFIED_MESH_G5_RECALL_FAIL")
        if precision + 1e-12 < threshold.min_precision:
            raise QualificationError("QUALIFIED_MESH_G5_PRECISION_FAIL")
        if hole - 1e-12 > threshold.max_largest_coherent_hole_fraction:
            raise QualificationError("QUALIFIED_MESH_G5_COHERENT_HOLE_FAIL")
        if interior - 1e-12 > threshold.max_interior_uncovered_fraction:
            raise QualificationError("QUALIFIED_MESH_G5_INTERIOR_UNCOVERED_FAIL")
        if row.get("status") != "PASS":
            raise QualificationError("QUALIFIED_MESH_G5_CELL_NOT_PASS")
    if actual != expected:
        raise QualificationError("QUALIFIED_MESH_G5_MATRIX_INCOMPLETE")
    if report.get("carrier_policy_hash") != carrier_policy.carrier_policy_lineage_hash:
        raise QualificationError("QUALIFIED_MESH_G5_CARRIER_POLICY_BINDING_MISMATCH")


def validate_qualified_mesh(value: QualifiedMeshIR, *, surface, partition, carrier_policy, envelope, policy) -> None:
    validate_mechanical_partition(partition, surface)
    validate_component_carrier_policy(carrier_policy, partition)
    validate_deformation_capability_envelope(envelope)
    validate_mesh_qualification_policy(policy)
    if value.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("QUALIFIED_MESH_SURFACE_LINEAGE_MISMATCH")
    if value.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("QUALIFIED_MESH_PARTITION_LINEAGE_MISMATCH")
    if value.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
        raise QualificationError("QUALIFIED_MESH_CARRIER_POLICY_LINEAGE_MISMATCH")
    if value.envelope_binding_hash != envelope.envelope_lineage_hash:
        raise QualificationError("QUALIFIED_MESH_ENVELOPE_LINEAGE_MISMATCH")
    if value.qualification_policy_hash != policy.qualification_policy_lineage_hash:
        raise QualificationError("QUALIFIED_MESH_POLICY_BINDING_MISMATCH")

    intrinsic = qualified_mesh_intrinsic_audit(value, surface=surface, partition=partition)
    if intrinsic["max_refinement_normal_ratio"] - 1e-12 > policy.g1_max_normal_refinement_ratio:
        raise QualificationError("QUALIFIED_MESH_G1_NORMAL_REFINEMENT_BOUND_FAIL")
    if intrinsic["max_refinement_tangent_to_normal"] - 1e-12 > policy.g1_max_tangential_to_normal_ratio:
        raise QualificationError("QUALIFIED_MESH_G1_TANGENTIAL_REFINEMENT_BOUND_FAIL")
    if intrinsic["rest_min_angle_deg"] + 1e-9 < policy.g3_min_angle_deg:
        raise QualificationError("QUALIFIED_MESH_G3_POLICY_MIN_ANGLE_FAIL")
    if intrinsic["rest_max_aspect_longest_over_min_altitude"] - 1e-9 > policy.g3_max_aspect_longest_over_min_altitude:
        raise QualificationError("QUALIFIED_MESH_G3_POLICY_ASPECT_FAIL")
    if value.qualification_report.get("intrinsic_audit_hash") != content_sha256(intrinsic):
        raise QualificationError("QUALIFIED_MESH_INTRINSIC_AUDIT_BINDING_MISMATCH")

    required = {"G1_SUPPORT_LINEAGE","G2_TOPOLOGY","G3_DEFORMATION","G4_COMPONENT_BOUNDARY","G5_MULTIVIEW_COVERAGE"}
    gates = dict(value.qualification_report.get("gates") or {})
    if set(gates) != required or any(gates[k] != "PASS" for k in required):
        raise QualificationError("QUALIFIED_MESH_REQUIRES_ALL_FIVE_GATES_PASS")
    if value.qualification_report.get("single_aggregate_score_authority") is not False:
        raise QualificationError("QUALIFIED_MESH_AGGREGATE_SCORE_AUTHORITY_FORBIDDEN")
    if value.qualification_report.get("g3_envelope_binding_hash") != envelope.envelope_lineage_hash:
        raise QualificationError("QUALIFIED_MESH_G3_ENVELOPE_BINDING_MISMATCH")
    if not value.qualification_report.get("g3_stress_probe_hash"):
        raise QualificationError("QUALIFIED_MESH_G3_STRESS_PROBE_MISSING")
    if value.qualification_report.get("g3_stress_probe_status") != "PASS":
        raise QualificationError("QUALIFIED_MESH_G3_STRESS_PROBE_NOT_PASS")
    if int(value.qualification_report.get("consequential_unknown_boundary_count", -1)) != 0:
        raise QualificationError("QUALIFIED_MESH_CONSEQUENTIAL_UNKNOWN_BOUNDARY")
    if any(row.decision == "UNKNOWN" for row in partition.boundary_constraints):
        if not value.qualification_report.get("unknown_boundary_analysis_hash"):
            raise QualificationError("QUALIFIED_MESH_G4_UNKNOWN_ANALYSIS_MISSING")

    component_ids = {component.component_id for component in partition.components}
    _validate_g5_matrix(value.qualification_report, component_ids=component_ids, carrier_policy=carrier_policy, policy=policy)
    if value.qualification_report.get("view_component_coverage_matrix_complete") is not True:
        raise QualificationError("QUALIFIED_MESH_VIEW_COMPONENT_COVERAGE_REQUIRED")
    if value.mesh_lineage_hash != qualified_mesh_lineage_hash(value):
        raise QualificationError("QUALIFIED_MESH_LINEAGE_HASH_MISMATCH")


def validate_canonical_mesh_candidate(value: CanonicalMeshCandidateIR, *, surface, partition, carrier_policy) -> None:
    validate_mechanical_partition(partition, surface)
    validate_component_carrier_policy(carrier_policy, partition)
    if value.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("MESH_CANDIDATE_SURFACE_LINEAGE_MISMATCH")
    if value.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("MESH_CANDIDATE_PARTITION_LINEAGE_MISMATCH")
    if value.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
        raise QualificationError("MESH_CANDIDATE_CARRIER_POLICY_LINEAGE_MISMATCH")
    if not value.producer_id or not value.producer_policy_hash:
        raise QualificationError("MESH_CANDIDATE_PRODUCER_AUTHORITY_MISSING")
    if value.candidate_lineage_hash != canonical_mesh_candidate_lineage_hash(value):
        raise QualificationError("MESH_CANDIDATE_LINEAGE_HASH_MISMATCH")
    if len(value.vertices) < 3 or not value.faces:
        raise QualificationError("MESH_CANDIDATE_EMPTY")

    known_surface_ids = {node.surface_id for node in surface.surface_nodes}
    component_ids = {component.component_id for component in partition.components}
    vertex_ids = set()
    for vertex in value.vertices:
        if not vertex.candidate_vertex_id or vertex.candidate_vertex_id in vertex_ids:
            raise QualificationError("MESH_CANDIDATE_VERTEX_ID_INVALID")
        vertex_ids.add(vertex.candidate_vertex_id)
        if vertex.component_id not in component_ids or not _vec_finite(vertex.P):
            raise QualificationError("MESH_CANDIDATE_VERTEX_INVALID")
        if not vertex.support_binding.coefficients:
            raise QualificationError("MESH_CANDIDATE_SUPPORT_MISSING")
        for sid, coeff in vertex.support_binding.coefficients:
            if sid not in known_surface_ids or not math.isfinite(float(coeff)):
                raise QualificationError("MESH_CANDIDATE_SUPPORT_INVALID")

    for face in value.faces:
        if len(face) != 3 or len(set(face)) != 3 or any(vid not in vertex_ids for vid in face):
            raise QualificationError("MESH_CANDIDATE_FACE_INVALID")
    for edge in value.edges:
        if len(edge) != 2 or edge[0] == edge[1] or any(vid not in vertex_ids for vid in edge):
            raise QualificationError("MESH_CANDIDATE_EDGE_INVALID")


def qualify_canonical_mesh_candidate(
    candidate: CanonicalMeshCandidateIR,
    *,
    surface,
    partition,
    carrier_policy,
    envelope,
    policy,
    qualification_report: Json,
) -> QualifiedMeshIR:
    """Promote one exact candidate into the sole product-geometry authority.

    Intrinsic G1/G2/G4 and rest-conditioning evidence is recomputed here; callers
    cannot inject their own intrinsic audit hash. External stress/raster evidence
    remains hash-bound in qualification_report and is validated fail-closed.
    """
    validate_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
    )
    validate_deformation_capability_envelope(envelope)
    validate_mesh_qualification_policy(policy)

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
    faces = tuple(tuple(id_map[vid] for vid in face) for face in candidate.faces)
    edges = tuple(tuple(id_map[vid] for vid in edge) for edge in candidate.edges)

    report = dict(qualification_report or {})
    report.pop("intrinsic_audit_hash", None)
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
        },
    )
    intrinsic = qualified_mesh_intrinsic_audit(mesh, surface=surface, partition=partition)
    mesh = replace(
        mesh,
        qualification_report={**report, "intrinsic_audit_hash": content_sha256(intrinsic)},
    )
    mesh = replace(mesh, mesh_lineage_hash=qualified_mesh_lineage_hash(mesh))
    validate_qualified_mesh(
        mesh,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
    )
    return mesh


def validate_qualified_presentation_graph(
    value: QualifiedPresentationGraphIR,
    *,
    carrier_policy: ComponentCarrierPolicyIR | None = None,
    puppet_state=None,
    skeleton=None,
    partition: MechanicalPartitionIR | None = None,
    envelope: DeformationCapabilityEnvelopeIR | None = None,
    mesh: QualifiedMeshIR | None = None,
    presentation_structure=None,
    appearance_set=None,
    composition_set=None,
) -> None:
    if (
        not value.skeleton_binding_hash
        or not value.mesh_binding_hash
        or not value.partition_binding_hash
        or not value.carrier_policy_binding_hash
        or not value.product_state_binding_hash
        or not value.presentation_structure_binding_hash
        or not value.appearance_set_binding_hash
        or not value.composition_set_binding_hash
    ):
        raise QualificationError("PRESENTATION_GRAPH_UPSTREAM_BINDING_MISSING")

    if puppet_state is not None:
        if value.product_state_binding_hash != puppet_state.product_state_hash:
            raise QualificationError("PRESENTATION_GRAPH_PRODUCT_STATE_MISMATCH")
        expected_state = {
            "skeleton_binding_hash": puppet_state.skeleton_lineage_hash,
            "mesh_binding_hash": puppet_state.mesh_lineage_hash,
            "partition_binding_hash": puppet_state.partition_lineage_hash,
            "carrier_policy_binding_hash": puppet_state.carrier_policy_lineage_hash,
        }
        for field_name, expected_hash in expected_state.items():
            if getattr(value, field_name) != expected_hash:
                raise QualificationError(f"PRESENTATION_GRAPH_STATE_BINDING_DRIFT:{field_name}")

    if skeleton is not None:
        if value.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
            raise QualificationError("PRESENTATION_GRAPH_SKELETON_MISMATCH")
        known_bones = {joint.canonical_joint_id for joint in skeleton.joints}
    else:
        known_bones = None

    if partition is not None:
        if value.partition_binding_hash != partition.partition_lineage_hash:
            raise QualificationError("PRESENTATION_GRAPH_PARTITION_MISMATCH")
        known_components = {component.component_id for component in partition.components}
    else:
        known_components = None

    if carrier_policy is not None:
        if value.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
            raise QualificationError("PRESENTATION_GRAPH_CARRIER_POLICY_MISMATCH")
        expected_carrier = {row.component_id: row.carrier_class for row in carrier_policy.decisions}
    else:
        expected_carrier = None

    if mesh is not None and value.mesh_binding_hash != mesh.mesh_lineage_hash:
        raise QualificationError("PRESENTATION_GRAPH_MESH_MISMATCH")

    if presentation_structure is not None:
        if value.presentation_structure_binding_hash != presentation_structure.structure_lineage_hash:
            raise QualificationError("PRESENTATION_GRAPH_STRUCTURE_BINDING_MISMATCH")
        if tuple(value.slots) != tuple(presentation_structure.slots):
            raise QualificationError("PRESENTATION_GRAPH_STRUCTURE_SLOT_DRIFT")
        if tuple(value.attachments) != tuple(presentation_structure.attachments):
            raise QualificationError("PRESENTATION_GRAPH_STRUCTURE_ATTACHMENT_DRIFT")
        structure_decisions = {row.decision_id: row for row in presentation_structure.decisions}
        graph_decisions = {row.decision_id: row for row in value.decisions}
        if any(graph_decisions.get(key) != row for key, row in structure_decisions.items()):
            raise QualificationError("PRESENTATION_GRAPH_STRUCTURE_DECISION_DRIFT")

    if appearance_set is not None:
        if value.appearance_set_binding_hash != appearance_set.appearance_set_hash:
            raise QualificationError("PRESENTATION_GRAPH_APPEARANCE_SET_BINDING_MISMATCH")
        app_by_view = {int(row.target_view_index): row for row in appearance_set.bindings}
    else:
        app_by_view = None

    if composition_set is not None:
        if value.composition_set_binding_hash != composition_set.composition_set_hash:
            raise QualificationError("PRESENTATION_GRAPH_COMPOSITION_SET_BINDING_MISMATCH")
        comp_by_view = {int(row.view_index): row for row in composition_set.views}
    else:
        comp_by_view = None

    slot_ids = [s.slot_id for s in value.slots]
    if not slot_ids or len(slot_ids) != len(set(slot_ids)) or len({s.setup_order for s in value.slots}) != len(value.slots):
        raise QualificationError("PRESENTATION_SLOT_SET_INVALID")
    slot_set = set(slot_ids)
    allowed = {sid: set() for sid in slot_set}
    attachment_ids = set()
    presented_components = set()
    for slot in value.slots:
        if not slot.bone_id:
            raise QualificationError("PRESENTATION_SLOT_BONE_BINDING_MISSING")
        if known_bones is not None and slot.bone_id not in known_bones:
            raise QualificationError("PRESENTATION_SLOT_UNKNOWN_BONE")
        if any(x not in KEYABLE_CHANNELS for x in slot.keyable_channels):
            raise QualificationError("PRESENTATION_KEYABLE_CHANNEL_INVALID")
    for attachment in value.attachments:
        if not attachment.attachment_id or attachment.attachment_id in attachment_ids:
            raise QualificationError("PRESENTATION_ATTACHMENT_ID_INVALID")
        attachment_ids.add(attachment.attachment_id)
        if attachment.slot_id not in slot_set:
            raise QualificationError("PRESENTATION_ATTACHMENT_SLOT_INVALID")
        if attachment.mechanical_class not in MECHANICAL_CLASSES or attachment.carrier_class not in PRESENTATION_CARRIER_CLASSES:
            raise QualificationError("PRESENTATION_ATTACHMENT_CLASS_INVALID")
        if not attachment.mechanical_component_ids or not attachment.carrier_binding_hash:
            raise QualificationError("PRESENTATION_ATTACHMENT_BINDING_MISSING")
        if known_components is not None and not set(attachment.mechanical_component_ids).issubset(known_components):
            raise QualificationError("PRESENTATION_ATTACHMENT_UNKNOWN_COMPONENT")
        if attachment.carrier_class != "CLIP":
            presented_components.update(attachment.mechanical_component_ids)
        if expected_carrier is not None and attachment.carrier_class != "CLIP":
            for component_id in attachment.mechanical_component_ids:
                if expected_carrier.get(component_id) != attachment.carrier_class:
                    raise QualificationError("PRESENTATION_ATTACHMENT_CARRIER_POLICY_DRIFT")
        if mesh is not None and attachment.carrier_class == "MESH" and attachment.carrier_binding_hash != mesh.mesh_lineage_hash:
            raise QualificationError("PRESENTATION_MESH_ATTACHMENT_BINDING_DRIFT")
        allowed[attachment.slot_id].add(attachment.attachment_id)
    if known_components is not None and presented_components != known_components:
        raise QualificationError("PRESENTATION_COMPONENT_COVERAGE_INCOMPLETE")
    for slot in value.slots:
        if slot.default_attachment_id is not None and slot.default_attachment_id not in allowed[slot.slot_id]:
            raise QualificationError("PRESENTATION_DEFAULT_ATTACHMENT_INVALID")

    if len(value.view_overlays) != 8 or tuple(sorted(v.view_index for v in value.view_overlays)) != tuple(range(8)):
        raise QualificationError("PRESENTATION_REQUIRES_EXACT_8_VIEW_OVERLAYS")
    ordered_overlays = tuple(sorted(value.view_overlays, key=lambda row: row.view_index))
    for overlay in ordered_overlays:
        if not overlay.camera_binding_hash or not overlay.appearance_binding_hash or not overlay.composition_binding_hash:
            raise QualificationError("PRESENTATION_VIEW_OVERLAY_BINDING_MISSING")
        if app_by_view is not None:
            row = app_by_view.get(int(overlay.view_index))
            if row is None or overlay.appearance_binding_hash != row.appearance_lineage_hash:
                raise QualificationError("PRESENTATION_VIEW_APPEARANCE_BINDING_DRIFT")
        if comp_by_view is not None:
            row = comp_by_view.get(int(overlay.view_index))
            if row is None or overlay.composition_binding_hash != row.composition_binding_hash:
                raise QualificationError("PRESENTATION_VIEW_COMPOSITION_BINDING_DRIFT")
    if envelope is not None:
        if tuple(row.camera_binding_hash for row in ordered_overlays) != tuple(envelope.camera_binding_hashes):
            raise QualificationError("PRESENTATION_VIEW_CAMERA_SET_MISMATCH")

    decision_ids = set()
    for decision in value.decisions:
        if not decision.decision_id or decision.decision_id in decision_ids:
            raise QualificationError("PRESENTATION_DECISION_ID_INVALID")
        decision_ids.add(decision.decision_id)
        if decision.authority_class not in PRESENTATION_EVIDENCE_CLASSES or not decision.evidence_refs:
            raise QualificationError("PRESENTATION_DECISION_EVIDENCE_INVALID")
    if value.qualification_report.get("status") != "PASS":
        raise QualificationError("PRESENTATION_GRAPH_NOT_QUALIFIED")
    if value.presentation_lineage_hash != qualified_presentation_lineage_hash(value):
        raise QualificationError("PRESENTATION_GRAPH_LINEAGE_HASH_MISMATCH")
