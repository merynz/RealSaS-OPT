from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

from .hashing import content_sha256
from .mesh.conditioning_v1 import triangle_rest_metric
from .types import QualificationError, RiggingSurfaceIR, SurfaceSupportBinding, Vec3

Json = dict[str, Any]
BOUNDARY_DECISIONS = {"SEPARATE", "PRESERVE_CONTINUITY", "UNKNOWN"}
MECHANICAL_CLASSES = {"RIGID", "DEFORMABLE"}
CARRIER_CLASSES = {"MESH", "PLANAR", "CLIP"}
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
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class MeshQualificationPolicyIR:
    g1_max_normal_refinement_ratio: float
    g1_max_tangential_to_normal_ratio: float
    g3_min_angle_deg: float
    g3_max_aspect_longest_over_min_altitude: float
    coverage_thresholds: tuple[CarrierCoverageThresholdIR, ...]
    qualification_policy_lineage_hash: str
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
        if row.carrier_class not in CARRIER_CLASSES or not row.evidence_refs:
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
    thresholds = {}
    for row in value.coverage_thresholds:
        if row.carrier_class not in CARRIER_CLASSES or row.carrier_class in thresholds:
            raise QualificationError("MESH_QUALIFICATION_POLICY_G5_CARRIER_INVALID")
        vals = (row.min_recall, row.min_precision, row.max_largest_coherent_hole_fraction)
        if any(not math.isfinite(float(x)) or float(x) < 0.0 or float(x) > 1.0 for x in vals):
            raise QualificationError("MESH_QUALIFICATION_POLICY_G5_THRESHOLD_INVALID")
        thresholds[row.carrier_class] = row
    if set(thresholds) != CARRIER_CLASSES:
        raise QualificationError("MESH_QUALIFICATION_POLICY_REQUIRES_ALL_CARRIER_CLASSES")
    if value.qualification_policy_lineage_hash != mesh_qualification_policy_lineage_hash(value):
        raise QualificationError("MESH_QUALIFICATION_POLICY_LINEAGE_HASH_MISMATCH")


def build_mesh_qualification_policy(*, g1_max_normal_refinement_ratio, g1_max_tangential_to_normal_ratio, g3_min_angle_deg, g3_max_aspect_longest_over_min_altitude, coverage_thresholds, metadata=None) -> MeshQualificationPolicyIR:
    value = MeshQualificationPolicyIR(
        float(g1_max_normal_refinement_ratio),
        float(g1_max_tangential_to_normal_ratio),
        float(g3_min_angle_deg),
        float(g3_max_aspect_longest_over_min_altitude),
        tuple(coverage_thresholds),
        "",
        metadata=dict(metadata or {}),
    )
    value = replace(value, qualification_policy_lineage_hash=mesh_qualification_policy_lineage_hash(value))
    validate_mesh_qualification_policy(value)
    return value


def validate_deformation_capability_envelope(value: DeformationCapabilityEnvelopeIR, *, known_joint_ids=None) -> None:
    if not value.skeleton_lineage_hash or not value.probe_plan_hash:
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
        if key in actual or key not in expected or carrier not in CARRIER_CLASSES:
            raise QualificationError("QUALIFIED_MESH_G5_MATRIX_ROW_INVALID")
        actual.add(key)
        if carrier_by_component.get(component_id) != carrier:
            raise QualificationError("QUALIFIED_MESH_G5_CARRIER_CLASS_DRIFT")
        recall = float(row.get("recall", float("nan")))
        precision = float(row.get("precision", float("nan")))
        hole = float(row.get("largest_coherent_hole_fraction", float("nan")))
        if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in (recall, precision, hole)):
            raise QualificationError("QUALIFIED_MESH_G5_METRIC_INVALID")
        threshold = thresholds[carrier]
        if recall + 1e-12 < threshold.min_recall:
            raise QualificationError("QUALIFIED_MESH_G5_RECALL_FAIL")
        if precision + 1e-12 < threshold.min_precision:
            raise QualificationError("QUALIFIED_MESH_G5_PRECISION_FAIL")
        if hole - 1e-12 > threshold.max_largest_coherent_hole_fraction:
            raise QualificationError("QUALIFIED_MESH_G5_COHERENT_HOLE_FAIL")
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


def validate_qualified_presentation_graph(value: QualifiedPresentationGraphIR, *, carrier_policy: ComponentCarrierPolicyIR | None = None) -> None:
    if not value.skeleton_binding_hash or not value.mesh_binding_hash or not value.partition_binding_hash or not value.carrier_policy_binding_hash:
        raise QualificationError("PRESENTATION_GRAPH_UPSTREAM_BINDING_MISSING")
    if carrier_policy is not None:
        if value.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
            raise QualificationError("PRESENTATION_GRAPH_CARRIER_POLICY_MISMATCH")
        expected_carrier = {row.component_id: row.carrier_class for row in carrier_policy.decisions}
    else:
        expected_carrier = None
    slot_ids = [s.slot_id for s in value.slots]
    if not slot_ids or len(slot_ids) != len(set(slot_ids)) or len({s.setup_order for s in value.slots}) != len(value.slots):
        raise QualificationError("PRESENTATION_SLOT_SET_INVALID")
    slot_set = set(slot_ids)
    allowed = {sid: set() for sid in slot_set}
    attachment_ids = set()
    for slot in value.slots:
        if any(x not in KEYABLE_CHANNELS for x in slot.keyable_channels):
            raise QualificationError("PRESENTATION_KEYABLE_CHANNEL_INVALID")
    for attachment in value.attachments:
        if not attachment.attachment_id or attachment.attachment_id in attachment_ids:
            raise QualificationError("PRESENTATION_ATTACHMENT_ID_INVALID")
        attachment_ids.add(attachment.attachment_id)
        if attachment.slot_id not in slot_set:
            raise QualificationError("PRESENTATION_ATTACHMENT_SLOT_INVALID")
        if attachment.mechanical_class not in MECHANICAL_CLASSES or attachment.carrier_class not in CARRIER_CLASSES:
            raise QualificationError("PRESENTATION_ATTACHMENT_CLASS_INVALID")
        if not attachment.mechanical_component_ids or not attachment.carrier_binding_hash:
            raise QualificationError("PRESENTATION_ATTACHMENT_BINDING_MISSING")
        if expected_carrier is not None:
            for component_id in attachment.mechanical_component_ids:
                if expected_carrier.get(component_id) != attachment.carrier_class:
                    raise QualificationError("PRESENTATION_ATTACHMENT_CARRIER_POLICY_DRIFT")
        allowed[attachment.slot_id].add(attachment.attachment_id)
    for slot in value.slots:
        if slot.default_attachment_id is not None and slot.default_attachment_id not in allowed[slot.slot_id]:
            raise QualificationError("PRESENTATION_DEFAULT_ATTACHMENT_INVALID")
    if len(value.view_overlays) != 8 or tuple(sorted(v.view_index for v in value.view_overlays)) != tuple(range(8)):
        raise QualificationError("PRESENTATION_REQUIRES_EXACT_8_VIEW_OVERLAYS")
    for overlay in value.view_overlays:
        if not overlay.camera_binding_hash or not overlay.appearance_binding_hash or not overlay.composition_binding_hash:
            raise QualificationError("PRESENTATION_VIEW_OVERLAY_BINDING_MISSING")
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
