from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

from .hashing import content_sha256
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


def validate_qualified_mesh(value: QualifiedMeshIR, *, surface, partition, envelope) -> None:
    validate_mechanical_partition(partition, surface)
    if value.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("QUALIFIED_MESH_SURFACE_LINEAGE_MISMATCH")
    if value.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("QUALIFIED_MESH_PARTITION_LINEAGE_MISMATCH")
    if value.envelope_binding_hash != envelope.envelope_lineage_hash:
        raise QualificationError("QUALIFIED_MESH_ENVELOPE_LINEAGE_MISMATCH")
    if not value.qualification_policy_hash:
        raise QualificationError("QUALIFIED_MESH_POLICY_BINDING_MISSING")
    required = {"G1_SUPPORT_LINEAGE","G2_TOPOLOGY","G3_DEFORMATION","G4_COMPONENT_BOUNDARY","G5_MULTIVIEW_COVERAGE"}
    gates = dict(value.qualification_report.get("gates") or {})
    if set(gates) != required or any(gates[k] != "PASS" for k in required):
        raise QualificationError("QUALIFIED_MESH_REQUIRES_ALL_FIVE_GATES_PASS")
    if value.qualification_report.get("single_aggregate_score_authority") is not False:
        raise QualificationError("QUALIFIED_MESH_AGGREGATE_SCORE_AUTHORITY_FORBIDDEN")
    if value.qualification_report.get("view_component_coverage_matrix_complete") is not True:
        raise QualificationError("QUALIFIED_MESH_VIEW_COMPONENT_COVERAGE_REQUIRED")
    if int(value.qualification_report.get("consequential_unknown_boundary_count", -1)) != 0:
        raise QualificationError("QUALIFIED_MESH_CONSEQUENTIAL_UNKNOWN_BOUNDARY")
    if value.mesh_lineage_hash != qualified_mesh_lineage_hash(value):
        raise QualificationError("QUALIFIED_MESH_LINEAGE_HASH_MISMATCH")


def validate_qualified_presentation_graph(value: QualifiedPresentationGraphIR) -> None:
    if not value.skeleton_binding_hash or not value.mesh_binding_hash or not value.partition_binding_hash:
        raise QualificationError("PRESENTATION_GRAPH_UPSTREAM_BINDING_MISSING")
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
