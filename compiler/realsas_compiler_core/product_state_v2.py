from __future__ import annotations

"""V2 complete puppet state: co-equal mechanics + appearance + presentation."""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

import numpy as np

from .hashing import content_sha256
from .product_authority_v1 import (
    PresentationAttachmentIR,
    PresentationDecisionEvidenceIR,
    PresentationSlotIR,
    PresentationViewOverlayIR,
)
from .visibility_v2 import VISIBILITY_CONTRACT_V2, VISIBILITY_CONTRACT_V2_HASH
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class QualifiedPresentationStructureV2IR:
    slots: tuple[PresentationSlotIR, ...]
    attachments: tuple[PresentationAttachmentIR, ...]
    decisions: tuple[PresentationDecisionEvidenceIR, ...]
    skeleton_binding_hash: str
    mesh_binding_hash: str
    mesh_skin_binding_hash: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    structure_hash: str
    schema_version: str = "RealSaS.QualifiedPresentationStructureIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def presentation_structure_v2_hash(value: QualifiedPresentationStructureV2IR) -> str:
    payload = value.to_dict()
    payload.pop("structure_hash", None)
    return content_sha256(payload)


def _face_groups(mesh, component_id: str, *, cut_face_pairs=()):
    cut_pairs = {
        tuple(sorted((int(a), int(b))))
        for a, b in cut_face_pairs
    }
    component_by_vertex = {
        str(vertex.canonical_mesh_vertex_id): str(vertex.component_id)
        for vertex in mesh.vertices
    }
    rows = []
    for face_index, face in enumerate(mesh.faces):
        components = {component_by_vertex[str(vertex_id)] for vertex_id in face}
        if len(components) != 1:
            raise QualificationError("PRESENTATION_V2_FACE_CROSSES_COMPONENT")
        if next(iter(components)) == str(component_id):
            rows.append((int(face_index), tuple(map(str, face))))
    if not rows:
        raise QualificationError("PRESENTATION_V2_COMPONENT_HAS_NO_FACE")

    edge_to_faces = {}
    for face_index, face in rows:
        for a, b in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_to_faces.setdefault(tuple(sorted((a, b))), []).append(face_index)
    neighbors = {face_index: set() for face_index, _face in rows}
    for incident in edge_to_faces.values():
        for a in incident:
            for b in incident:
                if a != b and tuple(sorted((int(a), int(b)))) not in cut_pairs:
                    neighbors[a].add(b)
    unseen = set(neighbors)
    groups = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        group = []
        while stack:
            current = stack.pop()
            group.append(current)
            for nxt in sorted(neighbors[current]):
                if nxt in unseen:
                    unseen.remove(nxt)
                    stack.append(nxt)
        groups.append(tuple(sorted(group)))
    return tuple(sorted(groups, key=lambda group: (group[0], len(group), group)))


RIGIDITY_NOOP_RELATIVE_EDGE_TOLERANCE = 1.0e-9
RIGIDITY_NOOP_PROBE_ROTATION_DEGREES = 37.0


def _axis_rotation(axis: int, angle_degrees: float) -> np.ndarray:
    angle = math.radians(float(angle_degrees))
    c = math.cos(angle)
    s = math.sin(angle)
    if int(axis) == 0:
        return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
    if int(axis) == 1:
        return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))
    if int(axis) == 2:
        return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
    raise QualificationError("PRESENTATION_V2_RIGIDITY_PROBE_AXIS_INVALID")


def _face_group_rigidity_noop_probe(
    face_indices,
    mesh,
    mesh_skin,
) -> dict:
    rows = {
        str(row.canonical_mesh_vertex_id): row
        for row in mesh_skin.rows
    }
    positions = {
        str(vertex.canonical_mesh_vertex_id): np.asarray(
            vertex.P,
            dtype=np.float64,
        )
        for vertex in mesh.vertices
    }
    vertex_ids = sorted(
        {
            str(vertex_id)
            for face_index in face_indices
            for vertex_id in mesh.faces[int(face_index)]
        }
    )
    if not vertex_ids or any(vertex_id not in positions for vertex_id in vertex_ids):
        raise QualificationError("PRESENTATION_V2_RIGIDITY_PROBE_VERTEX_MISSING")
    xyz = np.asarray([positions[vertex_id] for vertex_id in vertex_ids], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise QualificationError("PRESENTATION_V2_RIGIDITY_PROBE_POSITION_INVALID")
    local = {vertex_id: index for index, vertex_id in enumerate(vertex_ids)}
    if any(vertex_id not in rows for vertex_id in vertex_ids):
        raise QualificationError(
            "PRESENTATION_V2_RIGIDITY_PROBE_SKIN_ROW_MISSING"
        )

    edges = set()
    for face_index in face_indices:
        face = tuple(map(str, mesh.faces[int(face_index)]))
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges.add(tuple(sorted((local[a], local[b]))))
    if not edges:
        raise QualificationError("PRESENTATION_V2_RIGIDITY_PROBE_EDGE_EMPTY")
    edge_rows = np.asarray(sorted(edges), dtype=np.int64)
    rest_delta = xyz[edge_rows[:, 1]] - xyz[edge_rows[:, 0]]
    rest_length = np.linalg.norm(rest_delta, axis=1)
    span = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    scale = max(span, float(np.max(rest_length, initial=0.0)), 1.0e-12)
    if np.any(rest_length <= 1.0e-12 * scale):
        raise QualificationError("PRESENTATION_V2_RIGIDITY_PROBE_DEGENERATE_EDGE")

    joint_ids = sorted(
        {
            str(joint_id)
            for vertex_id in vertex_ids
            for joint_id, _weight in rows[vertex_id].influences
        }
    )
    if not joint_ids:
        raise QualificationError("PRESENTATION_V2_RIGIDITY_PROBE_JOINT_EMPTY")
    joint_index = {joint_id: i for i, joint_id in enumerate(joint_ids)}
    weights = np.zeros((len(vertex_ids), len(joint_ids)), dtype=np.float64)
    for vi, vertex_id in enumerate(vertex_ids):
        row = rows.get(vertex_id)
        if row is None or not row.influences:
            raise QualificationError("PRESENTATION_V2_RIGIDITY_PROBE_SKIN_ROW_MISSING")
        for joint_id, weight in row.influences:
            value = float(weight)
            if not math.isfinite(value) or value < 0.0:
                raise QualificationError("PRESENTATION_V2_RIGIDITY_PROBE_WEIGHT_INVALID")
            weights[vi, joint_index[str(joint_id)]] += value
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-9, rtol=0.0):
        raise QualificationError("PRESENTATION_V2_RIGIDITY_PROBE_SIMPLEX_DRIFT")

    max_error = 0.0
    worst_probe = ""
    translation = scale * np.asarray((0.31, -0.19, 0.13), dtype=np.float64)

    def measure(posed: np.ndarray, label: str) -> None:
        nonlocal max_error, worst_probe
        delta = posed[edge_rows[:, 1]] - posed[edge_rows[:, 0]]
        length = np.linalg.norm(delta, axis=1)
        error = float(np.max(np.abs(length - rest_length) / rest_length))
        if error > max_error:
            max_error = error
            worst_probe = label

    for joint_id, ji in joint_index.items():
        w = weights[:, ji : ji + 1]
        measure(xyz + w * translation[None, :], f"TRANSLATE:{joint_id}")
        for axis in range(3):
            rotation = _axis_rotation(axis, RIGIDITY_NOOP_PROBE_ROTATION_DEGREES)
            rotated = xyz @ rotation.T
            posed = xyz + w * (rotated - xyz)
            measure(posed, f"ROTATE_{axis}:{joint_id}")

    return {
        "mode": "INDEPENDENT_JOINT_LBS_RIGID_NOOP_V1",
        "vertex_count": int(len(vertex_ids)),
        "edge_count": int(len(edge_rows)),
        "influencing_joint_count": int(len(joint_ids)),
        "max_relative_edge_error": float(max_error),
        "tolerance": float(RIGIDITY_NOOP_RELATIVE_EDGE_TOLERANCE),
        "passed": bool(max_error <= RIGIDITY_NOOP_RELATIVE_EDGE_TOLERANCE),
        "worst_probe": worst_probe,
    }


def _face_group_class(
    face_indices,
    mesh,
    mesh_skin,
    *,
    owner_min: float,
    other_max: float,
):
    rows = {
        str(row.canonical_mesh_vertex_id): row
        for row in mesh_skin.rows
    }
    vertex_ids = sorted(
        {
            str(vertex_id)
            for face_index in face_indices
            for vertex_id in mesh.faces[int(face_index)]
        }
    )
    if not vertex_ids:
        raise QualificationError("PRESENTATION_V2_GROUP_VERTEX_EMPTY")
    common_owner = None
    minimum_owner = 1.0
    maximum_other = 0.0
    for vertex_id in vertex_ids:
        row = rows.get(vertex_id)
        if row is None or not row.influences:
            raise QualificationError("PRESENTATION_V2_MESH_SKIN_ROW_MISSING")
        ranked = sorted(
            ((str(joint), float(weight)) for joint, weight in row.influences),
            key=lambda item: (-item[1], item[0]),
        )
        if any(not math.isfinite(weight) or weight < 0.0 for _joint, weight in ranked):
            raise QualificationError("PRESENTATION_V2_SKIN_WEIGHT_INVALID")
        owner, owner_weight = ranked[0]
        other = sum(weight for joint, weight in ranked if joint != owner)
        if common_owner is None:
            common_owner = owner
        elif owner != common_owner:
            common_owner = ""
        minimum_owner = min(minimum_owner, owner_weight)
        maximum_other = max(maximum_other, other)
    legacy_rigid = (
        bool(common_owner)
        and minimum_owner >= float(owner_min)
        and maximum_other <= float(other_max)
    )
    noop = _face_group_rigidity_noop_probe(
        face_indices,
        mesh,
        mesh_skin,
    )
    noop_rigid = bool(noop["passed"])
    if legacy_rigid and not noop_rigid:
        raise QualificationError(
            "PRESENTATION_V2_LEGACY_RIGID_PREDICATE_NOOP_FAIL:"
            + str(noop["max_relative_edge_error"])
        )
    if noop_rigid and not common_owner:
        raise QualificationError(
            "PRESENTATION_V2_RIGID_NOOP_WITHOUT_COMMON_OWNER"
        )
    mechanical_class = "RIGID" if noop_rigid else "DEFORMABLE"
    return (
        mechanical_class,
        str(common_owner) if noop_rigid else "",
        float(minimum_owner),
        float(maximum_other),
        dict(noop),
        bool(legacy_rigid),
        bool(noop_rigid and not legacy_rigid),
    )


def build_presentation_structure_v2(
    *,
    skeleton,
    mesh,
    mesh_skin,
    partition,
    carrier_policy,
    min_rigid_owner_weight: float = 0.999,
    max_rigid_other_mass: float = 0.001,
    presentation_cut_face_pairs=(),
    presentation_partition_evidence_hash: str = "",
) -> QualifiedPresentationStructureV2IR:
    joint_ids = {str(joint.canonical_joint_id) for joint in skeleton.joints}
    carrier = {row.component_id: row.carrier_class for row in carrier_policy.decisions}
    slots = []
    attachments = []
    decisions = []
    setup_order = 0

    for component in sorted(partition.components, key=lambda row: row.component_id):
        carrier_class = carrier.get(component.component_id)
        if carrier_class != "MESH":
            raise QualificationError(
                f"PRESENTATION_V2_NONMESH_CARRIER_UNSUPPORTED:{component.component_id}"
            )
        for group_index, face_indices in enumerate(
            _face_groups(
                mesh,
                component.component_id,
                cut_face_pairs=presentation_cut_face_pairs,
            )
        ):
            (
                mechanical_class,
                rigid_owner,
                min_owner,
                max_other,
                rigidity_noop,
                legacy_rigid,
                legacy_false_negative,
            ) = _face_group_class(
                face_indices,
                mesh,
                mesh_skin,
                owner_min=min_rigid_owner_weight,
                other_max=max_rigid_other_mass,
            )
            bone_id = (
                rigid_owner
                if mechanical_class == "RIGID"
                else str(skeleton.root_id)
            )
            if bone_id not in joint_ids:
                raise QualificationError("PRESENTATION_V2_BONE_UNKNOWN")
            group_hash = content_sha256(
                {
                    "mesh": mesh.mesh_lineage_hash,
                    "component": component.component_id,
                    "face_indices": face_indices,
                    "presentation_partition_evidence_hash": str(
                        presentation_partition_evidence_hash
                    ),
                }
            )
            slot_id = "SLOT:" + content_sha256(
                {
                    "structure": "V2",
                    "component": component.component_id,
                    "group": group_hash,
                }
            )[:24]
            attachment_id = "ATT:" + content_sha256(
                {
                    "structure": "V2",
                    "component": component.component_id,
                    "group": group_hash,
                    "carrier": carrier_class,
                }
            )[:24]
            slots.append(
                PresentationSlotIR(
                    slot_id=slot_id,
                    bone_id=bone_id,
                    setup_order=setup_order,
                    default_attachment_id=attachment_id,
                    keyable_channels=("ATTACHMENT", "TINT", "ORDER", "VISIBILITY"),
                    metadata={
                        "mechanical_component_id": component.component_id,
                        "presentation_group_index": group_index,
                        "presentation_group_hash": group_hash,
                        "mesh_face_indices": face_indices,
                        "setup_order_role": "UI_ONLY_NOT_DEPTH_AUTHORITY",
                        "categorical_identity": None,
                    },
                )
            )
            attachments.append(
                PresentationAttachmentIR(
                    attachment_id=attachment_id,
                    slot_id=slot_id,
                    mechanical_component_ids=(component.component_id,),
                    mechanical_class=mechanical_class,
                    carrier_class="MESH",
                    carrier_binding_hash=mesh.mesh_lineage_hash,
                    metadata={
                        "presentation_group_hash": group_hash,
                        "mesh_face_indices": face_indices,
                        "rigid_owner_joint_id": rigid_owner,
                        "minimum_owner_weight": min_owner,
                        "maximum_other_mass": max_other,
                        "rigidity_noop_probe": rigidity_noop,
                        "legacy_weight_threshold_rigid": legacy_rigid,
                        "legacy_weight_threshold_false_negative": (
                            legacy_false_negative
                        ),
                        "rigidity_authority": (
                            "INDEPENDENT_JOINT_LBS_RIGID_NOOP_V1"
                        ),
                        "categorical_identity": None,
                        "detachability_authority": "UNPROVEN",
                    },
                )
            )
            decisions.append(
                PresentationDecisionEvidenceIR(
                    decision_id="DEC:" + content_sha256(
                        {
                            "component": component.component_id,
                            "group": group_hash,
                            "kind": "ROLE_FREE_GROUP",
                        }
                    )[:24],
                    decision_kind="SLOT_BINDING",
                    authority_class="MECHANICAL_PLUS_ROLE_FREE_APPEARANCE_BOUNDARY",
                    evidence_refs=(
                        partition.partition_lineage_hash,
                        mesh.mesh_lineage_hash,
                        mesh_skin.mesh_skin_lineage_hash,
                        str(presentation_partition_evidence_hash),
                        group_hash,
                    ),
                    metadata={
                        "categorical_recognition_used": False,
                        "presentation_group_derivation": "CONNECTED_FACE_ISLAND_AFTER_ROLE_FREE_APPEARANCE_BOUNDARY_CUTS",
                        "conceptual_object_identity_claimed": False,
                    },
                )
            )
            setup_order += 1

    value = QualifiedPresentationStructureV2IR(
        slots=tuple(slots),
        attachments=tuple(attachments),
        decisions=tuple(decisions),
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        mesh_skin_binding_hash=mesh_skin.mesh_skin_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        structure_hash="",
        metadata={
            "role_free": True,
            "appearance_authority_owned_elsewhere": True,
            "appearance_boundary_evidence_consumed": bool(
                presentation_partition_evidence_hash
            ),
            "presentation_partition_evidence_hash": str(
                presentation_partition_evidence_hash
            ),
            "presentation_cut_face_pairs": [
                [int(a), int(b)] for a, b in presentation_cut_face_pairs
            ],
            "depth_authority_owned_elsewhere": True,
            "categorical_recognition_used": False,
            "conceptual_object_identity_claimed": False,
            "mechanical_classification_scope": "PRESENTATION_FACE_GROUP",
            "rigid_classification_authority": (
                "INDEPENDENT_JOINT_LBS_RIGID_NOOP_V1"
            ),
            "legacy_weight_thresholds_are_diagnostic_not_final_authority": True,
            "rigid_noop_relative_edge_tolerance": (
                RIGIDITY_NOOP_RELATIVE_EDGE_TOLERANCE
            ),
        },
    )
    value = replace(value, structure_hash=presentation_structure_v2_hash(value))
    return value


@dataclass(frozen=True)
class QualifiedPresentationGraphV2IR:
    slots: tuple[PresentationSlotIR, ...]
    attachments: tuple[PresentationAttachmentIR, ...]
    view_overlays: tuple[PresentationViewOverlayIR, ...]
    decisions: tuple[PresentationDecisionEvidenceIR, ...]
    skeleton_binding_hash: str
    mesh_binding_hash: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    mechanical_state_binding_hash: str
    presentation_structure_binding_hash: str
    complete_appearance_asset_binding_hash: str
    complete_appearance_qualification_binding_hash: str
    composition_policy_binding_hash: str
    qualification_report: Json
    presentation_lineage_hash: str
    schema_version: str = "RealSaS.QualifiedPresentationGraphIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def presentation_graph_v2_hash(value: QualifiedPresentationGraphV2IR) -> str:
    payload = value.to_dict()
    payload.pop("presentation_lineage_hash", None)
    return content_sha256(payload)


@dataclass(frozen=True)
class CompletePuppetStateV2IR:
    mechanical_state_binding_hash: str
    skeleton_binding_hash: str
    mesh_binding_hash: str
    mesh_skin_binding_hash: str
    presentation_structure_binding_hash: str
    presentation_graph_binding_hash: str
    complete_appearance_asset_binding_hash: str
    complete_appearance_qualification_binding_hash: str
    output_direction_set_binding_hash: str
    qualification_ledger: tuple[Json, ...]
    complete_puppet_hash: str
    schema_version: str = "RealSaS.CompletePuppetStateIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def complete_puppet_state_hash(value: CompletePuppetStateV2IR) -> str:
    payload = value.to_dict()
    payload.pop("complete_puppet_hash", None)
    return content_sha256(payload)


def build_caa_bound_presentation_graph(
    *,
    structure: QualifiedPresentationStructureV2IR,
    product_state,
    skeleton,
    mesh,
    partition,
    carrier_policy,
    directions,
    appearance_asset_hash: str,
    appearance_qualification_hash: str,
):
    composition_hash = content_sha256(
        {
            "schema": "RealSaS.CompositionPolicy.v2",
            "visibility_contract_hash": VISIBILITY_CONTRACT_V2_HASH,
            "visibility_authority": VISIBILITY_CONTRACT_V2["authority"],
            "equal_depth_tie": VISIBILITY_CONTRACT_V2["exact_depth_tie"],
            "setup_order_is_physical_depth": False,
            "texture_alpha_selects_front_surface": False,
        }
    )
    overlays = tuple(
        PresentationViewOverlayIR(
            view_index=int(direction.direction_index),
            camera_binding_hash=str(direction.camera_binding_hash),
            appearance_binding_hash=str(appearance_asset_hash),
            composition_binding_hash=composition_hash,
            metadata={
                "appearance_authority": "CAA_V2",
                "visibility_authority": "CANONICAL_ZBUFFER_V2",
            },
        )
        for direction in sorted(
            directions.directions, key=lambda row: row.direction_index
        )
    )
    graph = QualifiedPresentationGraphV2IR(
        slots=structure.slots,
        attachments=structure.attachments,
        view_overlays=overlays,
        decisions=structure.decisions,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        mechanical_state_binding_hash=product_state.product_state_hash,
        presentation_structure_binding_hash=structure.structure_hash,
        complete_appearance_asset_binding_hash=str(appearance_asset_hash),
        complete_appearance_qualification_binding_hash=str(
            appearance_qualification_hash
        ),
        composition_policy_binding_hash=composition_hash,
        qualification_report={
            "status": "PASS_CAA_BOUND_PRESENTATION_V2",
            "role_free": True,
            "categorical_recognition_used": False,
            "appearance_authority": "COMPLETE_APPEARANCE_AUTHORITY_V2",
            "appearance_total": True,
            "visibility_authority": VISIBILITY_CONTRACT_V2["authority"],
            "visibility_contract_hash": VISIBILITY_CONTRACT_V2_HASH,
            "runtime_donor_search_forbidden": True,
        },
        presentation_lineage_hash="",
        metadata={
            "setup_order_not_depth_authority": True,
            "legacy_appearance_set_semantics_used": False,
        },
    )
    return replace(
        graph,
        presentation_lineage_hash=presentation_graph_v2_hash(graph),
    )


def build_complete_puppet_state_v2(
    *,
    mechanical_state,
    structure,
    presentation_graph,
    skeleton,
    mesh,
    mesh_skin,
    appearance_asset_hash: str,
    appearance_qualification_hash: str,
    output_direction_set_hash: str,
) -> CompletePuppetStateV2IR:
    ledger = (
        {"authority": "MECHANICAL_STATE", "hash": mechanical_state.product_state_hash},
        {"authority": "SKELETON", "hash": skeleton.skeleton_lineage_hash},
        {"authority": "MESH", "hash": mesh.mesh_lineage_hash},
        {"authority": "MESH_SKIN", "hash": mesh_skin.mesh_skin_lineage_hash},
        {"authority": "PRESENTATION_STRUCTURE", "hash": structure.structure_hash},
        {"authority": "PRESENTATION_GRAPH", "hash": presentation_graph.presentation_lineage_hash},
        {"authority": "COMPLETE_APPEARANCE_ASSET", "hash": appearance_asset_hash},
        {"authority": "COMPLETE_APPEARANCE_QUALIFICATION", "hash": appearance_qualification_hash},
        {"authority": "OUTPUT_DIRECTIONS", "hash": output_direction_set_hash},
    )
    value = CompletePuppetStateV2IR(
        mechanical_state_binding_hash=mechanical_state.product_state_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        mesh_skin_binding_hash=mesh_skin.mesh_skin_lineage_hash,
        presentation_structure_binding_hash=structure.structure_hash,
        presentation_graph_binding_hash=presentation_graph.presentation_lineage_hash,
        complete_appearance_asset_binding_hash=str(appearance_asset_hash),
        complete_appearance_qualification_binding_hash=str(appearance_qualification_hash),
        output_direction_set_binding_hash=str(output_direction_set_hash),
        qualification_ledger=ledger,
        complete_puppet_hash="",
        metadata={
            "geometry_mechanics_appearance_coequal": True,
            "runtime_generation_forbidden": True,
            "single_canonical_mesh_authority": True,
        },
    )
    return replace(value, complete_puppet_hash=complete_puppet_state_hash(value))


def presentation_structure_v2_from_dict(payload):
    slots = tuple(PresentationSlotIR(**row) for row in payload.get("slots") or ())
    attachments = tuple(
        PresentationAttachmentIR(**row) for row in payload.get("attachments") or ()
    )
    decisions = tuple(
        PresentationDecisionEvidenceIR(**row) for row in payload.get("decisions") or ()
    )
    value = QualifiedPresentationStructureV2IR(
        slots,
        attachments,
        decisions,
        str(payload["skeleton_binding_hash"]),
        str(payload["mesh_binding_hash"]),
        str(payload["mesh_skin_binding_hash"]),
        str(payload["partition_binding_hash"]),
        str(payload["carrier_policy_binding_hash"]),
        str(payload["structure_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.QualifiedPresentationStructureIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.structure_hash != presentation_structure_v2_hash(value):
        raise QualificationError("PRESENTATION_V2_STRUCTURE_HASH_DRIFT")
    return value


def complete_puppet_state_v2_from_dict(payload):
    value = CompletePuppetStateV2IR(
        mechanical_state_binding_hash=str(payload["mechanical_state_binding_hash"]),
        skeleton_binding_hash=str(payload["skeleton_binding_hash"]),
        mesh_binding_hash=str(payload["mesh_binding_hash"]),
        mesh_skin_binding_hash=str(payload["mesh_skin_binding_hash"]),
        presentation_structure_binding_hash=str(payload["presentation_structure_binding_hash"]),
        presentation_graph_binding_hash=str(payload["presentation_graph_binding_hash"]),
        complete_appearance_asset_binding_hash=str(payload["complete_appearance_asset_binding_hash"]),
        complete_appearance_qualification_binding_hash=str(payload["complete_appearance_qualification_binding_hash"]),
        output_direction_set_binding_hash=str(payload["output_direction_set_binding_hash"]),
        qualification_ledger=tuple(dict(row) for row in payload.get("qualification_ledger") or ()),
        complete_puppet_hash=str(payload["complete_puppet_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.CompletePuppetStateIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.complete_puppet_hash != complete_puppet_state_hash(value):
        raise QualificationError("COMPLETE_PUPPET_V2_HASH_DRIFT")
    return value


def presentation_graph_v2_from_dict(payload):
    def _slot(row):
        return PresentationSlotIR(
            slot_id=str(row["slot_id"]),
            bone_id=str(row["bone_id"]),
            setup_order=int(row["setup_order"]),
            default_attachment_id=(
                None
                if row.get("default_attachment_id") is None
                else str(row["default_attachment_id"])
            ),
            keyable_channels=tuple(map(str, row.get("keyable_channels") or ())),
            metadata=dict(row.get("metadata") or {}),
        )

    def _attachment(row):
        return PresentationAttachmentIR(
            attachment_id=str(row["attachment_id"]),
            slot_id=str(row["slot_id"]),
            mechanical_component_ids=tuple(
                map(str, row.get("mechanical_component_ids") or ())
            ),
            mechanical_class=str(row["mechanical_class"]),
            carrier_class=str(row["carrier_class"]),
            carrier_binding_hash=str(row["carrier_binding_hash"]),
            metadata=dict(row.get("metadata") or {}),
        )

    def _overlay(row):
        return PresentationViewOverlayIR(
            view_index=int(row["view_index"]),
            camera_binding_hash=str(row["camera_binding_hash"]),
            appearance_binding_hash=str(row["appearance_binding_hash"]),
            composition_binding_hash=str(row["composition_binding_hash"]),
            metadata=dict(row.get("metadata") or {}),
        )

    def _decision(row):
        return PresentationDecisionEvidenceIR(
            decision_id=str(row["decision_id"]),
            decision_kind=str(row["decision_kind"]),
            authority_class=str(row["authority_class"]),
            evidence_refs=tuple(map(str, row.get("evidence_refs") or ())),
            metadata=dict(row.get("metadata") or {}),
        )

    value = QualifiedPresentationGraphV2IR(
        slots=tuple(_slot(row) for row in payload.get("slots") or ()),
        attachments=tuple(
            _attachment(row) for row in payload.get("attachments") or ()
        ),
        view_overlays=tuple(
            _overlay(row) for row in payload.get("view_overlays") or ()
        ),
        decisions=tuple(_decision(row) for row in payload.get("decisions") or ()),
        skeleton_binding_hash=str(payload["skeleton_binding_hash"]),
        mesh_binding_hash=str(payload["mesh_binding_hash"]),
        partition_binding_hash=str(payload["partition_binding_hash"]),
        carrier_policy_binding_hash=str(payload["carrier_policy_binding_hash"]),
        mechanical_state_binding_hash=str(payload["mechanical_state_binding_hash"]),
        presentation_structure_binding_hash=str(
            payload["presentation_structure_binding_hash"]
        ),
        complete_appearance_asset_binding_hash=str(
            payload["complete_appearance_asset_binding_hash"]
        ),
        complete_appearance_qualification_binding_hash=str(
            payload["complete_appearance_qualification_binding_hash"]
        ),
        composition_policy_binding_hash=str(
            payload["composition_policy_binding_hash"]
        ),
        qualification_report=dict(payload.get("qualification_report") or {}),
        presentation_lineage_hash=str(payload["presentation_lineage_hash"]),
        schema_version=str(
            payload.get("schema_version") or "RealSaS.QualifiedPresentationGraphIR.v2"
        ),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.presentation_lineage_hash != presentation_graph_v2_hash(value):
        raise QualificationError("PRESENTATION_GRAPH_V2_HASH_DRIFT")
    return value
