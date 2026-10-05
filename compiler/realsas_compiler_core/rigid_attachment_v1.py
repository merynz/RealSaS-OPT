from __future__ import annotations

"""Compiler-side rigid attachment contract.

Rigid attachments are carrier components whose geometry must follow exactly one
qualified joint transform. They are deliberately excluded from deformable skin
interpolation. Variant/loadout visibility is presentation state, not geometry or
mechanical authority.
"""

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .types import QualificationError


@dataclass(frozen=True)
class RigidAttachmentBindingV1IR:
    attachment_id: str
    joint_id: str
    component_ids: tuple[int, ...]
    variant_group_id: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def validate_rigid_attachment_bindings_v1(
    bindings: Sequence[RigidAttachmentBindingV1IR],
    *,
    known_component_ids: Sequence[int],
    known_joint_ids: Sequence[str],
) -> None:
    known_components = {int(x) for x in known_component_ids}
    known_joints = {str(x) for x in known_joint_ids}
    seen_attachment_ids: set[str] = set()
    claimed_components: set[int] = set()

    for binding in tuple(bindings):
        aid = str(binding.attachment_id)
        joint = str(binding.joint_id)
        group = str(binding.variant_group_id)
        if not aid or aid in seen_attachment_ids:
            raise QualificationError("RIGID_ATTACHMENT_ID_INVALID")
        if not joint or joint not in known_joints:
            raise QualificationError("RIGID_ATTACHMENT_JOINT_INVALID")
        if not group:
            raise QualificationError("RIGID_ATTACHMENT_VARIANT_GROUP_INVALID")
        components = tuple(map(int, binding.component_ids))
        if not components or len(components) != len(set(components)):
            raise QualificationError("RIGID_ATTACHMENT_COMPONENT_SET_INVALID")
        if any(component not in known_components for component in components):
            raise QualificationError("RIGID_ATTACHMENT_COMPONENT_UNKNOWN")
        if claimed_components.intersection(components):
            raise QualificationError("RIGID_ATTACHMENT_COMPONENT_CLAIM_CONFLICT")
        if bool(binding.metadata.get("deformable_skin_interpolation_allowed", False)):
            raise QualificationError("RIGID_ATTACHMENT_DEFORMABLE_INTERPOLATION_FORBIDDEN")
        seen_attachment_ids.add(aid)
        claimed_components.update(components)


def apply_rigid_attachment_weights_v1(
    base_weights: np.ndarray,
    *,
    vertex_component_ids: np.ndarray,
    bindings: Sequence[RigidAttachmentBindingV1IR],
    joint_index_by_id: Mapping[str, int],
) -> np.ndarray:
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    components = np.asarray(vertex_component_ids, dtype=np.int64).reshape(-1)
    if weights.ndim != 2 or len(weights) != len(components):
        raise QualificationError("RIGID_ATTACHMENT_WEIGHT_SHAPE_INVALID")
    if not np.isfinite(weights).all():
        raise QualificationError("RIGID_ATTACHMENT_WEIGHT_NONFINITE")

    known_component_ids = tuple(sorted(set(map(int, components.tolist()))))
    validate_rigid_attachment_bindings_v1(
        bindings,
        known_component_ids=known_component_ids,
        known_joint_ids=tuple(map(str, joint_index_by_id)),
    )

    for binding in tuple(bindings):
        if str(binding.joint_id) not in joint_index_by_id:
            raise QualificationError("RIGID_ATTACHMENT_JOINT_INDEX_MISSING")
        joint_index = int(joint_index_by_id[str(binding.joint_id)])
        if not (0 <= joint_index < weights.shape[1]):
            raise QualificationError("RIGID_ATTACHMENT_JOINT_INDEX_INVALID")
        mask = np.isin(components, np.asarray(binding.component_ids, dtype=np.int64))
        if not bool(mask.any()):
            raise QualificationError("RIGID_ATTACHMENT_VERTEX_ACCOUNTING_EMPTY")
        weights[mask] = 0.0
        weights[mask, joint_index] = 1.0

    row_sum = weights.sum(axis=1)
    if not np.isfinite(row_sum).all() or float(np.max(np.abs(row_sum - 1.0))) > 1.0e-9:
        raise QualificationError("RIGID_ATTACHMENT_WEIGHT_SIMPLEX_INVALID")
    return weights


def build_rigid_loadout_face_mask_v1(
    faces: np.ndarray,
    *,
    vertex_component_ids: np.ndarray,
    bindings: Sequence[RigidAttachmentBindingV1IR],
    selected_attachment_ids: Sequence[str],
) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    components = np.asarray(vertex_component_ids, dtype=np.int64).reshape(-1)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise QualificationError("RIGID_ATTACHMENT_FACE_ARRAY_INVALID")
    if len(components) == 0 or np.any(faces < 0) or np.any(faces >= len(components)):
        raise QualificationError("RIGID_ATTACHMENT_FACE_INDEX_INVALID")

    known_component_ids = tuple(sorted(set(map(int, components.tolist()))))
    known_joint_ids = tuple(sorted({str(binding.joint_id) for binding in bindings}))
    validate_rigid_attachment_bindings_v1(
        bindings,
        known_component_ids=known_component_ids,
        known_joint_ids=known_joint_ids,
    )

    selected = tuple(map(str, selected_attachment_ids))
    if len(selected) != len(set(selected)):
        raise QualificationError("RIGID_ATTACHMENT_LOADOUT_DUPLICATE")
    by_id = {str(binding.attachment_id): binding for binding in bindings}
    if any(aid not in by_id for aid in selected):
        raise QualificationError("RIGID_ATTACHMENT_LOADOUT_UNKNOWN")

    groups: dict[str, str] = {}
    for aid in selected:
        group = str(by_id[aid].variant_group_id)
        if group in groups:
            raise QualificationError("RIGID_ATTACHMENT_LOADOUT_GROUP_CONFLICT")
        groups[group] = aid

    selected_set = set(selected)
    hidden_components: set[int] = set()
    for binding in bindings:
        if str(binding.attachment_id) not in selected_set:
            hidden_components.update(map(int, binding.component_ids))

    vertex_visible = ~np.isin(
        components,
        np.asarray(sorted(hidden_components), dtype=np.int64),
    )
    return np.all(vertex_visible[faces], axis=1)


def rigid_attachment_edge_report_v1(
    rest_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    vertex_component_ids: np.ndarray,
    binding: RigidAttachmentBindingV1IR,
) -> dict:
    rest = np.asarray(rest_vertices, dtype=np.float64)
    poses = np.asarray(posed_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    components = np.asarray(vertex_component_ids, dtype=np.int64).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != rest.shape:
        raise QualificationError("RIGID_ATTACHMENT_POSE_SHAPE_INVALID")

    vertex_mask = np.isin(components, np.asarray(binding.component_ids, dtype=np.int64))
    face_mask = np.all(vertex_mask[faces], axis=1)
    if not bool(face_mask.any()):
        raise QualificationError("RIGID_ATTACHMENT_FACE_ACCOUNTING_EMPTY")
    ff = faces[face_mask]
    edges = np.concatenate((ff[:, (0, 1)], ff[:, (1, 2)], ff[:, (2, 0)]), axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    rest_length = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    if float(np.min(rest_length)) <= 1.0e-12:
        raise QualificationError("RIGID_ATTACHMENT_DEGENERATE_EDGE")

    worst = 0.0
    for posed in poses:
        length = np.linalg.norm(posed[edges[:, 1]] - posed[edges[:, 0]], axis=1)
        worst = max(worst, float(np.max(np.abs(length / rest_length - 1.0))))
    return {
        "attachment_id": str(binding.attachment_id),
        "joint_id": str(binding.joint_id),
        "face_count": int(len(ff)),
        "edge_count": int(len(edges)),
        "max_abs_edge_ratio_minus_one": float(worst),
        "rigid_edge_preservation_pass": bool(worst <= 1.0e-9),
    }
