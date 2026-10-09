"""Connected compile-time 2D pose derived from the sealed G/motion witness.

The presentation style retains projected setup offsets and the existing proper
camera-twist rotations. Joint origins are transported through the actual frozen
parent tree, rather than preserving unrelated projected pivots. Explicit local
translation in the sealed motion is preserved. This derives a drawing pose; it
does not infer G, fit W, create missing material, or certify visual seam coverage.
"""
from dataclasses import dataclass

import numpy as np

from .camera_geometry_v2 import project_points_xyz_v3
from .types import QualificationError
from .visual_attachment_motion_v1 import ATTACHMENT_OPERATOR_ID, slot_rigid_transform_2d


OPERATOR_ID = "FROZEN_G_CONNECTED_CAMERA_TWIST_PRESENTATION_POSE_V1"
POLICY = {
    "schema": "RealSaS.ConnectedPresentationPosePolicy.v1",
    "operator_id": OPERATOR_ID,
    "hierarchy_authority": "EXACT_FROZEN_G_PARENT_INDICES",
    "rotation_style": "NEAREST_PROPER_2D_CAMERA_TWIST_OF_SEALED_GLOBAL_ROTATION",
    "length_style": "PROJECTED_SETUP_OFFSETS_TRANSPORTED_BY_PARENT_2D_ROTATION",
    "origin_style": "CONNECTED_PARENT_ORIGIN_PLUS_SEALED_LOCAL_TRANSLATION",
    "root_transport": "EXACT_PROJECTED_SEALED_ROOT_ORIGIN",
    "ambiguous_twist": "FAIL_CLOSED__NO_FRAME_HISTORY_FALLBACK",
    "maximum_joint_relation_residual_px": 1e-8,
    "mechanical_state_mutation_authorized": False,
    "runtime_pose_inference_authorized": False,
    "appearance_completion_authorized": False,
}


@dataclass(frozen=True)
class ConnectedPresentationPalette:
    rotations: np.ndarray
    translations: np.ndarray


def _geometry(axis_positions_source, axis_parents, skin_matrices_source):
    axis = np.asarray(axis_positions_source, dtype=np.float64)
    original_parents = np.asarray(axis_parents)
    matrices = np.asarray(skin_matrices_source, dtype=np.float64)
    if (axis.ndim != 2 or axis.shape[1] != 3 or not len(axis)
            or original_parents.shape != (len(axis),)
            or original_parents.dtype.kind not in "iu"
            or matrices.shape != (len(axis), 4, 4)
            or not np.isfinite(axis).all() or not np.isfinite(matrices).all()):
        raise QualificationError("PRESENTATION_CONNECTED_POSE_INPUT_INVALID")
    if (not np.allclose(matrices[:, 3, :], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or np.max(np.abs(matrices[:, :3, :3].transpose(0, 2, 1) @ matrices[:, :3, :3] - np.eye(3))) > 1e-6
            or np.max(np.abs(np.linalg.det(matrices[:, :3, :3]) - 1)) > 1e-6):
        raise QualificationError("PRESENTATION_CONNECTED_SEALED_TRANSFORM_INVALID")
    if np.any(original_parents < -1) or np.any(original_parents >= len(axis)):
        raise QualificationError("PRESENTATION_CONNECTED_POSE_TREE_INVALID")
    parents = original_parents.astype(np.int64)
    if (np.any(parents < -1) or np.any(parents >= len(axis))
            or np.count_nonzero(parents == -1) != 1
            or np.any(parents == np.arange(len(axis)))):
        raise QualificationError("PRESENTATION_CONNECTED_POSE_TREE_INVALID")
    remaining, order = set(range(len(axis))), []
    while remaining:
        ready = [i for i in sorted(remaining) if parents[i] < 0 or parents[i] in order]
        if not ready:
            raise QualificationError("PRESENTATION_CONNECTED_POSE_TREE_CYCLE")
        order.extend(ready)
        remaining.difference_update(ready)
    return axis, parents, matrices, order


def _transport(axis, parents, matrices, camera):
    rest = project_points_xyz_v3(axis, camera)[:, :2]
    own_xyz = np.einsum("jab,jb->ja", matrices, np.c_[axis, np.ones(len(axis))])[:, :3]
    own_xy = project_points_xyz_v3(own_xyz, camera)[:, :2]
    local = np.zeros_like(rest)
    children = np.flatnonzero(parents >= 0)
    parent_xyz = np.einsum("jab,jb->ja", matrices[parents[children]],
                           np.c_[axis[children], np.ones(len(children))])[:, :3]
    # Subtract projections, including the same viewport offset and scale. This
    # isolates explicit local translation; own rotations do not move a joint's
    # origin. The sealed witness remains authoritative for this component.
    local[children] = own_xy[children] - project_points_xyz_v3(parent_xyz, camera)[:, :2]
    return rest, own_xy, local


def compile_connected_palette(*, axis_positions_source, axis_parents,
                              skin_matrices_source, camera):
    axis, parents, matrices, order = _geometry(
        axis_positions_source, axis_parents, skin_matrices_source)
    rest, own, local = _transport(axis, parents, matrices, camera)
    rotations = np.asarray([slot_rigid_transform_2d(
        slot_rest_xyz=axis[j], skin_matrix_source=matrices[j], camera=camera)[0]
        for j in range(len(axis))])
    origins = np.zeros_like(rest)
    for j in order:
        p = int(parents[j])
        origins[j] = own[j] if p < 0 else (
            origins[p] + rotations[p] @ (rest[j] - rest[p]) + local[j])
    translations = origins - np.einsum("jab,jb->ja", rotations, rest)
    return ConnectedPresentationPalette(rotations, translations)


def apply_connected_palette(field, *, rest_source_xy, coefficients, palette):
    result = np.asarray(field, dtype=np.float64).copy()
    rest = np.asarray(rest_source_xy, dtype=np.float64)
    blend = np.asarray(coefficients, dtype=np.float64)
    rotations, translations = _palette_arrays(palette)
    if (result.shape != (len(rest), 3) or rest.shape != (len(rest), 2)
            or blend.shape != (len(rest), len(rotations))
            or not all(np.isfinite(a).all() for a in (result, rest, blend))
            or np.any(blend < 0)
            or not np.allclose(blend.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise QualificationError("PRESENTATION_CONNECTED_BLEND_INPUT_INVALID")
    # Canonical coefficients are consumed unchanged. Depth belongs to the
    # independent drawing/depth contract and is copied exactly.
    result[:, :2] = np.einsum("nj,jna->na", blend,
        np.einsum("jab,nb->jna", rotations, rest) + translations[:, None, :], optimize=True)
    return result


def apply_connected_attachment_slots(field, *, rest_source_xy,
                                     vertex_attachment_owner, attachments, palette):
    """Use the same connected pose for rigid equipped art and the body slots.

    The target slot mapping is already qualified input, not inferred from its
    current screen position or from an unarmed source motion preset. Prop depth
    remains with the independent visibility/depth authority.
    """
    result = np.asarray(field, dtype=np.float64).copy()
    rest = np.asarray(rest_source_xy, dtype=np.float64)
    owners = np.asarray(vertex_attachment_owner)
    rotations, translations = _palette_arrays(palette)
    items = attachments["attachments"]
    if (rest.ndim != 2 or rest.shape[1] != 2 or result.shape != (len(rest), 3)
            or owners.shape != (len(rest),) or owners.dtype.kind not in "iu"
            or not np.isfinite(rest).all() or not np.isfinite(result).all()
            or sorted(int(a["attachment_index"]) for a in items) != list(range(1, len(items) + 1))
            or np.any(owners < 0) or np.any(owners > len(items))):
        raise QualificationError("PRESENTATION_CONNECTED_ATTACHMENT_BINDING_INVALID")
    for item in items:
        if item.get("presentation_motion_model") != ATTACHMENT_OPERATOR_ID:
            raise QualificationError("PRESENTATION_CONNECTED_ATTACHMENT_MODEL_UNSUPPORTED")
        slot = item["target_slot_raw_index_fit_only"]
        if isinstance(slot, bool) or not isinstance(slot, (int, np.integer)) or not 0 <= slot < len(rotations):
            raise QualificationError("PRESENTATION_CONNECTED_ATTACHMENT_SLOT_INVALID")
        selected = owners == item["attachment_index"]
        result[selected, :2] = rest[selected] @ rotations[slot].T + translations[slot]
    return result


def _palette_arrays(palette):
    rotations = np.asarray(palette.rotations, dtype=np.float64)
    translations = np.asarray(palette.translations, dtype=np.float64)
    if (rotations.ndim != 3 or rotations.shape[1:] != (2, 2) or not len(rotations)
            or translations.shape != (len(rotations), 2)
            or not np.isfinite(rotations).all() or not np.isfinite(translations).all()
            or np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(2))) > 1e-8
            or np.max(np.abs(np.linalg.det(rotations) - 1)) > 1e-8):
        raise QualificationError("PRESENTATION_CONNECTED_PALETTE_INVALID")
    return rotations, translations


def prove_connected_palette(palette, *, axis_positions_source, axis_parents,
                            skin_matrices_source, camera):
    """Measure parent/child relations independently of palette compilation.

    This proof can reject the old independently fitted palette even when its
    own operator replay, triangles and raster parity all pass. It is a palette
    relation proof, not proof that separately repaired visual charts are welded.
    """
    axis, parents, matrices, _ = _geometry(
        axis_positions_source, axis_parents, skin_matrices_source)
    rotations, translations = _palette_arrays(palette)
    if len(rotations) != len(axis):
        raise QualificationError("PRESENTATION_CONNECTED_PALETTE_TREE_BINDING_DRIFT")
    rest, own, local = _transport(axis, parents, matrices, camera)
    origins = np.einsum("jab,jb->ja", rotations, rest) + translations
    root = int(np.flatnonzero(parents == -1)[0])
    children = np.flatnonzero(parents >= 0)
    parent_images = np.einsum("jab,jb->ja", rotations[parents[children]], rest[children]) + translations[parents[children]]
    joint_residual = float(np.max(np.linalg.norm(
        origins[children] - parent_images - local[children], axis=1), initial=0))
    root_residual = float(np.linalg.norm(origins[root] - own[root]))
    screen = np.asarray([camera.right, -np.asarray(camera.screen_up)])
    projected = screen[None] @ matrices[:, :3, :3] @ screen.T[None]
    c = projected[:, 0, 0] + projected[:, 1, 1]
    s = projected[:, 1, 0] - projected[:, 0, 1]
    norms = np.hypot(c, s)
    if np.any(norms < 1e-8):
        raise QualificationError("PRESENTATION_CONNECTED_CAMERA_TWIST_UNOBSERVABLE")
    expected = np.stack((c, -s, s, c), axis=1).reshape(-1, 2, 2) / norms[:, None, None]
    rotation_residual = float(np.max(np.abs(rotations - expected)))
    tolerance = POLICY["maximum_joint_relation_residual_px"]
    return {
        "connected_palette_relations_passed": joint_residual <= tolerance and root_residual <= tolerance and rotation_residual <= 1e-8,
        "maximum_parent_child_relation_residual_px": joint_residual,
        "maximum_root_transport_residual_px": root_residual,
        "maximum_rotation_style_residual": rotation_residual,
        "parent_child_relation_count": len(children),
        "visual_chart_contact_coverage_qualified": False,
    }
