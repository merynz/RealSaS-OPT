"""Slot-owned 2.5D depth for rigid target artwork.

The source art is a camera-facing rigid object. Its canonical rest surface relief
is frozen at bind time; camera twist rotates XY and the same canonical slot moves
the entire relief in depth. There is no animated mechanical depth per chart and
no added depth bias or triangle-enumeration draw order. Overlap still uses the
canonical ascending depth contract and ambiguous ties still fail closed.
"""
import numpy as np

from .camera_geometry_v2 import project_points_xyz_v3
from .types import QualificationError
from .visual_attachment_motion_v1 import ATTACHMENT_OPERATOR_ID, slot_rigid_transform_2d
from .visual_motion_blend_v1 import POLICY as MOTION_POLICY

OPERATOR_ID = "SOURCE_CHART_CANONICAL_2D_BLEND_WITH_SLOT_OWNED_DEPTH_V5"
DEPTH_OPERATOR_ID = "TARGET_SLOT_CAMERA_RIGID_2_5D_FROZEN_REST_RELIEF_V1"
POLICY = {**MOTION_POLICY,
    "schema": "RealSaS.VisualDeformationOperatorPolicy.v5",
    "operator_id": OPERATOR_ID,
    "attachment_depth_contract": DEPTH_OPERATOR_ID,
    "attachment_overlap_contract": "CANONICAL_REST_RELIEF_PLUS_COMMON_SLOT_DEPTH__NO_DRAW_ORDER_BIAS",
    "maximum_attachment_owner_field_residual": 1e-7,
}


def evaluate_slot_owned_depth(field, *, rest_depths, vertex_attachment_owner,
                              attachments, axis_positions_source,
                              skin_matrices_source, camera):
    result = np.asarray(field, dtype=np.float64).copy()
    rest = np.asarray(rest_depths, dtype=np.float64)
    owners = np.asarray(vertex_attachment_owner, dtype=np.int32)
    axis = np.asarray(axis_positions_source, dtype=np.float64)
    matrices = np.asarray(skin_matrices_source, dtype=np.float64)
    items = attachments["attachments"]
    if (result.shape != (len(rest), 3) or rest.ndim != 1 or owners.shape != rest.shape
            or axis.ndim != 2 or axis.shape[1] != 3 or matrices.shape != (len(axis), 4, 4)
            or not all(np.isfinite(x).all() for x in (result, rest, axis, matrices))
            or np.any(rest <= 0) or np.any(owners < 0) or np.any(owners > len(items))
            or sorted(int(a["attachment_index"]) for a in items) != list(range(1, len(items)+1))):
        raise QualificationError("PRESENTATION_ATTACHMENT_DEPTH_BINDING_INVALID")
    for item in items:
        if item.get("presentation_motion_model") != ATTACHMENT_OPERATOR_ID:
            raise QualificationError("PRESENTATION_ATTACHMENT_DEPTH_REQUIRES_RIGID_SLOT_OWNER")
        slot = int(item["target_slot_raw_index_fit_only"])
        if not 0 <= slot < len(axis):
            raise QualificationError("PRESENTATION_ATTACHMENT_DEPTH_SLOT_INVALID")
        # Reuse the proper-slot and observable-camera checks of XY ownership.
        slot_rigid_transform_2d(slot_rest_xyz=axis[slot], skin_matrix_source=matrices[slot], camera=camera)
        posed = (matrices[slot] @ np.r_[axis[slot], 1])[:3]
        z = project_points_xyz_v3(np.stack((axis[slot], posed)), camera)[:, 2]
        selected = owners == item["attachment_index"]
        result[selected, 2] = rest[selected] + (z[1] - z[0])
    if np.any(result[:, 2] <= 0):
        raise QualificationError("PRESENTATION_ATTACHMENT_DEPTH_BEHIND_CAMERA")
    return result


def attachment_owner_frame_metrics(actual, *, rest_source_xy, rest_depths,
                                   vertex_attachment_owner, attachments,
                                   axis_positions_source, skin_matrices_source, camera):
    """Independently prove shared slot XY and invariant rest-relief depth.

    Actual depth is never used as a depth baseline. A wrong slot, animated surface
    depth, or frame-local per-piece offset therefore cannot certify itself.
    """
    actual = np.asarray(actual, dtype=np.float64)
    rest_xy = np.asarray(rest_source_xy, dtype=np.float64)
    owners = np.asarray(vertex_attachment_owner, dtype=np.int32)
    expected = evaluate_slot_owned_depth(np.c_[rest_xy, rest_depths],
        rest_depths=rest_depths, vertex_attachment_owner=owners, attachments=attachments,
        axis_positions_source=axis_positions_source, skin_matrices_source=skin_matrices_source, camera=camera)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise QualificationError("PRESENTATION_ATTACHMENT_OWNER_FRAME_INVALID")
    xy_residual = depth_residual = 0.0
    for item in attachments["attachments"]:
        selected = owners == item["attachment_index"]
        slot = int(item["target_slot_raw_index_fit_only"])
        rotation, translation = slot_rigid_transform_2d(slot_rest_xyz=axis_positions_source[slot],
            skin_matrix_source=skin_matrices_source[slot], camera=camera)
        xy = rest_xy[selected] @ rotation.T + translation
        xy_residual = max(xy_residual, float(np.max(np.abs(actual[selected, :2] - xy), initial=0)))
        depth_residual = max(depth_residual, float(np.max(np.abs(actual[selected, 2] - expected[selected, 2]), initial=0)))
    limit = POLICY["maximum_attachment_owner_field_residual"]
    return {"attachment_ownership_passed": xy_residual <= limit and depth_residual <= limit,
            "maximum_attachment_xy_owner_residual": xy_residual,
            "maximum_attachment_depth_owner_residual": depth_residual}
