"""Compile rigid 2D artwork in a canonical target slot's local frame.

Source chart cuts are texture/addressing cuts, not independent motion owners.
Every vertex of a rigid attachment therefore shares one proper 2D rotation
and the projected canonical slot translation. The view-normal twist of the
sealed 3D slot rotation supplies that rotation; out-of-plane foreshortening is
not a deformation of an authored 2D rigid attachment. Canonical depth remains
the independently bound M-surface field. This is a compile-time presentation
operator, not a change to M, G, W, retargeting, or runtime skinning.
"""
import numpy as np

from .camera_geometry_v2 import project_points_xyz_v3
from .types import QualificationError
from .visual_domain_v2 import POLICY as DOMAIN_POLICY

OPERATOR_ID = "SOURCE_CHART_HARMONIC_WITH_SLOT_RIGID_2D_V3"
ATTACHMENT_OPERATOR_ID = "TARGET_SLOT_LOCAL_RIGID_2D_CAMERA_TWIST_V1"
POLICY = {**DOMAIN_POLICY,
    "schema": "RealSaS.VisualDeformationOperatorPolicy.v3",
    "operator_id": OPERATOR_ID,
    "attachment_motion_operator_id": ATTACHMENT_OPERATOR_ID,
    "attachment_xy_contract": "ONE_PROPER_2D_TRANSFORM_PER_TARGET_ATTACHMENT",
    "attachment_depth_contract": "UNCHANGED_CANONICAL_SURFACE_HARMONIC_DEPTH",
    "maximum_carrier_slot_relative_residual": 0.001,
    "minimum_camera_twist_norm": 1e-8,
    "maximum_slot_rotation_orthogonality_residual": 1e-6,
}


def slot_rigid_transform_2d(*, slot_rest_xyz, skin_matrix_source, camera):
    """Return R,t such that source pixels move as p @ R.T + t."""
    m = np.asarray(skin_matrix_source, dtype=np.float64)
    pivot = np.asarray(slot_rest_xyz, dtype=np.float64)
    if (m.shape != (4, 4) or pivot.shape != (3,) or not np.isfinite(m).all()
            or not np.isfinite(pivot).all()
            or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or np.max(np.abs(m[:3, :3].T @ m[:3, :3] - np.eye(3)))
                > POLICY["maximum_slot_rotation_orthogonality_residual"]
            or abs(np.linalg.det(m[:3, :3]) - 1) > 1e-6):
        raise QualificationError("PRESENTATION_ATTACHMENT_SLOT_TRANSFORM_INVALID")
    screen = np.asarray([camera.right, -np.asarray(camera.screen_up)])
    projected_rotation = screen @ m[:3, :3] @ screen.T
    c = projected_rotation[0, 0] + projected_rotation[1, 1]
    s = projected_rotation[1, 0] - projected_rotation[0, 1]
    norm = float(np.hypot(c, s))
    if norm < POLICY["minimum_camera_twist_norm"]:
        # A 180-degree out-of-plane swing has no uniquely defined view twist.
        # Do not choose a silent fallback or reuse a previous frame.
        raise QualificationError("PRESENTATION_ATTACHMENT_CAMERA_TWIST_UNOBSERVABLE")
    rotation = np.array([[c, -s], [s, c]]) / norm
    posed_pivot = (m @ np.r_[pivot, 1])[:3]
    rest_xy, posed_xy = project_points_xyz_v3(np.stack((pivot, posed_pivot)), camera)[:, :2]
    return rotation, posed_xy - rotation @ rest_xy


def attachment_slot_carrier_residual(attachments, *, rest_xyz, posed_xyz, skin_matrices_source):
    """Check the selected rigid target art actually follows its declared slot."""
    rest, posed = np.asarray(rest_xyz), np.asarray(posed_xyz)
    matrices = np.asarray(skin_matrices_source)
    if rest.shape != posed.shape or rest.ndim != 2 or rest.shape[1] != 3:
        raise QualificationError("PRESENTATION_ATTACHMENT_CARRIER_SHAPE_DRIFT")
    maximum = 0.0
    for item in attachments["attachments"]:
        if item.get("presentation_motion_model") != ATTACHMENT_OPERATOR_ID:
            continue
        ids = np.asarray(item["vertex_indices"], dtype=int)
        slot = int(item["target_slot_raw_index_fit_only"])
        if (not len(ids) or np.any(ids < 0) or np.any(ids >= len(rest))
                or slot < 0 or slot >= len(matrices)):
            raise QualificationError("PRESENTATION_ATTACHMENT_SLOT_BINDING_INVALID")
        scale = float(np.linalg.norm(np.ptp(rest[ids], axis=0)))
        if scale <= 1e-12:
            raise QualificationError("PRESENTATION_ATTACHMENT_REST_SCALE_DEGENERATE")
        expected = (np.c_[rest[ids], np.ones(len(ids))] @ matrices[slot].T)[:, :3]
        residual = float(np.max(np.linalg.norm(expected - posed[ids], axis=1))) / scale
        if not np.isfinite(residual):
            raise QualificationError("PRESENTATION_ATTACHMENT_SLOT_RESIDUAL_NONFINITE")
        maximum = max(maximum, residual)
    return maximum


def evaluate_attachment_motion(field, *, rest_source_xy, vertex_attachment_owner,
                               attachments, axis_positions_source,
                               skin_matrices_source, camera):
    result = np.asarray(field, dtype=np.float64).copy()
    rest = np.asarray(rest_source_xy, dtype=np.float64)
    owners = np.asarray(vertex_attachment_owner, dtype=np.int32)
    axis = np.asarray(axis_positions_source, dtype=np.float64)
    matrices = np.asarray(skin_matrices_source, dtype=np.float64)
    items = attachments["attachments"]
    if (result.shape != (len(rest), 3) or rest.shape != (len(rest), 2)
            or owners.shape != (len(rest),) or not np.isfinite(result).all()
            or axis.ndim != 2 or axis.shape[1] != 3
            or matrices.shape != (len(axis), 4, 4)
            or sorted(int(a["attachment_index"]) for a in items) != list(range(1, len(items)+1))
            or np.any(owners < 0) or np.any(owners > len(items))):
        raise QualificationError("PRESENTATION_ATTACHMENT_MOTION_BINDING_INVALID")
    for item in items:
        if item.get("presentation_motion_model") != ATTACHMENT_OPERATOR_ID:
            continue
        slot = int(item["target_slot_raw_index_fit_only"])
        if slot < 0 or slot >= len(axis):
            raise QualificationError("PRESENTATION_ATTACHMENT_SLOT_BINDING_INVALID")
        selected = owners == item["attachment_index"]
        rotation, translation = slot_rigid_transform_2d(slot_rest_xyz=axis[slot],
            skin_matrix_source=matrices[slot], camera=camera)
        result[selected, :2] = rest[selected] @ rotation.T + translation
    return result
