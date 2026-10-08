"""Bounded source-chart rigidity repair for unsafe 2D presentation motion.

The canonical skin field remains the motion authority. This operator only projects
body chart motion that violates the already-sealed Stage45 geometry limits onto
one proper unit-scale SE(2) transform per connected source-art domain. Target
attachments are excluded because their slot-local rigid motion has separate
ownership. Repair selection is the union over the complete sealed clip/frame
witness and is then applied to every frame of each selected domain, avoiding
frame-local mode switching.
"""
from __future__ import annotations

import numpy as np

from .types import QualificationError

OPERATOR_ID = "BOUNDED_BODY_DOMAIN_SE2_REPAIR_V1"
POLICY = {
    "schema": "RealSaS.VisualMotionSafetyRepairPolicy.v1",
    "operator_id": OPERATOR_ID,
    "repair_scope": "BODY_CONNECTED_SOURCE_DOMAIN_ONLY",
    "selection_scope": "UNION_OVER_COMPLETE_SEALED_CLIP_FRAME_WITNESS",
    "application_scope": "ALL_FRAMES_OF_SELECTED_DOMAIN",
    "fit": "LEAST_SQUARES_PROPER_RIGID_SE2__UNIT_SCALE",
    "minimum_signed_area_ratio": 0.05,
    "maximum_jacobian_condition": 16.0,
    "catastrophic_edge_ratio": 4.0,
    "target_attachment_domains_excluded": True,
    "mechanical_state_mutation_authorized": False,
    "runtime_inference_authorized": False,
}



def _as_inputs(rest_positions, visual_faces, domain_id, vertex_attachment_owner, frames_xy):
    rest = np.asarray(rest_positions, dtype=np.float64)
    faces = np.asarray(visual_faces, dtype=np.int64)
    domains = np.asarray(domain_id, dtype=np.int32)
    owners = np.asarray(vertex_attachment_owner, dtype=np.int32)
    frames = np.asarray(frames_xy, dtype=np.float64)
    if (rest.ndim != 2 or rest.shape[1] != 2 or not np.isfinite(rest).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or np.any(faces < 0) or np.any(faces >= len(rest))
            or domains.shape != (len(rest),) or owners.shape != (len(rest),)
            or np.any(domains < 0) or np.any(owners < 0)
            or frames.ndim != 3 or frames.shape[1:] != rest.shape
            or not len(frames) or not np.isfinite(frames).all()):
        raise QualificationError("VISUAL_MOTION_SAFETY_INPUT_INVALID")
    if np.any(domains[faces] != domains[faces[:, :1]]):
        raise QualificationError("VISUAL_MOTION_SAFETY_FACE_CROSSES_DOMAIN")
    if np.any(owners[faces] != owners[faces[:, :1]]):
        raise QualificationError("VISUAL_MOTION_SAFETY_FACE_CROSSES_OWNER")
    for domain in np.unique(domains):
        if len(np.unique(owners[domains == domain])) != 1:
            raise QualificationError("VISUAL_MOTION_SAFETY_DOMAIN_OWNER_AMBIGUOUS")
    return rest, faces, domains, owners, frames


def _triangle_metrics(rest, frames, faces):
    rb = np.stack((rest[faces[:, 1]] - rest[faces[:, 0]],
                   rest[faces[:, 2]] - rest[faces[:, 0]]), axis=2)
    if np.any(np.abs(np.linalg.det(rb)) <= 1.0e-12):
        raise QualificationError("VISUAL_MOTION_SAFETY_REST_DEGENERATE")
    pb = np.stack((frames[:, faces[:, 1]] - frames[:, faces[:, 0]],
                   frames[:, faces[:, 2]] - frames[:, faces[:, 0]]), axis=3)
    jac = pb @ np.linalg.inv(rb)[None, :, :, :]
    area = np.linalg.det(jac)
    singular = np.linalg.svd(jac, compute_uv=False)
    condition = np.divide(singular[:, :, 0], singular[:, :, 1],
                          out=np.full(area.shape, np.inf),
                          where=singular[:, :, 1] > 1.0e-12)
    return area, condition


def _unique_edges(faces):
    return np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]])), axis=1), axis=0)


def select_repair_domains(*, rest_positions, visual_faces, domain_id,
                          vertex_attachment_owner, frames_xy):
    """Return body-domain IDs that violate any sealed Stage45 geometry limit."""
    rest, faces, domains, owners, frames = _as_inputs(
        rest_positions, visual_faces, domain_id, vertex_attachment_owner, frames_xy)
    selected = []
    diagnostics = []
    for domain in np.unique(domains):
        vertices = np.flatnonzero(domains == domain)
        owner = int(owners[vertices[0]])
        face_index = np.flatnonzero(domains[faces[:, 0]] == domain)
        if not len(face_index):
            raise QualificationError("VISUAL_MOTION_SAFETY_UNUSED_DOMAIN")
        if owner != 0:
            diagnostics.append({"domain_id": int(domain), "owner": owner,
                                "selected": False, "reason": "TARGET_ATTACHMENT_EXCLUDED"})
            continue
        local_faces = faces[face_index]
        area, condition = _triangle_metrics(rest, frames, local_faces)
        edges = _unique_edges(local_faces)
        rest_length = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
        if np.any(rest_length <= 1.0e-12):
            raise QualificationError("VISUAL_MOTION_SAFETY_ZERO_REST_EDGE")
        posed_length = np.linalg.norm(frames[:, edges[:, 0]] - frames[:, edges[:, 1]], axis=2)
        ratio = np.maximum(posed_length / rest_length[None, :],
                           rest_length[None, :] / np.maximum(posed_length, 1.0e-30))
        area_bad = int(np.count_nonzero(area < POLICY["minimum_signed_area_ratio"]))
        condition_bad = int(np.count_nonzero(condition > POLICY["maximum_jacobian_condition"]))
        edge_bad = int(np.count_nonzero(ratio > POLICY["catastrophic_edge_ratio"]))
        choose = area_bad > 0 or condition_bad > 0 or edge_bad > 0
        if choose:
            selected.append(int(domain))
        diagnostics.append({
            "domain_id": int(domain), "owner": 0, "selected": choose,
            "area_failure_count": area_bad,
            "condition_failure_count": condition_bad,
            "catastrophic_edge_failure_count": edge_bad,
            "minimum_signed_area_ratio": float(np.min(area)),
            "maximum_jacobian_condition": float(np.max(condition)),
            "maximum_edge_ratio": float(np.max(ratio)),
        })
    return np.asarray(selected, dtype=np.int32), diagnostics


def _fit_proper_rigid(rest, target):
    x = np.asarray(rest, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2 or not len(x):
        raise QualificationError("VISUAL_MOTION_SAFETY_RIGID_FIT_INVALID")
    cx = x.mean(axis=0)
    cy = y.mean(axis=0)
    X = x - cx
    Y = y - cy
    if float(np.max(np.linalg.norm(X, axis=1), initial=0.0)) <= 1.0e-12:
        rotation = np.eye(2)
    else:
        covariance = (Y.T @ X) / float(len(x))
        u, _, vt = np.linalg.svd(covariance)
        sign = -1.0 if np.linalg.det(u @ vt) < 0.0 else 1.0
        correction = np.diag([1.0, sign])
        rotation = u @ correction @ vt
    translation = cy - rotation @ cx
    if (not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or np.max(np.abs(rotation.T @ rotation - np.eye(2))) > 1.0e-8
            or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-8):
        raise QualificationError("VISUAL_MOTION_SAFETY_RIGID_FIT_NOT_PROPER")
    return rotation, translation


def apply_repair_domains(*, rest_positions, frame_xy, domain_id,
                         vertex_attachment_owner, repair_domain_ids):
    rest = np.asarray(rest_positions, dtype=np.float64)
    frame = np.asarray(frame_xy, dtype=np.float64)
    domains = np.asarray(domain_id, dtype=np.int32)
    owners = np.asarray(vertex_attachment_owner, dtype=np.int32)
    repair = np.asarray(repair_domain_ids, dtype=np.int32)
    if (rest.shape != frame.shape or rest.ndim != 2 or rest.shape[1] != 2
            or domains.shape != (len(rest),) or owners.shape != (len(rest),)
            or not np.isfinite(rest).all() or not np.isfinite(frame).all()
            or len(np.unique(repair)) != len(repair)):
        raise QualificationError("VISUAL_MOTION_SAFETY_APPLY_INVALID")
    result = frame.copy()
    for domain in repair.tolist():
        vertices = np.flatnonzero(domains == int(domain))
        if not len(vertices) or np.any(owners[vertices] != 0):
            raise QualificationError("VISUAL_MOTION_SAFETY_REPAIR_SCOPE_INVALID")
        rotation, translation = _fit_proper_rigid(rest[vertices], frame[vertices])
        result[vertices] = rest[vertices] @ rotation.T + translation
    return result


def compile_motion_repair(*, rest_positions, visual_faces, domain_id,
                          vertex_attachment_owner, clip_fields):
    """Compile one witness-wide repair mask and apply it to all supplied clips.

    clip_fields maps a stable clip key to an array [frame, vertex, xyz]. Depth is
    copied exactly; only body XY for selected domains is projected to SE(2).
    """
    if not clip_fields:
        raise QualificationError("VISUAL_MOTION_SAFETY_CLIPS_REQUIRED")
    keys = list(clip_fields)
    fields = {key: np.asarray(clip_fields[key], dtype=np.float64) for key in keys}
    first_shape = None
    xy_rows = []
    for key in keys:
        value = fields[key]
        if value.ndim != 3 or value.shape[2] != 3 or not len(value) or not np.isfinite(value).all():
            raise QualificationError("VISUAL_MOTION_SAFETY_CLIP_FIELD_INVALID")
        if first_shape is None:
            first_shape = value.shape[1]
        if value.shape[1] != first_shape:
            raise QualificationError("VISUAL_MOTION_SAFETY_CLIP_VERTEX_DRIFT")
        xy_rows.append(value[:, :, :2])
    repair, selection = select_repair_domains(
        rest_positions=rest_positions, visual_faces=visual_faces, domain_id=domain_id,
        vertex_attachment_owner=vertex_attachment_owner,
        frames_xy=np.concatenate(xy_rows, axis=0))
    repaired = {}
    max_projection_residual = 0.0
    for key in keys:
        value = fields[key].copy()
        for fi in range(len(value)):
            baseline = value[fi, :, :2].copy()
            value[fi, :, :2] = apply_repair_domains(
                rest_positions=rest_positions, frame_xy=baseline, domain_id=domain_id,
                vertex_attachment_owner=vertex_attachment_owner,
                repair_domain_ids=repair)
            if len(repair):
                mask = np.isin(np.asarray(domain_id, dtype=np.int32), repair)
                max_projection_residual = max(max_projection_residual,
                    float(np.max(np.linalg.norm(value[fi, mask, :2] - baseline[mask], axis=1), initial=0.0)))
        repaired[key] = value
    return {
        "repair_domain_ids": repair,
        "selection_diagnostics": selection,
        "repaired_fields": repaired,
        "maximum_baseline_projection_residual_px": max_projection_residual,
    }
