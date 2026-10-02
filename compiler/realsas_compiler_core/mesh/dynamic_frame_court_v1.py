from __future__ import annotations

"""Intrinsic per-frame mechanical conditioning; independent of camera projection."""

import numpy as np

from .deformation_stress_v2 import _triangle_metrics_batch_exact
from ..product_authority_v1 import validate_mesh_qualification_policy
from ..types import QualificationError


def measure_dynamic_frame_geometry_v1(*, rest, posed, faces, policy):
    validate_mesh_qualification_policy(policy)
    rest=np.asarray(rest,dtype=np.float64)
    posed=np.asarray(posed,dtype=np.float64)
    faces=np.asarray(faces)
    if (rest.ndim!=2 or rest.shape[1]!=3 or posed.shape!=rest.shape
            or not np.isfinite(rest).all() or not np.isfinite(posed).all()
            or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces)
            or faces.dtype.kind not in "iu" or np.any(faces<0) or np.any(faces>=len(rest))):
        raise QualificationError("DYNAMIC_FRAME_COURT_INPUT_INVALID")
    area, condition, edge_min, edge_max, smin = _triangle_metrics_batch_exact(rest, posed, faces)
    finite = np.isfinite(area) & np.isfinite(condition) & np.isfinite(edge_min) & np.isfinite(edge_max) & np.isfinite(smin)
    masks = {
        "NONFINITE_DEFORMATION_METRIC": ~finite,
        "DYNAMIC_AREA_RATIO_BELOW_MIN": area < policy.g3_min_dynamic_area_ratio,
        "DYNAMIC_AREA_RATIO_ABOVE_MAX": area > policy.g3_max_dynamic_area_ratio,
        "DYNAMIC_CONDITION_NUMBER_ABOVE_MAX": condition > policy.g3_max_dynamic_condition_number,
    }
    bad = np.logical_or.reduce(tuple(masks.values()))
    bad_ids = np.flatnonzero(bad)
    # Stable deterministic diagnostic ranking, not a substitute for any hard gate.
    ranked = sorted(map(int, bad_ids), key=lambda i: (-float(np.nan_to_num(condition[i], nan=np.inf)), i))
    def number(value):
        return float(value) if np.isfinite(value) else None
    return {
        "schema": "RealSaS.DynamicFrameGeometryCourt.v1",
        "policy_hash": policy.qualification_policy_lineage_hash,
        "passed": not bool(np.any(bad)),
        "face_count": int(len(faces)),
        "bad_face_count": int(len(bad_ids)),
        "failure_counts": {k: int(np.count_nonzero(v)) for k, v in masks.items()},
        "bad_face_indices": bad_ids.tolist(),
        "minimum_area_ratio": number(np.min(area)),
        "maximum_area_ratio": number(np.max(area)),
        "maximum_condition_number": number(np.max(condition)),
        "minimum_edge_ratio": number(np.min(edge_min)),
        "maximum_edge_ratio": number(np.max(edge_max)),
        "worst_faces": [
            {"face_index": i, "area_ratio": number(area[i]), "condition": number(condition[i]),
             "edge_min": number(edge_min[i]), "edge_max": number(edge_max[i])}
            for i in ranked[:64]
        ],
        "projected_orientation_used": False,
    }


def sampled_pose_hash_v1(*, clip_id, time_seconds, rest_frame_set_hash, skin_matrices, posed):
    """Bind a sampled pose, never confuse a rest-frame hash with motion evidence."""
    import hashlib
    from ..hashing import content_sha256

    xyz = np.asarray(posed, dtype='<f8')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise QualificationError('SAMPLED_POSE_POSITION_INVALID')
    matrices = {}
    for joint_id, matrix in sorted(skin_matrices.items()):
        matrix = np.asarray(matrix, dtype='<f8')
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise QualificationError('SAMPLED_POSE_MATRIX_INVALID')
        matrices[str(joint_id)] = matrix.tolist()
    if not matrices or not np.isfinite(time_seconds):
        raise QualificationError('SAMPLED_POSE_BINDING_INVALID')
    return content_sha256({
        'schema': 'RealSaS.SampledPoseBinding.v1',
        'clip_id': str(clip_id), 'time_seconds': float(time_seconds),
        'rest_frame_set_hash': str(rest_frame_set_hash),
        'skin_matrices': matrices, 'posed_shape': list(xyz.shape),
        'posed_xyz_f64le_sha256': hashlib.sha256(xyz.tobytes(order='C')).hexdigest(),
    })
