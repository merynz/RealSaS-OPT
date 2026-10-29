"""Truth139 geometry supervision over the unchanged IRIS V3.1 dense carrier.

Direct F* remains the primary signed-geometry authority.  This module adds two bounded pieces of
training evidence that were deliberately preregistered before Truth139 fitting:

* one-sided wrong-sign stabilization only in the canonical near-zero F* band;
* world-space surface-normal evidence at exact source-ray hits.

The normal target is not substituted for F* and is never an observation input.  It teaches the
shared geometry state local surface orientation while TESSA/Compiler retain topology/mesh authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from models.iris.v5.direct_fstar_fit_v5 import DirectFStarFitPolicyV5, direct_fstar_loss_v5
from models.iris.v5.objective_tail_repair_v5 import near_zero_wrong_sign_loss_v5


PASS_RAY = "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION"


@dataclass(frozen=True)
class SurfaceNormalTruthV1:
    xyz: np.ndarray
    normal_world: np.ndarray
    view_index: np.ndarray
    face_index: np.ndarray
    valid: np.ndarray


def _normalization(value: Mapping[str, object]) -> tuple[np.ndarray, float]:
    schema = value.get("schema_version") or value.get("schema")
    if schema != "RealSaS.NormalizationDomainIR.v1":
        raise ValueError("IRIS_TRUTH139_NORMALIZATION_SCHEMA")
    center = np.asarray(value.get("center_xyz"), dtype=np.float64)
    half = float(value.get("half_extent", 0.0))
    if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(half) or half <= 0:
        raise ValueError("IRIS_TRUTH139_NORMALIZATION_INVALID")
    return center, half


def build_front_surface_normal_truth_v1(
    *,
    rest_vertices,
    faces,
    normalization: Mapping[str, object],
    packed_measurement: Mapping[str, object],
) -> SurfaceNormalTruthV1:
    """Derive sparse multi-view normal targets from already-qualified exact ray stacks.

    `canonical_hits` are depth-sorted by the geometric ray oracle.  Only the frontmost hit from a
    complete, tie-free, renderer-qualified ray is used, matching an ordinary visible normal map.
    Triangle winding is preserved as authored; orientation qualification remains a corpus-admission
    responsibility rather than being silently flipped per camera.
    """
    vertices = np.asarray(rest_vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or tri.ndim != 2
        or tri.shape[1:] != (3,)
        or not len(tri)
        or np.min(tri) < 0
        or np.max(tri) >= len(vertices)
    ):
        raise ValueError("IRIS_TRUTH139_NORMAL_MESH_INVALID")
    center, half = _normalization(normalization)
    if packed_measurement.get("schema") != "RealSaS.SourceRayAppearanceMeasurementCandidate.v1":
        raise ValueError("IRIS_TRUTH139_NORMAL_PACKED_SCHEMA")

    face_vertices = vertices[tri]
    face_cross = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    face_norm = np.linalg.norm(face_cross, axis=1)
    face_valid = np.isfinite(face_norm) & (face_norm > 1.0e-12)
    face_normal = np.zeros_like(face_cross)
    face_normal[face_valid] = face_cross[face_valid] / face_norm[face_valid, None]

    xyz_rows, normal_rows, view_rows, face_rows, valid_rows = [], [], [], [], []
    for ray in packed_measurement.get("rays") or ():
        measurement = dict(ray.get("measurement") or {})
        if measurement.get("status") != PASS_RAY or measurement.get("valid") is not True:
            continue
        if ray.get("complete") is not True or ray.get("has_depth_tie") is not False:
            continue
        hits = list(ray.get("canonical_hits") or ())
        if not hits:
            continue
        hit = hits[0]
        face = int(hit.get("face_index", -1))
        bary = np.asarray(hit.get("barycentric"), dtype=np.float64)
        if face < 0 or face >= len(tri) or bary.shape != (3,) or not np.isfinite(bary).all():
            raise ValueError("IRIS_TRUTH139_NORMAL_HIT_INVALID")
        if not np.isclose(float(bary.sum()), 1.0, atol=1.0e-7, rtol=0.0):
            raise ValueError("IRIS_TRUTH139_NORMAL_BARYCENTRIC_INVALID")
        point = np.sum(face_vertices[face] * bary[:, None], axis=0)
        xyz_rows.append((point - center) / half)
        normal_rows.append(face_normal[face])
        view_rows.append(int(ray.get("view_index", -1)))
        face_rows.append(face)
        valid_rows.append(bool(face_valid[face]))

    if not xyz_rows:
        raise ValueError("IRIS_TRUTH139_NORMAL_NO_QUALIFIED_VISIBLE_HITS")
    xyz = np.asarray(xyz_rows, dtype=np.float32)
    normal = np.asarray(normal_rows, dtype=np.float32)
    view = np.asarray(view_rows, dtype=np.int64)
    face_index = np.asarray(face_rows, dtype=np.int64)
    valid = np.asarray(valid_rows, dtype=np.bool_)
    if np.any(view < 0) or np.any(view >= 8):
        raise ValueError("IRIS_TRUTH139_NORMAL_VIEW_INVALID")
    if not np.any(valid):
        raise ValueError("IRIS_TRUTH139_NORMAL_ALL_DEGENERATE")
    return SurfaceNormalTruthV1(
        xyz=xyz,
        normal_world=normal,
        view_index=view,
        face_index=face_index,
        valid=valid,
    )


def surface_normal_cosine_loss_v1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.shape[-1] != 3 or valid.shape != predicted.shape[:-1]:
        raise ValueError("IRIS_TRUTH139_NORMAL_LOSS_SHAPE")
    mask = valid.bool()
    if not bool(mask.any()):
        raise ValueError("IRIS_TRUTH139_NORMAL_LOSS_EMPTY")
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("IRIS_TRUTH139_NORMAL_LOSS_NONFINITE")
    p = F.normalize(predicted.float(), dim=-1, eps=1.0e-8)
    t = F.normalize(target.float(), dim=-1, eps=1.0e-8)
    target_norm = torch.linalg.vector_norm(target.float(), dim=-1)
    if bool(torch.any(mask & (target_norm <= 1.0e-8))):
        raise ValueError("IRIS_TRUTH139_NORMAL_TARGET_ZERO")
    cosine = torch.sum(p * t, dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cosine[mask]).mean()


def truth139_geometry_objective_v1(
    *,
    predicted_field: torch.Tensor,
    target_fstar: torch.Tensor,
    predicted_normal: torch.Tensor,
    target_normal: torch.Tensor,
    normal_valid: torch.Tensor,
    field_clamp: float,
    huber_beta: float,
    near_zero_band: float,
    sign_weight: float,
    normal_weight: float,
) -> dict[str, torch.Tensor | int]:
    if not math.isfinite(float(sign_weight)) or float(sign_weight) <= 0.0:
        raise ValueError("IRIS_TRUTH139_SIGN_WEIGHT_INVALID")
    if not math.isfinite(float(normal_weight)) or float(normal_weight) <= 0.0:
        raise ValueError("IRIS_TRUTH139_NORMAL_WEIGHT_INVALID")
    direct = direct_fstar_loss_v5(
        predicted_field,
        target_fstar,
        policy=DirectFStarFitPolicyV5(field_clamp=float(field_clamp), huber_beta=float(huber_beta)),
    )
    sign = near_zero_wrong_sign_loss_v5(
        predicted_field,
        target_fstar,
        band=float(near_zero_band),
    )
    normal = surface_normal_cosine_loss_v1(predicted_normal, target_normal, normal_valid)
    total = direct["total"] + float(sign_weight) * sign["total"] + float(normal_weight) * normal
    return {
        "total": total,
        "direct_fstar": direct["total"],
        "near_zero_wrong_sign": sign["total"],
        "normal_cosine": normal,
        "sign_eligible_count": int(sign["eligible_count"]),
        "sign_wrong_count": int(sign["wrong_sign_count"]),
        "normal_valid_count": int(normal_valid.bool().sum().detach().cpu()),
    }
