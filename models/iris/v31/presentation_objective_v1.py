"""IRIS V3.1 objectives derived from PresentationResponseTeacherV1.

The teacher package is deliberately richer than this objective. Deterministic
geometric order remains a Compiler/oracle quantity; it is not counted as learned
intelligence merely because a network can reconstruct it from xyz+camera.

The first learned presentation program uses:
* observation-conditioned RGBA only where exactly one measured source hit
  contributes to the observed ray; overlapping translucent per-hit colours remain
  teacher/oracle evidence rather than fake deterministic student labels;
* heteroscedastic conditional prediction of non-rigid response and normalized
  response energy, preserving hidden-mechanism ambiguity without source-scale
  leakage or a fake semantic class;
* bounded conditional source-over attenuation on jointly contributing same-ray
  hits.

No cape/hair/cloth/weapon/owner label exists anywhere in this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


MECHANICAL_CHANNELS_V31_PRESENTATION = (
    "nonrigid_response_mean",
    "nonrigid_response_log_scale",
    "response_energy_mean",
    "response_energy_log_scale",
)
PAIR_CHANNELS_V31_PRESENTATION = ("occlusion_attenuation_logit",)
PASS_RAY = "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION"


@dataclass(frozen=True)
class PresentationTargetBatchV1:
    mechanical_xyz: np.ndarray
    mechanical_target: np.ndarray
    mechanical_valid: np.ndarray
    appearance_xyz: np.ndarray
    appearance_camera_index: np.ndarray
    appearance_rgba: np.ndarray
    appearance_valid: np.ndarray
    pair_xyz_a: np.ndarray
    pair_xyz_b: np.ndarray
    pair_camera_index: np.ndarray
    pair_target: np.ndarray
    pair_valid: np.ndarray


def _mesh(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if (
        v.ndim != 2
        or v.shape[1:] != (3,)
        or not np.isfinite(v).all()
        or f.ndim != 2
        or f.shape[1:] != (3,)
        or not len(f)
        or np.min(f) < 0
        or np.max(f) >= len(v)
    ):
        raise ValueError("IRIS_V31_PRESENTATION_MESH_INVALID")
    return v, f


def _normalization(value: Mapping[str, object]):
    schema = value.get("schema_version") or value.get("schema")
    if (
        schema != "RealSaS.NormalizationDomainIR.v1"
        or (
            value.get("schema_version") is not None
            and value.get("schema") is not None
            and value.get("schema_version") != value.get("schema")
        )
    ):
        raise ValueError("IRIS_V31_PRESENTATION_NORMALIZATION_SCHEMA")
    center = np.asarray(value.get("center_xyz"), dtype=np.float64)
    half = float(value.get("half_extent", 0.0))
    if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(half) or half <= 0:
        raise ValueError("IRIS_V31_PRESENTATION_NORMALIZATION_INVALID")
    return center, half


def _xyz(point, center, half):
    value = (np.asarray(point, dtype=np.float64) - center) / half
    if value.shape[-1:] != (3,) or not np.isfinite(value).all():
        raise ValueError("IRIS_V31_PRESENTATION_QUERY_INVALID")
    return value


def build_presentation_targets_v1(
    *,
    rest_vertices,
    faces,
    normalization: Mapping[str, object],
    presentation_arrays: Mapping[str, np.ndarray],
    packed_measurement: Mapping[str, object],
    contribution_floor: float = 1.0e-6,
) -> PresentationTargetBatchV1:
    vertices, tri = _mesh(rest_vertices, faces)
    center, half = _normalization(normalization)
    if not math.isfinite(float(contribution_floor)) or contribution_floor < 0:
        raise ValueError("IRIS_V31_PRESENTATION_CONTRIBUTION_FLOOR")
    face_count = len(tri)
    valid = np.asarray(presentation_arrays["surface_face_valid"], dtype=np.bool_)
    nonrigid = np.asarray(presentation_arrays["surface_nonrigid_response"], dtype=np.float64)
    energy_source_units = np.asarray(presentation_arrays["surface_response_energy"], dtype=np.float64)
    if valid.shape != (face_count,) or nonrigid.shape != valid.shape or energy_source_units.shape != valid.shape:
        raise ValueError("IRIS_V31_PRESENTATION_MECHANICAL_LAYOUT")
    if not np.isfinite(nonrigid[valid]).all() or not np.isfinite(energy_source_units[valid]).all():
        raise ValueError("IRIS_V31_PRESENTATION_MECHANICAL_NONFINITE")
    centroid = vertices[tri].mean(axis=1)
    mechanical_xyz = _xyz(centroid, center, half)
    # nonrigid is metric-strain based and therefore dimensionless. Response
    # energy is a source-world displacement magnitude; normalize it by the same
    # admitted half extent used for spatial queries so FIT8 cannot leak authored
    # asset scale into a purportedly generic mechanics target.
    energy_normalized = energy_source_units / half
    mechanical_target = np.stack((nonrigid, energy_normalized), axis=-1)

    if packed_measurement.get("schema") != "RealSaS.SourceRayAppearanceMeasurementCandidate.v1":
        raise ValueError("IRIS_V31_PRESENTATION_PACKED_SCHEMA")
    appearance_xyz, appearance_view, appearance_rgba, appearance_valid = [], [], [], []
    pair_xyz_a, pair_xyz_b, pair_view, pair_target, pair_valid = [], [], [], [], []
    for ray in packed_measurement.get("rays") or ():
        measured = dict(ray.get("measurement") or {})
        if measured.get("status") != PASS_RAY or measured.get("valid") is not True:
            continue
        hits = list(ray.get("canonical_hits") or ())
        alpha = np.asarray(measured.get("effective_alpha"), dtype=np.float64)
        rgb = np.asarray(measured.get("straight_linear_rgb"), dtype=np.float64)
        contribution = np.asarray(measured.get("contribution"), dtype=np.float64)
        if alpha.shape != (len(hits),) or rgb.shape != (len(hits), 3) or contribution.shape != (len(hits),):
            raise ValueError("IRIS_V31_PRESENTATION_RAY_LAYOUT")
        if (
            not np.isfinite(alpha).all()
            or not np.isfinite(rgb).all()
            or not np.isfinite(contribution).all()
            or np.any(alpha < 0)
            or np.any(alpha > 1)
            or np.any(contribution < 0)
        ):
            raise ValueError("IRIS_V31_PRESENTATION_RAY_VALUES")
        contributing = contribution > contribution_floor
        directly_identifiable_rgba = int(np.count_nonzero(contributing)) == 1
        points = []
        for index, hit in enumerate(hits):
            face = int(hit["face_index"])
            bary = np.asarray(hit["barycentric"], dtype=np.float64)
            if face < 0 or face >= face_count or bary.shape != (3,) or not np.isclose(bary.sum(), 1.0, atol=1e-7, rtol=0):
                raise ValueError("IRIS_V31_PRESENTATION_HIT_ADDRESS")
            point = np.sum(vertices[tri[face]] * bary[:, None], axis=0)
            normalized = _xyz(point, center, half)
            points.append(normalized)
            appearance_xyz.append(normalized)
            appearance_view.append(int(ray["view_index"]))
            appearance_rgba.append([float(rgb[index, 0]), float(rgb[index, 1]), float(rgb[index, 2]), float(alpha[index])])
            # A standalone teacher render can factor multiple translucent layers,
            # but the observed composite does not uniquely identify each per-hit
            # colour. Deterministic RGBA supervision is admitted only when this
            # ray has exactly one measurable source-over contributor.
            appearance_valid.append(bool(directly_identifiable_rgba and contributing[index]))
        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                conditional = bool(contributing[i] and contributing[j])
                # Emit both directions. Geometric order itself stays outside the
                # learned objective; only bounded measured attenuation is learned
                # as conditional evidence when both endpoints contribute.
                for a, b, attenuation in (
                    (i, j, float(alpha[i])),
                    (j, i, 0.0),
                ):
                    pair_xyz_a.append(points[a])
                    pair_xyz_b.append(points[b])
                    pair_view.append(int(ray["view_index"]))
                    pair_target.append([attenuation])
                    pair_valid.append(conditional)
    return PresentationTargetBatchV1(
        mechanical_xyz=np.asarray(mechanical_xyz, dtype=np.float32).reshape(-1, 3),
        mechanical_target=np.asarray(mechanical_target, dtype=np.float32).reshape(-1, 2),
        mechanical_valid=np.asarray(valid, dtype=np.bool_),
        appearance_xyz=np.asarray(appearance_xyz, dtype=np.float32).reshape(-1, 3),
        appearance_camera_index=np.asarray(appearance_view, dtype=np.int64),
        appearance_rgba=np.asarray(appearance_rgba, dtype=np.float32).reshape(-1, 4),
        appearance_valid=np.asarray(appearance_valid, dtype=np.bool_),
        pair_xyz_a=np.asarray(pair_xyz_a, dtype=np.float32).reshape(-1, 3),
        pair_xyz_b=np.asarray(pair_xyz_b, dtype=np.float32).reshape(-1, 3),
        pair_camera_index=np.asarray(pair_view, dtype=np.int64),
        pair_target=np.asarray(pair_target, dtype=np.float32).reshape(-1, 1),
        pair_valid=np.asarray(pair_valid, dtype=np.bool_),
    )


def heteroscedastic_mechanical_loss_v1(raw: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL for conditional response; uncertainty is part of the output."""
    if raw.shape[-1] != len(MECHANICAL_CHANNELS_V31_PRESENTATION) or target.shape != raw.shape[:-1] + (2,) or valid.shape != raw.shape[:-1]:
        raise ValueError("IRIS_V31_PRESENTATION_MECHANICAL_LOSS_SHAPE")
    mask = valid.bool()
    if not bool(mask.any()):
        raise ValueError("IRIS_V31_PRESENTATION_MECHANICAL_LOSS_EMPTY")
    mean = torch.stack((raw[..., 0], raw[..., 2]), dim=-1)
    log_scale = torch.stack((raw[..., 1], raw[..., 3]), dim=-1).clamp(-6.0, 3.0)
    residual = target.to(mean) - mean
    # log_scale parameterizes log variance, so 0.5*exp(-log_var)*r^2 + 0.5*log_var.
    nll = 0.5 * residual.square() * torch.exp(-log_scale) + 0.5 * log_scale
    return nll[mask].mean()


def appearance_rgba_loss_v1(predicted_rgba: torch.Tensor, target_rgba: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if predicted_rgba.shape != target_rgba.shape or predicted_rgba.shape[-1] != 4 or valid.shape != predicted_rgba.shape[:-1]:
        raise ValueError("IRIS_V31_PRESENTATION_RGBA_LOSS_SHAPE")
    mask = valid.bool()
    if not bool(mask.any()):
        raise ValueError("IRIS_V31_PRESENTATION_RGBA_LOSS_EMPTY")
    return F.smooth_l1_loss(predicted_rgba[mask], target_rgba.to(predicted_rgba)[mask])


def pair_probe_loss_v1(raw: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if raw.shape != target.shape or raw.shape[-1] != len(PAIR_CHANNELS_V31_PRESENTATION) or valid.shape != raw.shape[:-1]:
        raise ValueError("IRIS_V31_PRESENTATION_PAIR_LOSS_SHAPE")
    mask = valid.bool()
    if not bool(mask.any()):
        raise ValueError("IRIS_V31_PRESENTATION_PAIR_LOSS_EMPTY")
    return F.binary_cross_entropy_with_logits(raw[mask], target.to(raw)[mask])
