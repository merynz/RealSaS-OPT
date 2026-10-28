"""IRIS V3.1 typed presentation objectives."""
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


def heteroscedastic_mechanical_loss_v1(raw, target, valid):
    prediction = raw[..., (0, 2)]
    log_scale = raw[..., (1, 3)]
    residual = prediction - target
    value = 0.5 * residual.square() * torch.exp(-log_scale) + 0.5 * log_scale
    mask = valid[..., None].expand_as(value)
    return value[mask].mean() if mask.any() else value.sum() * 0.0


def appearance_rgba_loss_v1(raw, target, valid):
    value = F.smooth_l1_loss(raw, target, reduction="none")
    mask = valid[..., None].expand_as(value)
    return value[mask].mean() if mask.any() else value.sum() * 0.0


def pair_probe_loss_v1(raw, target, valid):
    value = F.binary_cross_entropy_with_logits(raw, target, reduction="none")
    mask = valid[..., None].expand_as(value)
    return value[mask].mean() if mask.any() else value.sum() * 0.0


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
    # Canonical Compiler IR dataclasses serialize their type marker as
    # `schema_version`; older hand-authored test/envelope payloads used `schema`.
    # They denote the same NormalizationDomainIR.v1 contract.  Accept either
    # spelling, but fail closed if both are present and disagree.
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
        hits = list(measured.get("hits") or ())
        view = int(ray.get("view_index", -1))
        if view < 0:
            raise ValueError("IRIS_V31_PRESENTATION_VIEW_INDEX")
        rgba = np.asarray(measured.get("full_rgba") or (), dtype=np.float64)
        if rgba.shape != (4,) or not np.isfinite(rgba).all():
            raise ValueError("IRIS_V31_PRESENTATION_RGBA")
        contributing = [
            hit for hit in hits
            if float(hit.get("source_over_contribution", 0.0)) > float(contribution_floor)
        ]
        if len(contributing) == 1:
            hit = contributing[0]
            face = int(hit["face_index"])
            if face < 0 or face >= face_count:
                raise ValueError("IRIS_V31_PRESENTATION_FACE_INDEX")
            point = np.asarray(hit["world_point"], dtype=np.float64)
            appearance_xyz.append(_xyz(point, center, half))
            appearance_view.append(view)
            appearance_rgba.append(rgba)
            appearance_valid.append(True)
        for a_index in range(len(contributing)):
            for b_index in range(a_index + 1, len(contributing)):
                a = contributing[a_index]
                b = contributing[b_index]
                fa = int(a["face_index"])
                fb = int(b["face_index"])
                if min(fa, fb) < 0 or max(fa, fb) >= face_count:
                    raise ValueError("IRIS_V31_PRESENTATION_PAIR_FACE_INDEX")
                ca = float(a["source_over_contribution"])
                cb = float(b["source_over_contribution"])
                denom = ca + cb
                if not math.isfinite(denom) or denom <= 0:
                    continue
                xa = _xyz(np.asarray(a["world_point"], dtype=np.float64), center, half)
                xb = _xyz(np.asarray(b["world_point"], dtype=np.float64), center, half)
                target_ab = np.asarray([cb / denom], dtype=np.float64)
                target_ba = np.asarray([ca / denom], dtype=np.float64)
                pair_xyz_a.extend((xa, xb))
                pair_xyz_b.extend((xb, xa))
                pair_view.extend((view, view))
                pair_target.extend((target_ab, target_ba))
                pair_valid.extend((True, True))

    return PresentationTargetBatchV1(
        mechanical_xyz=np.asarray(mechanical_xyz, dtype=np.float32),
        mechanical_target=np.asarray(mechanical_target, dtype=np.float32),
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
