from __future__ import annotations

"""Pure helpers for H-GEO-P3-SCALE-XFER.

No optimizer, model mutation, or TESSA invocation lives here.
"""

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tools.iris_vnext.h_geo_p1_geometry_v1 import _closest_points_to_triangles
from tools.iris_vnext.h_geo_p3_efreq_v1 import NormalizedCameraV1, project_points_v1

SCALE_XFER_EPS_IRIS = (
    1.0 / 1024.0,
    1.0 / 512.0,
    1.0 / 256.0,
    1.0 / 128.0,
    1.0 / 64.0,
    1.0 / 32.0,
)


def deterministic_unit_direction_v2(subject_index: int, query_index: int) -> np.ndarray:
    a = float((subject_index + 1) * 0.6180339887498948 + (query_index + 1) * 0.4142135623730950)
    b = float((subject_index + 1) * 0.2718281828459045 + (query_index + 1) * 0.1414213562373095)
    v = np.asarray(
        (math.sin(2.0 * math.pi * a), math.cos(2.0 * math.pi * a), math.sin(2.0 * math.pi * b)),
        dtype=np.float64,
    )
    return v / max(float(np.linalg.norm(v)), 1e-12)


def nearest_nonadjacent_candidates_v1(
    query_iris: np.ndarray,
    nearest_distance_iris: float,
    nearest_face: int,
    triangles: np.ndarray,
    face_normals: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    k: int = 4,
) -> list[dict[str, Any]]:
    q = np.asarray(query_iris, dtype=np.float64).reshape(3)
    tri = np.asarray(triangles, dtype=np.float64)
    mask = np.ones(len(tri), dtype=bool)
    mask[np.asarray(adjacency[int(nearest_face)], dtype=np.int64)] = False
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cp = _closest_points_to_triangles(q, tri[indices])
    dist = np.linalg.norm(cp - q[None], axis=1)
    order = np.argsort(dist, kind="mergesort")[: max(1, int(k))]
    n0 = np.asarray(face_normals[int(nearest_face)], dtype=np.float64)
    out: list[dict[str, Any]] = []
    for j in order:
        fi = int(indices[int(j)])
        n1 = np.asarray(face_normals[fi], dtype=np.float64)
        cosine = float(np.clip(abs(float(np.dot(n0, n1))), 0.0, 1.0))
        out.append(
            {
                "face_index": fi,
                "point_iris": cp[int(j)],
                "distance_iris": float(dist[int(j)]),
                "margin_over_nearest_iris": float(dist[int(j)] - float(nearest_distance_iris)),
                "normal_angle_up_to_sign_deg": float(np.degrees(np.arccos(cosine))),
            }
        )
    return out


def _sample_projected_tokens_v1(
    image_tokens: torch.Tensor,
    cameras: list[NormalizedCameraV1],
    points_iris: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    if image_tokens.shape[:3] != (1, 8, 1369) or image_tokens.shape[-1] != 384:
        raise ValueError(f"H_GEO_P3_SCALE_XFER_TOKEN_SHAPE_INVALID:{tuple(image_tokens.shape)}")
    if len(cameras) != 8:
        raise ValueError("H_GEO_P3_SCALE_XFER_CAMERA_COUNT_INVALID")
    points = np.asarray(points_iris, dtype=np.float64).reshape(-1, 3)
    coords = np.stack([project_points_v1(points, cam) for cam in cameras], axis=0)
    in_frame = np.max(np.abs(coords), axis=-1) <= 1.0
    grid = torch.from_numpy(coords.astype(np.float32)).to(image_tokens.device).unsqueeze(2)
    fmap = image_tokens[0].reshape(8, 37, 37, 384).permute(0, 3, 1, 2).contiguous()
    sampled = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="border", align_corners=False)
    sampled = sampled.squeeze(-1).permute(2, 0, 1).contiguous().float().cpu()
    cell_xy = np.floor((coords + 1.0) * (37.0 / 2.0)).astype(np.int64)
    cell_xy = np.clip(cell_xy, 0, 36)
    cell_id = cell_xy[..., 1] * 37 + cell_xy[..., 0]
    return sampled, coords, in_frame, cell_id


def paired_surface_token_stats_v1(
    image_tokens: torch.Tensor,
    cameras: list[NormalizedCameraV1],
    nearest_point_iris: np.ndarray,
    competitor_point_iris: np.ndarray,
) -> dict[str, float]:
    points = np.stack((nearest_point_iris, competitor_point_iris), axis=0)
    feat, coords, in_frame, cell_id = _sample_projected_tokens_v1(image_tokens, cameras, points)
    valid = in_frame[:, 0] & in_frame[:, 1]
    count = int(valid.sum())
    if count == 0:
        return {
            "both_in_frame_view_count": 0.0,
            "distinct_token_cell_view_count": 0.0,
            "same_token_cell_fraction": float("nan"),
            "median_token_cosine_distance": float("nan"),
            "maximum_token_cosine_distance": float("nan"),
            "median_projected_separation_token_cells": float("nan"),
        }
    f0 = feat[0, valid]
    f1 = feat[1, valid]
    cosine = F.cosine_similarity(f0, f1, dim=-1).clamp(-1.0, 1.0)
    cosine_distance = (1.0 - cosine).numpy()
    same = cell_id[valid, 0] == cell_id[valid, 1]
    sep = np.linalg.norm((coords[valid, 0] - coords[valid, 1]) * (37.0 / 2.0), axis=-1)
    return {
        "both_in_frame_view_count": float(count),
        "distinct_token_cell_view_count": float(np.sum(~same)),
        "same_token_cell_fraction": float(np.mean(same)),
        "median_token_cosine_distance": float(np.median(cosine_distance)),
        "maximum_token_cosine_distance": float(np.max(cosine_distance)),
        "median_projected_separation_token_cells": float(np.median(sep)),
    }


def whitened_rms_delta_v2(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(((a - b) / scale.clamp_min(1e-6)) ** 2, dim=-1))


def summarize_stage_delta_v1(
    plus: torch.Tensor,
    minus: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        whitened_rms_delta_v2(plus, center, scale)
        + whitened_rms_delta_v2(minus, center, scale)
    )


def classify_scale_transfer_v1(pooled: dict[str, Any], prereg: dict[str, Any]) -> tuple[dict[str, bool], str, str]:
    locked = prereg["locked_signals"]
    tail = pooled["tail"]
    control = pooled["control"]
    primary = "eps_0.00781250"

    target_cfg = locked["target_multisurface"]
    target = bool(
        tail["median_competing_surface_margin_iris"] <= float(target_cfg["pooled_tail_median_competing_margin_iris_lte"])
        or tail["fraction_second_sheet_within_1_over_128"] >= float(target_cfg["or_pooled_tail_fraction_with_second_sheet_within_1_over_128_gte"])
    )

    view_cfg = locked["view_input_resolution"]
    tail_max = tail["median_maximum_token_cosine_distance"]
    control_max = control["median_maximum_token_cosine_distance"]
    view = bool(
        tail["median_same_token_cell_fraction"] >= float(view_cfg["pooled_tail_median_same_token_cell_fraction_gte"])
        or tail["median_distinct_token_cell_view_count"] <= control["median_distinct_token_cell_view_count"] - float(view_cfg["or_pooled_tail_distinct_token_view_count_lte_control_minus"])
        or (
            np.isfinite(tail_max)
            and np.isfinite(control_max)
            and tail_max <= float(view_cfg["or_pooled_tail_max_token_cosine_distance_lte_control_ratio"]) * max(control_max, 1e-12)
        )
    )

    carrier_cfg = locked["carrier_transfer"]
    p3_to_p2 = tail[f"median_stage_delta_P3_H_GEO_{primary}"] / max(tail[f"median_stage_delta_P2_H_BASELINE_{primary}"], 1e-12)
    refined_to_bilinear = tail[f"median_stage_delta_P3_REFINED128_SAMPLED_PLANES_{primary}"] / max(tail[f"median_stage_delta_P3_BILINEAR128_SAMPLED_PLANES_{primary}"], 1e-12)
    carrier = bool(
        p3_to_p2 <= float(carrier_cfg["p3_h_geo_to_p2_h_tail_delta_ratio_at_1_over_128_lte"])
        and refined_to_bilinear <= float(carrier_cfg["and_refined128_to_bilinear128_tail_delta_ratio_at_1_over_128_lte"])
    )

    signals = {
        "target_multisurface": target,
        "view_input_resolution": view,
        "carrier_transfer": carrier,
    }
    count = sum(bool(v) for v in signals.values())
    if count >= 2:
        classification = "MIXED"
    elif target:
        classification = "TARGET_MULTISURFACE_LIMITED"
    elif view:
        classification = "VIEW_INPUT_RESOLUTION_LIMITED"
    elif carrier:
        classification = "CARRIER_TRANSFER_LIMITED"
    else:
        classification = "UNRESOLVED"
    next_action = prereg["interpretation"][classification + "_next"]
    return signals, classification, next_action


__all__ = [
    "SCALE_XFER_EPS_IRIS",
    "classify_scale_transfer_v1",
    "deterministic_unit_direction_v2",
    "nearest_nonadjacent_candidates_v1",
    "paired_surface_token_stats_v1",
    "summarize_stage_delta_v1",
    "whitened_rms_delta_v2",
]
