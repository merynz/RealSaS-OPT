from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def query_dense_ray_sdf_v3(
    model,
    scene_planes: torch.Tensor,
    ray_points_normalized: torch.Tensor,
    *,
    query_chunk: int = 131072,
) -> torch.Tensor:
    """Query the actual V3 signed field on dense camera-ray samples.

    ray_points_normalized must be [B,R,D,3]. The returned tensor is [B,R,D].
    This helper deliberately exposes the full depth axis because a source-background
    pixel certifies empty space along the entire admitted camera ray segment, not only
    at the minimum-SDF sample.
    """

    if ray_points_normalized.ndim != 4 or ray_points_normalized.shape[-1] != 3:
        raise ValueError("ray_points_normalized must be [B,R,D,3]")
    if ray_points_normalized.shape[0] != scene_planes.shape[0]:
        raise ValueError("ray points/scene plane batch mismatch")
    if int(query_chunk) <= 0:
        raise ValueError("query_chunk must be positive")
    if not torch.isfinite(ray_points_normalized).all():
        raise ValueError("ray_points_normalized must be finite")

    b, r, d, _ = ray_points_normalized.shape
    flat = ray_points_normalized.reshape(b, r * d, 3)
    sdf_parts: list[torch.Tensor] = []
    for start in range(0, int(flat.shape[1]), int(query_chunk)):
        row = model.query(scene_planes, flat[:, start : start + int(query_chunk)])["sdf"]
        if row.ndim != 2 or row.shape[0] != b:
            raise ValueError("V3 signed-field query returned unexpected shape")
        sdf_parts.append(row)
    sdf = torch.cat(sdf_parts, dim=1).reshape(b, r, d)
    if not torch.isfinite(sdf).all():
        raise ValueError("V3 signed-field query returned nonfinite values")
    return sdf


def background_full_ray_empty_space_barrier(
    sdf_samples: torch.Tensor,
    target_foreground: torch.Tensor,
    *,
    positive_margin: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Enforce pointwise empty-space authority on source-background camera rays.

    A qualified foreground pixel says only that some subject geometry is encountered
    on the admitted ray. A qualified background pixel is stronger: no point on the
    admitted ray segment may lie inside the subject. Therefore every sampled SDF point
    on a background ray must remain positive. This is a one-sided source-support
    constraint; it does not pull any inside-hull point toward the visual hull and is
    not mesh clipping or geometry replacement.
    """

    if sdf_samples.ndim != 3:
        raise ValueError("sdf_samples must be [B,R,D]")
    if target_foreground.shape != sdf_samples.shape[:2]:
        raise ValueError("target_foreground must be [B,R]")
    if sdf_samples.shape[-1] < 1:
        raise ValueError("background barrier requires depth samples")
    if not torch.isfinite(sdf_samples).all():
        raise ValueError("sdf_samples must be finite")
    target = target_foreground.to(dtype=sdf_samples.dtype)
    if torch.any((target != 0) & (target != 1)):
        raise ValueError("target_foreground must be binary")
    margin = float(positive_margin)
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("positive_margin must be finite and positive")

    background = target < 0.5
    count = int(background.sum().detach().cpu())
    if count == 0:
        zero = sdf_samples.sum() * 0.0
        return {
            "total": zero,
            "background_ray_count": torch.zeros((), dtype=torch.int64, device=sdf_samples.device),
            "violating_point_fraction": zero,
            "minimum_background_sdf": torch.full((), float("nan"), dtype=sdf_samples.dtype, device=sdf_samples.device),
        }

    bg = sdf_samples[background]
    violation = F.relu(margin - bg)
    total = violation.mean()
    violating = (bg < margin).to(dtype=sdf_samples.dtype).mean()
    return {
        "total": total,
        "background_ray_count": torch.as_tensor(count, dtype=torch.int64, device=sdf_samples.device),
        "violating_point_fraction": violating,
        "minimum_background_sdf": torch.amin(bg),
    }
