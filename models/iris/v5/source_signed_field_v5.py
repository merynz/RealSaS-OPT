from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates
import torch


@dataclass(frozen=True)
class SourceSignedFieldPolicyV5:
    """Exact F4 continuous source-field sampling contract.

    The canonical Knight policy fixes eight 1024x1024 source views and a bounded
    field in [-0.25, +0.25]. Smaller resolutions remain configurable only so the
    exact sampling semantics can be unit-tested without allocating product-size
    rasters.
    """

    field_clamp: float = 0.25
    unseen_value: float = 0.25
    required_views: int = 8
    required_resolution: int = 1024

    def validate(self) -> None:
        if int(self.required_views) <= 0:
            raise ValueError("required_views must be positive")
        if int(self.required_resolution) <= 1:
            raise ValueError("required_resolution must be > 1")
        if not math.isfinite(float(self.field_clamp)) or float(self.field_clamp) <= 0.0:
            raise ValueError("field_clamp must be finite and positive")
        if not math.isfinite(float(self.unseen_value)):
            raise ValueError("unseen_value must be finite")
        if abs(float(self.unseen_value)) > float(self.field_clamp):
            raise ValueError("unseen_value must lie inside the field clamp")


def _numpy_field_array(
    signed_2d: np.ndarray,
    policy: SourceSignedFieldPolicyV5,
) -> np.ndarray:
    policy.validate()
    field = np.asarray(signed_2d, dtype=np.float32)
    expected = (
        int(policy.required_views),
        int(policy.required_resolution),
        int(policy.required_resolution),
    )
    if field.shape != expected:
        raise ValueError(f"signed_2d must have shape {expected}, got {field.shape}")
    if not np.isfinite(field).all():
        raise ValueError("signed_2d contains non-finite values")
    return field


def _numpy_camera_arrays(
    *,
    center_xyz: np.ndarray,
    normalization_half_extent: float,
    camera_origins: np.ndarray,
    camera_right: np.ndarray,
    camera_screen_up: np.ndarray,
    camera_half_extent: np.ndarray,
    policy: SourceSignedFieldPolicyV5,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(center_xyz, dtype=np.float32).reshape(-1)
    origins = np.asarray(camera_origins, dtype=np.float32)
    right = np.asarray(camera_right, dtype=np.float32)
    up = np.asarray(camera_screen_up, dtype=np.float32)
    half = np.asarray(camera_half_extent, dtype=np.float32).reshape(-1)
    views = int(policy.required_views)
    if center.shape != (3,):
        raise ValueError("center_xyz must be [3]")
    if origins.shape != (views, 3) or right.shape != (views, 3) or up.shape != (views, 3):
        raise ValueError("camera origins/right/screen_up must be [V,3]")
    if half.shape != (views,):
        raise ValueError("camera_half_extent must be [V]")
    norm_half = float(normalization_half_extent)
    arrays = (center, origins, right, up, half)
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("camera/normalization arrays must be finite")
    if not math.isfinite(norm_half) or norm_half <= 0.0:
        raise ValueError("normalization_half_extent must be finite and positive")
    if np.any(half <= 0.0):
        raise ValueError("camera_half_extent must be positive")
    if np.any(np.linalg.norm(right, axis=1) <= 1e-12) or np.any(
        np.linalg.norm(up, axis=1) <= 1e-12
    ):
        raise ValueError("camera basis vectors must be nonzero")
    return center, norm_half, origins, right, up, half


def source_signed_distance_fields_v5(
    foreground_masks: np.ndarray,
    *,
    policy: SourceSignedFieldPolicyV5 = SourceSignedFieldPolicyV5(),
) -> np.ndarray:
    """Build S_v = EDT(background) - EDT(foreground), exactly as the F4 oracle."""

    policy.validate()
    masks = np.asarray(foreground_masks, dtype=bool)
    expected = (
        int(policy.required_views),
        int(policy.required_resolution),
        int(policy.required_resolution),
    )
    if masks.shape != expected:
        raise ValueError(f"foreground_masks must have shape {expected}, got {masks.shape}")
    if not np.all(np.any(masks, axis=(1, 2))):
        raise ValueError("every source view requires non-empty foreground")
    return np.stack(
        [
            (
                ndimage.distance_transform_edt(~masks[v])
                - ndimage.distance_transform_edt(masks[v])
            ).astype(np.float32)
            for v in range(int(policy.required_views))
        ],
        axis=0,
    )


def source_signed_field_numpy_v5(
    points_normalized: np.ndarray,
    *,
    signed_2d: np.ndarray,
    center_xyz: np.ndarray,
    normalization_half_extent: float,
    camera_origins: np.ndarray,
    camera_right: np.ndarray,
    camera_screen_up: np.ndarray,
    camera_half_extent: np.ndarray,
    policy: SourceSignedFieldPolicyV5 = SourceSignedFieldPolicyV5(),
) -> np.ndarray:
    """Reference F*(q): SciPy bilinear interpolation with nearest edge extension.

    Important exact F4 semantics:
    - projection support is closed: 0 <= sx,sy <= RES;
    - array sampling is at (sy-0.5, sx-0.5);
    - camera basis vectors are consumed as qualified, never silently normalized;
    - view values are scaled into normalized-3D units then MAX-aggregated;
    - unseen points receive +field_clamp and the final field is clamped.
    """

    field = _numpy_field_array(signed_2d, policy)
    center, norm_half, origins, right, up, cam_half = _numpy_camera_arrays(
        center_xyz=center_xyz,
        normalization_half_extent=normalization_half_extent,
        camera_origins=camera_origins,
        camera_right=camera_right,
        camera_screen_up=camera_screen_up,
        camera_half_extent=camera_half_extent,
        policy=policy,
    )
    q = np.asarray(points_normalized, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) == 0:
        raise ValueError("points_normalized must be non-empty [N,3]")
    if not np.isfinite(q).all():
        raise ValueError("points_normalized contains non-finite values")

    res = int(policy.required_resolution)
    pw = center[None, :] + q * np.float32(norm_half)
    value = np.full(len(q), -np.inf, dtype=np.float32)
    seen = np.zeros(len(q), dtype=bool)

    for view in range(int(policy.required_views)):
        d = pw - origins[view][None, :]
        gx = (d @ right[view]) / cam_half[view]
        gy = -(d @ up[view]) / cam_half[view]
        sx = (gx + 1.0) * 0.5 * res
        sy = (gy + 1.0) * 0.5 * res
        valid = (sx >= 0.0) & (sx <= res) & (sy >= 0.0) & (sy <= res)
        if not np.any(valid):
            continue
        ids = np.flatnonzero(valid)
        sampled = map_coordinates(
            field[view],
            np.stack([sy[ids] - 0.5, sx[ids] - 0.5], axis=0),
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float32)
        scale = np.float32(
            2.0 * float(cam_half[view]) / (float(res) * norm_half)
        )
        sampled *= scale
        value[ids] = np.maximum(value[ids], sampled)
        seen[ids] = True

    value[~seen] = np.float32(policy.unseen_value)
    return np.clip(
        value,
        -float(policy.field_clamp),
        float(policy.field_clamp),
    ).astype(np.float32)


def source_signed_field_torch_v5(
    points_normalized: torch.Tensor | np.ndarray,
    *,
    signed_2d: torch.Tensor | np.ndarray,
    center_xyz: torch.Tensor | np.ndarray,
    normalization_half_extent: float | torch.Tensor,
    camera_origins: torch.Tensor | np.ndarray,
    camera_right: torch.Tensor | np.ndarray,
    camera_screen_up: torch.Tensor | np.ndarray,
    camera_half_extent: torch.Tensor | np.ndarray,
    policy: SourceSignedFieldPolicyV5 = SourceSignedFieldPolicyV5(),
) -> torch.Tensor:
    """Torch parity implementation of F4's explicit manual bilinear sampler."""

    policy.validate()
    field = torch.as_tensor(signed_2d)
    device = field.device
    field = field.to(device=device, dtype=torch.float32)
    expected = (
        int(policy.required_views),
        int(policy.required_resolution),
        int(policy.required_resolution),
    )
    if tuple(field.shape) != expected:
        raise ValueError(f"signed_2d must have shape {expected}, got {tuple(field.shape)}")
    if not bool(torch.isfinite(field).all()):
        raise ValueError("signed_2d contains non-finite values")

    q = torch.as_tensor(points_normalized, device=device, dtype=torch.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.ndim != 2 or q.shape[1] != 3 or q.numel() == 0:
        raise ValueError("points_normalized must be non-empty [N,3]")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("points_normalized contains non-finite values")

    views = int(policy.required_views)
    res = int(policy.required_resolution)
    center = torch.as_tensor(center_xyz, device=device, dtype=torch.float32).reshape(-1)
    origins = torch.as_tensor(camera_origins, device=device, dtype=torch.float32)
    right = torch.as_tensor(camera_right, device=device, dtype=torch.float32)
    up = torch.as_tensor(camera_screen_up, device=device, dtype=torch.float32)
    cam_half = torch.as_tensor(camera_half_extent, device=device, dtype=torch.float32).reshape(-1)
    norm_half = torch.as_tensor(
        normalization_half_extent, device=device, dtype=torch.float32
    ).reshape(())

    if center.shape != (3,):
        raise ValueError("center_xyz must be [3]")
    if origins.shape != (views, 3) or right.shape != (views, 3) or up.shape != (views, 3):
        raise ValueError("camera origins/right/screen_up must be [V,3]")
    if cam_half.shape != (views,):
        raise ValueError("camera_half_extent must be [V]")
    for array in (center, origins, right, up, cam_half, norm_half):
        if not bool(torch.isfinite(array).all()):
            raise ValueError("camera/normalization tensors must be finite")
    if float(norm_half.detach().cpu()) <= 0.0 or bool(torch.any(cam_half <= 0.0)):
        raise ValueError("camera and normalization half extents must be positive")
    if bool(torch.any(torch.linalg.vector_norm(right, dim=1) <= 1e-12)) or bool(
        torch.any(torch.linalg.vector_norm(up, dim=1) <= 1e-12)
    ):
        raise ValueError("camera basis vectors must be nonzero")

    pw = center[None, :] + q * norm_half
    value = torch.full(
        (len(q),), -float("inf"), device=device, dtype=torch.float32
    )
    seen = torch.zeros(len(q), device=device, dtype=torch.bool)

    for view in range(views):
        d = pw - origins[view][None, :]
        gx = (d @ right[view]) / cam_half[view]
        gy = -(d @ up[view]) / cam_half[view]
        sx = (gx + 1.0) * 0.5 * res
        sy = (gy + 1.0) * 0.5 * res
        valid = (sx >= 0.0) & (sx <= res) & (sy >= 0.0) & (sy <= res)
        if not bool(valid.any()):
            continue

        ids = torch.nonzero(valid, as_tuple=False).squeeze(1)
        x = torch.clamp(sx[ids] - 0.5, 0.0, float(res - 1))
        y = torch.clamp(sy[ids] - 0.5, 0.0, float(res - 1))
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = torch.clamp(x0 + 1, max=res - 1)
        y1 = torch.clamp(y0 + 1, max=res - 1)
        wx = x - x0.float()
        wy = y - y0.float()

        f00 = field[view, y0, x0]
        f01 = field[view, y0, x1]
        f10 = field[view, y1, x0]
        f11 = field[view, y1, x1]
        sampled = (
            (1.0 - wy) * ((1.0 - wx) * f00 + wx * f01)
            + wy * ((1.0 - wx) * f10 + wx * f11)
        )
        scale = (2.0 * cam_half[view]) / (float(res) * norm_half)
        sampled = sampled * scale
        value[ids] = torch.maximum(value[ids], sampled)
        seen[ids] = True

    value = torch.where(
        seen,
        value,
        torch.full_like(value, float(policy.unseen_value)),
    )
    return torch.clamp(
        value,
        -float(policy.field_clamp),
        float(policy.field_clamp),
    )


def exterior_sign_audit_v5(
    field_values: torch.Tensor | np.ndarray,
    certified_exterior: torch.Tensor | np.ndarray,
) -> dict[str, int | float | bool]:
    """Independent R512 sign certificate. Magnitude is intentionally not audited."""

    values = np.asarray(
        torch.as_tensor(field_values).detach().cpu().numpy()
        if isinstance(field_values, torch.Tensor)
        else field_values,
        dtype=np.float64,
    ).reshape(-1)
    certified = np.asarray(
        torch.as_tensor(certified_exterior).detach().cpu().numpy()
        if isinstance(certified_exterior, torch.Tensor)
        else certified_exterior,
        dtype=bool,
    ).reshape(-1)
    if values.shape != certified.shape or values.size == 0:
        raise ValueError("field_values and certified_exterior must match and be non-empty")
    if not np.isfinite(values).all():
        raise ValueError("field_values contains non-finite values")
    count = int(certified.sum())
    if count <= 0:
        raise ValueError("exterior sign audit requires at least one certified point")
    nonpositive = int(np.count_nonzero(certified & (values <= 0.0)))
    return {
        "certified_exterior_count": count,
        "nonpositive_certified_exterior_count": nonpositive,
        "nonpositive_fraction": float(nonpositive / count),
        "passed": bool(nonpositive == 0),
    }
