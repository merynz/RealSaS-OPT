from __future__ import annotations

"""Color/alpha transport authority for Complete Appearance Authority V2.

Transport textures are straight-alpha sRGB RGBA8. All interpolation and
composition are performed in linear-light premultiplied RGBA. Conversion back
to sRGB straight RGBA8 occurs only at an explicit transport/output boundary.
"""

import numpy as np

from .types import QualificationError


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(x)):
        raise QualificationError("CAA_COLOR_SRGB_NONFINITE")
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(x)):
        raise QualificationError("CAA_COLOR_LINEAR_NONFINITE")
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * (x ** (1.0 / 2.4)) - 0.055,
    )


def straight_srgb_rgba_u8_to_premultiplied_linear(
    rgba_u8: np.ndarray,
) -> np.ndarray:
    rgba = np.asarray(rgba_u8, dtype=np.uint8)
    if rgba.shape[-1] != 4:
        raise QualificationError("CAA_COLOR_RGBA_SHAPE_INVALID")
    unit = rgba.astype(np.float64) / 255.0
    alpha = unit[..., 3:4]
    rgb_linear = srgb_to_linear(unit[..., :3])
    return np.concatenate((rgb_linear * alpha, alpha), axis=-1)


def premultiplied_linear_to_straight_srgb_u8(pm: np.ndarray) -> np.ndarray:
    value = np.asarray(pm, dtype=np.float64)
    if value.shape[-1] != 4 or np.any(~np.isfinite(value)):
        raise QualificationError("CAA_COLOR_PM_SHAPE_INVALID")
    alpha = np.clip(value[..., 3:4], 0.0, 1.0)
    rgb_linear = np.zeros_like(value[..., :3])
    nonzero = alpha[..., 0] > 1.0e-12
    if np.any(nonzero):
        rgb_linear[nonzero] = np.clip(
            value[..., :3][nonzero] / alpha[nonzero],
            0.0,
            1.0,
        )
    rgb_srgb = linear_to_srgb(rgb_linear)
    straight = np.concatenate((rgb_srgb, alpha), axis=-1)
    return np.clip(np.floor(straight * 255.0 + 0.5), 0.0, 255.0).astype(
        np.uint8
    )


def bilinear_premultiplied_linear_rgba(
    straight_srgb_rgba_u8: np.ndarray,
    uv_or_xy: np.ndarray,
    *,
    normalized: bool,
) -> np.ndarray:
    image = straight_srgb_rgba_u8_to_premultiplied_linear(
        straight_srgb_rgba_u8
    )
    points = np.asarray(uv_or_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise QualificationError("CAA_COLOR_BILINEAR_COORD_INVALID")
    height, width, _ = image.shape
    if normalized:
        x = np.clip(points[:, 0], 0.0, 1.0) * float(width - 1)
        y = np.clip(points[:, 1], 0.0, 1.0) * float(height - 1)
    else:
        x = np.clip(points[:, 0], 0.0, float(width - 1))
        y = np.clip(points[:, 1], 0.0, float(height - 1))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = (x - x0).reshape(-1, 1)
    ty = (y - y0).reshape(-1, 1)
    p00 = image[y0, x0]
    p10 = image[y0, x1]
    p01 = image[y1, x0]
    p11 = image[y1, x1]
    return (1.0 - ty) * ((1.0 - tx) * p00 + tx * p10) + ty * (
        (1.0 - tx) * p01 + tx * p11
    )


def source_sample_roundtrip_pm_error(
    straight_srgb_rgba_u8: np.ndarray,
    premultiplied_linear_truth: np.ndarray,
) -> np.ndarray:
    reconstructed = straight_srgb_rgba_u8_to_premultiplied_linear(
        straight_srgb_rgba_u8
    )
    truth = np.asarray(premultiplied_linear_truth, dtype=np.float64)
    if reconstructed.shape != truth.shape:
        raise QualificationError("CAA_COLOR_ROUNDTRIP_SHAPE_DRIFT")
    return np.max(np.abs(reconstructed - truth), axis=-1)
