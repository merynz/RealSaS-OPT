from __future__ import annotations

"""Explicit coordinate-domain conversion for IRIS <-> TESSA evidence.

IRIS scene/query coordinates use the normalized object cube [-1, 1].
TESSA structural geometry/tokenization uses [-0.5, 0.5].

These conversions are deterministic apparatus semantics, not learned evidence.
"""

import torch

IRIS_DOMAIN_ABS_MAX_V1 = 1.0
TESSA_DOMAIN_ABS_MAX_V1 = 0.5
IRIS_TO_TESSA_SCALE_V1 = 0.5
TESSA_TO_IRIS_SCALE_V1 = 2.0


def _validate_xyz(xyz: torch.Tensor, *, abs_max: float, name: str) -> None:
    if xyz.ndim < 1 or xyz.shape[-1] != 3:
        raise ValueError(f"{name}_SHAPE_INVALID")
    if not torch.is_floating_point(xyz) or not bool(torch.isfinite(xyz).all()):
        raise ValueError(f"{name}_NUMERIC_INVALID")
    if xyz.numel() and float(xyz.detach().abs().max()) > float(abs_max) + 1.0e-6:
        raise ValueError(f"{name}_DOMAIN_DRIFT")


def iris_xyz_to_tessa_v1(xyz_iris: torch.Tensor) -> torch.Tensor:
    _validate_xyz(xyz_iris, abs_max=IRIS_DOMAIN_ABS_MAX_V1, name="IRIS_XYZ")
    return xyz_iris * IRIS_TO_TESSA_SCALE_V1


def tessa_xyz_to_iris_v1(xyz_tessa: torch.Tensor) -> torch.Tensor:
    _validate_xyz(xyz_tessa, abs_max=TESSA_DOMAIN_ABS_MAX_V1, name="TESSA_XYZ")
    return xyz_tessa * TESSA_TO_IRIS_SCALE_V1


def triplane_cell_center_xy_tessa_v1(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return fine carrier cell centers in TESSA coordinates.

    `LocalEvidenceFieldV1.sample_triplanes` samples the carrier with
    `align_corners=False`. Therefore index i of an N-cell axis corresponds to
    IRIS coordinate ((i + 0.5) * 2/N - 1), not endpoint-linspace(-1, 1, N).
    TESSA receives the same physical positions scaled by 0.5.
    """

    if height <= 0 or width <= 0:
        raise ValueError("TRIPLANE_CELL_CENTER_SHAPE_INVALID")
    y_iris = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (
        2.0 / float(height)
    ) - 1.0
    x_iris = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (
        2.0 / float(width)
    ) - 1.0
    yy, xx = torch.meshgrid(y_iris, x_iris, indexing="ij")
    return torch.stack((xx, yy), dim=-1) * IRIS_TO_TESSA_SCALE_V1
