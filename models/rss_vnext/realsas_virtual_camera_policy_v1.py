from __future__ import annotations

"""Fixed virtual target cameras for learned canonical and RGB/normal images.

These are intentionally NOT original measured camera estimates. Their exact
coordinate contract is shared by canonicalization, generation and S-LRM.
"""

import math

import torch
from torch import Tensor


def target_canonical_and_six_cameras_v1(
    batch_size: int, *, device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    radius: float = 2., orthographic_half_extent: float = 1.,
) -> tuple[Tensor, Tensor]:
    if (
        type(batch_size) is not int or not 1 <= batch_size <= 8
        or dtype not in (torch.float32, torch.float64)
        or not 0.2 <= radius <= 20.
        or not 0.05 <= orthographic_half_extent <= 20.
    ):
        raise ValueError("RSS_VNEXT_VIRTUAL_CAMERA_POLICY_INVALID")
    # Match the six-view S-LRM training target bank exactly. The old
    # 60-degree uniform ring disagreed with the -90/-45/0/45/90/180
    # training cameras, creating avoidable checkpoint inference drift.
    azimuth_degrees = (-90., -45., 0., 45., 90., 180.)
    views = torch.zeros((batch_size, 6, 13), dtype=dtype, device=device)
    for index in range(6):
        angle = math.radians(azimuth_degrees[index])
        radial = torch.tensor(
            (math.sin(angle), 0., math.cos(angle)),
            dtype=dtype, device=device,
        )
        right = torch.tensor(
            (-math.cos(angle), 0., math.sin(angle)),
            dtype=dtype, device=device,
        )
        views[:, index, :3] = radial * radius
        views[:, index, 3:6] = right
        views[:, index, 7] = 1.
        views[:, index, 9:12] = -radial
        views[:, index, 12] = orthographic_half_extent
    axes = torch.stack((
        views[:, :, 3:6], views[:, :, 6:9], views[:, :, 9:12],
    ), dim=-2)
    eye = torch.eye(3, dtype=dtype, device=device)
    if (
        not torch.allclose(axes @ axes.transpose(-1, -2), eye, atol=1e-5, rtol=0)
        or bool((torch.linalg.det(axes) < .999).any())
    ):
        raise RuntimeError("RSS_VNEXT_VIRTUAL_CAMERA_FRAME_NOT_ORTHONORMAL")
    # Canonical FRONT/BACK are virtual target poses at 0 and 180 degrees;
    # original user views may have unknown camera, pose and perspective.
    return views[:, [2, 5]].contiguous(), views


__all__ = ["target_canonical_and_six_cameras_v1"]
