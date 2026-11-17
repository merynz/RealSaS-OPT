from __future__ import annotations

"""Bounded, exhaustive IRIS sign scan for one signed semantic surface layer.

No source labels, source mesh, or separate geometry model enter this forward
path. Every fine-grid vertex is queried exactly once for the sign certificate,
but only vertices/cubes touching the surface are subsequently materialized.

This is an INFERENCE primitive. Gradients for production-scale training need a
separately qualified active-set checkpointing policy; no accidental full-grid
autograd graph is built here.
"""

import math
from typing import Callable

import torch
from torch import Tensor

SPARSE_SIGN_SCAN_SCHEMA = "RealSaS.RSSvNext.ExhaustiveSignedGridScan.v1"


def qualify_sparse_resolution_v1(
    resolution: tuple[int, int, int],
) -> tuple[int, int, int]:
    if (
        type(resolution) is not tuple or len(resolution) != 3
        or any(type(x) is not int or not 2 <= x <= 512 for x in resolution)
        or math.prod(resolution) > 32_000_000
        or math.prod(x + 1 for x in resolution) > 33_000_000
    ):
        raise ValueError("RSS_VNEXT_SPARSE_SCAN_GRID_BUDGET")
    return resolution


def global_grid_xyz_v1(
    global_vertex_ids: Tensor, *,
    resolution: tuple[int, int, int],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Exact XYZ namespace of PinnedFlexiCubesV1 anisotropic grid."""
    nx, ny, nz = qualify_sparse_resolution_v1(resolution)
    n = (nx + 1) * (ny + 1) * (nz + 1)
    ids = global_vertex_ids
    if (
        ids.ndim != 1 or ids.dtype != torch.long
        or bool((ids < 0).any()) or bool((ids >= n).any())
        or dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16)
    ):
        raise ValueError("RSS_VNEXT_SPARSE_SCAN_GLOBAL_VERTEX_DOMAIN")
    x = torch.div(ids, (ny + 1) * (nz + 1), rounding_mode="floor")
    y = torch.div(ids, nz + 1, rounding_mode="floor") % (ny + 1)
    z = ids % (nz + 1)
    # Original pinned regular XYZ constructor is float32 before scene casting.
    return torch.stack((
        x.float() * (2.0 / nx) - 1.0,
        y.float() * (2.0 / ny) - 1.0,
        z.float() * (2.0 / nz) - 1.0,
    ), dim=-1).to(dtype=dtype)


@torch.no_grad()
def scan_exhaustive_signed_grid_v1(
    query_signed_field: Callable[[Tensor], Tensor], *,
    resolution: tuple[int, int, int],
    device: torch.device,
    scene_dtype: torch.dtype,
    query_chunk: int = 16384,
) -> Tensor:
    """Returns CPU boolean [nx+1,ny+1,nz+1], not pseudo-labels.

    query_signed_field receives scene XYZ [1,Q,3] and returns [1,Q].
    A coarse scan alone is never a certificate of all fine surface cubes.
    The caller must re-query compact active corners, compare every sign and
    use PinnedSparseFlexiCubesV1's exhaustive cube-address validation.
    """
    nx, ny, nz = qualify_sparse_resolution_v1(resolution)
    if (
        type(query_chunk) is not int or not 1 <= query_chunk <= 131072
        or not isinstance(device, torch.device)
    ):
        raise ValueError("RSS_VNEXT_SPARSE_SCAN_CHUNK_POLICY")
    count = (nx + 1) * (ny + 1) * (nz + 1)
    negative = torch.empty(count, dtype=torch.bool, device="cpu")
    for start in range(0, count, query_chunk):
        stop = min(count, start + query_chunk)
        ids = torch.arange(start, stop, device=device, dtype=torch.long)
        points = global_grid_xyz_v1(
            ids, resolution=resolution, dtype=scene_dtype,
        )[None]
        signed = query_signed_field(points)
        if (
            not isinstance(signed, Tensor)
            or signed.shape != (1, stop - start)
            or signed.device != device
            or not bool(torch.isfinite(signed).all())
        ):
            raise ValueError("RSS_VNEXT_SPARSE_SCAN_FIELD_DOMAIN")
        negative[start:stop].copy_((signed[0] < 0).to("cpu"))
    if not bool(negative.any()) or not bool((~negative).any()):
        raise ValueError("RSS_VNEXT_SPARSE_SCAN_NO_SIGN_CHANGE_ABSTAIN")
    return negative.reshape(nx + 1, ny + 1, nz + 1)


__all__ = [
    "SPARSE_SIGN_SCAN_SCHEMA",
    "qualify_sparse_resolution_v1",
    "global_grid_xyz_v1",
    "scan_exhaustive_signed_grid_v1",
]
