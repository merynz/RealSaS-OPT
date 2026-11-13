from __future__ import annotations

"""Mixed-precision-safe no-grad query helpers for IRIS V4 diagnostics.

Training forward passes remain governed by the caller's autocast context. These helpers
exist for no-grad hard-negative discovery, where cached scene planes may be bfloat16
while freshly constructed camera-ray samples are float32. PyTorch grid_sample requires
matching input/grid dtypes outside autocast, so diagnostic queries are promoted to the
model parameter dtype (normally float32) before evaluation.
"""

import torch

from models.iris.v3.negative_space_barrier_v3 import query_dense_ray_sdf_v3


@torch.no_grad()
def query_hard_negative_ray_min_sdf_v4(
    model,
    scene_planes: torch.Tensor,
    ray_points_normalized: torch.Tensor,
    *,
    query_chunk: int = 131072,
) -> torch.Tensor:
    """Return [B,R] ray-min SDF for hard-negative discovery in model parameter dtype.

    This helper is intentionally no-grad and is not part of the optimized geometry
    objective. It changes neither the learned-field architecture nor loss semantics;
    it only makes hard-negative refresh numerically well-defined across autocast
    boundaries.
    """

    if int(query_chunk) <= 0:
        raise ValueError("query_chunk must be positive")
    try:
        reference = next(model.parameters())
    except StopIteration as exc:
        raise ValueError("hard-negative query requires a parameterized model") from exc

    planes = scene_planes.to(device=reference.device, dtype=reference.dtype)
    points = ray_points_normalized.to(device=reference.device, dtype=reference.dtype)
    sdf = query_dense_ray_sdf_v3(
        model,
        planes,
        points,
        query_chunk=int(query_chunk),
    )
    return torch.amin(sdf.float(), dim=-1)
