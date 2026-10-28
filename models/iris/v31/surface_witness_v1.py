from __future__ import annotations

"""Deterministic geometry-native surface witness sampling for IRIS V3.1.

This module carries only the selected XYZ-FPS sampler. It deliberately does not
select extraction resolution, observability thresholds, iso-surface algorithm,
or a production witness count.
"""

from dataclasses import dataclass
import math

import torch


SAMPLER_ID_V1 = "XYZ_FPS"
COURT_STATUS_V1 = "GEOMETRY_NATIVE_SAMPLER_SELECTED_TINY_V1"
COURT_RUN_ID_V1 = 40312976720
COURT_SOURCE_SHA_V1 = "1b9fa8687d304aa247c829bae1ff512355644956"
COURT_ARTIFACT_DIGEST_V1 = "a61cc8b32ec7c30c4896e2632fa70ad92239eb4c8ab3e0f32246f6e81a7ebaa3"


@dataclass(frozen=True)
class SurfaceWitnessSamplerReceiptV1:
    sampler_id: str = SAMPLER_ID_V1
    court_status: str = COURT_STATUS_V1
    court_run_id: int = COURT_RUN_ID_V1
    court_source_sha: str = COURT_SOURCE_SHA_V1
    court_artifact_digest_sha256: str = COURT_ARTIFACT_DIGEST_V1
    production_resolution_selected: bool = False
    production_witness_count_selected: bool = False
    product_authority_changed: bool = False
    model_release_changed: bool = False
    genericity_claimed: bool = False
    schema: str = "RealSaS.IRISSurfaceWitnessSamplerReceipt.v1"


def _validated_cpu_rows(
    xyz: torch.Tensor,
    normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.device]:
    if not isinstance(xyz, torch.Tensor) or not isinstance(normal, torch.Tensor):
        raise TypeError("IRIS_V31_XYZ_FPS_TENSORS_REQUIRED")
    if xyz.ndim != 2 or xyz.shape[-1] != 3 or normal.shape != xyz.shape or xyz.shape[0] <= 0:
        raise ValueError("IRIS_V31_XYZ_FPS_SHAPE_INVALID")
    if not torch.is_floating_point(xyz) or not torch.is_floating_point(normal):
        raise ValueError("IRIS_V31_XYZ_FPS_FLOAT_REQUIRED")
    if not bool(torch.isfinite(xyz).all()) or not bool(torch.isfinite(normal).all()):
        raise ValueError("IRIS_V31_XYZ_FPS_NONFINITE")
    norms = torch.linalg.vector_norm(normal.detach().float(), dim=-1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1.0e-4, rtol=0.0)):
        raise ValueError("IRIS_V31_XYZ_FPS_UNIT_NORMAL_REQUIRED")
    device = xyz.device
    xyz_cpu = xyz.detach().to(device="cpu", dtype=torch.float64).contiguous()
    normal_cpu = normal.detach().to(device="cpu", dtype=torch.float64).contiguous()
    rows = [
        tuple(float(v) for v in torch.cat((xyz_cpu[i], normal_cpu[i])).tolist())
        for i in range(int(xyz_cpu.shape[0]))
    ]
    if len(set(rows)) != len(rows):
        raise ValueError("IRIS_V31_XYZ_FPS_DUPLICATE_CANONICAL_ROW")
    return xyz_cpu, normal_cpu, device


def xyz_fps_indices_v31(
    xyz: torch.Tensor,
    normal: torch.Tensor,
    *,
    count: int,
) -> torch.Tensor:
    """Return exact input indices under selected deterministic XYZ-FPS semantics.

    Geometry distance uses xyz only. Normals participate only in the canonical
    lexicographic ordering/tie-break `(x,y,z,nx,ny,nz)` selected by the court.
    Sampling is evaluated on CPU float64 so CUDA kernel ordering cannot become a
    hidden witness-selection authority.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("IRIS_V31_XYZ_FPS_COUNT_INVALID")
    xyz_cpu, normal_cpu, output_device = _validated_cpu_rows(xyz, normal)
    n = int(xyz_cpu.shape[0])

    canonical_rows = torch.cat((xyz_cpu, normal_cpu), dim=-1).tolist()
    order_list = sorted(
        range(n),
        key=lambda i: tuple(float(v) for v in canonical_rows[i]),
    )
    order = torch.tensor(order_list, dtype=torch.long)
    if count >= n:
        return order.to(device=output_device)

    xyz_c = xyz_cpu[order]
    diagonal = float(
        torch.linalg.vector_norm(
            xyz_c.max(dim=0).values - xyz_c.min(dim=0).values
        )
    )
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("IRIS_V31_XYZ_FPS_BBOX_DEGENERATE")
    xyz_c = xyz_c / diagonal

    selected = [0]
    min_distance = torch.full((n,), float("inf"), dtype=torch.float64)
    for _ in range(1, count):
        anchor = xyz_c[selected[-1]]
        distance = ((xyz_c - anchor) ** 2).sum(dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        min_distance[torch.tensor(selected, dtype=torch.long)] = -1.0
        best = float(min_distance.max())
        if not math.isfinite(best) or best < 0.0:
            raise RuntimeError("IRIS_V31_XYZ_FPS_SELECTION_INVALID")
        # xyz_c is already in canonical lexicographic order; first exact max is
        # therefore the preregistered deterministic tie-break.
        next_index = int(torch.nonzero(min_distance == best, as_tuple=False)[0, 0])
        selected.append(next_index)

    selected_input = order[torch.tensor(selected, dtype=torch.long)]
    if int(selected_input.numel()) != count or int(torch.unique(selected_input).numel()) != count:
        raise RuntimeError("IRIS_V31_XYZ_FPS_QUOTA_DRIFT")
    return selected_input.to(device=output_device)
