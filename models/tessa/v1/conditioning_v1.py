from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from compiler.realsas_compiler_core.types import RiggingSurfaceIR


TESSA_SURFACE_FEATURE_DIM_V1 = 17


@dataclass(frozen=True)
class TESSAConditioningV1:
    """Deterministic RiggingSurfaceIR -> TESSA tensor adapter.

    The adapter preserves GSA as geometry/evidence authority.  It does not use
    teacher topology, teacher rig, teacher skin, or current Stage18 faces.
    """

    features: torch.Tensor  # [N,17]
    surface_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    center: tuple[float, float, float]
    scale: float
    source_geometry_lineage_hash: str


def _normal(node) -> tuple[float, float, float, float]:
    n = node.derived_normal
    if n is None:
        return 0.0, 0.0, 0.0, 0.0
    x = torch.tensor(tuple(map(float, n)), dtype=torch.float64)
    length = float(torch.linalg.vector_norm(x))
    if not math.isfinite(length) or length <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    x = x / length
    return float(x[0]), float(x[1]), float(x[2]), 1.0


def build_tessa_conditioning_v1(
    surface: RiggingSurfaceIR,
    *,
    component_by_surface_id: dict[str, str] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> TESSAConditioningV1:
    if not surface.surface_nodes:
        raise ValueError("TESSA_GSA_SURFACE_EMPTY")
    if not surface.geometry_lineage_hash:
        raise ValueError("TESSA_GSA_GEOMETRY_LINEAGE_MISSING")

    ordered = tuple(sorted(surface.surface_nodes, key=lambda n: str(n.surface_id)))
    ids = tuple(str(n.surface_id) for n in ordered)
    if len(ids) != len(set(ids)):
        raise ValueError("TESSA_GSA_SURFACE_ID_DUPLICATE")

    xyz = torch.tensor([tuple(map(float, n.P)) for n in ordered], dtype=torch.float64)
    if not torch.isfinite(xyz).all():
        raise ValueError("TESSA_GSA_XYZ_NONFINITE")
    lo = xyz.amin(dim=0)
    hi = xyz.amax(dim=0)
    center_t = 0.5 * (lo + hi)
    scale = float((hi - lo).amax())
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("TESSA_GSA_SCALE_DEGENERATE")
    xyz = (xyz - center_t[None, :]) / scale

    degree = {sid: 0 for sid in ids}
    for rel in surface.local_relations:
        a, b = str(rel.a_surface_id), str(rel.b_surface_id)
        if a in degree:
            degree[a] += 1
        if b in degree:
            degree[b] += 1
    max_degree = max(max(degree.values()), 1)

    rows = []
    components = []
    component_by_surface_id = component_by_surface_id or {}
    for index, node in enumerate(ordered):
        nx, ny, nz, normal_valid = _normal(node)
        view_mask = [0.0] * 8
        for view in node.support_views:
            vi = int(view)
            if vi < 0 or vi >= 8:
                raise ValueError("TESSA_GSA_SUPPORT_VIEW_OUT_OF_RANGE")
            view_mask[vi] = 1.0
        persistence = 1.0 if node.persistence_group_id is not None else 0.0
        sid = str(node.surface_id)
        row = [
            float(xyz[index, 0]),
            float(xyz[index, 1]),
            float(xyz[index, 2]),
            nx, ny, nz,
            normal_valid,
            *view_mask,
            float(degree[sid]) / float(max_degree),
            persistence,
        ]
        if len(row) != TESSA_SURFACE_FEATURE_DIM_V1:
            raise AssertionError("TESSA_SURFACE_FEATURE_DIM_DRIFT")
        rows.append(row)
        components.append(str(component_by_surface_id.get(sid, "UNASSIGNED")))

    features = torch.tensor(rows, dtype=dtype, device=device)
    return TESSAConditioningV1(
        features=features,
        surface_ids=ids,
        component_ids=tuple(components),
        center=tuple(map(float, center_t.tolist())),
        scale=scale,
        source_geometry_lineage_hash=str(surface.geometry_lineage_hash),
    )


def denormalize_tessa_xyz_v1(
    xyz_normalized: torch.Tensor,
    conditioning: TESSAConditioningV1,
) -> torch.Tensor:
    center = torch.tensor(
        conditioning.center,
        dtype=xyz_normalized.dtype,
        device=xyz_normalized.device,
    )
    return xyz_normalized * float(conditioning.scale) + center
