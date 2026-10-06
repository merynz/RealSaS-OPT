from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from compiler.realsas_compiler_core.preproduct_authority_v1 import NormalizationDomainIR
from compiler.realsas_compiler_core.types import RiggingSurfaceIR


TESSA_SURFACE_FEATURE_DIM_V1 = 17


@dataclass(frozen=True)
class TESSAConditioningV1:
    """Deterministic Stage08 + RiggingSurfaceIR -> TESSA tensor adapter.

    Stage08 NormalizationDomainIR owns the canonical object coordinate frame.
    GSA is geometry/evidence authority inside that frame. TESSA must never
    recompute a new object frame from the finite GSA sample cloud because doing
    so can exclude valid source surface points that lie between/just outside GSA
    samples. Teacher topology, rig, skin and Stage18 faces are never inputs.
    """

    features: torch.Tensor  # [N,17]
    surface_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    center: tuple[float, float, float]
    scale: float  # full world extent represented by TESSA [-0.5,+0.5]
    source_geometry_lineage_hash: str
    normalization_hash: str
    coordinate_frame: str


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
    normalization: NormalizationDomainIR,
    component_by_surface_id: dict[str, str] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> TESSAConditioningV1:
    if not surface.surface_nodes:
        raise ValueError("TESSA_GSA_SURFACE_EMPTY")
    if not surface.geometry_lineage_hash:
        raise ValueError("TESSA_GSA_GEOMETRY_LINEAGE_MISSING")
    if not normalization.normalization_hash:
        raise ValueError("TESSA_NORMALIZATION_HASH_MISSING")
    if str(normalization.coordinate_frame) != "REALSAS_OBJECT_FRAME":
        raise ValueError("TESSA_NORMALIZATION_FRAME_UNSUPPORTED")

    center_t = torch.tensor(tuple(map(float, normalization.center_xyz)), dtype=torch.float64)
    half_extent = float(normalization.half_extent)
    if center_t.shape != (3,) or not torch.isfinite(center_t).all():
        raise ValueError("TESSA_NORMALIZATION_CENTER_INVALID")
    if not math.isfinite(half_extent) or half_extent <= 1e-12:
        raise ValueError("TESSA_NORMALIZATION_HALF_EXTENT_INVALID")

    # Stage08 canonical normalization is world = center + normalized * half_extent,
    # with canonical normalized coordinates in [-1,+1]. TESSA's coordinate token
    # vocabulary is [-0.5,+0.5], so use a full scale of 2*half_extent. This is an
    # exact frame conversion, not padding, clipping or a teacher-derived fit.
    scale = 2.0 * half_extent

    ordered = tuple(sorted(surface.surface_nodes, key=lambda n: str(n.surface_id)))
    ids = tuple(str(n.surface_id) for n in ordered)
    if len(ids) != len(set(ids)):
        raise ValueError("TESSA_GSA_SURFACE_ID_DUPLICATE")

    world_xyz = torch.tensor([tuple(map(float, n.P)) for n in ordered], dtype=torch.float64)
    if not torch.isfinite(world_xyz).all():
        raise ValueError("TESSA_GSA_XYZ_NONFINITE")
    xyz = (world_xyz - center_t[None, :]) / scale
    if torch.any(xyz < -0.500001) or torch.any(xyz > 0.500001):
        lo = tuple(map(float, xyz.amin(dim=0).tolist()))
        hi = tuple(map(float, xyz.amax(dim=0).tolist()))
        raise ValueError(f"TESSA_GSA_OUTSIDE_CANONICAL_NORMALIZATION:{lo}:{hi}")

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
            nx,
            ny,
            nz,
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
        scale=float(scale),
        source_geometry_lineage_hash=str(surface.geometry_lineage_hash),
        normalization_hash=str(normalization.normalization_hash),
        coordinate_frame=str(normalization.coordinate_frame),
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
