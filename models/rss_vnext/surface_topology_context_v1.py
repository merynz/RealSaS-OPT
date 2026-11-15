from __future__ import annotations

"""Differentiable IRIS FlexiCubes face incidence context for TESSA.

The input is a predicted surface, not a source mesh and not canonical topology.
No teacher joints/skin or privileged source geometry enter this path.  Vertex
incidence and individual face geometry make the existing TESSA surface stream
sensitive to connectivity, not merely the positions of isolated samples.
"""

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def iris_surface_incidence_v1(
    xyz: Tensor, faces: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return (vertex [1,V,8], face [1,F,10]) typed geometry descriptors.

    Vertex: area-weighted normal xyz, log1p incident-face count, log1p area,
    incident-edge mean and standard deviation, mean face quality.
    Face: centroid xyz, unit normal xyz, three edge lengths, face area.
    The descriptors preserve gradients to predicted vertices, never faces
    (face connectivity is discrete by construction).
    """
    if (
        xyz.ndim != 3 or xyz.shape[0] != 1 or xyz.shape[-1] != 3
        or faces.ndim != 2 or faces.shape[-1] != 3
        or faces.dtype != torch.long or not faces.numel()
        or faces.device != xyz.device or not torch.is_floating_point(xyz)
        or not bool(torch.isfinite(xyz).all())
    ):
        raise ValueError("RSS_VNEXT_IRIS_SURFACE_INCIDENCE_INPUT")
    n = xyz.shape[1]
    if n < 3 or int(faces.min()) < 0 or int(faces.max()) >= n:
        raise ValueError("RSS_VNEXT_IRIS_SURFACE_INCIDENCE_FACE_ADDRESS")
    a, b, c = faces.unbind(-1)
    tri = xyz[0, faces]  # [F,3,3]
    e01, e12, e20 = (
        torch.linalg.vector_norm(tri[:, 1] - tri[:, 0], dim=-1),
        torch.linalg.vector_norm(tri[:, 2] - tri[:, 1], dim=-1),
        torch.linalg.vector_norm(tri[:, 0] - tri[:, 2], dim=-1),
    )
    cross = torch.cross(
        tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1,
    )
    double_area = torch.linalg.vector_norm(cross, dim=-1)
    area = double_area * 0.5
    normal = cross / double_area.clamp_min(1.e-10).unsqueeze(-1)
    quality = (
        4.0 * math.sqrt(3.0) * area
        / (e01.square() + e12.square() + e20.square()).clamp_min(1.e-10)
    ).clamp(0.0, 1.0)
    count = xyz.new_zeros(n)
    area_sum = xyz.new_zeros(n)
    normal_sum = xyz.new_zeros(n, 3)
    edges_sum = xyz.new_zeros(n)
    edges_sq_sum = xyz.new_zeros(n)
    quality_sum = xyz.new_zeros(n)
    for vertices, first, second in (
        (a, e01, e20),
        (b, e01, e12),
        (c, e12, e20),
    ):
        count = count.index_add(0, vertices, torch.ones_like(area))
        area_sum = area_sum.index_add(0, vertices, area / 3.0)
        normal_sum = normal_sum.index_add(0, vertices, cross)
        edges_sum = edges_sum.index_add(0, vertices, first + second)
        edges_sq_sum = edges_sq_sum.index_add(
            0, vertices, first.square() + second.square(),
        )
        quality_sum = quality_sum.index_add(0, vertices, quality)
    mean_edge = edges_sum / (2.0 * count).clamp_min(1.0)
    mean_sq_edge = edges_sq_sum / (2.0 * count).clamp_min(1.0)
    edge_std = (mean_sq_edge - mean_edge.square()).clamp_min(0.0).sqrt()
    vertex = torch.cat((
        F.normalize(normal_sum, dim=-1, eps=1.e-10),
        count.log1p().unsqueeze(-1),
        area_sum.log1p().unsqueeze(-1),
        mean_edge.unsqueeze(-1),
        edge_std.unsqueeze(-1),
        (quality_sum / count.clamp_min(1.0)).unsqueeze(-1),
    ), dim=-1)
    face = torch.cat((
        tri.mean(dim=1), normal,
        torch.stack((e01, e12, e20), dim=-1),
        area.unsqueeze(-1),
    ), dim=-1)
    if vertex.shape != (n, 8) or face.shape != (len(faces), 10):
        raise RuntimeError("RSS_VNEXT_IRIS_SURFACE_INCIDENCE_OUTPUT")
    if not bool(torch.isfinite(vertex).all() and torch.isfinite(face).all()):
        raise ValueError("RSS_VNEXT_IRIS_SURFACE_INCIDENCE_NONFINITE")
    return vertex[None], face[None]


__all__ = ["iris_surface_incidence_v1"]
