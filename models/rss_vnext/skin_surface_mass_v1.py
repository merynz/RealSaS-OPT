from __future__ import annotations

"""Canonical TESSA surface-measure conditioning for MIRA's *teacher encoder*.

The source-native skin weights still come from independently qualified source
data. The mesh here is the frozen TESSA product mesh, used only to determine
how vertex samples represent surface area. This tensor MUST NOT qualify
source geometry, animation, attachment, or any release evidence.

Both codec-first MIRA training and stationary AXIS token handoff use this same
rule; otherwise equal weight arrays would produce different FSQ target IDs.
"""

import torch
from torch import Tensor

POOLING_ABI = "RealSaS.RSSvNext.MIRATESSASurfaceMass.v2"


def tessa_vertex_surface_mass_v1(rest_vertices: Tensor, faces: Tensor) -> Tensor:
    """Detached per-vertex lumped triangle area, shape [1,V].

    Unreferenced vertices have zero mass and cannot dominate EITHER latent
    aggregator (mean or masked max). Uses TESSA's canonical face ordering; masks and
    source permissions remain independent concerns.
    """
    if (
        not isinstance(rest_vertices, Tensor)
        or not isinstance(faces, Tensor)
        or rest_vertices.ndim != 3
        or rest_vertices.shape[0] != 1
        or rest_vertices.shape[-1] != 3
        or not torch.is_floating_point(rest_vertices)
        or faces.ndim != 2
        or faces.shape[-1] != 3
        or faces.dtype != torch.long
        or faces.shape[0] < 1
        or faces.device != rest_vertices.device
    ):
        raise ValueError("RSS_VNEXT_MIRA_SURFACE_MASS_ABI_INVALID")
    n = rest_vertices.shape[1]
    if (
        n < 3
        or not bool(torch.isfinite(rest_vertices).all())
        or int(faces.min()) < 0
        or int(faces.max()) >= n
        or bool((faces[:, 0] == faces[:, 1]).any())
        or bool((faces[:, 0] == faces[:, 2]).any())
        or bool((faces[:, 1] == faces[:, 2]).any())
    ):
        raise ValueError("RSS_VNEXT_MIRA_SURFACE_MASS_MESH_INVALID")
    with torch.no_grad():
        positions = rest_vertices.detach()[0].to(dtype=torch.float32)
        tri = positions[faces]
        areas = 0.5 * torch.linalg.cross(
            tri[:, 1] - tri[:, 0],
            tri[:, 2] - tri[:, 0],
        ).norm(dim=-1)
        if not bool(torch.isfinite(areas).all()):
            raise ValueError("RSS_VNEXT_MIRA_SURFACE_MASS_NONFINITE")
        # Degenerate triangles carry no surface measure; if all triangles
        # degenerate we cannot define a surface-conditioning distribution.
        masses = torch.zeros((n,), device=positions.device, dtype=torch.float32)
        masses.index_add_(
            0,
            faces.reshape(-1),
            (areas[:, None].expand(-1, 3) / 3.0).reshape(-1),
        )
        total = masses.sum()
        if not bool(torch.isfinite(total)) or float(total) <= 1e-10:
            raise ValueError("RSS_VNEXT_MIRA_SURFACE_MASS_NO_VALID_AREA")
        return (masses / total).unsqueeze(0)


__all__ = ["POOLING_ABI", "tessa_vertex_surface_mass_v1"]
