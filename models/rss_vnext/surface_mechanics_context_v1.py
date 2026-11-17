from __future__ import annotations

"""Shared H(x) -> full-width, face-addressed mechanics evidence for AXIS/MIRA.

A canonical indexed mesh is not a sufficient statistic of the image/scene.
This adapter preserves the ORIGINAL undetached H(x) in its output *as a
separate summand*.  It augments it with surface-identified role, region,
articulation/affordance, component and exact face-incidence evidence.
The Compiler is never called during this differentiable model calculation.

Semantic roles are never treated as cloth material or rig ground truth.
Component ID integers are only used to aggregate within one component;
their numeric values do not become learned category IDs.

This is a learnable v1 candidate, NOT proof of a trained release.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .animation_ready_mesh import RSSvNextStructuralMeshCandidate


class RSSvNextSurfaceMechanicsContextV1(nn.Module):
    def __init__(
        self, *,
        carrier_dim: int,
        role_count: int,
        region_count: int,
        affordance_count: int,
    ) -> None:
        super().__init__()
        if min(carrier_dim, role_count, region_count, affordance_count) < 1:
            raise ValueError("RSS_VNEXT_MECHANICS_CONTEXT_DIMENSION")
        self.carrier_dim = carrier_dim
        self.role_count = role_count
        self.region_count = region_count
        self.affordance_count = affordance_count
        # H and component H; 3 typed posteriors; normal, relative geometry,
        # degree, boundary. Same H dimension at output => existing MIRA ABI.
        width = 2 * carrier_dim + role_count + region_count + affordance_count + 8
        self.conditioner = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, carrier_dim),
            nn.GELU(),
            nn.Linear(carrier_dim, carrier_dim),
        )
        # Preserve the existing full-width raw-H MIRA input at initialization.
        # Trained context enters only via an explicitly checkpointed adapter.
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def forward(self, mesh: RSSvNextStructuralMeshCandidate) -> Tensor:
        mesh.validate()
        h = mesh.local_features
        xyz = mesh.rest_vertices_iris
        faces = mesh.faces
        roles = mesh.presentation_role_logits
        regions = mesh.semantic_logits
        afford = mesh.affordance_logits
        if (
            h.shape[0] != 1 or h.shape[-1] != self.carrier_dim
            or roles.shape != (1, h.shape[1], self.role_count)
            or regions.shape != (1, h.shape[1], self.region_count)
            or afford.shape != (1, h.shape[1], self.affordance_count)
            or any(t.device != h.device for t in (
                xyz, faces, roles, regions, afford, mesh.normals,
                mesh.vertex_component_ids, mesh.face_component_ids,
            ))
            or any(t.dtype != h.dtype for t in (
                xyz, roles, regions, afford, mesh.normals,
            ))
            or not bool(torch.isfinite(h).all())
            or not bool(torch.isfinite(roles).all())
            or not bool(torch.isfinite(regions).all())
            or not bool(torch.isfinite(afford).all())
        ):
            raise ValueError("RSS_VNEXT_MECHANICS_CONTEXT_ADDRESS_OR_FIELD")
        v = h.shape[1]
        components, inverse = torch.unique(
            mesh.vertex_component_ids, sorted=True, return_inverse=True,
        )
        ncomp = components.numel()
        counter = h.new_zeros(ncomp, 1).index_add(
            0, inverse, h.new_ones(v, 1),
        ).clamp_min(1.0)
        component_h = h.new_zeros(ncomp, self.carrier_dim).index_add(
            0, inverse, h[0],
        ) / counter
        component_xyz = xyz.new_zeros(ncomp, 3).index_add(
            0, inverse, xyz[0],
        ) / counter
        rel_xyz = xyz[0] - component_xyz[inverse]
        face_vertices = faces.reshape(-1)
        valence = h.new_zeros(v, 1).index_add(
            0, face_vertices, h.new_ones(face_vertices.numel(), 1),
        )
        # Boundary edges: sorted vertex pairs with exactly one incident face.
        # This detects open hems without inferring garment material.
        edges = torch.cat((
            faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
        ), dim=0).sort(dim=-1).values
        unique_edges, edge_occurrences = torch.unique(
            edges, dim=0, return_counts=True,
        )
        if bool((edge_occurrences > 2).any()):
            raise ValueError("RSS_VNEXT_MECHANICS_CONTEXT_NONMANIFOLD_EDGE")
        boundaries = unique_edges[edge_occurrences == 1].reshape(-1)
        on_boundary = h.new_zeros(v, 1)
        if boundaries.numel():
            on_boundary.index_fill_(0, boundaries, 1.0)
        normalized_valence = valence / (1.0 + valence)
        dense = torch.cat((
            h[0], component_h[inverse],
            F.softmax(roles[0], dim=-1),
            F.softmax(regions[0], dim=-1),
            F.softmax(afford[0], dim=-1),
            mesh.normals[0], rel_xyz,
            normalized_valence, on_boundary,
        ), dim=-1)
        # New adapter can learn differential skin at spatially coincident
        # body+cloth vertices while retaining all raw neural scene features.
        residual = self.conditioner(dense)[None]
        out = h + residual
        if out.shape != h.shape or not bool(torch.isfinite(out).all()):
            raise RuntimeError("RSS_VNEXT_MECHANICS_CONTEXT_OUTPUT_DRIFT")
        return out


__all__ = ["RSSvNextSurfaceMechanicsContextV1"]
