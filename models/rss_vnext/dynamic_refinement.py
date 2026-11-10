from __future__ import annotations

"""RSS.vNext motion-conditioned structural mesh refinement (candidate-only).

One shared carrier, one exact mesh, trainable refinement that reads predicted
joint/skin effects. Source poses are ONLY teacher-side loss evidence; runtime
refinement never consumes privileged teacher motions. No topology, UV, vertex
or component identity is silently rewritten.

This stage refines continuous geometry + per-vertex skin weights. If the
discrete faces are wrong, the upstream structural candidate must abstain
or generate an alternative: coordinate refinement cannot solve bad topology.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .animation_ready_mesh import RSSvNextStructuralMeshCandidate


@dataclass(frozen=True)
class RSSvNextDynamicRefinementConfig:
    carrier_dim: int
    joint_dim: int
    width: int = 256
    layers: int = 3
    max_vertex_delta: float = 0.025
    max_weight_logit_delta: float = 0.5

    def validate(self):
        if min(self.carrier_dim, self.joint_dim, self.width, self.layers) <= 0:
            raise ValueError("RSS_VNEXT_DYNAMIC_DIM_INVALID")
        if not 0 < self.max_vertex_delta <= 0.1 or not 0 < self.max_weight_logit_delta <= 3:
            raise ValueError("RSS_VNEXT_DYNAMIC_DELTA_INVALID")


@dataclass
class RSSvNextDynamicRefinementOutput:
    vertices: Tensor        # [1,V,3], exact original vertex IDs
    weights: Tensor         # [1,V,J], exact original joint order
    displacement: Tensor    # [1,V,3], bounded
    weight_delta_logits: Tensor
    readiness_logits: Tensor  # [1,V,4], learned evidence only
    source_mesh: RSSvNextStructuralMeshCandidate
    authority: str = "UNQUALIFIED_DYNAMIC_MESH_PROPOSAL"

    def validate(self):
        src = self.source_mesh
        if self.vertices.shape != src.rest_vertices_iris.shape or self.displacement.shape != self.vertices.shape:
            raise ValueError("RSS_VNEXT_DYNAMIC_VERTEX_IDENTITY")
        if self.weights.ndim != 3 or self.weights.shape[:2] != self.vertices.shape[:2]:
            raise ValueError("RSS_VNEXT_DYNAMIC_WEIGHT_DOMAIN")
        if self.readiness_logits.shape != self.vertices.shape[:2] + (4,):
            raise ValueError("RSS_VNEXT_DYNAMIC_QUALITY_SHAPE")
        if not all(bool(torch.isfinite(x).all()) for x in (self.vertices, self.weights, self.displacement, self.readiness_logits)):
            raise ValueError("RSS_VNEXT_DYNAMIC_NUMERIC")
        if bool((self.weights < -1e-6).any()) or not torch.allclose(
            self.weights.sum(-1), torch.ones_like(self.weights[..., 0]), atol=1e-5
        ):
            raise ValueError("RSS_VNEXT_DYNAMIC_WEIGHT_SIMPLEX")
        if self.authority != "UNQUALIFIED_DYNAMIC_MESH_PROPOSAL":
            raise ValueError("RSS_VNEXT_DYNAMIC_PREMATURE_AUTHORITY")


class _ComponentSafeEdgeBlock(nn.Module):
    """Local mesh-neighbor aggregation, O(E) edges; never joins components."""

    def __init__(self, width: int):
        super().__init__()
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.edge = nn.Sequential(
            nn.Linear(4, width), nn.GELU(), nn.Linear(width, width),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width * 2),
            nn.GELU(), nn.Linear(width * 2, width),
        )
        self.out_norm = nn.LayerNorm(width)
        self.scale = math.sqrt(width)

    def forward(self, x: Tensor, xyz: Tensor, edge_src: Tensor, edge_dst: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("RSS_VNEXT_DYNAMIC_EDGE_BATCH_ONE")
        if edge_src.numel() == 0:
            return x
        feat = x[0]
        delta = xyz[0, edge_dst] - xyz[0, edge_src]
        edge_context = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), dim=-1)
        score = ((self.q(feat[edge_dst]) * self.k(feat[edge_src])).sum(-1)
                 / self.scale)
        # Sparse neighbor softmax per destination, stable against degree.
        max_per_dst = torch.full(
            (x.shape[1],), -1e10, dtype=score.dtype, device=score.device,
        ).scatter_reduce(0, edge_dst, score, reduce="amax", include_self=True)
        unnormalized = torch.exp(score - max_per_dst[edge_dst])
        denom = torch.zeros(
            x.shape[1], device=score.device, dtype=score.dtype,
        ).index_add(0, edge_dst, unnormalized)
        alpha = unnormalized / denom[edge_dst].clamp_min(1e-8)
        msg = self.v(feat[edge_src]) + self.edge(edge_context)
        gathered = torch.zeros_like(feat).index_add(
            0, edge_dst, alpha[:, None] * msg,
        )
        update = self.mix(torch.cat((feat, gathered), dim=-1))
        return self.out_norm(x + update.unsqueeze(0))


def _indexed_edges(faces: Tensor, vertex_component_ids: Tensor) -> tuple[Tensor, Tensor]:
    if faces.ndim != 2 or faces.shape[-1] != 3 or not faces.numel():
        raise ValueError("RSS_VNEXT_DYNAMIC_FACE_SET_INVALID")
    directed = torch.cat(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0,
    )
    directed = torch.cat((directed, directed.flip(-1)), dim=0).unique(dim=0)
    src, dst = directed[:, 0], directed[:, 1]
    if bool((vertex_component_ids[src] != vertex_component_ids[dst]).any()):
        raise ValueError("RSS_VNEXT_DYNAMIC_CROSS_COMPONENT_EDGE")
    return src, dst


class RSSvNextDynamicRefinement(nn.Module):
    """Refines rest shape *and* influences conditioned on predicted mechanics.

    Inputs:
      mesh: produced by source-observed FRONT/BACK + structural proposal only;
      weights: predicted [1,V,J] simplex on this same mesh (not teacher weights);
      joints: predicted mechanics hidden [1,J,C], ordered like weights.
    Outputs: [1,V,3] refined XYZ, [1,V,J] refined influence distribution.
    """
    def __init__(self, cfg: RSSvNextDynamicRefinementConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        # Geometry H, rest point, initial skin entropy and local bone aggregation
        self.vertex_in = nn.Sequential(
            nn.LayerNorm(cfg.carrier_dim + cfg.joint_dim + 4),
            nn.Linear(cfg.carrier_dim + cfg.joint_dim + 4, cfg.width),
            nn.GELU(), nn.LayerNorm(cfg.width),
        )
        self.blocks = nn.ModuleList(
            [_ComponentSafeEdgeBlock(cfg.width) for _ in range(cfg.layers)]
        )
        self.vertex_head = nn.Linear(cfg.width, 3)
        self.vertex_context = nn.Linear(cfg.width, cfg.joint_dim)
        self.joint_project = nn.Linear(cfg.joint_dim, cfg.joint_dim)
        self.quality_head = nn.Linear(cfg.width, 4)
        # Conservative zero-start: cannot make huge edits to trained static geometry.
        nn.init.zeros_(self.vertex_head.weight)
        nn.init.zeros_(self.vertex_head.bias)
        nn.init.zeros_(self.vertex_context.weight)
        nn.init.zeros_(self.vertex_context.bias)

    def forward(
        self,
        mesh: RSSvNextStructuralMeshCandidate,
        weights: Tensor,
        joint_features: Tensor,
    ) -> RSSvNextDynamicRefinementOutput:
        mesh.validate()
        xyz = mesh.rest_vertices_iris
        if weights.ndim != 3 or weights.shape[:2] != xyz.shape[:2]:
            raise ValueError("RSS_VNEXT_DYNAMIC_WEIGHT_VERTEX_BINDING")
        if joint_features.ndim != 3 or joint_features.shape[0] != 1 or joint_features.shape[1] != weights.shape[-1]:
            raise ValueError("RSS_VNEXT_DYNAMIC_JOINT_BINDING")
        if joint_features.shape[-1] != self.cfg.joint_dim or mesh.local_features.shape[-1] != self.cfg.carrier_dim:
            raise ValueError("RSS_VNEXT_DYNAMIC_FEATURE_WIDTH")
        if not all(bool(torch.isfinite(x).all()) for x in (weights, joint_features)):
            raise ValueError("RSS_VNEXT_DYNAMIC_NONFINITE")
        if bool((weights < -1e-6).any()) or bool((weights.sum(-1) - 1).abs().max() > 1e-4):
            raise ValueError("RSS_VNEXT_DYNAMIC_INFLUENCE_SIMPLEX")
        src, dst = _indexed_edges(mesh.faces, mesh.vertex_component_ids)
        aggregated_joint = weights @ joint_features
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1, keepdim=True)
        x = self.vertex_in(torch.cat((
            mesh.local_features, xyz, entropy, aggregated_joint,
        ), dim=-1))
        for block in self.blocks:
            x = block(x, xyz, src, dst)
        # Bounded vertex displacement, preserves vertex count, face connectivity
        # and component IDs; it is NOT a discrete remeshing optimizer.
        room = (1 - xyz.abs()).clamp_min(0)
        displacement = room * self.cfg.max_vertex_delta * torch.tanh(self.vertex_head(x))
        refined_xyz = xyz + displacement

        # Dense normalized weights, computed with the original declared J order.
        # Forward uses differentiable log-probabilities (no hard labels).
        projected = self.vertex_context(x)  # [1,V,C]
        joint = self.joint_project(joint_features)
        logits_delta = (projected @ joint.transpose(1, 2)) / math.sqrt(self.cfg.joint_dim)
        logits_delta = self.cfg.max_weight_logit_delta * torch.tanh(logits_delta)
        refined_weights = torch.softmax(weights.clamp_min(1e-8).log() + logits_delta, dim=-1)
        result = RSSvNextDynamicRefinementOutput(
            vertices=refined_xyz, weights=refined_weights,
            displacement=displacement, weight_delta_logits=logits_delta,
            readiness_logits=self.quality_head(x), source_mesh=mesh,
        )
        result.validate()
        return result


__all__ = [
    "RSSvNextDynamicRefinementConfig", "RSSvNextDynamicRefinementOutput",
    "RSSvNextDynamicRefinement",
]
