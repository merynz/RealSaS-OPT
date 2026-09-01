from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from realsas_compiler_core.types import (
    QualifiedSkeletonIR,
    RiggingSurfaceIR,
    SkinInfluenceProposal,
    SkinProposalIR,
)


@dataclass(frozen=True)
class ArachneDemoConfig:
    hidden_dim: int = 192
    depth: int = 4
    top_k: int = 8
    temperature: float = 1.0


def canonical_joint_order(skeleton: QualifiedSkeletonIR) -> tuple[str, ...]:
    by_parent: dict[str | None, list[str]] = {}
    for joint in skeleton.joints:
        by_parent.setdefault(joint.parent_canonical_id, []).append(joint.canonical_joint_id)
    for children in by_parent.values():
        children.sort()
    out: list[str] = []
    queue = [skeleton.root_id]
    seen: set[str] = set()
    while queue:
        joint_id = queue.pop(0)
        if joint_id in seen:
            raise ValueError("cycle/duplicate in qualified skeleton")
        seen.add(joint_id)
        out.append(joint_id)
        queue.extend(by_parent.get(joint_id, ()))
    if len(out) != len(skeleton.joints):
        raise ValueError("qualified skeleton is disconnected")
    return tuple(out)


def _normalized_pair(surface_p: torch.Tensor, joint_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pmin = surface_p.amin(dim=0, keepdim=True)
    pmax = surface_p.amax(dim=0, keepdim=True)
    center = 0.5 * (pmin + pmax)
    scale = (pmax - pmin).amax().clamp_min(1e-6)
    return (surface_p - center) / scale, (joint_p - center) / scale


class ArachneDemoV1(nn.Module):
    """Generic geometry+skeleton-conditioned surface influence proposer.

    The model proposes semantic influence evidence. Legal references, simplex
    authority, bounded correction and final skin identity remain Compiler-owned.
    """

    def __init__(self, cfg: ArachneDemoConfig = ArachneDemoConfig()):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        input_dim = 14
        for i in range(cfg.depth):
            layers.extend(
                [
                    nn.Linear(input_dim if i == 0 else cfg.hidden_dim, cfg.hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(cfg.hidden_dim, 1))
        self.pair_mlp = nn.Sequential(*layers)

    def forward(
        self,
        surface_p: torch.Tensor,
        joint_p: torch.Tensor,
        parent_p: torch.Tensor,
        is_root: torch.Tensor,
    ) -> torch.Tensor:
        if surface_p.ndim != 2 or surface_p.shape[-1] != 3:
            raise ValueError("surface_p must be [N,3]")
        if joint_p.ndim != 2 or joint_p.shape[-1] != 3:
            raise ValueError("joint_p must be [J,3]")
        if parent_p.shape != joint_p.shape or is_root.shape != (joint_p.shape[0],):
            raise ValueError("bad skeleton conditioning shapes")

        surface_n, joint_n = _normalized_pair(surface_p, joint_p)
        _, parent_n = _normalized_pair(surface_p, parent_p)
        n, j = surface_n.shape[0], joint_n.shape[0]
        s = surface_n[:, None, :].expand(n, j, 3)
        q = joint_n[None, :, :].expand(n, j, 3)
        rel = s - q
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        parent_rel = (joint_n - parent_n)[None, :, :].expand(n, j, 3)
        root = is_root[None, :, None].expand(n, j, 1)
        features = torch.cat([s, q, rel, dist, parent_rel, root], dim=-1)
        return self.pair_mlp(features).squeeze(-1) / self.cfg.temperature

    @torch.no_grad()
    def propose(
        self,
        surface: RiggingSurfaceIR,
        skeleton: QualifiedSkeletonIR,
        *,
        device: torch.device | str = "cpu",
    ) -> SkinProposalIR:
        device = torch.device(device)
        self.eval()
        if not surface.surface_nodes:
            raise ValueError("Arachne requires a non-empty surface")
        order = canonical_joint_order(skeleton)
        by_joint = {j.canonical_joint_id: j for j in skeleton.joints}
        surface_p = torch.tensor([n.P for n in surface.surface_nodes], dtype=torch.float32, device=device)
        joint_p = torch.tensor([by_joint[j].position for j in order], dtype=torch.float32, device=device)
        parent_p_rows = []
        root_flags = []
        for jid in order:
            joint = by_joint[jid]
            if joint.parent_canonical_id is None:
                parent_p_rows.append(joint.position)
                root_flags.append(1.0)
            else:
                parent_p_rows.append(by_joint[joint.parent_canonical_id].position)
                root_flags.append(0.0)
        parent_p = torch.tensor(parent_p_rows, dtype=torch.float32, device=device)
        is_root = torch.tensor(root_flags, dtype=torch.float32, device=device)
        logits = self(surface_p, joint_p, parent_p, is_root)
        weights = torch.softmax(logits, dim=-1)

        influences: list[SkinInfluenceProposal] = []
        top_k = min(max(1, self.cfg.top_k), len(order))
        for row_index, node in enumerate(surface.surface_nodes):
            row = weights[row_index]
            values, indices = torch.topk(row, k=top_k, largest=True, sorted=True)
            values = values / values.sum().clamp_min(1e-12)
            for value, index in zip(values.tolist(), indices.tolist()):
                influences.append(
                    SkinInfluenceProposal(
                        surface_id=node.surface_id,
                        canonical_joint_id=order[index],
                        weight=float(value),
                    )
                )

        return SkinProposalIR(
            influences=tuple(influences),
            surface_binding_hash=surface.geometry_lineage_hash,
            skeleton_binding_hash=skeleton.skeleton_lineage_hash,
            model_provenance="RealSaS.ArachneDemoV1.experimental",
            metadata={
                "generalization_claim": False,
                "semantic_influence_owner": "ARACHNE_DEMO_LEARNED_PROPOSAL",
                "simplex_authority_owner": "COMPILER",
                "joint_serialization": order,
                "top_k": top_k,
            },
        )
