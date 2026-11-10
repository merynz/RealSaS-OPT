from __future__ import annotations

"""Source-preserving *counterfactual* topology candidates for motion training.

Research apparatus, NOT a product remesher and NOT a Truth Foundry schema.
Only a strictly internal, qualified, unprotected, component-consistent
near-coplanar two-triangle edge may be flipped. A generated counterfactual is
still UNQUALIFIED until static and motion courts evaluate it.

The purpose is to create distinct structural token policies which share the
same source geometry/rig/skin/pose supervision, so the model can learn *why*
one face decision moves better, instead of blindly imitating source faces.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class RSSvNextEdgeFlipCandidate:
    faces: Tensor                   # exactly same V, F, component indexing
    removed_edge: tuple[int, int]
    new_edge: tuple[int, int]
    face_indices: tuple[int, int]
    authority: str = "UNQUALIFIED_COUNTERFACTUAL_TRAINING_CANDIDATE"


def propose_component_safe_edge_flips(
    vertices: Tensor,
    faces: Tensor,
    component_id: Tensor,
    *,
    protected_edges: frozenset[tuple[int, int]],
    protection_court_identity: str,
    max_candidates: int = 8,
    min_normal_agreement: float = 0.995,
    min_triangle_area2: float = 1e-8,
) -> tuple[RSSvNextEdgeFlipCandidate, ...]:
    """Enumerate bounded candidate connectivity alternatives.

    The protection court MUST certify a set of edges that cannot be rewired
    because they are source seams, control/attachment boundaries or otherwise
    have mechanically distinct ownership. This function does not infer that.
    """
    if not protection_court_identity:
        raise ValueError("RSS_VNEXT_EDGE_FLIP_PROTECTION_COURT_REQUIRED")
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or not torch.isfinite(vertices).all():
        raise ValueError("RSS_VNEXT_EDGE_FLIP_VERTEX_ARRAY_INVALID")
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.dtype != torch.long or not faces.numel():
        raise ValueError("RSS_VNEXT_EDGE_FLIP_FACE_ARRAY_INVALID")
    if component_id.shape != (vertices.shape[0],) or component_id.dtype != torch.long:
        raise ValueError("RSS_VNEXT_EDGE_FLIP_COMPONENT_ARRAY_INVALID")
    if bool((faces < 0).any()) or int(faces.max()) >= vertices.shape[0]:
        raise ValueError("RSS_VNEXT_EDGE_FLIP_FACE_RANGE")
    if max_candidates < 1 or max_candidates > 64:
        raise ValueError("RSS_VNEXT_EDGE_FLIP_BUDGET_INVALID")
    if not 0.9 <= min_normal_agreement <= 1:
        raise ValueError("RSS_VNEXT_EDGE_FLIP_NORMAL_POLICY")
    if not 0 < min_triangle_area2 < 1:
        raise ValueError("RSS_VNEXT_EDGE_FLIP_AREA_POLICY")
    for edge in protected_edges:
        if len(edge) != 2 or edge[0] >= edge[1] or edge[0] < 0 or edge[1] >= vertices.shape[0]:
            raise ValueError("RSS_VNEXT_EDGE_FLIP_PROTECTED_EDGE_INVALID")

    # All local directed edges; reject non-manifold (>2) adjacency outright.
    edge_to_incident: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for fi, (u, v, w) in enumerate(faces.tolist()):
        if len({u, v, w}) != 3:
            raise ValueError("RSS_VNEXT_EDGE_FLIP_DEGENERATE_SOURCE_FACE")
        if not (component_id[u] == component_id[v] == component_id[w]):
            raise ValueError("RSS_VNEXT_EDGE_FLIP_SOURCE_CROSS_COMPONENT_FACE")
        for a, b, c in ((u, v, w), (v, w, u), (w, u, v)):
            edge_to_incident.setdefault(tuple(sorted((a, b))), []).append((fi, a, b, c))
    if any(len(inc) > 2 for inc in edge_to_incident.values()):
        raise ValueError("RSS_VNEXT_EDGE_FLIP_NONMANIFOLD_SOURCE")
    candidates: list[RSSvNextEdgeFlipCandidate] = []

    for shared in sorted(edge_to_incident):
        if shared in protected_edges:
            continue
        inc = edge_to_incident[shared]
        if len(inc) != 2:
            continue    # only internal manifold edges
        (fi, a, b, c), (fj, a2, b2, d) = inc
        if fi == fj or (a2, b2) != (b, a) or c == d:
            continue    # opposite winding required
        # No pre-existing diagonal or unrelated edge can become duplicated.
        new_edge = tuple(sorted((c, d)))
        if new_edge in edge_to_incident or new_edge in protected_edges:
            continue
        p = vertices
        tri1 = p[faces[fi]]
        tri2 = p[faces[fj]]
        n1 = torch.cross(tri1[1] - tri1[0], tri1[2] - tri1[0], dim=-1)
        n2 = torch.cross(tri2[1] - tri2[0], tri2[2] - tri2[0], dim=-1)
        if min(float(n1.norm()), float(n2.norm())) <= min_triangle_area2:
            continue
        n1 = F.normalize(n1, dim=-1)
        n2 = F.normalize(n2, dim=-1)
        if float((n1 * n2).sum()) < min_normal_agreement:
            continue
        # New triangle orientation preserves local outward-facing normals.
        faces_i = torch.tensor((c, d, b), dtype=torch.long, device=faces.device)
        faces_j = torch.tensor((d, c, a), dtype=torch.long, device=faces.device)
        nt1, nt2 = p[faces_i], p[faces_j]
        nn1 = torch.cross(nt1[1] - nt1[0], nt1[2] - nt1[0], dim=-1)
        nn2 = torch.cross(nt2[1] - nt2[0], nt2[2] - nt2[0], dim=-1)
        if (min(float(nn1.norm()), float(nn2.norm())) <= min_triangle_area2
                or float((F.normalize(nn1, dim=-1) * n1).sum()) < min_normal_agreement
                or float((F.normalize(nn2, dim=-1) * n1).sum()) < min_normal_agreement):
            continue
        candidate = faces.clone()
        candidate[fi] = faces_i
        candidate[fj] = faces_j
        # No face duplicated after rewire, even if topology is unusual.
        canonical = [tuple(sorted(row)) for row in candidate.tolist()]
        if len(set(canonical)) != len(canonical):
            continue
        candidates.append(RSSvNextEdgeFlipCandidate(
            faces=candidate,
            removed_edge=shared, new_edge=new_edge,
            face_indices=(fi, fj),
        ))
        if len(candidates) >= max_candidates:
            break
    return tuple(candidates)


__all__ = ["RSSvNextEdgeFlipCandidate", "propose_component_safe_edge_flips"]
