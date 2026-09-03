from __future__ import annotations

import torch


def _fallback_world_knn(q_points: torch.Tensor, valid_mask: torch.Tensor, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-Q fallback only; production should pass the sparse field graph."""
    B, Q, D, _ = q_points.shape
    if Q > 4096:
        raise ValueError("large-Q regularizer requires sparse neighbor graph from EvidenceFieldV2")
    ref = q_points[:, :, D // 2]
    valid_q = valid_mask.any(dim=-1)
    d = torch.cdist(ref.float(), ref.float())
    d = d.masked_fill(~valid_q[:, None, :], float("inf"))
    eye = torch.eye(Q, device=q_points.device, dtype=torch.bool)[None]
    d = d.masked_fill(eye, float("inf"))
    kk = min(max(1, int(k)), max(1, Q - 1))
    idx = torch.topk(d, k=kk, dim=-1, largest=False, sorted=True).indices
    b = torch.arange(B, device=q_points.device)[:, None, None]
    mask = valid_q[:, :, None] & valid_q[b, idx] & torch.isfinite(torch.gather(d, -1, idx))
    return idx, mask


def isotropic_world_regularizer_v2(
    q_points: torch.Tensor,
    score_logits: torch.Tensor,
    neighbor_pairs: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    *,
    neighbor_indices: torch.Tensor | None = None,
    neighbor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximately isotropic local smoothness in common world coordinates.

    XY/carrier edges come from a geometry-derived local graph; depth edges connect
    adjacent hypotheses on the same analytic ray. Distances are normalized by a
    robust world-space median, preserving uniform scale and rigid-rotation
    equivariance. Q enumeration is never used as neighborhood authority.
    """
    if q_points.ndim != 4 or score_logits.shape != q_points.shape[:3]:
        raise ValueError("q/score shape mismatch")
    B, Q, D, _ = q_points.shape
    if valid_mask is None:
        valid_mask = torch.ones_like(score_logits, dtype=torch.bool)
    if valid_mask.shape != score_logits.shape:
        raise ValueError("valid_mask shape mismatch")

    terms = []
    distances = []

    if neighbor_pairs is not None:
        if neighbor_pairs.ndim != 2 or neighbor_pairs.shape[-1] != 2:
            raise ValueError("neighbor_pairs must be [E,2]")
        a, b = neighbor_pairs[:, 0].long(), neighbor_pairs[:, 1].long()
        if (a < 0).any() or (b < 0).any() or (a >= Q).any() or (b >= Q).any():
            raise ValueError("neighbor index out of range")
        qa, qb = q_points[:, a], q_points[:, b]
        sa, sb = score_logits[:, a], score_logits[:, b]
        ev = valid_mask[:, a] & valid_mask[:, b]
        dist = torch.linalg.norm(qa - qb, dim=-1)
        terms.append(((sa - sb).square(), ev, dist))
        if ev.any():
            distances.append(dist[ev])
    else:
        if neighbor_indices is None or neighbor_mask is None:
            neighbor_indices, neighbor_mask = _fallback_world_knn(q_points, valid_mask)
        if neighbor_indices.ndim != 3 or neighbor_mask.shape != neighbor_indices.shape or neighbor_indices.shape[:2] != (B, Q):
            raise ValueError("sparse neighbor graph shape mismatch")
        if (neighbor_indices < 0).any() or (neighbor_indices >= Q).any():
            raise ValueError("sparse neighbor index out of range")
        bi = torch.arange(B, device=q_points.device)[:, None, None]
        qn = q_points[bi, neighbor_indices]
        sn = score_logits[bi, neighbor_indices]
        vn = valid_mask[bi, neighbor_indices]
        qi = q_points[:, :, None]
        si = score_logits[:, :, None]
        vi = valid_mask[:, :, None]
        ev = neighbor_mask[..., None].bool() & vi & vn
        dist = torch.linalg.norm(qi - qn, dim=-1)
        terms.append(((si - sn).square(), ev, dist))
        if ev.any():
            distances.append(dist[ev])

    if D > 1:
        ev = valid_mask[..., :-1] & valid_mask[..., 1:]
        dist = torch.linalg.norm(q_points[..., 1:, :] - q_points[..., :-1, :], dim=-1)
        diff2 = (score_logits[..., 1:] - score_logits[..., :-1]).square()
        terms.append((diff2, ev, dist))
        if ev.any():
            distances.append(dist[ev])

    if not distances:
        return score_logits.sum() * 0.0
    scale = torch.cat([x.reshape(-1) for x in distances]).median().clamp_min(1e-8)
    values = []
    for diff2, mask, dist in terms:
        if mask.any():
            normalized_distance = (dist / scale).clamp_min(1e-4)
            values.append((diff2 / normalized_distance)[mask])
    if not values:
        return score_logits.sum() * 0.0
    return torch.cat([x.reshape(-1) for x in values]).mean()
