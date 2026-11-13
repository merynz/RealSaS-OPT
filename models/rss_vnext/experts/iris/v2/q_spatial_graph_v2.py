from __future__ import annotations

import torch


def anchor_grid_pixel_indices_v2(anchor_grid: torch.Tensor, resolution: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover nearest native pixel-center indices from normalized GRID_XY coordinates."""
    if anchor_grid.ndim != 3 or anchor_grid.shape[-1] != 2:
        raise ValueError("anchor_grid must be [B,Q,2]")
    if resolution < 2:
        raise ValueError("resolution must be >=2")
    x = torch.round((anchor_grid[..., 0] + 1.0) * 0.5 * float(resolution) - 0.5).long()
    y = torch.round((anchor_grid[..., 1] + 1.0) * 0.5 * float(resolution) - 0.5).long()
    if ((x < 0) | (x >= resolution) | (y < 0) | (y >= resolution)).any():
        raise ValueError("anchor grid contains coordinates outside native raster")
    return x, y


def build_anchor_neighbor_graph_v2(
    anchor_grid: torch.Tensor,
    valid_q: torch.Tensor,
    *,
    max_neighbors: int = 8,
    resolution: int = 1024,
    search_offsets_px: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a sparse, permutation-equivariant local graph without QxQ distances.

    Canonical q rays originate on the native anchor raster. A dense integer lookup
    therefore gives exact local adjacency at several possible lattice strides in
    O(B*resolution^2 + B*Q) memory/time rather than O(Q^2). Candidate neighbors
    are searched symmetrically in eight directions and later weighted by exact
    world-space distances, so array/raster enumeration is never an edge source.
    """
    if valid_q.ndim != 2 or anchor_grid.shape[:2] != valid_q.shape:
        raise ValueError("anchor_grid/valid_q shape mismatch")
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be positive")
    offsets = tuple(int(x) for x in search_offsets_px)
    if not offsets or min(offsets) <= 0 or tuple(sorted(set(offsets))) != offsets:
        raise ValueError("search offsets must be positive unique ascending integers")
    B, Q = valid_q.shape
    x, y = anchor_grid_pixel_indices_v2(anchor_grid, resolution)
    directions = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
    all_idx = []
    all_mask = []
    for b in range(B):
        lookup = torch.full((resolution, resolution), -1, dtype=torch.long, device=anchor_grid.device)
        q_ids = torch.arange(Q, device=anchor_grid.device)
        vb = valid_q[b].bool()
        lookup[y[b, vb], x[b, vb]] = q_ids[vb]
        candidates = []
        masks = []
        for radius in offsets:
            for dx, dy in directions:
                xx = x[b] + dx * radius
                yy = y[b] + dy * radius
                inside = (xx >= 0) & (xx < resolution) & (yy >= 0) & (yy < resolution)
                safe_x = xx.clamp(0, resolution - 1)
                safe_y = yy.clamp(0, resolution - 1)
                idx = lookup[safe_y, safe_x]
                good = inside & vb & (idx >= 0) & (idx != q_ids)
                candidates.append(idx.clamp_min(0))
                masks.append(good)
        cand = torch.stack(candidates, dim=-1)
        mask = torch.stack(masks, dim=-1)
        C = cand.shape[-1]
        priority = torch.arange(C, device=cand.device, dtype=torch.long)[None].expand(Q, C)
        ranked = torch.where(mask, priority, torch.full_like(priority, C + 1))
        k = min(int(max_neighbors), C)
        choice = torch.topk(ranked, k=k, dim=-1, largest=False, sorted=True).indices
        idx = torch.gather(cand, -1, choice)
        good = torch.gather(mask, -1, choice)
        if k < max_neighbors:
            pad = max_neighbors - k
            idx = torch.cat([idx, torch.zeros((Q, pad), dtype=torch.long, device=idx.device)], dim=-1)
            good = torch.cat([good, torch.zeros((Q, pad), dtype=torch.bool, device=good.device)], dim=-1)
        all_idx.append(idx)
        all_mask.append(good)
    return torch.stack(all_idx, dim=0), torch.stack(all_mask, dim=0)
