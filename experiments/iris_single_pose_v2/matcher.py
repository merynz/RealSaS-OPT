from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.nn.functional as F

from coords import grid_to_cell_index, cell_index_to_grid


@dataclass(frozen=True)
class MatcherConfig:
    coarse_keep: int = 32
    p_rescue_keep: int = 8
    final_topk: int = 8
    coarse_row_half_width_cells: int = 1
    fine_radius_cells: int = 4
    fine_per_basin: int = 2
    alpha_threshold: float = 0.20
    rrf_k: float = 60.0


@dataclass
class Basin:
    coarse_index: int
    coarse_coord: np.ndarray
    coarse_rank: int
    p_rank: int
    basin_rank_score: float
    admitted_by: str


@dataclass
class MatchCandidate:
    coord: np.ndarray
    basin_index: int
    basin_rank: int
    fine_local_rank: int
    fine_similarity: float
    p_distance: float


@dataclass
class MatchResult:
    candidates: List[MatchCandidate]
    basins: List[Basin]
    coarse_field_hw: tuple[int, int]
    fine_field_hw: tuple[int, int]
    source_query: np.ndarray

    @property
    def top_coords(self):
        if not self.candidates: return np.zeros((0, 2), np.float32)
        return np.asarray([c.coord for c in self.candidates], np.float32)


def _sample_one(field: torch.Tensor, view: int, coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0: return np.zeros((0, field.shape[2]), np.float32)
    g = torch.as_tensor(coords, dtype=field.dtype, device=field.device)[None, :, None, :]
    s = F.grid_sample(field[:, view], g, mode="bilinear", padding_mode="zeros", align_corners=False)
    return s[0, :, :, 0].T.detach().float().cpu().numpy()


def _norm(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _alpha_mask_at(alpha: torch.Tensor, hw: tuple[int, int], threshold: float) -> np.ndarray:
    a = F.interpolate(alpha, size=hw, mode="bilinear", align_corners=False)[0, 0]
    return (a > threshold).detach().cpu().numpy()


def _candidate_cells(mask: np.ndarray):
    yy, xx = np.nonzero(mask); h, w = mask.shape
    gx = 2.0 * (xx.astype(np.float32) + 0.5) / float(w) - 1.0
    gy = 2.0 * (yy.astype(np.float32) + 0.5) / float(h) - 1.0
    return yy.astype(np.int32), xx.astype(np.int32), np.stack([gx, gy], axis=-1).astype(np.float32)


def _rank_positions(order, n):
    r = np.empty(n, np.int32); r[order] = np.arange(1, n + 1, dtype=np.int32); return r


def _rrf(rank_a, rank_b, k): return 1.0 / (k + rank_a) + 1.0 / (k + rank_b)


def match_query(outputs, images, source_view: int, target_view: int, query_xy, cfg: MatcherConfig = MatcherConfig()) -> MatchResult:
    """D3-free correspondence. Global admission uses Z_coarse/P only; Z_fine is local-only."""
    if outputs["Z_coarse"].shape[0] != 1: raise ValueError("matcher currently expects batch size 1")
    qxy = np.asarray(query_xy, np.float32).reshape(1, 2); zc = outputs["Z_coarse"]; zf = outputs["Z_fine"]; pfield = outputs["P"]
    hc, wc = zc.shape[-2:]; hf, wf = zf.shape[-2:]
    alpha_t = images[0, target_view, 3:4][None]
    coarse_mask = _alpha_mask_at(alpha_t, (hc, wc), cfg.alpha_threshold); cy, cx, ccoords = _candidate_cells(coarse_mask)
    if len(ccoords) == 0: return MatchResult([], [], (hc, wc), (hf, wf), qxy[0])

    q_row = int(grid_to_cell_index(qxy[0, 1], hc)); row_ok = np.abs(cy - q_row) <= cfg.coarse_row_half_width_cells
    valid_ids = np.flatnonzero(row_ok)
    if len(valid_ids) == 0: valid_ids = np.arange(len(ccoords), dtype=np.int32)

    qzc = _norm(_sample_one(zc, source_view, qxy))[0]; qzf = _norm(_sample_one(zf, source_view, qxy))[0]; qp = _sample_one(pfield, source_view, qxy)[0]
    tzc = _norm(_sample_one(zc, target_view, ccoords)); tp = _sample_one(pfield, target_view, ccoords)
    zscore = tzc @ qzc; pdist = np.linalg.norm(tp - qp[None], axis=-1)
    z_order_local = valid_ids[np.argsort(-zscore[valid_ids], kind="stable")]; p_order_local = valid_ids[np.argsort(pdist[valid_ids], kind="stable")]
    rz = _rank_positions(np.argsort(-zscore[valid_ids], kind="stable"), len(valid_ids)); rp = _rank_positions(np.argsort(pdist[valid_ids], kind="stable"), len(valid_ids))
    rank_z_full = {int(valid_ids[i]): int(rz[i]) for i in range(len(valid_ids))}; rank_p_full = {int(valid_ids[i]): int(rp[i]) for i in range(len(valid_ids))}

    top_z = set(map(int, z_order_local[:min(cfg.coarse_keep, len(z_order_local))])); top_p = set(map(int, p_order_local[:min(cfg.p_rescue_keep, len(p_order_local))]))
    admitted = top_z | top_p; basins = []
    for idx in admitted:
        src = "Zc+P" if idx in top_z and idx in top_p else ("P" if idx in top_p else "Zc")
        basins.append(Basin(idx, ccoords[idx], rank_z_full[idx], rank_p_full[idx], float(_rrf(rank_z_full[idx], rank_p_full[idx], cfg.rrf_k)), src))
    basins.sort(key=lambda b: (-b.basin_rank_score, b.coarse_rank, b.p_rank, b.coarse_index))

    fine_mask = _alpha_mask_at(alpha_t, (hf, wf), cfg.alpha_threshold); basin_candidates = []
    for brank, basin in enumerate(basins, start=1):
        center_x = int(grid_to_cell_index(basin.coarse_coord[0], wf)); center_y = int(grid_to_cell_index(basin.coarse_coord[1], hf))
        ys = range(max(0, center_y-cfg.fine_radius_cells), min(hf, center_y+cfg.fine_radius_cells+1)); xs = range(max(0, center_x-cfg.fine_radius_cells), min(wf, center_x+cfg.fine_radius_cells+1))
        cells = [(y, x) for y in ys for x in xs if fine_mask[y, x]]
        if not cells: basin_candidates.append([]); continue
        coords = np.asarray([[float(cell_index_to_grid(x, wf)), float(cell_index_to_grid(y, hf))] for y, x in cells], np.float32)
        tzf = _norm(_sample_one(zf, target_view, coords)); fp = _sample_one(pfield, target_view, coords)
        fs = tzf @ qzf; pd = np.linalg.norm(fp - qp[None], axis=-1); order = np.argsort(-fs, kind="stable"); local = []
        for local_rank, j in enumerate(order[:min(cfg.fine_per_basin, len(order))], start=1):
            local.append(MatchCandidate(coords[j], basin.coarse_index, brank, local_rank, float(fs[j]), float(pd[j])))
        basin_candidates.append(local)

    candidates = []; depth = 0
    while len(candidates) < cfg.final_topk:
        added = False
        for local in basin_candidates:
            if depth < len(local):
                candidates.append(local[depth]); added = True
                if len(candidates) >= cfg.final_topk: break
        if not added: break
        depth += 1
    return MatchResult(candidates, basins, (hc, wc), (hf, wf), qxy[0])
