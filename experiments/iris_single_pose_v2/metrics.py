from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from coords import grid_to_pixel_center, grid_to_cell_index


@dataclass(frozen=True)
class ResolutionContract:
    authority_resolution: int = 1024
    input_resolution: int = 1024
    coarse_stride: int = 8
    fine_stride: int = 2

    @property
    def coarse_resolution(self): return self.input_resolution // self.coarse_stride
    @property
    def fine_resolution(self): return self.input_resolution // self.fine_stride


def native_pixel_error(pred_grid, truth_grid, authority_resolution: int = 1024):
    p = grid_to_pixel_center(np.asarray(pred_grid, np.float32), authority_resolution)
    t = grid_to_pixel_center(np.asarray(truth_grid, np.float32), authority_resolution)
    return np.linalg.norm(p - t, axis=-1)


def topk_native_hit(candidate_grids, truth_grid, k: int, tol_native_px: float, authority_resolution: int = 1024):
    c = np.asarray(candidate_grids, np.float32)[:k]
    if len(c) == 0: return False
    e = native_pixel_error(c, np.asarray(truth_grid, np.float32)[None], authority_resolution)
    return bool(np.any(e <= float(tol_native_px)))


def exact_coarse_cell_hit(candidate_coarse_coords, truth_grid, coarse_resolution: int, k: int):
    c = np.asarray(candidate_coarse_coords, np.float32)[:k]
    if len(c) == 0: return False
    tx = int(grid_to_cell_index(float(truth_grid[0]), coarse_resolution)); ty = int(grid_to_cell_index(float(truth_grid[1]), coarse_resolution))
    for q in c:
        if int(grid_to_cell_index(float(q[0]), coarse_resolution)) == tx and int(grid_to_cell_index(float(q[1]), coarse_resolution)) == ty: return True
    return False


def summarize_errors(errors):
    a = np.asarray(errors, np.float64); a = a[np.isfinite(a)]
    if not len(a): return {"n":0,"mean":None,"median":None,"p90":None,"p95":None}
    return {"n":int(len(a)),"mean":float(np.mean(a)),"median":float(np.median(a)),"p90":float(np.percentile(a,90)),"p95":float(np.percentile(a,95))}


def normal_angle_deg(pred_n, gt_n):
    p = np.asarray(pred_n, np.float64); g = np.asarray(gt_n, np.float64)
    p = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12); g = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(p*g, axis=-1), -1.0, 1.0)))
