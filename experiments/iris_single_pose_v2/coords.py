from __future__ import annotations

import numpy as np
import torch


def pixel_center_to_grid(pixel_xy, resolution: int):
    """Pixel-center coordinates [0,R-1] -> grid_sample coordinates [-1,1], align_corners=False."""
    r = float(resolution)
    if torch.is_tensor(pixel_xy):
        return 2.0 * (pixel_xy.to(torch.float32) + 0.5) / r - 1.0
    p = np.asarray(pixel_xy, np.float32)
    return (2.0 * (p + 0.5) / r - 1.0).astype(np.float32)


def grid_to_pixel_center(grid_xy, resolution: int):
    """grid_sample coordinates [-1,1] -> pixel-center coordinates, align_corners=False."""
    r = float(resolution)
    if torch.is_tensor(grid_xy):
        return (grid_xy.to(torch.float32) + 1.0) * (r / 2.0) - 0.5
    g = np.asarray(grid_xy, np.float32)
    return ((g + 1.0) * (r / 2.0) - 0.5).astype(np.float32)


def field_cell_centers(height: int, width: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return [H,W,2] grid_sample coords of field-cell centers."""
    yy = torch.arange(height, device=device, dtype=dtype)
    xx = torch.arange(width, device=device, dtype=dtype)
    gy = 2.0 * (yy + 0.5) / float(height) - 1.0
    gx = 2.0 * (xx + 0.5) / float(width) - 1.0
    gy, gx = torch.meshgrid(gy, gx, indexing="ij")
    return torch.stack([gx, gy], dim=-1)


def grid_to_cell_index(grid_coord, cells: int):
    """Map one normalized coordinate to the containing align_corners=False cell index."""
    if torch.is_tensor(grid_coord):
        idx = torch.floor((grid_coord + 1.0) * 0.5 * float(cells)).to(torch.long)
        return idx.clamp(0, cells - 1)
    g = np.asarray(grid_coord, np.float32)
    idx = np.floor((g + 1.0) * 0.5 * float(cells)).astype(np.int64)
    return np.clip(idx, 0, cells - 1)


def cell_index_to_grid(index, cells: int):
    if torch.is_tensor(index):
        return 2.0 * (index.to(torch.float32) + 0.5) / float(cells) - 1.0
    i = np.asarray(index, np.float32)
    return (2.0 * (i + 0.5) / float(cells) - 1.0).astype(np.float32)


def grid_distance_in_pixels(a, b, resolution: int):
    """Euclidean image-plane distance measured in pixels of `resolution`."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        aa = a if torch.is_tensor(a) else torch.as_tensor(a)
        bb = b if torch.is_tensor(b) else torch.as_tensor(b, device=aa.device)
        pa = grid_to_pixel_center(aa, resolution)
        pb = grid_to_pixel_center(bb, resolution)
        return torch.linalg.norm(pa - pb, dim=-1)
    pa = grid_to_pixel_center(a, resolution)
    pb = grid_to_pixel_center(b, resolution)
    return np.linalg.norm(pa - pb, axis=-1)
