from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F


def sample_dense(field: torch.Tensor, xy_px: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    B, V, C, H, W = field.shape
    N = xy_px.shape[2]
    grid = xy_px.clone().to(field.dtype)
    grid[..., 0] = grid[..., 0] / float(image_size - 1) * 2.0 - 1.0
    grid[..., 1] = grid[..., 1] / float(image_size - 1) * 2.0 - 1.0
    y = F.grid_sample(field.reshape(B * V, C, H, W), grid.reshape(B * V, N, 1, 2), mode='bilinear', padding_mode='border', align_corners=True)
    return y[:, :, :, 0].permute(0, 2, 1).reshape(B, V, N, C)


def sample_g1_predictions(outputs: Dict[str, torch.Tensor], xy: torch.Tensor, image_size: int = 256):
    result = {k: sample_dense(outputs[k], xy, image_size) for k in ['point','normal','log_sigma','visibility_logit']}
    if 'descriptor_z' in outputs:
        result['descriptor_z'] = sample_dense(outputs['descriptor_z'], xy, image_size)
        result['descriptor_log_sigma'] = sample_dense(outputs['descriptor_log_sigma'], xy, image_size)
    return result
