from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .contracts_v1 import IRISVNextConfigV1
from .flexicubes_v1 import FLEXICUBES_PARAMETER_SEMANTICS, split_flexicubes_parameters_v1


def _plane_coordinates(size: int, *, device, dtype) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((xx, yy), dim=-1)
    plane_id = torch.eye(3, device=device, dtype=dtype)[:, None, None, :].expand(
        3, size, size, 3
    )
    return torch.cat((xy[None].expand(3, -1, -1, -1), plane_id), dim=-1)


class CameraAwareViewBankVNextV1(nn.Module):
    def __init__(self, cfg: IRISVNextConfigV1):
        super().__init__()
        self.cfg = cfg
        self.image_projection = nn.Linear(cfg.image_token_dim, cfg.scene_dim)
        self.camera_projection = nn.Sequential(
            nn.Linear(cfg.camera_feature_dim, cfg.camera_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.camera_embed_dim, cfg.scene_dim),
        )
        self.xy_projection = nn.Sequential(
            nn.Linear(2, cfg.camera_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.camera_embed_dim, cfg.scene_dim),
        )
        self.norm = nn.LayerNorm(cfg.scene_dim)

    def forward(self, image_tokens: torch.Tensor, token_xy: torch.Tensor, camera_features: torch.Tensor) -> torch.Tensor:
        if image_tokens.ndim != 4:
            raise ValueError("IRIS_VNEXT_IMAGE_TOKENS_RANK_INVALID")
        b, v, t, _ = image_tokens.shape
        if token_xy.shape != (b, v, t, 2):
            raise ValueError("IRIS_VNEXT_TOKEN_XY_SHAPE_INVALID")
        if camera_features.ndim != 3 or camera_features.shape[:2] != (b, v) or camera_features.shape[-1] != self.cfg.camera_feature_dim:
            raise ValueError("IRIS_VNEXT_CAMERA_FEATURE_SHAPE_INVALID")
        z = self.image_projection(image_tokens) + self.camera_projection(camera_features)[:, :, None, :] + self.xy_projection(token_xy)
        return self.norm(z).reshape(b, v * t, self.cfg.scene_dim)


class SharedSceneTriplaneVNextV1(nn.Module):
    def __init__(self, cfg: IRISVNextConfigV1):
        super().__init__()
        self.cfg = cfg
        p, d = cfg.plane_size, cfg.scene_dim
        self.scene_queries = nn.Parameter(torch.randn(3, p, p, d) / math.sqrt(d))
        self.query_position = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerDecoderLayer(d_model=d, nhead=cfg.attention_heads, dim_feedforward=d * cfg.feedforward_multiplier, dropout=cfg.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.scene_layers, norm=nn.LayerNorm(d))
        self.upsample = nn.ConvTranspose2d(d, cfg.plane_channels, kernel_size=2, stride=2)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.ndim != 3 or memory.shape[-1] != self.cfg.scene_dim:
            raise ValueError("IRIS_VNEXT_SCENE_MEMORY_SHAPE_INVALID")
        b = memory.shape[0]
        p, d = self.cfg.plane_size, self.cfg.scene_dim
        pos = _plane_coordinates(p, device=memory.device, dtype=memory.dtype)
        q = self.scene_queries.to(memory) + self.query_position(pos)
        q = q.reshape(1, 3 * p * p, d).expand(b, -1, -1)
        scene = self.decoder(q, memory).reshape(b, 3, p, p, d).permute(0, 1, 4, 2, 3).contiguous()
        up = self.upsample(scene.reshape(b * 3, d, p, p))
        return up.reshape(b, 3, self.cfg.plane_channels, up.shape[-2], up.shape[-1])


class ResidualEvidenceBlockV1(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm, self.fc1, self.fc2 = nn.LayerNorm(width), nn.Linear(width, width), nn.Linear(width, width)
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(F.silu(self.fc1(self.norm(x))))
        return (x + h) * self.scale


class LocalEvidenceFieldV1(nn.Module):
    """Shared local substrate H(x) plus geometry/extraction readouts."""
    def __init__(self, cfg: IRISVNextConfigV1):
        super().__init__()
        self.cfg = cfg
        width = 3 * cfg.plane_channels
        self.input_norm = nn.LayerNorm(width)
        self.input_projection = nn.Linear(width, cfg.field_hidden_dim)
        self.blocks = nn.ModuleList([ResidualEvidenceBlockV1(cfg.field_hidden_dim) for _ in range(cfg.field_residual_blocks)])
        self.feature_projection = nn.Identity() if cfg.field_hidden_dim == cfg.local_feature_dim else nn.Linear(cfg.field_hidden_dim, cfg.local_feature_dim)
        self.final_norm = nn.LayerNorm(cfg.local_feature_dim)
        self.field_head = nn.Linear(cfg.local_feature_dim, 1)
        self.uncertainty_head = nn.Linear(cfg.local_feature_dim, 1)
        self.delta_head = nn.Linear(cfg.local_feature_dim, 3)

    @staticmethod
    def sample_triplanes(scene_planes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if scene_planes.ndim != 5 or scene_planes.shape[1] != 3:
            raise ValueError("IRIS_VNEXT_TRIPLANE_SHAPE_INVALID")
        if points.ndim != 3 or points.shape[0] != scene_planes.shape[0] or points.shape[-1] != 3:
            raise ValueError("IRIS_VNEXT_QUERY_POINT_SHAPE_INVALID")
        b, _, c, _, _ = scene_planes.shape
        x, y, z = points.unbind(dim=-1)
        grids = torch.stack((torch.stack((x, y), dim=-1), torch.stack((x, z), dim=-1), torch.stack((y, z), dim=-1)), dim=1)
        sampled = F.grid_sample(scene_planes.reshape(b * 3, c, scene_planes.shape[-2], scene_planes.shape[-1]), grids.reshape(b * 3, points.shape[1], 1, 2), mode="bilinear", padding_mode="border", align_corners=False)
        return sampled.squeeze(-1).reshape(b, 3, c, points.shape[1]).permute(0, 3, 1, 2).reshape(b, points.shape[1], 3 * c)

    def forward(self, scene_planes: torch.Tensor, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = F.silu(self.input_projection(self.input_norm(self.sample_triplanes(scene_planes, points))))
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(self.feature_projection(h))
        return {
            "H": h,
            "field": self.field_head(h).squeeze(-1),
            "log_uncertainty": self.uncertainty_head(h).squeeze(-1).clamp(-8.0, 4.0),
            "extraction_delta": torch.tanh(self.delta_head(h)) * float(self.cfg.maximum_extraction_delta),
        }


class CubeExtractionHeadV1(nn.Module):
    """Raw FlexiCubes beta12/alpha8/gamma1 degrees of freedom.

    These outputs are not supervised labels and are never MIRA skin weights.
    They receive gradients through the pinned differentiable extractor.
    """
    output_dim = 21
    parameter_semantics = FLEXICUBES_PARAMETER_SEMANTICS

    def __init__(self, cfg: IRISVNextConfigV1):
        super().__init__()
        self.local_feature_dim = cfg.local_feature_dim
        self.weight_scale = float(cfg.flexicubes_weight_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(8 * cfg.local_feature_dim),
            nn.Linear(8 * cfg.local_feature_dim, cfg.topology_width),
            nn.GELU(),
            nn.Linear(cfg.topology_width, self.output_dim),
        )

    def forward(self, corner_features: torch.Tensor) -> torch.Tensor:
        if corner_features.ndim != 4 or corner_features.shape[-2:] != (8, self.local_feature_dim):
            raise ValueError("IRIS_VNEXT_CUBE_CORNER_FEATURE_SHAPE_INVALID")
        return self.net(corner_features.flatten(-2))

    def interpreted(self, corner_features: torch.Tensor):
        return split_flexicubes_parameters_v1(
            self.forward(corner_features), weight_scale=self.weight_scale
        )


class IRISVNextSceneCoreV1(nn.Module):
    architecture_id = "RealSaS.IRIS.VNext.SharedSceneCore.v1"
    input_semantics = "FROZEN_DINOV2_S_IMAGE_TOKENS_PLUS_CAMERAS"
    output_semantics = "SHARED_SCENE_TRIPLANE_EVIDENCE"
    def __init__(self, cfg: IRISVNextConfigV1 = IRISVNextConfigV1()):
        super().__init__(); cfg.validate(); self.config = cfg
        self.view_bank = CameraAwareViewBankVNextV1(cfg); self.scene = SharedSceneTriplaneVNextV1(cfg); self.field = LocalEvidenceFieldV1(cfg); self.cube_extraction = CubeExtractionHeadV1(cfg)
    def encode_scene(self, image_tokens, token_xy, camera_features):
        return self.scene(self.view_bank(image_tokens, token_xy, camera_features))
    def query(self, scene_planes, points):
        return self.field(scene_planes, points)
