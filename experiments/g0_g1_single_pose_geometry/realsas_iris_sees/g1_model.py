from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, Any
import torch
from torch import nn

from .model import (
    SEESConfig,
    SharedImageEncoder,
    GlobalMultiViewFusion,
    DenseFusionDecoder,
    TimeConditionedGeometryHead,
    DescriptorHead,
    CameraResidualHead,
    canonical_orthographic_camera_features,
)


@dataclass(frozen=True)
class G1Config:
    image_size: int = 128
    base_dim: int = 64
    token_dim: int = 192
    descriptor_dim: int = 64
    transformer_depth: int = 4
    transformer_heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.0
    input_channels: int = 4
    expose_descriptor_z: bool = True

    def sees_config(self) -> SEESConfig:
        return SEESConfig(
            image_size=self.image_size,
            base_dim=self.base_dim,
            token_dim=self.token_dim,
            descriptor_dim=self.descriptor_dim,
            transformer_depth=self.transformer_depth,
            transformer_heads=self.transformer_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            camera_residual_enabled=False,
            input_channels=self.input_channels,
        )


class IRISG1SinglePose(nn.Module):
    """Canonical G1 baseline: A×8 raster -> P/N/V/U (+ optional dormant Z).

    This is deliberately the smallest surgery on N1D:
      - remove the input pose dimension;
      - evaluate only target-time A geometry;
      - omit all A<->B transport, differential, scene-flow, F/R/D/GFDR paths;
      - optionally expose legacy descriptor Z for preservation/diagnostics only.

    GlobalMultiViewFusion is retained byte-structurally, including pose_embed.
    With P=1 only pose index 0 is used. This preserves warm-start fidelity.
    """
    def __init__(self, cfg: G1Config = G1Config()):
        super().__init__()
        self.cfg = cfg
        scfg = cfg.sees_config()
        self.encoder = SharedImageEncoder(scfg.base_dim, scfg.token_dim, scfg.input_channels)
        self.fusion = GlobalMultiViewFusion(
            scfg.token_dim, scfg.transformer_depth, scfg.transformer_heads,
            scfg.mlp_ratio, scfg.dropout,
        )
        self.camera_token = nn.Sequential(
            nn.Linear(6, scfg.token_dim), nn.GELU(), nn.Linear(scfg.token_dim, scfg.token_dim)
        )
        self.decoder = DenseFusionDecoder(scfg.base_dim, scfg.token_dim, out_dim=128)
        self.geometry = TimeConditionedGeometryHead(128)
        self.descriptor = DescriptorHead(128, scfg.descriptor_dim) if cfg.expose_descriptor_z else None
        self.camera_residual = CameraResidualHead(scfg.token_dim)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 5 or images.shape[1] != 8 or images.shape[2] != self.cfg.input_channels:
            raise ValueError(f"expected [B,8,{self.cfg.input_channels},H,W], got {tuple(images.shape)}")
        B, V, C, H, W = images.shape
        x = images.reshape(B * V, C, H, W)
        f4, f8, f16 = self.encoder(x)
        h4, w4 = f4.shape[-2:]
        h8, w8 = f8.shape[-2:]
        h16, w16 = f16.shape[-2:]
        f4 = f4.view(B, 1, V, -1, h4, w4)
        f8 = f8.view(B, 1, V, -1, h8, w8)
        f16 = f16.view(B, 1, V, -1, h16, w16)

        cam = canonical_orthographic_camera_features(1, 1, images.device, images.dtype)[:, :, 0, 0]
        cam6 = cam.view(1, 1, 8, 6).expand(B, 1, 8, 6)
        global16 = self.fusion(f16, cam6)
        residual = self.camera_residual(global16)[:, 0]

        dense_views = []
        for v in range(8):
            g = global16[:, 0, v] + self.camera_token(cam6[:, 0, v])[:, :, None, None]
            dense_views.append(self.decoder(f4[:, 0, v], f8[:, 0, v], g))
        dense = torch.stack(dense_views, dim=1)
        _, _, Fd, Hd, Wd = dense.shape
        flat = dense.reshape(B * V, Fd, Hd, Wd)

        geo = self.geometry(flat, 0)
        out: Dict[str, torch.Tensor] = {
            "dense_feature": dense,
            "point": geo["point"].view(B, V, 3, Hd, Wd),
            "normal": geo["normal"].view(B, V, 3, Hd, Wd),
            "log_sigma": geo["log_sigma"].view(B, V, 1, Hd, Wd),
            "visibility_logit": geo["visibility_logit"].view(B, V, 1, Hd, Wd),
            "camera_residual_diagnostic": residual,
        }
        if self.descriptor is not None:
            z = self.descriptor(flat)
            out["descriptor_z"] = z["descriptor"].view(B, V, -1, Hd, Wd)
            out["descriptor_log_sigma"] = z["descriptor_log_sigma"].view(B, V, 1, Hd, Wd)
        return out


def migrate_n1d_state_dict(model: IRISG1SinglePose, n1d_state: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """Load every shape-compatible retained N1D tensor and report all archive-only source tensors."""
    dst = model.state_dict()
    loadable = {}
    archived_only = []
    mismatched = []
    for key, value in n1d_state.items():
        if key in dst:
            if tuple(dst[key].shape) == tuple(value.shape):
                loadable[key] = value
            else:
                mismatched.append({"key": key, "source": list(value.shape), "dest": list(dst[key].shape)})
        else:
            archived_only.append(key)
    result = model.load_state_dict(loadable, strict=False)
    return {
        "loaded_tensor_count": len(loadable),
        "destination_tensor_count": len(dst),
        "loaded_fraction": len(loadable) / max(len(dst), 1),
        "missing_destination_keys": sorted(result.missing_keys),
        "unexpected_source_keys": sorted(result.unexpected_keys),
        "archived_only_source_keys": sorted(archived_only),
        "shape_mismatches": mismatched,
    }
