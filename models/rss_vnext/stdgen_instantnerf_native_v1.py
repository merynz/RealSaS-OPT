from __future__ import annotations

"""Checkpoint-key-compatible InstantNeRF feature backbone for RSS.vNext.

Upstream reference: TencentARC/InstantMesh, configs/instant-nerf-large.yaml,
src/models/{lrm.py,decoder/transformer.py,encoder/dino.py} (Apache-2.0).
This is an independent PyTorch implementation of the feature computation.
It does NOT import InstantMesh's EG3D/NVIDIA-proprietary renderer.

The published 16-layer/80-channel checkpoint is accepted only by the
strict SHA, key and tensor-shape court in stdgen_instantnerf_pretrain_court_v1.
Native triplanes are a *pretrained proposal*, not automatically an RSS IRIS
source-truth H(x) or a qualified semantic field.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .stdgen_instantnerf_pretrain_court_v1 import StdGENInstantNeRFArchitectureV1


class _ReferenceCrossSelfBlock(nn.Module):
    """Published transformer parameter layout and cross->self->MLP order."""
    def __init__(self, width: int, context_dim: int, heads: int, eps: float = 1.e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.cross_attn = nn.MultiheadAttention(
            width, heads, kdim=context_dim, vdim=context_dim,
            bias=False, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(
            width, heads, bias=False, batch_first=True,
        )
        self.norm3 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4*width), nn.GELU(), nn.Dropout(0.),
            nn.Linear(4*width, width), nn.Dropout(0.),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = x + self.cross_attn(self.norm1(x), cond, cond, need_weights=False)[0]
        y = self.norm2(x)
        x = x + self.self_attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm3(x))


class StdGENNativeTriplaneTransformerV1(nn.Module):
    """The released 16-block query/deconv layout; dimensions are not guessed."""
    def __init__(self, cfg: StdGENInstantNeRFArchitectureV1):
        super().__init__()
        w = cfg.transformer_width
        s = cfg.triplane_base_size
        self.pos_embed = nn.Parameter(torch.empty(1, 3*s*s, w))
        nn.init.normal_(self.pos_embed, std=w**-.5)
        self.layers = nn.ModuleList(
            [_ReferenceCrossSelfBlock(w, cfg.encoder_feature_dim,
                                      cfg.transformer_heads)
             for _ in range(cfg.transformer_layers)]
        )
        self.norm = nn.LayerNorm(w, eps=1.e-6)
        self.deconv = nn.ConvTranspose2d(w, cfg.triplane_channels, 2, stride=2)
        self.low_res = s
        self.high_res = cfg.triplane_high_size

    def forward(self, image_feats: Tensor) -> Tensor:
        if (image_feats.ndim != 3 or
                image_feats.shape[-1] != self.layers[0].cross_attn.kdim or
                image_feats.shape[1] < 1 or
                not torch.isfinite(image_feats).all()):
            raise ValueError("RSS_VNEXT_INSTANTNERF_IMAGE_FEATURE_CONTRACT")
        batch = image_feats.shape[0]
        x = self.pos_embed.expand(batch, -1, -1)
        for layer in self.layers:
            x = layer(x, image_feats)
        s = self.low_res
        x = self.norm(x).reshape(batch, 3, s, s, -1)
        x = x.permute(1, 0, 4, 2, 3).reshape(3*batch, -1, s, s)
        planes = self.deconv(x)
        if planes.shape[-2:] != (self.high_res, self.high_res):
            raise ValueError("RSS_VNEXT_INSTANTNERF_HIGHRES_PLANES_DRIFT")
        return planes.reshape(3, batch, -1, self.high_res, self.high_res).permute(
            1, 0, 2, 3, 4
        ).contiguous()


class _StdGENNativeOSGDecoder(nn.Module):
    """Parameter-bearing field only; no third-party ray renderer is imported."""
    def __init__(self, plane_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3*plane_channels, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        out = self.net(features)
        return {
            "rgb": (torch.sigmoid(out[..., 1:]) * 1.002 - .001),
            "sigma": out[..., :1],
        }


class _StdGENNativeSynthesizer(nn.Module):
    """Holds released decoder weights, not its license-restricted renderer."""
    def __init__(self, cfg: StdGENInstantNeRFArchitectureV1):
        super().__init__()
        self.decoder = _StdGENNativeOSGDecoder(cfg.triplane_channels)

    def forward_points(self, planes: Tensor, points: Tensor) -> dict[str, Tensor]:
        if (planes.ndim != 5 or planes.shape[1] != 3 or
                points.ndim != 3 or points.shape[:1] != planes.shape[:1] or
                points.shape[-1] != 3 or
                not torch.isfinite(points).all()):
            raise ValueError("RSS_VNEXT_INSTANTNERF_QUERY_DOMAIN")
        b, _, c, h, w = planes.shape
        x,y,z = points.unbind(dim=-1)
        # Plane ordering XY / XZ / YZ is the pretrain convention.
        uv = torch.stack((
            torch.stack((x,y), dim=-1),
            torch.stack((x,z), dim=-1),
            torch.stack((y,z), dim=-1),
        ), dim=1)
        samples = F.grid_sample(
            planes.reshape(3*b, c, h, w),
            uv.reshape(3*b, points.shape[1], 1, 2),
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        features = samples.squeeze(-1).reshape(b, 3, c, -1).permute(
            0, 3, 1, 2
        ).reshape(b, points.shape[1], 3*c)
        return self.decoder(features)


class _StdGENNativeCameraDINO(nn.Module):
    """DINO ViT-B/16 with actual per-layer camera AdaLN, NOT plain ViT."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        dim = int(base_model.config.hidden_size)
        self.camera_embedder = nn.Sequential(
            nn.Linear(16, dim), nn.SiLU(), nn.Linear(dim, dim),
        )
        for layer in self.model.encoder.layer:
            if hasattr(layer, "adaLN_modulation"):
                raise ValueError("RSS_VNEXT_INSTANTNERF_UNEXPECTED_ADALN_EXISTING")
            layer.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4*dim))

    @staticmethod
    def _mod(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1. + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, images: Tensor, camera16: Tensor) -> Tensor:
        if (images.ndim != 5 or images.shape[2] != 3 or
                camera16.shape != (*images.shape[:2],16) or
                not torch.isfinite(images).all() or
                not torch.isfinite(camera16).all() or
                bool((images < 0).any()) or bool((images > 1).any())):
            raise ValueError("RSS_VNEXT_INSTANTNERF_CAMERA_DINO_INPUT")
        b,v,_,h,w = images.shape
        pixel = images.reshape(b*v,3,h,w)
        # Exact default DINOv1 ViTImageProcessor normalization (no resize).
        mean = pixel.new_tensor((.485,.456,.406))[None,:,None,None]
        std = pixel.new_tensor((.229,.224,.225))[None,:,None,None]
        pixel = (pixel - mean) / std
        cond = self.camera_embedder(camera16.reshape(b*v,16))
        model = self.model
        tokens = model.embeddings(pixel, interpolate_pos_encoding=True)
        for layer in model.encoder.layer:
            shift_a,scale_a,shift_m,scale_m = layer.adaLN_modulation(cond).chunk(4,dim=-1)
            normalized = self._mod(layer.layernorm_before(tokens),shift_a,scale_a)
            # Use parameter-bearing ViT attention without vendor code. The
            # attention layer performs the original QKV calculation.
            attn = layer.attention(normalized)[0]
            tokens = tokens + attn
            mlp_in = self._mod(layer.layernorm_after(tokens),shift_m,scale_m)
            tokens = layer.output(layer.intermediate(mlp_in),tokens)
        return model.layernorm(tokens)


class StdGENNativeInstantNeRFV1(nn.Module):
    """Full pretraining STATE layout; only scene planes and point decode exposed."""
    def __init__(self, *, dino_model: nn.Module,
                 architecture: StdGENInstantNeRFArchitectureV1 =
                     StdGENInstantNeRFArchitectureV1()):
        super().__init__()
        architecture.check_published()
        if int(dino_model.config.hidden_size) != architecture.encoder_feature_dim:
            raise ValueError("RSS_VNEXT_INSTANTNERF_DINO_WIDTH_MISMATCH")
        if len(dino_model.encoder.layer) != 12:
            raise ValueError("RSS_VNEXT_INSTANTNERF_DINO_DEPTH_MISMATCH")
        self.encoder = _StdGENNativeCameraDINO(dino_model)
        self.transformer = StdGENNativeTriplaneTransformerV1(architecture)
        self.synthesizer = _StdGENNativeSynthesizer(architecture)

    def forward_planes(self, images: Tensor, cameras16: Tensor) -> Tensor:
        memory = self.encoder(images, cameras16)
        b,v = images.shape[:2]
        return self.transformer(memory.reshape(b,v*memory.shape[1],-1))

    def query_native_field(self, planes: Tensor, points: Tensor) -> dict[str, Tensor]:
        return self.synthesizer.forward_points(planes, points)


def build_stdgen_native_instantnerf_v1(*, local_dino_config_dir: str
                                      ) -> StdGENNativeInstantNeRFV1:
    """Offline construction; never downloads a model or weights implicitly.

    A local DINO config is the source of the full ViT block shapes. The
    pretrained InstantNeRF checkpoint is loaded only after the strict SHA,
    complete key/shape and rights checks at the independent court.
    """
    try:
        from transformers import ViTConfig, ViTModel
        config = ViTConfig.from_pretrained(local_dino_config_dir, local_files_only=True)
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError("RSS_VNEXT_INSTANTNERF_OFFLINE_DINO_CONFIG_REQUIRED") from exc
    if (config.hidden_size != 768 or config.num_hidden_layers != 12 or
            config.num_attention_heads != 12 or config.patch_size != 16 or
            config.intermediate_size != 3072 or config.image_size != 224):
        raise ValueError("RSS_VNEXT_INSTANTNERF_DINO_CONFIG_DRIFT")
    return StdGENNativeInstantNeRFV1(
        dino_model=ViTModel(config, add_pooling_layer=False),
    )


__all__ = ["StdGENNativeInstantNeRFV1", "StdGENNativeTriplaneTransformerV1",
           "build_stdgen_native_instantnerf_v1"]
