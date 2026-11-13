from __future__ import annotations

"""Independently authored Era3D-style UNet feature-scale row-wise attention.

Reference: Era3D NeurIPS 2024 and the CVPR 2025 multi-view reference §3.1. Unlike the
earlier external noisy-latent row mixer, this operator executes WITHIN each
Diffusers Transformer2DModel feature stage (down, mid and up), at that
stage's native resolution. It expects six orthographic, same-elevation
camera views in matched normal/RGB order. This is a cleanroom approximation
with necessary semantics, NOT bytewise Era3D parity or trained quality.

Zero-initialized output projects to an exact identity residual at install.
No Era3D code or task weights are imported. Original SD2.1 2D weights
are retained; the separate 8-channel input conversion reuses the original
four-channel pretrained weights.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .realsas_era3d_unclip_microstep_v1 import (
    expand_pretrained_unclip_input_conv_v1,
)


class Era3DUNetFeatureRowAttentionV1(nn.Module):
    """Each fixed scanline attends over 6 views x {normal, RGB} x width.

    With aligned orthographic projections this row grouping corresponds to
    the same world-space height. This is invalid for arbitrary unequal
    elevation or perspective cameras; those must not enter this baseline.
    """
    def __init__(
        self, feature_channels: int, *, hidden_width: int = 64,
        heads: int = 4, row_chunk: int = 2, max_row_tokens: int = 2048,
    ) -> None:
        super().__init__()
        if (
            feature_channels < 1 or hidden_width < 4 or heads < 1
            or hidden_width % heads or row_chunk < 1 or row_chunk > 8
            or max_row_tokens < 12
        ):
            raise ValueError("RSS_VNEXT_ERA3D_FEATURE_ROW_CONFIG")
        self.feature_channels=feature_channels
        self.row_chunk=row_chunk
        self.max_row_tokens=max_row_tokens
        self.to_attention=nn.Linear(feature_channels,hidden_width)
        self.view_modality_codes=nn.Embedding(12,hidden_width)
        self.cross_view=nn.MultiheadAttention(
            hidden_width,heads,batch_first=True,
        )
        self.to_feature=nn.Linear(hidden_width,feature_channels)
        nn.init.zeros_(self.to_feature.weight)
        nn.init.zeros_(self.to_feature.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim!=4:
            raise ValueError("RSS_VNEXT_ERA3D_FEATURE_ROW_DOMAIN")
        batch,c,h,w=features.shape
        if (
            batch<12 or batch%12 or batch//12>8
            or c!=self.feature_channels or 12*w>self.max_row_tokens
            or h<1 or not bool(torch.isfinite(features).all())
        ):
            raise ValueError("RSS_VNEXT_ERA3D_FEATURE_ROW_RESOURCE_OR_CAMERA_ALIGNMENT")
        b=batch//12
        # Flatten order is NORMAL (views 0..5), RGB (views 0..5).
        # Never flatten H with W; each y is independently coupled.
        features5=features.reshape(b,12,c,h,w).permute(0,3,1,4,2)
        delta=[]
        dtype=self.to_attention.weight.dtype
        view=self.view_modality_codes.weight.to(dtype)[None,None,:,None,:]
        for start in range(0,h,self.row_chunk):
            rows=features5[:,start:start+self.row_chunk]
            count=rows.shape[1]
            emb=self.to_attention(rows.to(dtype))+view
            seq=emb.reshape(b*count,12*w,-1)
            mixed,_=self.cross_view(seq,seq,seq,need_weights=False)
            change=self.to_feature(mixed).reshape(b,count,12,w,c)
            delta.append(change)
        correction=torch.cat(delta,dim=1).permute(0,2,4,1,3)
        return features + correction.reshape_as(features).to(features.dtype)


class Era3DUNetTransformerRowAdapterV1(nn.Module):
    """Wrap a Diffusers Transformer2DModel without replacing base weights."""
    def __init__(self, transformer: nn.Module, feature_channels: int,
                 *, hidden_width: int = 64) -> None:
        super().__init__()
        self.original_transformer=transformer
        self.rowwise_features=Era3DUNetFeatureRowAttentionV1(
            feature_channels,hidden_width=hidden_width,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output=self.original_transformer(*args,**kwargs)
        if isinstance(output,tuple):
            if not output or not isinstance(output[0],Tensor):
                raise ValueError("RSS_VNEXT_ERA3D_TRANSFORMER_OUTPUT_SHAPE")
            return (self.rowwise_features(output[0]),*output[1:])
        sample=getattr(output,"sample",None)
        if not isinstance(sample,Tensor):
            raise ValueError("RSS_VNEXT_ERA3D_TRANSFORMER_OUTPUT_SHAPE")
        # Diffusers Transformer2DModelOutput is a dataclass BaseOutput whose
        # constructor accepts sample. Avoid mutating the original output.
        return type(output)(sample=self.rowwise_features(sample))


def install_era3d_cleanroom_unet_row_attention_v1(
    denoiser: nn.Module, *, hidden_width: int = 64,
) -> tuple[str,...]:
    """Prepare pretrained SD2.1 UnCLIP UNet for independent Era3D-style fit.

    External attention over input-noise latents is explicitly *not* accepted.
    Targets must be aligned six-view orthographic RGB and normal renderings.
    The returned paths are all stage-wise Transformer2DModel consumers.
    """
    if getattr(denoiser,"rowwise_attention_in_unet_features",False):
        raise ValueError("RSS_VNEXT_ERA3D_ROW_ATTENTION_ALREADY_INSTALLED")
    conv=getattr(denoiser,"conv_in",None)
    config=getattr(denoiser,"config",None)
    if (
        type(conv) is not nn.Conv2d or conv.in_channels!=4
        or config is None or getattr(config,"in_channels",None)!=4
        or not callable(getattr(denoiser,"register_to_config",None))
    ):
        raise ValueError("RSS_VNEXT_ERA3D_EXPECT_SD21_UNCLIP_4CHANNEL_UNET")
    targets=[]
    for path,module in denoiser.named_modules():
        if type(module).__name__ == "Transformer2DModel":
            channels=getattr(getattr(module,"config",None),"in_channels",None)
            if not isinstance(channels,int) or channels<1:
                raise ValueError("RSS_VNEXT_ERA3D_TRANSFORMER_CHANNEL_CONTRACT")
            targets.append((path,module,channels))
    if not targets or len(targets)>80:
        raise ValueError("RSS_VNEXT_ERA3D_NO_VALID_UNET_FEATURE_TRANSFORMERS")
    # All shape/config checks complete before touching pretrained weights.
    converted=expand_pretrained_unclip_input_conv_v1(conv)
    wrappers=[]
    for path,module,channels in targets:
        wrappers.append((path,Era3DUNetTransformerRowAdapterV1(
            module,channels,hidden_width=hidden_width,
        )))
    denoiser.conv_in=converted
    denoiser.register_to_config(in_channels=8)
    for path,wrapper in wrappers:
        name,_,leaf=path.rpartition(".")
        parent=denoiser.get_submodule(name) if name else denoiser
        setattr(parent,leaf,wrapper)
    denoiser.requires_grad_(True)
    denoiser.rowwise_attention_in_unet_features=True
    return tuple(path for path,_,_ in targets)


__all__=[
    "Era3DUNetFeatureRowAttentionV1",
    "Era3DUNetTransformerRowAdapterV1",
    "install_era3d_cleanroom_unet_row_attention_v1",
]
