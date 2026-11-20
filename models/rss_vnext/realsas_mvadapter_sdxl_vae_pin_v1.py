from __future__ import annotations

"""Official MV-Adapter SDXL VAE pin: strong pretrained fp16-stable ancestor.

MV-Adapter's actual SDXL I2MV training config sets
  pretrained_vae_name_or_path: madebyollin/sdxl-vae-fp16-fix
(not the SDXL base VAE in default Diffusers pipelines).
This model card currently advertises MIT; the pinned revision license and
downstream obligations require source-review, not automated legal approval.

This loader never downloads or deserializes unsafe pickle. Identical SHA-pinned
VAE bytes must be used in BOTH A/B loss encoding and released sampling.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

SCHEMA="RealSaS.RSSvNext.MVAdapterSDXLFixedVAEAncestor.v1"
UPSTREAM_REPO="madebyollin/sdxl-vae-fp16-fix"
UPSTREAM_LICENSE="MIT"
WEIGHT="diffusion_pytorch_model.safetensors"


def _sha(v:object,n:int)->bool:
    return type(v) is str and len(v)==n and all(
        ch in "0123456789abcdef" for ch in v
    )


def _file_digest(path:Path)->str:
    digest=sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PinnedMVAdapterSDXLVAEV1:
    snapshot_dir:str
    pinned_revision:str
    checkpoint_sha256:str
    model_config_sha256:str
    independent_license_review_sha256:str
    source_repo:str=UPSTREAM_REPO
    license_id:str=UPSTREAM_LICENSE
    schema:str=SCHEMA

    def verify(self)->dict[str,str]:
        if (
            self.schema!=SCHEMA
            or self.source_repo!=UPSTREAM_REPO
            or self.license_id!=UPSTREAM_LICENSE
            or not _sha(self.pinned_revision,40)
            or not all(_sha(x,64) for x in (
                self.checkpoint_sha256,self.model_config_sha256,
                self.independent_license_review_sha256,
            ))
        ):
            raise ValueError("RSS_VNEXT_MVADAPTER_VAE_RIGHTS_OR_REVISION_PIN")
        folder=Path(self.snapshot_dir).expanduser()
        if folder.is_symlink() or not folder.is_dir():
            raise RuntimeError("RSS_VNEXT_MVADAPTER_VAE_LOCAL_PIN_MISSING")
        weight=folder/WEIGHT
        config=folder/"config.json"
        if (
            not weight.is_file() or weight.is_symlink()
            or not config.is_file() or config.is_symlink()
            or _file_digest(weight)!=self.checkpoint_sha256
            or _file_digest(config)!=self.model_config_sha256
        ):
            raise RuntimeError("RSS_VNEXT_MVADAPTER_VAE_CHECKPOINT_BYTES_DRIFT")
        data=json.loads(config.read_text())
        if (
            data.get("_class_name")!="AutoencoderKL"
            or data.get("in_channels")!=3
            or data.get("out_channels")!=3
            or data.get("latent_channels")!=4
        ):
            raise RuntimeError("RSS_VNEXT_MVADAPTER_VAE_INCOMPATIBLE_SDXL_CODEC")
        return {
            "schema":SCHEMA,
            "vae_ancestry":UPSTREAM_REPO,
            "vae_weight_sha256":self.checkpoint_sha256,
            "license_id":UPSTREAM_LICENSE,
            "license_human_review_not_automatically_proven":"true",
        }


def load_official_mvadapter_fp16_stable_vae_v1(
    pin:PinnedMVAdapterSDXLVAEV1, *,
    device:str,dtype:Any,
):
    pin.verify()
    from diffusers import AutoencoderKL
    vae=AutoencoderKL.from_pretrained(
        str(Path(pin.snapshot_dir).expanduser().resolve()),
        local_files_only=True,use_safetensors=True,torch_dtype=dtype,
    )
    if (
        vae.config.in_channels!=3
        or vae.config.out_channels!=3
        or vae.config.latent_channels!=4
    ):
        raise RuntimeError("RSS_VNEXT_MVADAPTER_VAE_LOADED_ARCH_DRIFT")
    vae=vae.to(device=device,dtype=dtype).eval()
    vae.requires_grad_(False)
    vae._rss_vnext_vae_ancestor_sha256=pin.checkpoint_sha256
    return vae


def bind_mvadapter_vae_to_pipelines_v1(
    pin:PinnedMVAdapterSDXLVAEV1,*,
    stage_b_pipeline:Any,
    device:str,dtype:Any,
):
    """Replace SDXL base pipeline VAE with the official training VAE.

    Use the returned SAME VAE instance for Stage A and Stage B encodes/decodes.
    It does NOT unfreeze or train the upstream VAE.
    """
    if not hasattr(stage_b_pipeline,"vae"):
        raise ValueError("RSS_VNEXT_MVADAPTER_PIPELINE_VAE_SLOT_REQUIRED")
    actual=load_official_mvadapter_fp16_stable_vae_v1(
        pin,device=device,dtype=dtype,
    )
    stage_b_pipeline.vae=actual
    return actual


__all__=[
    "SCHEMA","UPSTREAM_REPO","UPSTREAM_LICENSE","WEIGHT",
    "PinnedMVAdapterSDXLVAEV1",
    "load_official_mvadapter_fp16_stable_vae_v1",
    "bind_mvadapter_vae_to_pipelines_v1",
]
