from __future__ import annotations

"""Fail-closed ORIGINAL InstantNeRF ancestry court for StdGEN S-LRM.

StdGEN CVPR2025 initializes NeRF S-LRM from the InstantMesh InstantNeRF
large model; source release uses DINO ViT-B/16 (not RSS DINOv2-S),
transformer width 1024, 16 layers and heads, triplane 32->64, 80 channels,
128 ray samples. The public model card for TencentARC/InstantMesh supplies
the instant_nerf_large.ckpt artifact with SHA256 below. Source is Apache-2
at repository level but some renderer files carry separate NVIDIA
proprietary headers: this tool loads ONLY model tensor data through
weights_only Torch, and imports NO upstream code.

An official checkpoint is never considered compatible with RSS IRIS
until all target model keys/shapes agree; no overlap-based partial load.
The hash/shape check does not constitute legal review or training quality.
A synthetic fixture identity exists only to test the algorithm.
"""

from dataclasses import dataclass
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

import torch
from torch import nn

_OFFICIAL_SHA="530a344ea9b856e012492b9644ab73532ddf7c06d548f60c489e49a528322fb9"
_HF_MODEL="TencentARC/InstantMesh/instant_nerf_large.ckpt"
_REAL_ID="STDGEN_2025_PRETRAINED_INSTANTNERF_LARGE_16L_VITB16"
_TEST_ID="SYNTHETIC_CHECKPOINT_UNIT_TEST_NO_UPSTREAM_PROOF"


@dataclass(frozen=True)
class StdGENInstantNeRFArchitectureV1:
    encoder_name:str="facebook/dino-vitb16"
    encoder_feature_dim:int=768
    transformer_width:int=1024
    transformer_layers:int=16
    transformer_heads:int=16
    triplane_base_size:int=32
    triplane_high_size:int=64
    triplane_channels:int=80
    ray_samples:int=128

    def check_published(self) -> None:
        if self != StdGENInstantNeRFArchitectureV1():
            raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_PUBLISHED_ARCHITECTURE_DRIFT")


@dataclass(frozen=True)
class StdGENInstantNeRFCheckpointReceiptV1:
    checkpoint_sha256:str
    model_card_source:str
    source_architecture_id:str
    extracted_tensor_count:int
    target_tensor_count:int
    exact_key_and_shape_match:bool
    test_fixture_only:bool
    source_code_imported:bool
    approved_as_source_observation:bool
    pretrained_model_loaded:bool


def _file_digest(path:Path) ->str:
    h=sha256()
    with path.open("rb") as f:
        while chunk:=f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_keys(
    path:Path, *,
    expected_sha256:str,unit_test_override:bool,
) ->tuple[str,dict[str,tuple[int,...]]]:
    if (
        not path.is_file() or path.is_symlink() or path.suffix!=".ckpt"
        or path.stat().st_size>2_500_000_000 or path.stat().st_size<100
    ):
        raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_CHECKPOINT_FILE_DOMAIN")
    digest=_file_digest(path)
    if unit_test_override:
        if expected_sha256==_OFFICIAL_SHA or digest!=expected_sha256:
            raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_TEST_HASH_CONTRACT")
    elif expected_sha256!=_OFFICIAL_SHA or digest!=_OFFICIAL_SHA:
        raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_PINNED_SHA256_MISMATCH")
    # Only metadata allocations: 1.51GB official checkpoint must NOT
    # materialize its tensors merely to inspect and compare shapes.
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            payload=torch.load(
                path,map_location="cpu",weights_only=True,mmap=True,
            )
    except (RuntimeError,ValueError,TypeError,KeyError) as exc:
        raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_SAFE_CHECKPOINT_PARSE") from exc
    if not isinstance(payload,Mapping) or not isinstance(payload.get("state_dict"),Mapping):
        raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_CHECKPOINT_STATE_DICT_REQUIRED")
    sd=payload["state_dict"]
    prefix="lrm_generator."
    keys={
        k[len(prefix):]:tuple(int(x) for x in v.shape)
        for k,v in sd.items()
        if isinstance(k,str) and k.startswith(prefix) and isinstance(v,torch.Tensor)
    }
    if not keys or len(keys)!=len(set(keys)):
        raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_PREFIXED_TENSORS_REQUIRED")
    return digest,keys


def inspect_stdgen_instantnerf_checkpoint_compatibility_v1(
    path:str|Path,model:nn.Module, *,
    expected_sha256:str=_OFFICIAL_SHA,
    unit_test_override:bool=False,
)->StdGENInstantNeRFCheckpointReceiptV1:
    """Compare all original InstantNeRF state tensor keys, not just shapes.

    Never auto-remap an InstantNeRF weight into a loosely named RSS IRIS
    tensor. Exact compatibility is mandatory before load.
    """
    if not isinstance(model,nn.Module):
        raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_NATIVE_MODEL_REQUIRED")
    filename=Path(path)
    digest,keys=_checkpoint_keys(
        filename,expected_sha256=expected_sha256,
        unit_test_override=unit_test_override,
    )
    if not unit_test_override:
        StdGENInstantNeRFArchitectureV1().check_published()
        positions=keys.get("transformer.pos_embed")
        deconv=keys.get("transformer.deconv.weight")
        if positions!=(1,3*32*32,1024) or deconv!=(1024,80,2,2):
            raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_16L_TRIPLANE_SHAPE_MISMATCH")
        for i in range(16):
            anchor=f"transformer.layers.{i}.self_attn.in_proj_weight"
            if keys.get(anchor)!=(3072,1024):
                raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_ATTENTION_LAYER_SHAPE_DRIFT")
    target={k:tuple(v.shape) for k,v in model.state_dict().items()}
    if keys!=target:
        missing=sorted(set(keys)^set(target))
        bad=sorted(k for k in set(keys)&set(target) if keys[k]!=target[k])
        excerpt=", ".join((missing+bad)[:4])
        raise ValueError(
            "RSS_VNEXT_STDGEN_INSTANTNERF_NOT_LOAD_COMPATIBLE_WITH_TARGET_MODEL:"+excerpt
        )
    return StdGENInstantNeRFCheckpointReceiptV1(
        checkpoint_sha256=digest,
        model_card_source=_TEST_ID if unit_test_override else _HF_MODEL,
        source_architecture_id=_TEST_ID if unit_test_override else _REAL_ID,
        extracted_tensor_count=len(keys),
        target_tensor_count=len(target),
        exact_key_and_shape_match=True,
        test_fixture_only=unit_test_override,
        source_code_imported=False,
        approved_as_source_observation=False,
        pretrained_model_loaded=False,
    )


def load_verified_stdgen_instantnerf_weights_v1(
    path:str|Path,model:nn.Module,
    *,compatibility:StdGENInstantNeRFCheckpointReceiptV1,
)->None:
    """Strict pretrain load, and ONLY when the official hash was verified."""
    if (
        type(compatibility) is not StdGENInstantNeRFCheckpointReceiptV1
        or compatibility.test_fixture_only
        or compatibility.checkpoint_sha256!=_OFFICIAL_SHA
        or compatibility.source_architecture_id!=_REAL_ID
        or not compatibility.exact_key_and_shape_match
        or compatibility.pretrained_model_loaded
        or not isinstance(model,nn.Module)
    ):
        raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_REAL_CHECKPOINT_ADMISSION_REQUIRED")
    if _file_digest(Path(path))!=_OFFICIAL_SHA:
        raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_LOAD_DIGEST_DRIFT")
    # Safe Torch loader accepts only tensors / containers / primitives.
    contents=torch.load(Path(path),map_location="cpu",weights_only=True,mmap=True)
    prefix="lrm_generator."
    state={k[len(prefix):]:v for k,v in contents["state_dict"].items()
           if isinstance(k,str) and k.startswith(prefix)}
    if state.keys()!=model.state_dict().keys():
        raise ValueError("RSS_VNEXT_STDGEN_INSTANTNERF_MODEL_KEY_DRIFT_ON_LOAD")
    model.load_state_dict(state,strict=True)


__all__=[
    "StdGENInstantNeRFArchitectureV1",
    "StdGENInstantNeRFCheckpointReceiptV1",
    "inspect_stdgen_instantnerf_checkpoint_compatibility_v1",
    "load_verified_stdgen_instantnerf_weights_v1",
]
