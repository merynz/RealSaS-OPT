from __future__ import annotations

"""Pinned source-native diffusion teachers. No student image as ground truth.

A/B training receives frozen image tokens / source-rendered targets prepared
by the separate data foundry. This court compares actual tensor BYTES to an
independently admitted source-native registry. Generated Stage A/B predictions
do not receive this authority; they remain inference hypotheses.

A stage A receipt MUST also bind the original-two-image identity and frozen
source-encoder weights. The batch token hash alone cannot certify original
provenance; this upstream link is enforced by the separate source foundry.
"""

from hashlib import sha256
from typing import Any, Mapping
import json

import torch
from torch import Tensor

STAGES=("A_CANONICAL_SDXL","B_PAIRED_RGB_NORMAL_MVADAPTER")
SCHEMA="RealSaS.RSSvNext.SourceQualifiedSDXLMVAdapterStageAB.v2"
AUTHORITY="QUALIFIED_INDEPENDENT_NATIVE_3D_RENDER_NOT_GENERATED"


def _hex(value:object)->bool:
    return type(value) is str and len(value)==64 and all(
        c in "0123456789abcdef" for c in value
    )


def source_tensor_bytes_sha256_v2(t:Tensor)->str:
    if not isinstance(t,Tensor) or not torch.is_floating_point(t):
        raise ValueError("RSS_VNEXT_STAGE_AB_TEACHER_TENSOR_INVALID")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("RSS_VNEXT_STAGE_AB_TEACHER_NONFINITE")
    raw=t.detach().contiguous().cpu()
    head=json.dumps({
        "shape":list(raw.shape),"dtype":str(raw.dtype),
    },sort_keys=True,separators=(",",":")).encode()
    return sha256(head+b"\0"+raw.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def source_stageab_tensor_digests_v2(stage:str,batch:Any)->dict[str,str]:
    if stage not in STAGES:
        raise ValueError("RSS_VNEXT_STAGE_AB_TRUTH_STAGE_INVALID")
    fields=(
        (
            "source_tokens","target_canonical_rgba","target_valid",
            "target_cameras","canonical_pose_queries",
        )
        if stage=="A_CANONICAL_SDXL" else
        (
            "canonical_rgba","rgb","normal_encoded_rgb",
            "alpha","valid","target_cameras","plucker_controls",
        )
    )
    return {
        field+"_sha256":source_tensor_bytes_sha256_v2(getattr(batch,field))
        for field in fields
    }


def qualify_stageab_training_source_v2(
    *,stage:str,batch:Any,
    dataset_manifest_sha256:str,
    trusted_source_registry:Mapping[str,Mapping[str,Any]],
)->str:
    """Reject unregistered/altered targets before any optimizer operation."""
    if (
        stage not in STAGES or not _hex(dataset_manifest_sha256)
        or not isinstance(trusted_source_registry,Mapping)
    ):
        raise ValueError("RSS_VNEXT_STAGE_AB_TRUTH_REGISTRY_POLICY")
    key=getattr(batch,"source_render_qualification_key",None)
    receipt=trusted_source_registry.get(key)
    if type(receipt) not in (dict,):
        raise ValueError("RSS_VNEXT_STAGE_AB_SOURCE_TRUTH_NOT_REGISTERED")
    required={
        "schema":SCHEMA,"admission_status":AUTHORITY,
        "stage":stage,
        "dataset_manifest_sha256":dataset_manifest_sha256,
        "source_render_qualification_key":key,
        "generated_model_prediction_as_teacher":False,
        "source_3d_renders_independently_audited":True,
        **source_stageab_tensor_digests_v2(stage,batch),
    }
    if (
        any(receipt.get(k)!=v for k,v in required.items())
        or not _hex(receipt.get("source_asset_sha256"))
        or not _hex(receipt.get("original_two_rgba_sha256"))
        or not _hex(receipt.get("native_source_receipt_sha256"))
        or (
            stage=="A_CANONICAL_SDXL" and
            not _hex(receipt.get("frozen_source_encoder_sha256"))
        )
    ):
        raise ValueError("RSS_VNEXT_STAGE_AB_NATIVE_TEACHER_RECEIPT_DRIFT")
    return sha256(json.dumps(
        receipt,sort_keys=True,separators=(",",":"),
        allow_nan=False,
    ).encode()).hexdigest()


def digest_stageab_registry_v2(
    registry:Mapping[str,Mapping[str,Any]],
)->str:
    if not isinstance(registry,Mapping) or not registry:
        raise ValueError("RSS_VNEXT_STAGE_AB_EMPTY_TRUSTED_REGISTRY")
    return sha256(json.dumps({
        str(k):dict(v) for k,v in registry.items()
    },sort_keys=True,separators=(",",":"),
        allow_nan=False,
    ).encode()).hexdigest()


__all__=[
    "SCHEMA","AUTHORITY","STAGES",
    "source_tensor_bytes_sha256_v2",
    "source_stageab_tensor_digests_v2",
    "qualify_stageab_training_source_v2",
    "digest_stageab_registry_v2",
]
