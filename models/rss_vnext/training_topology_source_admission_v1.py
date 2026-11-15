from __future__ import annotations

"""Source-native TESSA teacher token admission for production training.

A model-produced mesh token sequence cannot certify itself. The trainer
accepts topology supervision only when a launcher-owned registry binds the
exact source asset, original images, target token bytes, and validity mask.
This mirrors the existing AXIS product-matched mesh teacher policy, but
also protects TESSA-only phases that have no AXIS loss active.
"""

from typing import Mapping, Callable
from torch import Tensor
import torch

SCHEMA="RealSaS.RSSvNext.SourceTopologyTeacherAdmission.v1"
STATUS="QUALIFIED_SOURCE_NATIVE_TESSA_TOPOLOGY"


def validate_source_topology_teacher_v1(
    *,
    token_ids:Tensor|None,
    valid_mask:Tensor|None,
    admission_key:str,
    trusted_registry:Mapping[str,Mapping[str,object]],
    dataset_manifest_sha256:str,
    source_qualification_key:str,
    original_rgba:Tensor,
    input_camera_features:Tensor,
    tensor_digest:Callable[[Tensor],str],
)->None:
    if (
        token_ids is None or valid_mask is None
        or type(admission_key) is not str or not admission_key
        or not isinstance(token_ids,Tensor) or token_ids.dtype!=torch.long
        or token_ids.ndim!=2 or token_ids.shape[1]<3
        or not isinstance(valid_mask,Tensor) or valid_mask.dtype!=torch.bool
        or valid_mask.shape!=(token_ids.shape[0],)
        or not bool(valid_mask.any())
    ):
        raise ValueError("RSS_VNEXT_TESSA_SOURCE_TOPOLOGY_BATCH_INCOMPLETE")
    row=trusted_registry.get(admission_key)
    if row is None:
        raise ValueError("RSS_VNEXT_TESSA_SOURCE_TOPOLOGY_UNTRUSTED")
    expected={
        "schema":SCHEMA,
        "admission_status":STATUS,
        "dataset_manifest_sha256":dataset_manifest_sha256,
        "source_qualification_key":source_qualification_key,
        "observed_rgba_sha256":tensor_digest(original_rgba),
        "input_camera_features_sha256":tensor_digest(input_camera_features),
        "topology_teacher_tokens_sha256":tensor_digest(token_ids),
        "topology_valid_mask_sha256":tensor_digest(valid_mask),
        "teacher_semantics":"SOURCE_NATIVE_INDEXED_MESH_NOT_STUDENT_PREDICTION",
        "source_mesh_reindex_verified":True,
    }
    if any(row.get(k)!=v for k,v in expected.items()):
        raise ValueError("RSS_VNEXT_TESSA_SOURCE_TOPOLOGY_RECEIPT_DRIFT")
    for key in ("source_asset_sha256","source_native_faces_sha256"):
        value=row.get(key)
        if (type(value) is not str or len(value)!=64
            or any(ch not in "0123456789abcdef" for ch in value)):
            raise ValueError("RSS_VNEXT_TESSA_SOURCE_TOPOLOGY_ASSET_IDENTITY_INVALID")


__all__=["SCHEMA","STATUS","validate_source_topology_teacher_v1"]
