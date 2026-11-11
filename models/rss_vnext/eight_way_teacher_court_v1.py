from __future__ import annotations

"""Fail-closed, source-bound 2->8 appearance teacher admission.

A string starting with 'QUALIFIED_SOURCE_RENDER' is NOT proof. The training
job must receive a trusted, immutable registry of source-render qualification
records, independently produced by the corpus/source court. This module checks
that every teacher pixel, mask, source input, and camera ring matches that
registry. It does not mint source truth, qualify artist labels, or accept
novel views as extra student observations.

The proof source (native scene, lineage, rights and rendered frames) must be
verified by the external admission builder; matching hashes alone cannot prove
that a scene is the correct original.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Mapping

import torch
from torch import Tensor

from .multiview_construction import canonical_novel_camera_ring


_SCHEMA = "RealSaS.RSSvNext.EightWayNativeSourceRenderReceipt.v1"
_VIEW_ORDER = (1, 2, 3, 5, 6, 7)
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True,
                   separators=(",", ":"), allow_nan=False).encode("utf8")
    ).hexdigest()


def tensor_bytes_sha256_v1(value: Tensor) -> str:
    """Hash tensor content, dtype and exact shape; masked edits remain detectable."""
    if not isinstance(value, Tensor) or value.ndim == 0:
        raise ValueError("RSS_VNEXT_TEACHER_TENSOR_INVALID")
    if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
        raise ValueError("RSS_VNEXT_TEACHER_TENSOR_NONFINITE")
    array = value.detach().to("cpu").contiguous().numpy()
    meta = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf8")
    return sha256(meta + b"\0" + array.tobytes()).hexdigest()


@dataclass(frozen=True)
class RSSvNextEightWayTeacherReceiptV1:
    """Record content stored in the independently approved source registry.

    No function here can grant source authority by creating this dataclass.
    The registry MUST be supplied by a separate trusted corpus loader.
    """

    source_asset_sha256: str
    native_scene_sha256: str
    native_render_artifact_sha256: str
    source_qualification_key: str
    dataset_manifest_sha256: str
    front_back_rgba_sha256: str
    source_cameras_sha256: str
    novel_cameras_sha256: str
    novel_rgba_sha256: str
    novel_normal_sha256: str
    novel_rgb_mask_sha256: str
    novel_normal_mask_sha256: str
    provenance_kind: str = "ORIGINAL_NATIVE_SOURCE_RENDER"
    target_view_indices: tuple[int, ...] = _VIEW_ORDER
    schema_version: str = _SCHEMA
    record_hash: str = ""

    def as_record(self) -> dict:
        return asdict(self)


def source_teacher_receipt_hash_v1(value: RSSvNextEightWayTeacherReceiptV1) -> str:
    record = value.as_record()
    record.pop("record_hash")
    return _digest(record)


def expected_eight_way_tensor_hashes_v1(
    *,
    observed_rgba: Tensor,
    source_cameras: Tensor,
    novel_rgba: Tensor,
    novel_normal: Tensor,
    rgb_mask: Tensor,
    normal_mask: Tensor,
    index: int,
) -> dict[str, str]:
    """Return *unqualified* content fingerprints; NEVER grants admission."""
    if (observed_rgba.ndim != 5 or observed_rgba.shape[1:3] != (2, 4)
            or source_cameras.shape != (observed_rgba.shape[0], 2, 13)):
        raise ValueError("RSS_VNEXT_TEACHER_OBSERVED_INPUT_DOMAIN")
    b = observed_rgba.shape[0]
    if not (0 <= index < b):
        raise ValueError("RSS_VNEXT_TEACHER_BATCH_INDEX")
    if (novel_rgba.ndim != 5 or novel_rgba.shape[:3] != (b, 6, 4)
            or novel_normal.shape != (b, 6, 3, *novel_rgba.shape[-2:])
            or rgb_mask.shape != (b, 6, *novel_rgba.shape[-2:])
            or normal_mask.shape != rgb_mask.shape
            or rgb_mask.dtype != torch.bool or normal_mask.dtype != torch.bool):
        raise ValueError("RSS_VNEXT_TEACHER_RENDER_SHAPES")
    ring = canonical_novel_camera_ring(source_cameras, 8)
    return {
        "front_back_rgba_sha256": tensor_bytes_sha256_v1(observed_rgba[index]),
        "source_cameras_sha256": tensor_bytes_sha256_v1(source_cameras[index]),
        "novel_cameras_sha256": tensor_bytes_sha256_v1(ring.novel_cameras[index]),
        "novel_rgba_sha256": tensor_bytes_sha256_v1(novel_rgba[index]),
        "novel_normal_sha256": tensor_bytes_sha256_v1(novel_normal[index]),
        "novel_rgb_mask_sha256": tensor_bytes_sha256_v1(rgb_mask[index]),
        "novel_normal_mask_sha256": tensor_bytes_sha256_v1(normal_mask[index]),
    }


def validate_eight_way_teacher_receipts_v1(
    *,
    observed_rgba: Tensor,
    source_cameras: Tensor,
    novel_rgba: Tensor,
    novel_normal: Tensor,
    rgb_mask: Tensor,
    normal_mask: Tensor,
    receipts: tuple[RSSvNextEightWayTeacherReceiptV1, ...] | None,
    trusted_registry: Mapping[str, Mapping[str, object]] | None,
    dataset_manifest_sha256: str,
    source_qualification_key: str,
) -> str:
    """Return a *derived* provenance string only after external record match.

    A sample-provided receipt is never authoritative by itself. The trusted
    registry must contain an exact external record under its content hash.
    """
    batch = observed_rgba.shape[0]
    if receipts is None or len(receipts) != batch or not trusted_registry:
        raise ValueError("RSS_VNEXT_TEACHER_TRUSTED_REGISTRY_REQUIRED")
    if not _SHA.fullmatch(dataset_manifest_sha256):
        raise ValueError("RSS_VNEXT_TEACHER_DATASET_DIGEST")
    for i, receipt in enumerate(receipts):
        if not isinstance(receipt, RSSvNextEightWayTeacherReceiptV1):
            raise ValueError("RSS_VNEXT_TEACHER_RECEIPT_TYPE")
        if (receipt.schema_version != _SCHEMA
                or receipt.provenance_kind != "ORIGINAL_NATIVE_SOURCE_RENDER"
                or receipt.target_view_indices != _VIEW_ORDER):
            raise ValueError("RSS_VNEXT_TEACHER_PROVENANCE_OR_ORDER")
        for digest in (
            receipt.source_asset_sha256, receipt.native_scene_sha256,
            receipt.native_render_artifact_sha256, receipt.dataset_manifest_sha256,
            receipt.record_hash,
        ):
            if not isinstance(digest, str) or not _SHA.fullmatch(digest):
                raise ValueError("RSS_VNEXT_TEACHER_RECEIPT_SHA256")
        if (receipt.dataset_manifest_sha256 != dataset_manifest_sha256
                or receipt.source_qualification_key != source_qualification_key):
            raise ValueError("RSS_VNEXT_TEACHER_SOURCE_OR_DATASET_DRIFT")
        if receipt.record_hash != source_teacher_receipt_hash_v1(receipt):
            raise ValueError("RSS_VNEXT_TEACHER_RECEIPT_HASH_DRIFT")
        stored = trusted_registry.get(receipt.record_hash)
        if stored is None or dict(stored) != receipt.as_record():
            raise ValueError("RSS_VNEXT_TEACHER_EXTERNAL_ADMISSION_MISSING")
        expected = expected_eight_way_tensor_hashes_v1(
            observed_rgba=observed_rgba, source_cameras=source_cameras,
            novel_rgba=novel_rgba, novel_normal=novel_normal,
            rgb_mask=rgb_mask, normal_mask=normal_mask, index=i,
        )
        for key, fingerprint in expected.items():
            if getattr(receipt, key) != fingerprint:
                raise ValueError("RSS_VNEXT_TEACHER_PAYLOAD_DRIFT:" + key)
    return "QUALIFIED_SOURCE_RENDER:EXTERNAL_REGISTRY:" + _digest(
        [r.record_hash for r in receipts]
    )


__all__ = [
    "RSSvNextEightWayTeacherReceiptV1",
    "tensor_bytes_sha256_v1",
    "source_teacher_receipt_hash_v1",
    "expected_eight_way_tensor_hashes_v1",
    "validate_eight_way_teacher_receipts_v1",
]
