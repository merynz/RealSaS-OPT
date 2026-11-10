from __future__ import annotations

"""2-view -> one pinned RSS.vNext model -> sealed, replayable Compiler proposal.

This is the only permitted checkpoint-backed research inference entrypoint.
It does not skip the existing Stage19/carrier/graphics/product qualification
gates or create a playable ProductRevision merely from a model prediction.
The last-written receipt commits to the source tensor archive and proposal.
"""

from hashlib import sha256
from pathlib import Path
import json
import os

import numpy as np
import torch
from torch import Tensor, nn

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rig_constraint_schema_v1 import RIG_CONSTRAINT_OPERATOR_IDS_V1
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,
    validate_inference_evidence_v1,
)
from .inference_checkpoint_v1 import load_inference_checkpoint_v1
from .inference_evidence_export_v1 import (
    export_inference_evidence_v1, _source_tensor_digest,
)

_RECEIPT_SCHEMA = "RealSaS.RSSvNext.SourceToCompilerProposalReceipt.v1"
_RECEIPT_FILES = ("observed_source_tensors.npz", "neural_evidence.json", "inference_receipt.json")
_MAX_SOURCE_BYTES = 128 * 1024 * 1024


def _digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(value)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _source_preflight(rgba: Tensor, cameras: Tensor) -> None:
    if rgba.ndim != 5 or rgba.shape[:3] != (1, 2, 4):
        raise ValueError("RSS_VNEXT_RUN_TWO_SOURCE_RGBA_REQUIRED")
    if cameras.shape != (1, 2, 13):
        raise ValueError("RSS_VNEXT_RUN_CAMERA_13D_PAIR_REQUIRED")
    if rgba.dtype != torch.float32 or cameras.dtype != torch.float32:
        raise ValueError("RSS_VNEXT_RUN_FLOAT32_CONTRACT_REQUIRED")
    if rgba.device != cameras.device:
        raise ValueError("RSS_VNEXT_RUN_SOURCE_CAMERA_DEVICE_DRIFT")
    if rgba.shape[-1] < 14 or rgba.shape[-2] < 14:
        # DINOv2 uses 14-pixel patches; tiny test encoder can use 8x8.
        raise ValueError("RSS_VNEXT_RUN_SOURCE_RESOLUTION_TOO_SMALL")
    if rgba.shape[-1] > 2048 or rgba.shape[-2] > 2048:
        raise ValueError("RSS_VNEXT_RUN_SOURCE_RESOLUTION_TOO_LARGE")
    if not bool(torch.isfinite(rgba).all()) or not bool(torch.isfinite(cameras).all()):
        raise ValueError("RSS_VNEXT_RUN_NONFINITE_INPUT")
    if bool(((rgba < 0) | (rgba > 1)).any()):
        raise ValueError("RSS_VNEXT_RUN_RGBA_OUTSIDE_NORMALIZED_RANGE")
    if rgba.numel() * 4 + cameras.numel() * 4 > _MAX_SOURCE_BYTES:
        raise ValueError("RSS_VNEXT_RUN_SOURCE_ARCHIVE_TOO_LARGE")


@torch.no_grad()
def run_two_view_to_compiler_proposal_v1(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    vision_encoder: nn.Module,
    observed_rgba: Tensor,
    camera_features: Tensor,
    output_directory: str | Path,
    max_topology_tokens: int | None = None,
    generate_controls: bool = True,
    product_mode: bool = True,
) -> dict:
    """Execute pinned checkpoint and commit the proposal with two source frames.

    ABSTAIN is a successful evidence outcome when learned topology fails.
    A successful neural output remains unqualified until existing courts pass.
    """
    _source_preflight(observed_rgba, camera_features)
    target = Path(output_directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    # Refuse accidental reuse of an existing receipt: revision lineage is immutable.
    if any((target / name).exists() for name in _RECEIPT_FILES):
        raise RuntimeError("RSS_VNEXT_RUN_OUTPUT_REVISION_ALREADY_EXISTS")
    model = load_inference_checkpoint_v1(
        checkpoint_path,
        expected_sha256=expected_checkpoint_sha256,
        vision_encoder=vision_encoder,
        device=observed_rgba.device,
        product_mode=product_mode,
    )
    prediction = model.propose_teacher_free(
        observed_rgba,
        camera_features,
        max_topology_tokens=max_topology_tokens,
        generate_controls=generate_controls,
        executable_constraint_operators=(
            tuple(RIG_CONSTRAINT_OPERATOR_IDS_V1) if generate_controls else None
        ),
    )
    evidence = export_inference_evidence_v1(
        prediction,
        original_rgba=observed_rgba,
        original_camera_features=camera_features,
        checkpoint_sha256=expected_checkpoint_sha256,
    )
    validate_inference_evidence_v1(
        evidence, expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    source_path, evidence_path, receipt_path = (
        target / _RECEIPT_FILES[0],
        target / _RECEIPT_FILES[1],
        target / _RECEIPT_FILES[2],
    )
    source_tmp = source_path.with_name(source_path.name + ".tmp")
    try:
        with source_tmp.open("wb") as fp:
            np.savez_compressed(
                fp,
                observed_rgba=observed_rgba.detach().cpu().numpy(),
                camera_features=camera_features.detach().cpu().numpy(),
            )
        os.replace(source_tmp, source_path)
    finally:
        source_tmp.unlink(missing_ok=True)
    payload = json.dumps(
        evidence.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    _atomic_bytes(evidence_path, payload)
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "status": evidence.mesh_status,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "source_archive_sha256": _digest_file(source_path),
        "evidence_file_sha256": _digest_file(evidence_path),
        "evidence_hash": evidence.evidence_hash,
        "observed_view_count": 2,
        "product_revision_minted": False,
        "qualification_status": "COMPILER_EVIDENCE_SEALED_NOT_PRODUCT_QUALIFIED",
    }
    receipt["receipt_hash"] = content_sha256(receipt)
    _atomic_bytes(
        receipt_path,
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    return receipt


def read_compiler_proposal_receipt_v1(
    directory: str | Path,
    *,
    expected_checkpoint_sha256: str,
) -> tuple[dict, RSSvNextInferenceEvidenceIRV1]:
    """Validate persisted artifacts before handing evidence to Compiler courts."""
    root = Path(directory).resolve()
    source_path, evidence_path, receipt_path = (
        root / _RECEIPT_FILES[0], root / _RECEIPT_FILES[1], root / _RECEIPT_FILES[2]
    )
    receipt = json.loads(receipt_path.read_text("utf-8"))
    if receipt.get("schema") != _RECEIPT_SCHEMA:
        raise RuntimeError("RSS_VNEXT_RECEIPT_SCHEMA_DRIFT")
    claimed = dict(receipt)
    claimed_hash = claimed.pop("receipt_hash", None)
    if claimed_hash != content_sha256(claimed):
        raise RuntimeError("RSS_VNEXT_RECEIPT_HASH_DRIFT")
    if (
        receipt.get("checkpoint_sha256") != expected_checkpoint_sha256
        or receipt.get("observed_view_count") != 2
        or receipt.get("product_revision_minted") is not False
        or receipt.get("qualification_status") != "COMPILER_EVIDENCE_SEALED_NOT_PRODUCT_QUALIFIED"
    ):
        raise RuntimeError("RSS_VNEXT_RECEIPT_AUTHORITY_DRIFT")
    if (
        _digest_file(source_path) != receipt.get("source_archive_sha256")
        or _digest_file(evidence_path) != receipt.get("evidence_file_sha256")
    ):
        raise RuntimeError("RSS_VNEXT_RECEIPT_ARTIFACT_BYTES_DRIFT")
    raw = json.loads(evidence_path.read_text("utf-8"))
    evidence = RSSvNextInferenceEvidenceIRV1(**raw)
    validate_inference_evidence_v1(
        evidence, expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    if evidence.evidence_hash != receipt["evidence_hash"] or evidence.mesh_status != receipt["status"]:
        raise RuntimeError("RSS_VNEXT_RECEIPT_EVIDENCE_DRIFT")
    if source_path.stat().st_size > _MAX_SOURCE_BYTES:
        raise RuntimeError("RSS_VNEXT_RECEIPT_SOURCE_BUNDLE_TOO_LARGE")
    with np.load(source_path, allow_pickle=False) as archive:
        if set(archive.files) != {"observed_rgba", "camera_features"}:
            raise RuntimeError("RSS_VNEXT_RECEIPT_SOURCE_FIELDS_DRIFT")
        rgba = torch.from_numpy(archive["observed_rgba"])
        cam = torch.from_numpy(archive["camera_features"])
    _source_preflight(rgba, cam)
    if (
        _source_tensor_digest(rgba[0, 0]) != evidence.observed_front_sha256
        or _source_tensor_digest(rgba[0, 1]) != evidence.observed_back_sha256
        or _source_tensor_digest(cam) != evidence.camera_binding_sha256
    ):
        raise RuntimeError("RSS_VNEXT_RECEIPT_OBSERVATION_HASH_DRIFT")
    return receipt, evidence


__all__ = [
    "run_two_view_to_compiler_proposal_v1",
    "read_compiler_proposal_receipt_v1",
]
