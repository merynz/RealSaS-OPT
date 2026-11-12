from __future__ import annotations

"""2-view -> one pinned RSS.vNext model -> sealed, replayable Compiler proposal.

This is the only permitted checkpoint-backed research inference entrypoint.
It does not skip the existing Stage19/carrier/graphics/product qualification
gates or create a playable ProductRevision merely from a model prediction.
The last-written receipt commits to the source tensor archive and proposal.
"""

from hashlib import sha256
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile
import json
import os

import numpy as np
import torch
from torch import Tensor, nn

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rss_vnext_source_observation_v1 import (
    RSSvNextTwoViewSourceObservationV1,
    canonicalize_two_view_source_v1,
    MAX_BYTES as _MAX_CANONICAL_SOURCE_BYTES,
)
from compiler.realsas_compiler_core.rig_constraint_schema_v1 import RIG_CONSTRAINT_OPERATOR_IDS_V1
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,
    validate_inference_evidence_v1,
)
from .inference_checkpoint_v1 import load_inference_checkpoint_v1
from .embedded_mira_decoder_v1 import (
    SCHEMA as EMBEDDED_MIRA_SCHEMA,
    encode_embedded_mira_decoder_v1,
)
from .model_config_archive_v1 import (
    encode_rss_vnext_unified_model_config_v1,
    decode_rss_vnext_unified_model_config_v1,
)
from .neural_scene_archive_v1 import (
    SCHEMA as NEURAL_SCENE_SCHEMA,
    encode_neural_scene_carrier_v1,
    read_neural_scene_carrier_v1,
)
from .inference_evidence_export_v1 import (
    export_inference_evidence_v1, _source_tensor_digest,
)

_RECEIPT_SCHEMA = "RealSaS.RSSvNext.SourceToCompilerProposalReceipt.v2"
_RECEIPT_FILES = ("observed_source_tensors.npz", "neural_evidence.json", "inference_receipt.json")
_ORIGINAL_SOURCE_FILE = "original_two_view_source.npz"
_NEURAL_SCENE_FILE = "rss_vnext_neural_scene.npz"
_UNIFIED_CONFIG_FILE = "rss_vnext_unified_model_config.json"
_EMBEDDED_MIRA_FILE = "rss_vnext_embedded_mira_decoder.pt"
_MAX_SOURCE_BYTES = 128 * 1024 * 1024


def _unified_config_digest_v1(cfg) -> str:
    # Exact architecture_digest used by RSS training/checkpoint admission.
    return sha256(json.dumps(
        asdict(cfg), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf8")).hexdigest()


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
    source_observation: RSSvNextTwoViewSourceObservationV1 | None = None,
) -> dict:
    """Execute pinned checkpoint and commit the proposal with two source frames.

    ABSTAIN is a successful evidence outcome when learned topology fails.
    A successful neural output remains unqualified until existing courts pass.
    """
    _source_preflight(observed_rgba, camera_features)
    if product_mode and source_observation is None:
        raise RuntimeError("RSS_VNEXT_PRODUCT_CANONICAL_SOURCE_REQUIRED")
    if source_observation is not None:
        if not isinstance(source_observation, RSSvNextTwoViewSourceObservationV1):
            raise RuntimeError("RSS_VNEXT_SOURCE_CANONICAL_TYPE_REQUIRED")
        normalized, cameras = source_observation.to_model_tensors(
            device=str(observed_rgba.device),
        )
        if not torch.equal(observed_rgba, normalized) or not torch.equal(camera_features, cameras):
            raise RuntimeError("RSS_VNEXT_SOURCE_NORMALIZATION_OR_CAMERA_DRIFT")
    target = Path(output_directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    # Refuse accidental reuse of an existing receipt: revision lineage is immutable.
    if any((target / name).exists() for name in (*_RECEIPT_FILES, _ORIGINAL_SOURCE_FILE, _NEURAL_SCENE_FILE, _UNIFIED_CONFIG_FILE, _EMBEDDED_MIRA_FILE)):
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
    # Export the actual embedded MIRA FSQ + spatial decoder weights from this
    # SAME unified checkpoint. No fake independent MIRA model release or
    # launcher-supplied codec digest is allowed to establish skin-code meaning.
    mira_decoder_bytes = encode_embedded_mira_decoder_v1(
        model, parent_unified_checkpoint_sha256=expected_checkpoint_sha256,
    )
    _atomic_bytes(target / _EMBEDDED_MIRA_FILE, mira_decoder_bytes)
    # Persist the exact codec/model architecture separately so Compiler
    # can replay TESSA and the joint region schema WITHOUT loading a second
    # checkpoint or guessing hidden training configuration.
    config_bytes = encode_rss_vnext_unified_model_config_v1(
        model.cfg, checkpoint_sha256=expected_checkpoint_sha256,
    )
    _atomic_bytes(target / _UNIFIED_CONFIG_FILE, config_bytes)
    # Persist every model-owned neural carrier field independently of the
    # canonical decision path. A legal topology ABSTAIN still carries this
    # scene so Semantic, Graphics and Mechanics never depend on Mesh success.
    carrier_bytes = encode_neural_scene_carrier_v1(
        prediction["state"], evidence=evidence,
    )
    read_neural_scene_carrier_v1(carrier_bytes, evidence=evidence)
    carrier_path = target / _NEURAL_SCENE_FILE
    _atomic_bytes(carrier_path, carrier_bytes)
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
        "unified_model_config_sha256": _unified_config_digest_v1(model.cfg),
        "unified_model_config_file_sha256": _digest_file(target / _UNIFIED_CONFIG_FILE),
        "embedded_mira_decoder_schema": EMBEDDED_MIRA_SCHEMA,
        "embedded_mira_decoder_file_sha256": _digest_file(target / _EMBEDDED_MIRA_FILE),
        "embedded_mira_independent_release_required": False,
        "embedded_mira_discrete_runtime_parity_qualified": False,
        "source_archive_sha256": _digest_file(source_path),
        "evidence_file_sha256": _digest_file(evidence_path),
        "neural_scene_schema": NEURAL_SCENE_SCHEMA,
        "neural_scene_file_sha256": _digest_file(carrier_path),
        "neural_scene_retained_after_mesh_abstain": True,
        "neural_scene_product_authority": False,
        "evidence_hash": evidence.evidence_hash,
        "observed_view_count": 2,
        "product_revision_minted": False,
        "qualification_status": "COMPILER_EVIDENCE_SEALED_NOT_PRODUCT_QUALIFIED",
    }
    if source_observation is not None:
        original = target / _ORIGINAL_SOURCE_FILE
        original_tmp = original.with_name(original.name + ".tmp")
        try:
            with original_tmp.open("wb") as fp:
                np.savez_compressed(
                    fp,
                    native_rgba=source_observation.native_rgba,
                    camera_features=source_observation.camera_features,
                    source_rights_claim_sha256=np.asarray(
                        [source_observation.source_rights_claim_sha256],
                    ),
                )
            os.replace(original_tmp, original)
        finally:
            original_tmp.unlink(missing_ok=True)
        receipt["canonical_source_binding_sha256"] = source_observation.source_binding_sha256
        receipt["canonical_source_file_sha256"] = _digest_file(original)
        receipt["source_rights_claim_sha256"] = source_observation.source_rights_claim_sha256
        receipt["source_rights_verified"] = False
        receipt["source_normalization"] = "RGBA_UINT8_DIV_255_FLOAT32_NO_COMPOSITE_NO_RECENTER"
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
    _carrier_decoder_follows: bool = False,
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
        or type(receipt.get("unified_model_config_sha256")) is not str
        or len(receipt["unified_model_config_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in receipt["unified_model_config_sha256"])
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
    if (
        receipt.get("neural_scene_schema") != NEURAL_SCENE_SCHEMA
        or receipt.get("neural_scene_retained_after_mesh_abstain") is not True
        or receipt.get("neural_scene_product_authority") is not False
    ):
        raise RuntimeError("RSS_VNEXT_RECEIPT_NEURAL_CARRIER_POLICY_DRIFT")
    mira_path = root / _EMBEDDED_MIRA_FILE
    if (
        receipt.get("embedded_mira_decoder_schema") != EMBEDDED_MIRA_SCHEMA
        or receipt.get("embedded_mira_independent_release_required") is not False
        or receipt.get("embedded_mira_discrete_runtime_parity_qualified") is not False
        or not mira_path.is_file()
        or mira_path.stat().st_size > 512 * 1024 * 1024
        or _digest_file(mira_path) != receipt.get("embedded_mira_decoder_file_sha256")
    ):
        raise RuntimeError("RSS_VNEXT_RECEIPT_EMBEDDED_MIRA_PARENT_OR_BYTES_DRIFT")
    cfg_path = root / _UNIFIED_CONFIG_FILE
    if (
        not cfg_path.is_file()
        or _digest_file(cfg_path) != receipt.get("unified_model_config_file_sha256")
    ):
        raise RuntimeError("RSS_VNEXT_RECEIPT_UNIFIED_CONFIG_BYTES_DRIFT")
    decode_rss_vnext_unified_model_config_v1(
        cfg_path.read_bytes(),
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=receipt["unified_model_config_sha256"],
    )
    carrier_path = root / _NEURAL_SCENE_FILE
    if (
        not carrier_path.is_file()
        or carrier_path.stat().st_size > 512 * 1024 * 1024
        or _digest_file(carrier_path) != receipt.get("neural_scene_file_sha256")
    ):
        raise RuntimeError("RSS_VNEXT_RECEIPT_NEURAL_CARRIER_BYTES_DRIFT")
    raw = json.loads(evidence_path.read_text("utf-8"))
    evidence = RSSvNextInferenceEvidenceIRV1(**raw)
    validate_inference_evidence_v1(
        evidence, expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    if evidence.evidence_hash != receipt["evidence_hash"] or evidence.mesh_status != receipt["status"]:
        raise RuntimeError("RSS_VNEXT_RECEIPT_EVIDENCE_DRIFT")
    if not _carrier_decoder_follows:
        read_neural_scene_carrier_v1(carrier_path.read_bytes(), evidence=evidence)
    # _carrier_decoder_follows is an internal performance seam for consumers
    # that immediately run the SAME bound neural-carrier court themselves.
    # The public/default receipt reader never skips full archive validation.
    if source_path.stat().st_size > _MAX_SOURCE_BYTES:
        raise RuntimeError("RSS_VNEXT_RECEIPT_SOURCE_BUNDLE_TOO_LARGE")
    with np.load(source_path, allow_pickle=False) as archive:
        if set(archive.files) != {"observed_rgba", "camera_features"}:
            raise RuntimeError("RSS_VNEXT_RECEIPT_SOURCE_FIELDS_DRIFT")
        rgba = torch.from_numpy(archive["observed_rgba"])
        cam = torch.from_numpy(archive["camera_features"])
    _source_preflight(rgba, cam)
    canonical_fields = {
        "canonical_source_binding_sha256", "canonical_source_file_sha256",
        "source_rights_claim_sha256", "source_rights_verified",
        "source_normalization",
    }
    present = canonical_fields.intersection(receipt)
    if present and present != canonical_fields:
        raise RuntimeError("RSS_VNEXT_RECEIPT_CANONICAL_SOURCE_FIELDS_INCOMPLETE")
    original_path = root / _ORIGINAL_SOURCE_FILE
    if present:
        if receipt["source_rights_verified"] is not False:
            raise RuntimeError("RSS_VNEXT_SOURCE_RIGHTS_PREMATURE_AUTHORITY")
        if receipt["source_normalization"] != "RGBA_UINT8_DIV_255_FLOAT32_NO_COMPOSITE_NO_RECENTER":
            raise RuntimeError("RSS_VNEXT_SOURCE_NORMALIZATION_POLICY_DRIFT")
        if (
            not original_path.is_file()
            or original_path.stat().st_size > _MAX_CANONICAL_SOURCE_BYTES
            or _digest_file(original_path) != receipt["canonical_source_file_sha256"]
        ):
            raise RuntimeError("RSS_VNEXT_RECEIPT_CANONICAL_SOURCE_BYTES_DRIFT")
        # Bound decompressed NumPy entries before loading untrusted archives.
        with ZipFile(original_path) as zf:
            entries = zf.infolist()
            if (
                len(entries) != 3
                or sum(entry.file_size for entry in entries) > _MAX_CANONICAL_SOURCE_BYTES
                or any(entry.file_size > _MAX_CANONICAL_SOURCE_BYTES for entry in entries)
            ):
                raise RuntimeError("RSS_VNEXT_CANONICAL_SOURCE_ARCHIVE_LIMIT")
        with np.load(original_path, allow_pickle=False) as raw:
            if set(raw.files) != {
                "native_rgba", "camera_features", "source_rights_claim_sha256",
            }:
                raise RuntimeError("RSS_VNEXT_CANONICAL_SOURCE_ARCHIVE_FIELDS_DRIFT")
            rights = raw["source_rights_claim_sha256"]
            if rights.shape != (1,):
                raise RuntimeError("RSS_VNEXT_CANONICAL_SOURCE_RIGHTS_FIELD_INVALID")
            canonical = canonicalize_two_view_source_v1(
                raw["native_rgba"], raw["camera_features"],
                source_rights_claim_sha256=str(rights[0]),
            )
        normalized_rgba, normalized_camera = canonical.to_model_tensors()
        if (
            not torch.equal(normalized_rgba, rgba)
            or not torch.equal(normalized_camera, cam)
            or canonical.source_binding_sha256 != receipt["canonical_source_binding_sha256"]
            or canonical.source_rights_claim_sha256 != receipt["source_rights_claim_sha256"]
        ):
            raise RuntimeError("RSS_VNEXT_CANONICAL_SOURCE_OBSERVATION_BINDING_DRIFT")
    elif original_path.exists():
        raise RuntimeError("RSS_VNEXT_RECEIPT_UNDECLARED_SOURCE_ARTIFACT")
    if (
        _source_tensor_digest(rgba[0, 0]) != evidence.observed_front_sha256
        or _source_tensor_digest(rgba[0, 1]) != evidence.observed_back_sha256
        or _source_tensor_digest(cam) != evidence.camera_binding_sha256
    ):
        raise RuntimeError("RSS_VNEXT_RECEIPT_OBSERVATION_HASH_DRIFT")
    return receipt, evidence


@torch.no_grad()
def run_canonical_two_view_to_compiler_proposal_v1(
    *,
    source_observation: RSSvNextTwoViewSourceObservationV1,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    vision_encoder: nn.Module,
    output_directory: str | Path,
    max_topology_tokens: int | None = None,
    generate_controls: bool = True,
    product_mode: bool = True,
) -> dict:
    """Only canonical source entry into production-oriented RSS.vNext inference.

    A rights *claim* and a model prediction are not qualification receipts.
    Source input is never replaced by synthetic camera or novel-view evidence.
    """
    rgba, camera = source_observation.to_model_tensors()
    return run_two_view_to_compiler_proposal_v1(
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        vision_encoder=vision_encoder,
        observed_rgba=rgba,
        camera_features=camera,
        output_directory=output_directory,
        max_topology_tokens=max_topology_tokens,
        generate_controls=generate_controls,
        product_mode=product_mode,
        source_observation=source_observation,
    )


def read_compiler_neural_scene_v1(
    directory: str | Path,
    *, expected_checkpoint_sha256: str,
) -> dict[str, Tensor]:
    """Independent consumer read, verified against the SAME source+model court."""
    _, evidence = read_compiler_proposal_receipt_v1(
        directory, expected_checkpoint_sha256=expected_checkpoint_sha256,
        _carrier_decoder_follows=True,
    )
    return read_neural_scene_carrier_v1(
        (Path(directory).resolve() / _NEURAL_SCENE_FILE).read_bytes(),
        evidence=evidence,
    )


def read_compiler_expert_fanout_v1(
    directory: str | Path,
    *,
    expected_checkpoint_sha256: str,
    model_config,
):
    """Read original RSS two-view + unified scene + typed specialist evidence.

    Receipt verifies all CAS files/observations; the fan-out court decodes the
    large shared neural archive exactly ONCE. Mesh need not exist. No alternate
    IRIS V3 independent checkpoint or fixed humanoid semantic adapter.
    """
    from compiler.realsas_compiler_core.rss_vnext_expert_fanout_v1 import (
        open_rss_vnext_expert_fanout_v1,
    )
    receipt, evidence = read_compiler_proposal_receipt_v1(
        directory,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        _carrier_decoder_follows=True,
    )
    if _unified_config_digest_v1(model_config) != receipt["unified_model_config_sha256"]:
        raise RuntimeError("RSS_VNEXT_RECEIPT_UNIFIED_MODEL_CONFIG_DRIFT")
    return open_rss_vnext_expert_fanout_v1(
        evidence,
        (Path(directory).resolve() / _NEURAL_SCENE_FILE).read_bytes(),
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_model_config_sha256=receipt["unified_model_config_sha256"],
        model_config=model_config,
    )


__all__ = [
    "read_compiler_expert_fanout_v1",
    "read_compiler_neural_scene_v1",
    "run_two_view_to_compiler_proposal_v1",
    "run_canonical_two_view_to_compiler_proposal_v1",
    "read_compiler_proposal_receipt_v1",
]
