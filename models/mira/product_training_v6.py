from __future__ import annotations

"""Source-qualified FSQ skin codec training on exact product runtime features.

Matches SkinTokens' codec-first dependency: the MIRA FSQ representation must
be trained on source mesh/joint/skin truth before AXIS can learn its code IDs.
This input is NOT guessed from neural geometry or a Compiler repair. No masked
unknown weight is implicitly zero; only fully admitted source skin matrices
may enter the present dense codec.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import torch
from torch import Tensor

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR, canonical_mesh_lineage_hash_v1,
    validate_canonical_mesh_domain_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalRigIRV2, canonical_rig_lineage_hash_v2,
    validate_canonical_rig_domain_v2,
)
from models.mira.conditioning_v6 import (
    MIRA_V6_CONDITIONING_ID, MIRA_V6_VERTEX_FEATURE_DIM, MIRA_V6_JOINT_FEATURE_DIM,
    build_mira_conditioning_v6,
)
from models.mira.skin_fsq_codec_v6 import (
    MIRASkinFSQCodecV6, MIRA_V6_CODEC_ID,
)


def skin_truth_tensor_sha256_v6(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise ValueError("MIRA_PRODUCT_SOURCE_TENSOR_REQUIRED")
    cpu = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype":str(cpu.dtype),"shape":list(cpu.shape)},
        sort_keys=True,separators=(",",":"),
    ).encode()
    return sha256(header+b"\0"+cpu.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class MIRAProductCodecTrainingReportV6:
    loss: Tensor
    reconstruction_loss: Tensor
    quantization_loss: Tensor
    token_ids: Tensor
    conditioning_id: str
    mesh_lineage_hash: str
    deform_structure_hash: str
    source_weight_sha256: str
    product_qualification_granted: bool = False


def qualified_mira_product_codec_objective_v6(
    codec: MIRASkinFSQCodecV6,
    *,
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    source_dense_weights: Tensor,
    trusted_receipts: Mapping[str, Mapping[str, Any]],
    receipt_key: str,
    dataset_manifest_sha256: str,
    source_qualification_key: str,
) -> MIRAProductCodecTrainingReportV6:
    """Train the exact 7D/25D MIRA decoder used by product.skin V6."""
    if (
        codec.config.vertex_feature_dim != MIRA_V6_VERTEX_FEATURE_DIM
        or codec.config.joint_feature_dim != MIRA_V6_JOINT_FEATURE_DIM
    ):
        raise ValueError("MIRA_PRODUCT_TRAIN_FEATURE_DIM_NOT_RUNTIME_COMPATIBLE")
    if not codec.training:
        raise ValueError("MIRA_PRODUCT_TRAIN_CODEC_NOT_IN_TRAIN_MODE")
    if (
        mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh)
        or rig.rig_lineage_hash != canonical_rig_lineage_hash_v2(rig)
        or rig.mesh_lineage_hash != mesh.mesh_lineage_hash
    ):
        raise ValueError("MIRA_PRODUCT_TRAIN_CANONICAL_PARENT_DRIFT")
    validate_canonical_mesh_domain_v1(mesh)
    validate_canonical_rig_domain_v2(rig)
    conditioning = build_mira_conditioning_v6(
        mesh, rig, device=next(codec.parameters()).device,
    )
    if conditioning.conditioning_id != MIRA_V6_CONDITIONING_ID:
        raise ValueError("MIRA_PRODUCT_TRAIN_CONDITIONING_DRIFT")
    expected = (1, len(mesh.vertices), len(rig.joints))
    if (
        source_dense_weights.shape != expected
        or not torch.is_floating_point(source_dense_weights)
        or not bool(torch.isfinite(source_dense_weights).all())
        or bool(((source_dense_weights < 0) | (source_dense_weights > 1)).any())
        or not torch.allclose(
            source_dense_weights.sum(-1), torch.ones(expected[:2],device=source_dense_weights.device),
            atol=1e-6,rtol=0,
        )
    ):
        raise ValueError("MIRA_PRODUCT_TRAIN_SOURCE_SKIN_SIMPLEX_OR_COVERAGE_INVALID")
    if source_dense_weights.device != conditioning.vertex_features.device:
        raise ValueError("MIRA_PRODUCT_TRAIN_SOURCE_SKIN_DEVICE_DRIFT")
    receipt = trusted_receipts.get(receipt_key)
    digest = skin_truth_tensor_sha256_v6(source_dense_weights)
    if (
        receipt is None
        or receipt.get("admission_status") != "QUALIFIED_FULL_SOURCE_SKIN"
        or receipt.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or receipt.get("source_qualification_key") != source_qualification_key
        or receipt.get("source_weight_sha256") != digest
        or receipt.get("mesh_lineage_hash") != mesh.mesh_lineage_hash
        or receipt.get("deform_structure_hash") != rig.deform_structure_hash
        or receipt.get("conditioning_id") != MIRA_V6_CONDITIONING_ID
        or receipt.get("codec_architecture_id") != MIRA_V6_CODEC_ID
        or receipt.get("source_semantics") != "SOURCE_SKIN_WEIGHT_MATRIX_NOT_NEURAL_PSEUDOLABEL"
    ):
        raise ValueError("MIRA_PRODUCT_TRAIN_UNADMITTED_WEIGHT_OR_PARENT")
    result = codec(
        source_dense_weights,
        conditioning.vertex_features,
        conditioning.joint_features,
    )
    if not bool(torch.isfinite(result["loss"])):
        raise ValueError("MIRA_PRODUCT_TRAIN_NONFINITE_OBJECTIVE")
    return MIRAProductCodecTrainingReportV6(
        loss=result["loss"],
        reconstruction_loss=result["reconstruction_loss"],
        quantization_loss=result["quantization_loss"],
        token_ids=result["token_ids"],
        conditioning_id=conditioning.conditioning_id,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        source_weight_sha256=digest,
    )


__all__ = [
    "MIRAProductCodecTrainingReportV6",
    "skin_truth_tensor_sha256_v6",
    "qualified_mira_product_codec_objective_v6",
]
