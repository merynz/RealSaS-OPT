from __future__ import annotations

"""Training utilities for the MIRA V6 discrete skin-field codec.

The codec is trained against dense carrier truth, then frozen. The exact frozen
codec produces the RealSaS discrete skin-code proposal consumed as AXIS V6 causal
teacher truth. This keeps AXIS/MIRA coupled without giving Compiler any code-search
role.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from compiler.realsas_compiler_core.skin_code_ir_v1 import (
    JointSkinCodeRowIR,
    SkinCodeProposalIR,
    seal_skin_code_proposal_v1,
)
from models.mira.skin_fsq_codec_v6 import MIRASkinFSQCodecV6


@dataclass(frozen=True)
class MIRAV6LossWeights:
    reconstruction: float = 1.0
    quantization: float = 0.05
    entropy: float = 0.001


def mira_v6_codec_objective(
    model: MIRASkinFSQCodecV6,
    weights: Tensor,
    vertex_features: Tensor,
    joint_features: Tensor,
    *,
    loss_weights: MIRAV6LossWeights = MIRAV6LossWeights(),
) -> dict[str, Tensor]:
    result = model(weights, vertex_features, joint_features)
    reconstructed = result["reconstructed_weights"]
    entropy = -(reconstructed.clamp_min(model.config.eps) * reconstructed.clamp_min(model.config.eps).log()).sum(dim=-1).mean()
    total = (
        loss_weights.reconstruction * result["reconstruction_loss"]
        + loss_weights.quantization * result["quantization_loss"]
        + loss_weights.entropy * entropy
    )
    target = weights / weights.sum(dim=-1, keepdim=True).clamp_min(model.config.eps)
    abs_error = (reconstructed - target).abs()
    return {
        "total": total,
        "reconstruction": result["reconstruction_loss"],
        "quantization": result["quantization_loss"],
        "entropy": entropy,
        "mae": abs_error.mean(),
        "max_abs": abs_error.amax(),
        "simplex_max_abs": (reconstructed.sum(dim=-1) - 1.0).abs().amax(),
        "code_ids": result["token_ids"],
        "reconstructed_weights": reconstructed,
    }


@torch.no_grad()
def encode_axis_teacher_codes_v6(
    model: MIRASkinFSQCodecV6,
    weights: Tensor,
    vertex_features: Tensor,
    joint_features: Tensor,
    *,
    proposal_joint_ids: Sequence[str],
    rig_program_proposal_hash: str,
    carrier_evidence_hash: str,
    carrier_topology_hash: str,
    carrier_geometry_hash: str,
    codec_checkpoint_hash: str,
    model_provenance: str,
) -> SkinCodeProposalIR:
    """Freeze dense truth into the exact discrete code vocabulary used by AXIS."""
    model.eval()
    if weights.ndim != 3 or weights.shape[0] != 1:
        raise ValueError("MIRA_V6_TEACHER_EXPORT_BATCH_ONE_REQUIRED")
    if len(proposal_joint_ids) != weights.shape[-1]:
        raise ValueError("MIRA_V6_TEACHER_EXPORT_JOINT_CARDINALITY_DRIFT")
    if len(set(map(str, proposal_joint_ids))) != len(proposal_joint_ids):
        raise ValueError("MIRA_V6_TEACHER_EXPORT_JOINT_ID_DUPLICATE")
    digest = str(codec_checkpoint_hash).strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("MIRA_V6_TEACHER_EXPORT_CODEC_HASH_INVALID")

    code_ids, _, _ = model.encode_tokens(weights, vertex_features, joint_features)
    if tuple(code_ids.shape) != (1, len(proposal_joint_ids), model.config.tokens_per_joint):
        raise RuntimeError("MIRA_V6_TEACHER_EXPORT_CODE_SHAPE_DRIFT")
    rows = tuple(
        JointSkinCodeRowIR(
            proposal_joint_id=str(jid),
            code_ids=tuple(int(v) for v in code_ids[0, i].tolist()),
            metadata={"joint_sequence_index": i, "teacher_generated_by_frozen_codec": True},
        )
        for i, jid in enumerate(proposal_joint_ids)
    )
    return seal_skin_code_proposal_v1(SkinCodeProposalIR(
        rows=rows,
        rig_program_proposal_hash=str(rig_program_proposal_hash),
        carrier_evidence_hash=str(carrier_evidence_hash),
        carrier_topology_hash=str(carrier_topology_hash),
        carrier_geometry_hash=str(carrier_geometry_hash),
        codec_architecture_id=model.architecture_id,
        codec_checkpoint_hash=digest,
        codebook_size=int(model.config.vocab_size),
        codes_per_joint=int(model.config.tokens_per_joint),
        model_provenance=str(model_provenance),
        metadata={
            "source": "DENSE_SKIN_TRUTH_ENCODED_BY_FROZEN_MIRA_V6_CODEC",
            "compiler_code_search_allowed": False,
            "compiler_code_repair_allowed": False,
            "axis_teacher_only": True,
        },
    ))


__all__ = ["MIRAV6LossWeights", "encode_axis_teacher_codes_v6", "mira_v6_codec_objective"]
