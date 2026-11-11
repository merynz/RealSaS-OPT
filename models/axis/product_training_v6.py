from __future__ import annotations

"""AXIS product-aligned trainer: exact RiggingSurface 47D runtime conditioning.

The unified RSS.vNext shared-H AXIS auxiliary objective is NOT a substitute for
this product worker. ModelRelease accepts the AXIS runtime checkpoint only with
the exact 47D RiggingSurface adapter and the separately frozen MIRA codec hash.
This module trains no constraint/control labels without independent source
interventions and never turns synthetic inputs into scientific qualification.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Any

import torch
from torch import Tensor

from compiler.realsas_compiler_core.types import RiggingSurfaceIR
from compiler.realsas_compiler_core.rig_constraint_schema_v1 import RIG_CONSTRAINT_OPERATOR_IDS_V1
from models.axis.rig_program_model_v6 import (
    AXISRigProgramModelV6,
    AXISSequenceV6,
    AXIS_V6_PRODUCT_ARCHITECTURE_ID,
    RigTokenKindV6,
    qualified_axis_rig_skin_sequence_loss_v6,
)
from models.axis.scene_conditioning_v6 import (
    AXIS_V6_SURFACE_SCENE_DIM, AXIS_V6_SCENE_ADAPTER_ID,
    condition_axis_v6_from_rigging_surface,
)


def axis_product_teacher_sequence_sha256_v6(sequence: AXISSequenceV6) -> str:
    """Teacher bytes include every typed token/attribute, not just kind IDs."""
    fields = {}
    for name in sequence.__dataclass_fields__:
        tensor = getattr(sequence, name)
        if not isinstance(tensor, Tensor):
            raise ValueError("AXIS_PRODUCT_TEACHER_TENSOR_REQUIRED")
        t = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {"dtype": str(t.dtype), "shape": list(t.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode()
        fields[name] = sha256(header + b"\0" + t.numpy().tobytes()).hexdigest()
    return sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class AXISProductTrainingStepReportV6:
    loss: Tensor
    scene_adapter_id: str
    scene_tensorization_hash: str
    teacher_sequence_sha256: str
    skin_codec_checkpoint_sha256: str
    control_intervention_qualified: bool
    model_release_authority_granted: bool = False


def qualified_axis_product_teacher_objective_v6(
    model: AXISRigProgramModelV6,
    *,
    source_surface: RiggingSurfaceIR,
    teacher: AXISSequenceV6,
    receipt: Mapping[str, Any],
    dataset_manifest_sha256: str,
    source_qualification_key: str,
    codec_checkpoint_sha256: str,
    tokens_per_joint: int,
) -> AXISProductTrainingStepReportV6:
    """Actual product worker conditioning, no learned/shared-carrier surrogate."""
    if model.config.scene_dim != AXIS_V6_SURFACE_SCENE_DIM:
        raise ValueError("AXIS_PRODUCT_TRAIN_CONDITIONING_WIDTH_NOT_RUNTIME_COMPATIBLE")
    if not model.training:
        raise ValueError("AXIS_PRODUCT_TRAIN_MODEL_NOT_IN_TRAIN_MODE")
    scene = condition_axis_v6_from_rigging_surface(
        source_surface, device=next(model.parameters()).device,
    )
    if scene.scene_adapter_id != AXIS_V6_SCENE_ADAPTER_ID:
        raise ValueError("AXIS_PRODUCT_TRAIN_SCENE_ADAPTER_DRIFT")
    seq_hash = axis_product_teacher_sequence_sha256_v6(teacher)
    t = scene.surface_tensor
    if (
        receipt.get("admission_status") != "QUALIFIED_SOURCE_RIG_SKIN_TOKEN_TEACHER"
        or receipt.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or receipt.get("source_qualification_key") != source_qualification_key
        or receipt.get("architecture_id") != AXIS_V6_PRODUCT_ARCHITECTURE_ID
        or receipt.get("scene_adapter_id") != AXIS_V6_SCENE_ADAPTER_ID
        or receipt.get("surface_geometry_lineage_hash") != source_surface.geometry_lineage_hash
        or receipt.get("surface_tensorization_hash") != t.tensorization_hash
        or receipt.get("teacher_sequence_sha256") != seq_hash
        or receipt.get("mira_codec_checkpoint_sha256") != codec_checkpoint_sha256
        or receipt.get("teacher_semantics") != "SOURCE_RIG_AND_PINNED_FSQ_NOT_INFERRED_CONTROLS"
    ):
        raise ValueError("AXIS_PRODUCT_TRAIN_UNADMITTED_TEACHER_OR_CONDITIONING_DRIFT")
    if not isinstance(codec_checkpoint_sha256, str) or len(codec_checkpoint_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in codec_checkpoint_sha256
    ):
        raise ValueError("AXIS_PRODUCT_TRAIN_FSQ_CHECKPOINT_NOT_PINNED")
    needs_control = bool((
        (teacher.kind_ids == int(RigTokenKindV6.CONTROL))
        | (teacher.kind_ids == int(RigTokenKindV6.CONSTRAINT))
        | (teacher.kind_ids == int(RigTokenKindV6.ANCHOR))
    ).any())
    intervention = bool(
        receipt.get("intervention_qualification_status") == "QUALIFIED_SOURCE_CONTROL_CONSEQUENCE"
        and isinstance(receipt.get("source_intervention_sha256"), str)
        and len(receipt["source_intervention_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in receipt["source_intervention_sha256"])
    )
    if needs_control and not intervention:
        raise ValueError("AXIS_PRODUCT_TRAIN_CONTROL_INTERVENTION_REQUIRED")
    loss, _parts = qualified_axis_rig_skin_sequence_loss_v6(
        model,
        scene.scene_memory,
        teacher,
        tokens_per_joint=tokens_per_joint,
        executable_constraint_operators=RIG_CONSTRAINT_OPERATOR_IDS_V1,
        control_intervention_qualified=intervention,
    )
    return AXISProductTrainingStepReportV6(
        loss=loss,
        scene_adapter_id=scene.scene_adapter_id,
        scene_tensorization_hash=t.tensorization_hash,
        teacher_sequence_sha256=seq_hash,
        skin_codec_checkpoint_sha256=codec_checkpoint_sha256,
        control_intervention_qualified=intervention,
    )


__all__ = [
    "AXISProductTrainingStepReportV6",
    "axis_product_teacher_sequence_sha256_v6",
    "qualified_axis_product_teacher_objective_v6",
]
