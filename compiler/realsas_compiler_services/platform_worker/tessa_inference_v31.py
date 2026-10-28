from __future__ import annotations

from dataclasses import dataclass

import torch

from compiler.realsas_compiler_services.platform_worker.model_binding import (
    MaterializedModelReleaseRef,
    require_materialized_model_release,
)
from models.tessa.v31.checkpoint_v31 import load_checkpoint_v31
from models.tessa.v31.model_v31 import TESSAStructuralizerV31


TESSA_MODEL_FAMILY_V31 = "TESSA"
TESSA_ARCHITECTURE_ID_V31 = TESSAStructuralizerV31.architecture_id


@dataclass(frozen=True)
class TESSAV31InferenceBinding:
    model_ref: MaterializedModelReleaseRef
    model: TESSAStructuralizerV31
    schema: str = "RealSaS.TESSAV31InferenceBinding.v1"


def load_tessa_v31_inference(
    manifest: dict,
    *,
    device: torch.device | str = "cpu",
) -> TESSAV31InferenceBinding:
    ref = require_materialized_model_release(
        manifest,
        family=TESSA_MODEL_FAMILY_V31,
        architecture_id=TESSA_ARCHITECTURE_ID_V31,
    )
    model = load_checkpoint_v31(ref.checkpoint_path, device=device)
    if model.architecture_id != ref.architecture_id:
        raise RuntimeError("TESSA_V31_INFERENCE_ARCHITECTURE_BINDING_DRIFT")
    if model.attention_semantics != TESSAStructuralizerV31.attention_semantics:
        raise RuntimeError("TESSA_V31_INFERENCE_ATTENTION_SEMANTICS_DRIFT")
    return TESSAV31InferenceBinding(model_ref=ref, model=model)
