from __future__ import annotations

from dataclasses import dataclass

import torch

from compiler.realsas_compiler_services.platform_worker.model_binding import (
    MaterializedModelReleaseRef,
    require_materialized_model_release,
)
from models.iris.vnext.checkpoint_v1 import load_checkpoint_v1
from models.iris.vnext.inference_input_v1 import IRISVNextSceneInputsV1
from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3


IRIS_MODEL_FAMILY = "IRIS"
IRIS_ARCHITECTURE_ID = IRISVNextEvidenceModelV3.architecture_id


@dataclass(frozen=True)
class IRISVNextInferenceBindingV1:
    model_ref: MaterializedModelReleaseRef
    model: IRISVNextEvidenceModelV3
    schema: str = "RealSaS.IRISVNextInferenceBinding.v1"


def load_iris_vnext_inference_v1(
    manifest: dict,
    *,
    device: torch.device | str = "cpu",
) -> IRISVNextInferenceBindingV1:
    """Load the exact Attempt-bound IRIS V3 checkpoint for inference only."""

    ref = require_materialized_model_release(
        manifest,
        family=IRIS_MODEL_FAMILY,
        architecture_id=IRIS_ARCHITECTURE_ID,
    )
    model = load_checkpoint_v1(ref.checkpoint_path, device=device)
    if model.architecture_id != ref.architecture_id:
        raise RuntimeError("IRIS_VNEXT_INFERENCE_ARCHITECTURE_BINDING_DRIFT")
    return IRISVNextInferenceBindingV1(model_ref=ref, model=model)


@torch.no_grad()
def encode_scene_v1(
    binding: IRISVNextInferenceBindingV1,
    inputs: IRISVNextSceneInputsV1,
) -> torch.Tensor:
    """Execute the first behaviorally effective IRIS V3 inference boundary."""

    if not isinstance(binding, IRISVNextInferenceBindingV1):
        raise TypeError("IRIS_VNEXT_INFERENCE_BINDING_REQUIRED")
    inputs.validate()
    model = binding.model
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("IRIS_VNEXT_INFERENCE_MODEL_NOT_FROZEN")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("IRIS_VNEXT_INFERENCE_MODEL_EMPTY") from exc
    planes = model.encode_scene(
        inputs.image_tokens.to(device=device),
        inputs.token_xy.to(device=device),
        inputs.camera_features.to(device=device),
    )
    expected = (
        inputs.image_tokens.shape[0],
        3,
        model.config.plane_channels,
        model.config.plane_size * 2,
        model.config.plane_size * 2,
    )
    if tuple(planes.shape) != expected:
        raise RuntimeError("IRIS_VNEXT_INFERENCE_SCENE_SHAPE_DRIFT")
    if not torch.is_floating_point(planes) or not bool(torch.isfinite(planes).all()):
        raise RuntimeError("IRIS_VNEXT_INFERENCE_SCENE_INVALID")
    return planes
