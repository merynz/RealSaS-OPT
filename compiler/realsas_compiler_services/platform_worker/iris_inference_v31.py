from __future__ import annotations

from dataclasses import dataclass

import torch

from compiler.realsas_compiler_services.platform_worker.model_binding import (
    MaterializedModelReleaseRef,
    require_materialized_model_release,
)
from models.iris.v31.checkpoint_v31 import load_checkpoint_v31
from models.iris.v31.inference_input_v31 import IRISV31SceneInputs
from models.iris.v31.model_v31 import IRISV31


IRIS_MODEL_FAMILY_V31 = "IRIS"
IRIS_ARCHITECTURE_ID_V31 = IRISV31.architecture_id


@dataclass(frozen=True)
class IRISV31InferenceBinding:
    model_ref: MaterializedModelReleaseRef
    model: IRISV31
    schema: str = "RealSaS.IRISV31InferenceBinding.v1"


def load_iris_v31_inference(
    manifest: dict,
    *,
    device: torch.device | str = "cpu",
) -> IRISV31InferenceBinding:
    ref = require_materialized_model_release(
        manifest,
        family=IRIS_MODEL_FAMILY_V31,
        architecture_id=IRIS_ARCHITECTURE_ID_V31,
    )
    model = load_checkpoint_v31(ref.checkpoint_path, device=device)
    if model.architecture_id != ref.architecture_id:
        raise RuntimeError("IRIS_V31_INFERENCE_ARCHITECTURE_BINDING_DRIFT")
    return IRISV31InferenceBinding(model_ref=ref, model=model)


@torch.no_grad()
def encode_carrier_v31(
    binding: IRISV31InferenceBinding,
    inputs: IRISV31SceneInputs,
) -> torch.Tensor:
    if not isinstance(binding, IRISV31InferenceBinding):
        raise TypeError("IRIS_V31_INFERENCE_BINDING_REQUIRED")
    inputs.validate()
    model = binding.model
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("IRIS_V31_INFERENCE_MODEL_NOT_FROZEN")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("IRIS_V31_INFERENCE_MODEL_EMPTY") from exc
    carrier = model.encode(
        final=inputs.final.to(device=device),
        layer2=inputs.layer2.to(device=device),
        alpha=inputs.alpha.to(device=device),
        cameras=inputs.camera_features.to(device=device),
    )
    expected = (
        inputs.final.shape[0],
        model.config.carrier_channels,
        model.config.carrier_resolution,
        model.config.carrier_resolution,
        model.config.carrier_resolution,
    )
    if tuple(carrier.shape) != expected:
        raise RuntimeError("IRIS_V31_INFERENCE_CARRIER_SHAPE_DRIFT")
    if not torch.is_floating_point(carrier) or not bool(torch.isfinite(carrier).all()):
        raise RuntimeError("IRIS_V31_INFERENCE_CARRIER_INVALID")
    return carrier
