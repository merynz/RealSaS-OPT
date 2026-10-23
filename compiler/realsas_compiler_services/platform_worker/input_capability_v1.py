from __future__ import annotations

"""Current v1 source-observation lowering for the replaceable product Input capability.

The capability identity is intentionally not tied to eight views. This v1 source
schema is the currently admitted scientific contract; later source/model-input
schemas may replace it without changing Platform/CAS/Attempt authority machinery.
"""

from dataclasses import dataclass
import io
import json
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch

from compiler.realsas_compiler_services.platform_worker.scene_input_contract_v1 import (
    SHAPE_INPUT_ARTIFACT_TYPE_V1,
    encode_shape_input_v1,
)
from models.iris.v2.dinov2_foundation_v2 import (
    DINOv2FoundationAuthorityV2,
    DINOv2LoadedAuthorityV2,
    load_exact_dinov2_s_v2,
)
from models.iris.vnext.inference_input_v1 import (
    build_scene_inputs_from_rgba_v1,
    camera_features_v1,
)


INPUT_EXECUTOR_CONTRACT_V1 = "RealSaS.ProductInputInference.v1"
SOURCE_OBSERVATION_ARTIFACT_TYPE_V1 = "RealSaS.SourceObservation"
SOURCE_OBSERVATION_SCHEMA_V1 = "RealSaS.SourceObservation.v1"
INPUT_OUTPUT_AUTHORITY_V1 = "VERIFIED_SOURCE_DERIVED_SHAPE_INPUT"


@dataclass(frozen=True)
class SourceObservationV1:
    native_rgba: torch.Tensor
    camera_bundle: dict[str, Any]
    schema: str = SOURCE_OBSERVATION_SCHEMA_V1

    def validate(self) -> None:
        # Eight views are a v1 scientific contract, not a Platform invariant.
        if self.native_rgba.ndim != 5 or tuple(self.native_rgba.shape) != (1, 8, 4, 1024, 1024):
            raise ValueError("PRODUCT_INPUT_V1_RGBA_SHAPE_INVALID")
        if self.native_rgba.dtype != torch.uint8:
            raise ValueError("PRODUCT_INPUT_V1_RGBA_DTYPE_INVALID")
        camera = dict(self.camera_bundle)
        if camera.get("schema") != "RealSaS.CameraProjectionBundle.v1":
            raise ValueError("PRODUCT_INPUT_V1_CAMERA_SCHEMA_INVALID")
        if camera.get("per_view_recenter", False) is not False or camera.get("per_view_pose_change", False) is not False:
            raise ValueError("PRODUCT_INPUT_V1_PER_VIEW_CAMERA_MUTATION_FORBIDDEN")
        order = list(camera.get("view_order") or ())
        if len(order) != 8 or len(set(map(str, order))) != 8:
            raise ValueError("PRODUCT_INPUT_V1_VIEW_ORDER_INVALID")
        camera_features_v1(camera, batch_size=1, device="cpu")


def encode_source_observation_v1(
    *,
    native_rgba: torch.Tensor | np.ndarray,
    camera_bundle: Mapping[str, Any],
) -> bytes:
    rgba = torch.as_tensor(native_rgba).detach().cpu().contiguous()
    if rgba.ndim == 4:
        rgba = rgba.unsqueeze(0)
    source = SourceObservationV1(native_rgba=rgba, camera_bundle=dict(camera_bundle))
    source.validate()
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        schema=np.asarray([SOURCE_OBSERVATION_SCHEMA_V1]),
        rgba=rgba.numpy(),
        camera_json=np.asarray(
            [json.dumps(source.camera_bundle, sort_keys=True, separators=(",", ":"))]
        ),
    )
    return stream.getvalue()


def decode_source_observation_v1(payload: bytes) -> SourceObservationV1:
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise RuntimeError("PRODUCT_INPUT_V1_SOURCE_BYTES_REQUIRED")
    try:
        with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as bundle:
            if set(bundle.files) != {"schema", "rgba", "camera_json"}:
                raise RuntimeError("PRODUCT_INPUT_V1_SOURCE_FIELDS_DRIFT")
            schema = str(bundle["schema"].reshape(-1)[0])
            if schema != SOURCE_OBSERVATION_SCHEMA_V1:
                raise RuntimeError("PRODUCT_INPUT_V1_SOURCE_SCHEMA_DRIFT")
            rgba_np = np.asarray(bundle["rgba"])
            if rgba_np.dtype != np.uint8:
                raise RuntimeError("PRODUCT_INPUT_V1_SOURCE_RGBA_DTYPE_DRIFT")
            camera = json.loads(str(bundle["camera_json"].reshape(-1)[0]))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("PRODUCT_INPUT_V1_SOURCE_DECODE_FAILED") from exc
    source = SourceObservationV1(
        native_rgba=torch.from_numpy(rgba_np.copy()),
        camera_bundle=dict(camera),
    )
    source.validate()
    return source


@dataclass(frozen=True)
class InputInferenceResultV1:
    payload: bytes
    diagnostics: dict[str, Any]
    schema: str = "RealSaS.ProductInputInferenceResult.v1"


@torch.no_grad()
def run_input_inference_v1(
    source: SourceObservationV1,
    *,
    dino_model: torch.nn.Module,
    loaded_authority: DINOv2LoadedAuthorityV2,
    device: torch.device | str = "cpu",
) -> InputInferenceResultV1:
    """Lower exact v1 source observations into the admitted frozen IRIS input frame."""
    source.validate()
    if dino_model.training or any(parameter.requires_grad for parameter in dino_model.parameters()):
        raise RuntimeError("PRODUCT_INPUT_V1_DINO_NOT_FROZEN")
    started = perf_counter()
    scene_inputs = build_scene_inputs_from_rgba_v1(
        dino_model=dino_model,
        loaded_authority=loaded_authority,
        native_rgba=source.native_rgba.to(device=device),
        camera_bundle=source.camera_bundle,
    )
    payload = encode_shape_input_v1(scene_inputs)
    return InputInferenceResultV1(
        payload=payload,
        diagnostics={
            "schema": "RealSaS.ProductInputInferenceDiagnostics.v1",
            "authority": INPUT_OUTPUT_AUTHORITY_V1,
            "output_artifact_type": SHAPE_INPUT_ARTIFACT_TYPE_V1,
            "source_schema": SOURCE_OBSERVATION_SCHEMA_V1,
            "foundation_authority_sha256": scene_inputs.foundation_authority_sha256,
            "view_count": 8,
            "native_resolution": 1024,
            "elapsed_seconds": float(perf_counter() - started),
            "source_truth_copied_into_shape": False,
            "platform_view_count_invariant_claimed": False,
        },
    )


def load_product_dino_v1(
    *,
    source_dir: str,
    weight_path: str,
    device: torch.device | str = "cpu",
):
    if not str(source_dir).strip() or not str(weight_path).strip():
        raise RuntimeError("PRODUCT_INPUT_V1_DINO_PATHS_REQUIRED")
    return load_exact_dinov2_s_v2(
        source_dir,
        weight_path,
        device=device,
        authority=DINOv2FoundationAuthorityV2(),
    )
