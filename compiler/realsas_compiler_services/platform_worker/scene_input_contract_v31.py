from __future__ import annotations

"""Portable Input->Shape artifact contract for IRIS V3.1 model-ready evidence."""

import io

import numpy as np
import torch

from models.iris.v31.inference_input_v31 import IRISV31SceneInputs


SHAPE_INPUT_ARTIFACT_TYPE_V31 = "RealSaS.ShapeInput"
SHAPE_INPUT_SCHEMA_V31 = "RealSaS.ShapeInput.v3_1"


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)


def encode_shape_input_v31(inputs: IRISV31SceneInputs) -> bytes:
    inputs.validate()
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        schema=np.asarray([SHAPE_INPUT_SCHEMA_V31]),
        foundation_authority_sha256=np.asarray([inputs.foundation_authority_sha256]),
        alpha_resampling_policy_sha256=np.asarray([inputs.alpha_resampling_policy_sha256]),
        final=_numpy(inputs.final),
        layer2=_numpy(inputs.layer2),
        alpha=_numpy(inputs.alpha),
        camera_features=_numpy(inputs.camera_features),
    )
    return stream.getvalue()


def decode_shape_input_v31(payload: bytes) -> IRISV31SceneInputs:
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise RuntimeError("SHAPE_INPUT_V31_BYTES_REQUIRED")
    try:
        with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as bundle:
            required = {
                "schema",
                "foundation_authority_sha256",
                "alpha_resampling_policy_sha256",
                "final",
                "layer2",
                "alpha",
                "camera_features",
            }
            if set(bundle.files) != required:
                raise RuntimeError("SHAPE_INPUT_V31_FIELDS_DRIFT")
            if str(bundle["schema"].reshape(-1)[0]) != SHAPE_INPUT_SCHEMA_V31:
                raise RuntimeError("SHAPE_INPUT_V31_SCHEMA_DRIFT")
            value = IRISV31SceneInputs(
                final=torch.from_numpy(np.asarray(bundle["final"], dtype=np.float32).copy()),
                layer2=torch.from_numpy(np.asarray(bundle["layer2"], dtype=np.float32).copy()),
                alpha=torch.from_numpy(np.asarray(bundle["alpha"], dtype=np.float32).copy()),
                camera_features=torch.from_numpy(
                    np.asarray(bundle["camera_features"], dtype=np.float32).copy()
                ),
                foundation_authority_sha256=str(
                    bundle["foundation_authority_sha256"].reshape(-1)[0]
                ),
                alpha_resampling_policy_sha256=str(
                    bundle["alpha_resampling_policy_sha256"].reshape(-1)[0]
                ),
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("SHAPE_INPUT_V31_DECODE_FAILED") from exc
    value.validate()
    return value
