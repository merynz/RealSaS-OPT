from __future__ import annotations

"""Versioned transport contract between product Input and Shape capabilities.

This module owns only the portable artifact boundary. It intentionally does not
own source-view cardinality policy, DINO execution, IRIS inference, TESSA
inference, or Compiler authority. A future Input/IRIS revision may introduce a
new schema without changing Platform capability identity.
"""

import io

import numpy as np
import torch

from models.iris.vnext.inference_input_v1 import IRISVNextSceneInputsV1


SHAPE_INPUT_ARTIFACT_TYPE_V1 = "RealSaS.ShapeInput"
SHAPE_INPUT_SCHEMA_V1 = "RealSaS.ShapeInput.v1"


def _tensor_numpy(value: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    out = value.detach().cpu().contiguous().numpy()
    return out if dtype is None else out.astype(dtype, copy=False)


def encode_shape_input_v1(inputs: IRISVNextSceneInputsV1) -> bytes:
    """Encode the current model-ready scene input without pickle or path authority."""
    inputs.validate()
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        schema=np.asarray([SHAPE_INPUT_SCHEMA_V1]),
        foundation_authority_sha256=np.asarray([inputs.foundation_authority_sha256]),
        image_tokens=_tensor_numpy(inputs.image_tokens, np.float32),
        token_xy=_tensor_numpy(inputs.token_xy, np.float32),
        camera_features=_tensor_numpy(inputs.camera_features, np.float32),
    )
    return stream.getvalue()


def decode_shape_input_v1(payload: bytes) -> IRISVNextSceneInputsV1:
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise RuntimeError("SHAPE_INPUT_BYTES_REQUIRED")
    try:
        with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as bundle:
            required = {
                "schema",
                "foundation_authority_sha256",
                "image_tokens",
                "token_xy",
                "camera_features",
            }
            if set(bundle.files) != required:
                raise RuntimeError("SHAPE_INPUT_FIELDS_DRIFT")
            schema = str(bundle["schema"].reshape(-1)[0])
            if schema != SHAPE_INPUT_SCHEMA_V1:
                raise RuntimeError("SHAPE_INPUT_SCHEMA_DRIFT")
            authority = str(bundle["foundation_authority_sha256"].reshape(-1)[0])
            value = IRISVNextSceneInputsV1(
                image_tokens=torch.from_numpy(
                    np.asarray(bundle["image_tokens"], dtype=np.float32).copy()
                ),
                token_xy=torch.from_numpy(
                    np.asarray(bundle["token_xy"], dtype=np.float32).copy()
                ),
                camera_features=torch.from_numpy(
                    np.asarray(bundle["camera_features"], dtype=np.float32).copy()
                ),
                foundation_authority_sha256=authority,
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("SHAPE_INPUT_DECODE_FAILED") from exc
    value.validate()
    return value
