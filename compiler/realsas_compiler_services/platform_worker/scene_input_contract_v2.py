from __future__ import annotations

"""CAS-native SourceObservation -> model-ready ShapeInput evidence boundary.

V2 changes only transport/provenance, not the frozen IRIS scene feature ABI.
Input owns the source CAS digest; Shape receives it alongside DINO features.
The digest is an identity binding, NOT rights or camera/source qualification.
"""

from dataclasses import dataclass
import io
from zipfile import ZipFile

import numpy as np
import torch

from models.iris.vnext.inference_input_v1 import IRISVNextSceneInputsV1


SHAPE_INPUT_ARTIFACT_TYPE_V2 = "RealSaS.ShapeInput"
SHAPE_INPUT_SCHEMA_V2 = "RealSaS.ShapeInput.v2"
_MAX_SHAPE_INPUT_BYTES_V2 = 256 * 1024 * 1024
_REQUIRED_FIELDS_V2 = frozenset({
    "schema", "foundation_authority_sha256",
    "source_observation_content_sha256",
    "image_tokens", "token_xy", "camera_features",
})


def _require_sha256(value: str) -> str:
    if (
        type(value) is not str or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise RuntimeError("SHAPE_INPUT_V2_SOURCE_CAS_ID_INVALID")
    return value


@dataclass(frozen=True)
class SourceBoundShapeInputV2:
    inputs: IRISVNextSceneInputsV1
    source_observation_content_sha256: str
    schema: str = SHAPE_INPUT_SCHEMA_V2

    def validate(self) -> None:
        if self.schema != SHAPE_INPUT_SCHEMA_V2:
            raise RuntimeError("SHAPE_INPUT_V2_SCHEMA_DRIFT")
        _require_sha256(self.source_observation_content_sha256)
        self.inputs.validate()


def encode_shape_input_v2(
    inputs: IRISVNextSceneInputsV1, *,
    source_observation_content_sha256: str,
) -> bytes:
    value = SourceBoundShapeInputV2(
        inputs=inputs,
        source_observation_content_sha256=source_observation_content_sha256,
    )
    value.validate()
    def arr(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        schema=np.asarray([SHAPE_INPUT_SCHEMA_V2]),
        foundation_authority_sha256=np.asarray([inputs.foundation_authority_sha256]),
        source_observation_content_sha256=np.asarray([source_observation_content_sha256]),
        image_tokens=arr(inputs.image_tokens),
        token_xy=arr(inputs.token_xy),
        camera_features=arr(inputs.camera_features),
    )
    return stream.getvalue()


def decode_shape_input_v2(payload: bytes) -> SourceBoundShapeInputV2:
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise RuntimeError("SHAPE_INPUT_V2_BYTES_REQUIRED")
    raw = bytes(payload)
    if len(raw) > _MAX_SHAPE_INPUT_BYTES_V2:
        raise RuntimeError("SHAPE_INPUT_V2_RESOURCE_LIMIT")
    try:
        with ZipFile(io.BytesIO(raw)) as zf:
            entries = zf.infolist()
            if (
                len(entries) != len(_REQUIRED_FIELDS_V2)
                or {e.filename for e in entries} != {name + ".npy" for name in _REQUIRED_FIELDS_V2}
                or sum(e.file_size for e in entries) > _MAX_SHAPE_INPUT_BYTES_V2
                or any(e.file_size > _MAX_SHAPE_INPUT_BYTES_V2 for e in entries)
            ):
                raise RuntimeError("SHAPE_INPUT_V2_ARCHIVE_FIELDS_OR_RESOURCE_LIMIT")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != _REQUIRED_FIELDS_V2:
                raise RuntimeError("SHAPE_INPUT_V2_FIELDS_DRIFT")
            if str(archive["schema"].reshape(-1)[0]) != SHAPE_INPUT_SCHEMA_V2:
                raise RuntimeError("SHAPE_INPUT_V2_SCHEMA_DRIFT")
            arrays = {
                name: np.asarray(archive[name])
                for name in ("image_tokens", "token_xy", "camera_features")
            }
            if any(a.dtype != np.float32 for a in arrays.values()):
                raise RuntimeError("SHAPE_INPUT_V2_TENSOR_DTYPE_DRIFT")
            model_input = IRISVNextSceneInputsV1(
                image_tokens=torch.from_numpy(arrays["image_tokens"].copy()),
                token_xy=torch.from_numpy(arrays["token_xy"].copy()),
                camera_features=torch.from_numpy(arrays["camera_features"].copy()),
                foundation_authority_sha256=str(archive["foundation_authority_sha256"].reshape(-1)[0]),
            )
            value = SourceBoundShapeInputV2(
                inputs=model_input,
                source_observation_content_sha256=str(
                    archive["source_observation_content_sha256"].reshape(-1)[0]
                ),
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("SHAPE_INPUT_V2_DECODE_FAILED") from exc
    value.validate()
    return value


__all__ = [
    "SHAPE_INPUT_ARTIFACT_TYPE_V2", "SHAPE_INPUT_SCHEMA_V2",
    "SourceBoundShapeInputV2", "encode_shape_input_v2", "decode_shape_input_v2",
]
