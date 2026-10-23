from __future__ import annotations

"""Model-independent codec for immutable SourceObservation.v1 bytes.

Input may lower these bytes into learned-model tensors; Mesh may derive admitted
foreground materialization from the same source. Neither consumer owns source
truth. This module intentionally imports no Torch/model package.
"""

from dataclasses import dataclass
import hashlib
import io
import json
from typing import Any, Mapping

import numpy as np


SOURCE_OBSERVATION_SCHEMA_V1 = "RealSaS.SourceObservation.v1"


@dataclass(frozen=True)
class SourceObservationArraysV1:
    native_rgba: np.ndarray
    camera_bundle: dict[str, Any]
    schema: str = SOURCE_OBSERVATION_SCHEMA_V1

    def validate(self) -> None:
        rgba = np.asarray(self.native_rgba)
        if rgba.shape != (1, 8, 4, 1024, 1024):
            raise RuntimeError("SOURCE_OBSERVATION_V1_RGBA_SHAPE_INVALID")
        if rgba.dtype != np.uint8:
            raise RuntimeError("SOURCE_OBSERVATION_V1_RGBA_DTYPE_INVALID")
        camera = dict(self.camera_bundle)
        if camera.get("schema") != "RealSaS.CameraProjectionBundle.v1":
            raise RuntimeError("SOURCE_OBSERVATION_V1_CAMERA_SCHEMA_INVALID")
        if camera.get("per_view_recenter", False) is not False:
            raise RuntimeError("SOURCE_OBSERVATION_V1_PER_VIEW_RECENTER_FORBIDDEN")
        if camera.get("per_view_pose_change", False) is not False:
            raise RuntimeError("SOURCE_OBSERVATION_V1_PER_VIEW_POSE_CHANGE_FORBIDDEN")
        order = tuple(map(str, camera.get("view_order") or ()))
        if len(order) != 8 or len(set(order)) != 8:
            raise RuntimeError("SOURCE_OBSERVATION_V1_VIEW_ORDER_INVALID")


def encode_source_observation_arrays_v1(
    *, native_rgba: np.ndarray, camera_bundle: Mapping[str, Any]
) -> bytes:
    value = SourceObservationArraysV1(
        native_rgba=np.asarray(native_rgba, dtype=np.uint8),
        camera_bundle=dict(camera_bundle),
    )
    value.validate()
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        schema=np.asarray([SOURCE_OBSERVATION_SCHEMA_V1]),
        rgba=value.native_rgba,
        camera_json=np.asarray(
            [json.dumps(value.camera_bundle, sort_keys=True, separators=(",", ":"))]
        ),
    )
    return stream.getvalue()


def decode_source_observation_arrays_v1(payload: bytes) -> SourceObservationArraysV1:
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise RuntimeError("SOURCE_OBSERVATION_V1_BYTES_REQUIRED")
    try:
        with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as bundle:
            if set(bundle.files) != {"schema", "rgba", "camera_json"}:
                raise RuntimeError("SOURCE_OBSERVATION_V1_FIELDS_DRIFT")
            schema = str(bundle["schema"].reshape(-1)[0])
            if schema != SOURCE_OBSERVATION_SCHEMA_V1:
                raise RuntimeError("SOURCE_OBSERVATION_V1_SCHEMA_DRIFT")
            rgba = np.asarray(bundle["rgba"])
            if rgba.dtype != np.uint8:
                raise RuntimeError("SOURCE_OBSERVATION_V1_RGBA_DTYPE_DRIFT")
            camera = json.loads(str(bundle["camera_json"].reshape(-1)[0]))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("SOURCE_OBSERVATION_V1_DECODE_FAILED") from exc
    value = SourceObservationArraysV1(
        native_rgba=rgba.copy(), camera_bundle=dict(camera), schema=schema
    )
    value.validate()
    return value


def derive_foreground_masks_v1(
    source: SourceObservationArraysV1,
    *,
    expected_sha256_by_view: Mapping[int, str],
) -> dict[int, bytes]:
    """Materialize admission-equivalent alpha>0 masks and bind to Stage07 SHA."""
    source.validate()
    expected = {int(key): str(value) for key, value in expected_sha256_by_view.items()}
    if set(expected) != set(range(8)):
        raise RuntimeError("SOURCE_OBSERVATION_V1_FOREGROUND_AUTHORITY_INCOMPLETE")
    output: dict[int, bytes] = {}
    rgba = np.asarray(source.native_rgba, dtype=np.uint8)
    for view_index in range(8):
        alpha = rgba[0, view_index, 3]
        raw = (alpha > 0).astype(np.uint8, copy=False).tobytes(order="C")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected[view_index]:
            raise RuntimeError(
                f"SOURCE_OBSERVATION_V1_FOREGROUND_AUTHORITY_DRIFT:{view_index}"
            )
        output[view_index] = raw
    return output


__all__ = [
    "SOURCE_OBSERVATION_SCHEMA_V1",
    "SourceObservationArraysV1",
    "decode_source_observation_arrays_v1",
    "derive_foreground_masks_v1",
    "encode_source_observation_arrays_v1",
]
