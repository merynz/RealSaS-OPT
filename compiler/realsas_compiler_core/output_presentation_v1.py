from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .camera_authority_v1 import QualifiedCameraSetIR, validate_qualified_camera_set
from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class OutputPresentationDirectionIR:
    direction_index: int
    direction_id: str
    yaw_degrees: float
    camera_binding_hash: str
    schema_version: str = "RealSaS.OutputPresentationDirectionIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class OutputPresentationDirectionSetIR:
    directions: tuple[OutputPresentationDirectionIR, ...]
    source_camera_set_binding_hash: str
    direction_set_hash: str
    schema_version: str = "RealSaS.OutputPresentationDirectionSetIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def output_direction_set_hash(value: OutputPresentationDirectionSetIR) -> str:
    payload = value.to_dict()
    payload.pop("direction_set_hash", None)
    return content_sha256(payload)


def validate_output_direction_set(value: OutputPresentationDirectionSetIR) -> None:
    rows = tuple(sorted(value.directions, key=lambda x: x.direction_index))
    if len(rows) != 8 or tuple(x.direction_index for x in rows) != tuple(range(8)):
        raise QualificationError("OUTPUT_DIRECTION_SET_REQUIRES_EXACT_8_DIRECTIONS")
    if tuple(x.direction_id for x in rows) != tuple(f"V{i}" for i in range(8)):
        raise QualificationError("OUTPUT_DIRECTION_IDS_MUST_BE_V0_V7")
    if len({x.camera_binding_hash for x in rows}) != 8:
        raise QualificationError("OUTPUT_DIRECTION_CAMERA_BINDINGS_NOT_UNIQUE")
    if value.direction_set_hash != output_direction_set_hash(value):
        raise QualificationError("OUTPUT_DIRECTION_SET_HASH_MISMATCH")


def build_output_direction_set(camera_set: QualifiedCameraSetIR) -> OutputPresentationDirectionSetIR:
    validate_qualified_camera_set(camera_set)
    cameras = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))
    rows = tuple(
        OutputPresentationDirectionIR(
            direction_index=int(c.view_index),
            direction_id=f"V{int(c.view_index)}",
            yaw_degrees=float((int(c.view_index) * 45) % 360),
            camera_binding_hash=str(camera_set.camera_binding_hashes[int(c.view_index)]),
            metadata={"presentation_direction": True, "source_observation_role": False},
        )
        for c in cameras
    )
    value = OutputPresentationDirectionSetIR(
        directions=rows,
        source_camera_set_binding_hash=camera_set.camera_set_hash,
        direction_set_hash="",
        metadata={
            "output_direction_count": 8,
            "source_observation_authority_separate": True,
            "first_knight_v2_reuses_exact_output_camera_geometry": True,
            "future_source_view_count_may_differ": True,
        },
    )
    value = replace(value, direction_set_hash=output_direction_set_hash(value))
    validate_output_direction_set(value)
    return value


def output_direction_set_from_dict(payload: Mapping[str, Any]) -> OutputPresentationDirectionSetIR:
    rows = tuple(
        OutputPresentationDirectionIR(
            direction_index=int(row["direction_index"]),
            direction_id=str(row["direction_id"]),
            yaw_degrees=float(row["yaw_degrees"]),
            camera_binding_hash=str(row["camera_binding_hash"]),
            schema_version=str(row.get("schema_version") or "RealSaS.OutputPresentationDirectionIR.v1"),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("directions") or ()
    )
    value = OutputPresentationDirectionSetIR(
        directions=rows,
        source_camera_set_binding_hash=str(payload["source_camera_set_binding_hash"]),
        direction_set_hash=str(payload["direction_set_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.OutputPresentationDirectionSetIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    validate_output_direction_set(value)
    return value
