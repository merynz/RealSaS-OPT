from __future__ import annotations

"""Typed source-owned visual runtime projection authority.

This is the Stage42 projection contract for source-owned 2D presentation
geometry. It is intentionally separate from RuntimeProjectionIR.v2, whose
geometry contract is the canonical mechanical mesh.

The IR transports already-qualified Stage37 visual topology plus compiler-
evaluated per-frame 2D positions. It does not authorize runtime triangulation,
appearance generation, donor search, model fitting, or visual-mesh rebuilding.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]

SOURCE_OWNED_VISUAL_RUNTIME_PROJECTION_SCHEMA = (
    "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"
)


@dataclass(frozen=True)
class SourceOwnedVisualRuntimeViewV1IR:
    view_index: int
    view_id: str
    camera: Json
    source_width: int
    source_height: int
    texture_path: str
    texture_sha256: str
    visual_mesh_npz_path: str
    visual_mesh_npz_sha256: str
    visual_mesh_hash: str
    visual_vertex_count: int
    visual_face_count: int
    schema_version: str = "RealSaS.SourceOwnedVisualRuntimeViewIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class SourceOwnedVisualRuntimeClipV1IR:
    clip_id: str
    duration_seconds: float
    loop: bool
    frame_count: int
    array_prefix: str
    schema_version: str = "RealSaS.SourceOwnedVisualRuntimeClipIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class SourceOwnedVisualRuntimeProjectionV1IR:
    complete_puppet_binding_hash: str
    mechanical_state_binding_hash: str
    mechanical_mesh_binding_hash: str
    qualified_visual_presentation_binding_hash: str
    dynamic_motion_binding_hash: str
    appearance_asset_binding_hash: str
    appearance_qualification_binding_hash: str
    camera_set_binding_hash: str
    visual_deformation_operator_id: str
    visual_deformation_policy_hash: str
    projection_npz_path: str
    projection_npz_sha256: str
    views: tuple[SourceOwnedVisualRuntimeViewV1IR, ...]
    clips: tuple[SourceOwnedVisualRuntimeClipV1IR, ...]
    projection_hash: str
    schema_version: str = SOURCE_OWNED_VISUAL_RUNTIME_PROJECTION_SCHEMA
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


def source_owned_visual_runtime_projection_hash(
    value: SourceOwnedVisualRuntimeProjectionV1IR,
) -> str:
    payload = value.to_dict()
    payload.pop("projection_hash", None)
    return content_sha256(payload)


def _require_hash(value: str, label: str) -> None:
    if len(str(value)) != 64:
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_HASH_INVALID:" + str(label)
        )


def validate_source_owned_visual_runtime_projection(
    value: SourceOwnedVisualRuntimeProjectionV1IR,
) -> None:
    if value.schema_version != SOURCE_OWNED_VISUAL_RUNTIME_PROJECTION_SCHEMA:
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_PROJECTION_SCHEMA_DRIFT"
        )
    views = tuple(sorted(value.views, key=lambda row: int(row.view_index)))
    if (
        len(views) != 8
        or tuple(int(row.view_index) for row in views) != tuple(range(8))
        or tuple(str(row.view_id) for row in views)
        != tuple(f"V{i}" for i in range(8))
    ):
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_REQUIRES_EXACT_V0_V7"
        )
    if not value.clips:
        raise QualificationError("SOURCE_VISUAL_RUNTIME_CLIP_SET_EMPTY")
    if len({str(row.clip_id) for row in value.clips}) != len(value.clips):
        raise QualificationError("SOURCE_VISUAL_RUNTIME_CLIP_ID_DUPLICATE")
    if any(int(row.frame_count) <= 0 for row in value.clips):
        raise QualificationError("SOURCE_VISUAL_RUNTIME_CLIP_FRAME_EMPTY")
    if any(
        int(row.source_width) <= 0
        or int(row.source_height) <= 0
        or int(row.visual_vertex_count) < 3
        or int(row.visual_face_count) < 1
        for row in views
    ):
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_VIEW_CARDINALITY_INVALID"
        )
    for label, digest in (
        ("COMPLETE_PUPPET", value.complete_puppet_binding_hash),
        ("MECHANICAL_STATE", value.mechanical_state_binding_hash),
        ("MECHANICAL_MESH", value.mechanical_mesh_binding_hash),
        (
            "QUALIFIED_VISUAL_PRESENTATION",
            value.qualified_visual_presentation_binding_hash,
        ),
        ("DYNAMIC_MOTION", value.dynamic_motion_binding_hash),
        ("APPEARANCE_ASSET", value.appearance_asset_binding_hash),
        (
            "APPEARANCE_QUALIFICATION",
            value.appearance_qualification_binding_hash,
        ),
        ("CAMERA_SET", value.camera_set_binding_hash),
        ("DEFORMATION_POLICY", value.visual_deformation_policy_hash),
        ("PROJECTION_NPZ", value.projection_npz_sha256),
        ("PROJECTION", value.projection_hash),
    ):
        _require_hash(str(digest), label)
    for row in views:
        for label, digest in (
            ("TEXTURE", row.texture_sha256),
            ("VISUAL_MESH_NPZ", row.visual_mesh_npz_sha256),
            ("VISUAL_MESH", row.visual_mesh_hash),
        ):
            _require_hash(str(digest), f"V{row.view_index}:{label}")
    if not str(value.visual_deformation_operator_id):
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_DEFORMATION_OPERATOR_MISSING"
        )
    if (
        value.projection_hash
        != source_owned_visual_runtime_projection_hash(value)
    ):
        raise QualificationError(
            "SOURCE_VISUAL_RUNTIME_PROJECTION_HASH_DRIFT"
        )


def source_owned_visual_runtime_projection_from_dict(
    payload: Mapping[str, Any],
) -> SourceOwnedVisualRuntimeProjectionV1IR:
    views = tuple(
        SourceOwnedVisualRuntimeViewV1IR(
            view_index=int(row["view_index"]),
            view_id=str(row["view_id"]),
            camera=dict(row["camera"]),
            source_width=int(row["source_width"]),
            source_height=int(row["source_height"]),
            texture_path=str(row["texture_path"]),
            texture_sha256=str(row["texture_sha256"]),
            visual_mesh_npz_path=str(row["visual_mesh_npz_path"]),
            visual_mesh_npz_sha256=str(row["visual_mesh_npz_sha256"]),
            visual_mesh_hash=str(row["visual_mesh_hash"]),
            visual_vertex_count=int(row["visual_vertex_count"]),
            visual_face_count=int(row["visual_face_count"]),
            schema_version=str(
                row.get("schema_version")
                or "RealSaS.SourceOwnedVisualRuntimeViewIR.v1"
            ),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("views") or ()
    )
    clips = tuple(
        SourceOwnedVisualRuntimeClipV1IR(
            clip_id=str(row["clip_id"]),
            duration_seconds=float(row["duration_seconds"]),
            loop=bool(row["loop"]),
            frame_count=int(row["frame_count"]),
            array_prefix=str(row["array_prefix"]),
            schema_version=str(
                row.get("schema_version")
                or "RealSaS.SourceOwnedVisualRuntimeClipIR.v1"
            ),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("clips") or ()
    )
    value = SourceOwnedVisualRuntimeProjectionV1IR(
        complete_puppet_binding_hash=str(
            payload["complete_puppet_binding_hash"]
        ),
        mechanical_state_binding_hash=str(
            payload["mechanical_state_binding_hash"]
        ),
        mechanical_mesh_binding_hash=str(
            payload["mechanical_mesh_binding_hash"]
        ),
        qualified_visual_presentation_binding_hash=str(
            payload["qualified_visual_presentation_binding_hash"]
        ),
        dynamic_motion_binding_hash=str(
            payload["dynamic_motion_binding_hash"]
        ),
        appearance_asset_binding_hash=str(
            payload["appearance_asset_binding_hash"]
        ),
        appearance_qualification_binding_hash=str(
            payload["appearance_qualification_binding_hash"]
        ),
        camera_set_binding_hash=str(payload["camera_set_binding_hash"]),
        visual_deformation_operator_id=str(
            payload["visual_deformation_operator_id"]
        ),
        visual_deformation_policy_hash=str(
            payload["visual_deformation_policy_hash"]
        ),
        projection_npz_path=str(payload["projection_npz_path"]),
        projection_npz_sha256=str(payload["projection_npz_sha256"]),
        views=views,
        clips=clips,
        projection_hash=str(payload["projection_hash"]),
        schema_version=str(
            payload.get("schema_version")
            or SOURCE_OWNED_VISUAL_RUNTIME_PROJECTION_SCHEMA
        ),
        metadata=dict(payload.get("metadata") or {}),
    )
    validate_source_owned_visual_runtime_projection(value)
    return value
