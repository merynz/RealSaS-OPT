from __future__ import annotations

"""Qualified source-owned visual presentation topology for RealSaS V2.

This module owns the typed Stage37 result that sits between the Stage18
source-owned visual substrate and Stage42 runtime visual-motion compilation.

It deliberately does NOT choose or qualify a dynamic visual deformation
operator. Stage37 freezes source-preserving per-view presentation topology and
its exact Stage35 mechanical qualification lineage. Stage42 may evaluate a
selected deformation implementation over this topology; Stage45 remains the
shipping visual-quality authority.
"""

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hashing import content_sha256
from .types import QualificationError
from .visual_mesh_arap_v1 import VisualMesh2D, visual_mesh_semantic_hash

Json = dict[str, Any]

QUALIFIED_VISUAL_PRESENTATION_VIEW_SCHEMA = (
    "RealSaS.QualifiedVisualPresentationViewIR.v1"
)
QUALIFIED_VISUAL_PRESENTATION_SET_SCHEMA = (
    "RealSaS.QualifiedVisualPresentationSetIR.v1"
)


@dataclass(frozen=True)
class QualifiedVisualPresentationViewIR:
    view_index: int
    direction_id: str
    width: int
    height: int
    vertex_count: int
    face_count: int
    region_count: int
    mesh_npz_path: str
    mesh_npz_sha256: str
    source_visual_mesh_hash: str
    source_raster_sha256: str
    source_foreground_mask_sha256: str
    visual_mesh_hash: str
    view_hash: str
    schema_version: str = QUALIFIED_VISUAL_PRESENTATION_VIEW_SCHEMA
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedVisualPresentationSetIR:
    source_visual_mesh_set_binding_hash: str
    observation_set_binding_hash: str
    output_direction_set_binding_hash: str
    mechanical_mesh_binding_hash: str
    skin_topology_compatibility_report_hash: str
    appearance_asset_binding_hash: str
    appearance_qualification_binding_hash: str
    views: tuple[QualifiedVisualPresentationViewIR, ...]
    set_hash: str
    schema_version: str = QUALIFIED_VISUAL_PRESENTATION_SET_SCHEMA
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


def _without(mapping: Mapping[str, Any], *keys: str) -> Json:
    out = dict(mapping)
    for key in keys:
        out.pop(key, None)
    return out


def qualified_visual_presentation_view_hash(
    value: QualifiedVisualPresentationViewIR,
) -> str:
    return content_sha256(_without(value.to_dict(), "view_hash"))


def qualified_visual_presentation_set_hash(
    value: QualifiedVisualPresentationSetIR,
) -> str:
    return content_sha256(_without(value.to_dict(), "set_hash"))


def _require_hash(value: str, label: str) -> None:
    if len(str(value)) != 64:
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_HASH_INVALID:" + str(label)
        )


def validate_qualified_visual_presentation_view(
    value: QualifiedVisualPresentationViewIR,
) -> None:
    if (
        int(value.view_index) < 0
        or int(value.width) <= 0
        or int(value.height) <= 0
        or int(value.vertex_count) < 3
        or int(value.face_count) < 1
        or int(value.region_count) < 1
    ):
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_VIEW_CARDINALITY_INVALID"
        )
    if str(value.direction_id) != f"V{int(value.view_index)}":
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_DIRECTION_ID_DRIFT"
        )
    for label, digest in (
        ("MESH_NPZ", value.mesh_npz_sha256),
        ("SOURCE_VISUAL_MESH", value.source_visual_mesh_hash),
        ("SOURCE_RASTER", value.source_raster_sha256),
        ("SOURCE_FOREGROUND", value.source_foreground_mask_sha256),
        ("VISUAL_MESH", value.visual_mesh_hash),
        ("VIEW", value.view_hash),
    ):
        _require_hash(str(digest), label)
    if value.view_hash != qualified_visual_presentation_view_hash(value):
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_VIEW_HASH_DRIFT"
        )


def validate_qualified_visual_presentation_set(
    value: QualifiedVisualPresentationSetIR,
) -> None:
    rows = tuple(sorted(value.views, key=lambda row: int(row.view_index)))
    if (
        len(rows) != 8
        or tuple(int(row.view_index) for row in rows) != tuple(range(8))
    ):
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_REQUIRES_V0_V7"
        )
    for label, digest in (
        ("SOURCE_VISUAL_SET", value.source_visual_mesh_set_binding_hash),
        ("OBSERVATION_SET", value.observation_set_binding_hash),
        ("OUTPUT_DIRECTIONS", value.output_direction_set_binding_hash),
        ("MECHANICAL_MESH", value.mechanical_mesh_binding_hash),
        (
            "SKIN_TOPOLOGY_COMPATIBILITY",
            value.skin_topology_compatibility_report_hash,
        ),
        ("APPEARANCE_ASSET", value.appearance_asset_binding_hash),
        (
            "APPEARANCE_QUALIFICATION",
            value.appearance_qualification_binding_hash,
        ),
        ("SET", value.set_hash),
    ):
        _require_hash(str(digest), label)
    for row in rows:
        validate_qualified_visual_presentation_view(row)
    if value.set_hash != qualified_visual_presentation_set_hash(value):
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_SET_HASH_DRIFT"
        )


def qualified_visual_presentation_set_from_dict(
    payload: Mapping[str, Any],
) -> QualifiedVisualPresentationSetIR:
    views = tuple(
        QualifiedVisualPresentationViewIR(
            view_index=int(row["view_index"]),
            direction_id=str(row["direction_id"]),
            width=int(row["width"]),
            height=int(row["height"]),
            vertex_count=int(row["vertex_count"]),
            face_count=int(row["face_count"]),
            region_count=int(row["region_count"]),
            mesh_npz_path=str(row["mesh_npz_path"]),
            mesh_npz_sha256=str(row["mesh_npz_sha256"]),
            source_visual_mesh_hash=str(row["source_visual_mesh_hash"]),
            source_raster_sha256=str(row["source_raster_sha256"]),
            source_foreground_mask_sha256=str(
                row["source_foreground_mask_sha256"]
            ),
            visual_mesh_hash=str(row["visual_mesh_hash"]),
            view_hash=str(row["view_hash"]),
            schema_version=str(
                row.get("schema_version")
                or QUALIFIED_VISUAL_PRESENTATION_VIEW_SCHEMA
            ),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("views") or ()
    )
    value = QualifiedVisualPresentationSetIR(
        source_visual_mesh_set_binding_hash=str(
            payload["source_visual_mesh_set_binding_hash"]
        ),
        observation_set_binding_hash=str(
            payload["observation_set_binding_hash"]
        ),
        output_direction_set_binding_hash=str(
            payload["output_direction_set_binding_hash"]
        ),
        mechanical_mesh_binding_hash=str(
            payload["mechanical_mesh_binding_hash"]
        ),
        skin_topology_compatibility_report_hash=str(
            payload["skin_topology_compatibility_report_hash"]
        ),
        appearance_asset_binding_hash=str(
            payload["appearance_asset_binding_hash"]
        ),
        appearance_qualification_binding_hash=str(
            payload["appearance_qualification_binding_hash"]
        ),
        views=views,
        set_hash=str(payload["set_hash"]),
        schema_version=str(
            payload.get("schema_version")
            or QUALIFIED_VISUAL_PRESENTATION_SET_SCHEMA
        ),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.schema_version != QUALIFIED_VISUAL_PRESENTATION_SET_SCHEMA:
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_SET_SCHEMA_DRIFT"
        )
    validate_qualified_visual_presentation_set(value)
    return value


def load_qualified_visual_presentation_view(
    value: QualifiedVisualPresentationViewIR,
) -> dict[str, np.ndarray | VisualMesh2D]:
    path = Path(value.mesh_npz_path).expanduser().resolve()
    if not path.is_file():
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_VIEW_NPZ_MISSING"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != value.mesh_npz_sha256:
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_VIEW_NPZ_BYTES_DRIFT"
        )
    with np.load(path, allow_pickle=False) as data:
        required = {
            "positions",
            "faces",
            "uv",
            "vertex_region_id",
            "face_region_id",
            "region_labels",
            "seed_region_labels",
        }
        if not required.issubset(data.files):
            raise QualificationError(
                "QUALIFIED_VISUAL_PRESENTATION_VIEW_ARRAY_MISSING"
            )
        positions = np.asarray(data["positions"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.uint32)
        uv = np.asarray(data["uv"], dtype=np.float64)
        vertex_region_id = np.asarray(
            data["vertex_region_id"], dtype=np.int32
        )
        face_region_id = np.asarray(data["face_region_id"], dtype=np.int32)
        region_labels = np.asarray(data["region_labels"], dtype=np.int32)
        seed_region_labels = np.asarray(
            data["seed_region_labels"], dtype=np.int32
        )
    mesh = VisualMesh2D(
        positions=positions,
        faces=faces,
        uv=uv,
        width=int(value.width),
        height=int(value.height),
    )
    if (
        positions.shape != (int(value.vertex_count), 2)
        or faces.shape != (int(value.face_count), 3)
        or uv.shape != (int(value.vertex_count), 2)
        or vertex_region_id.shape != (int(value.vertex_count),)
        or face_region_id.shape != (int(value.face_count),)
        or region_labels.shape != (int(value.height), int(value.width))
        or seed_region_labels.shape
        != (int(value.height), int(value.width))
        or not np.isfinite(positions).all()
        or not np.isfinite(uv).all()
        or np.any(faces >= int(value.vertex_count))
    ):
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_VIEW_ARRAY_SHAPE_DRIFT"
        )
    if visual_mesh_semantic_hash(mesh) != value.visual_mesh_hash:
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_VIEW_MESH_HASH_DRIFT"
        )
    if np.any(vertex_region_id < 0) or np.any(face_region_id < 0):
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_REGION_ID_INVALID"
        )
    if len(set(map(int, face_region_id.tolist()))) != int(value.region_count):
        raise QualificationError(
            "QUALIFIED_VISUAL_PRESENTATION_REGION_COUNT_DRIFT"
        )
    return {
        "mesh": mesh,
        "vertex_region_id": vertex_region_id,
        "face_region_id": face_region_id,
        "region_labels": region_labels,
        "seed_region_labels": seed_region_labels,
    }
