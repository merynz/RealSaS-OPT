from __future__ import annotations

"""Deterministic observation-pixel atlas binding without appearance reconstruction.

The source appearance remains the authority for donor pixels. This module only moves
those donor coordinates into a declared atlas panel and recomputes normalized material
UVs; it never selects a new donor, changes topology, or synthesizes pixels.
"""

from math import isfinite

from .types import QualificationError
from .v4 import build_appearance_binding, validate_appearance_binding
from .v4_types import AppearanceCornerBinding


def _required_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise QualificationError(f"APPEARANCE_ATLAS_INVALID_{name}")
    return value


def _atlas_uv(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    if not (isfinite(x) and isfinite(y)):
        raise QualificationError("APPEARANCE_ATLAS_NONFINITE_COORDINATE")
    if x < -1e-9 or x > float(width - 1) + 1e-9 or y < -1e-9 or y > float(height - 1) + 1e-9:
        raise QualificationError("APPEARANCE_ATLAS_COORDINATE_OUTSIDE_TEXTURE")
    return ((x + 0.5) / float(width), 1.0 - (y + 0.5) / float(height))


def rebind_local_observed_appearance_to_atlas_panel(
    source_appearance,
    *,
    source_width: int,
    source_height: int,
    atlas_width: int,
    atlas_height: int,
    panel_x_offset: int,
    atlas_payload_hash: str,
):
    validate_appearance_binding(source_appearance)
    source_width = _required_positive("SOURCE_WIDTH", source_width)
    source_height = _required_positive("SOURCE_HEIGHT", source_height)
    atlas_width = _required_positive("ATLAS_WIDTH", atlas_width)
    atlas_height = _required_positive("ATLAS_HEIGHT", atlas_height)
    panel_x_offset = int(panel_x_offset)
    if panel_x_offset < 0 or panel_x_offset + source_width > atlas_width or source_height > atlas_height:
        raise QualificationError("APPEARANCE_ATLAS_PANEL_OUTSIDE_TEXTURE")
    if not str(atlas_payload_hash or "").strip():
        raise QualificationError("APPEARANCE_ATLAS_PAYLOAD_HASH_REQUIRED")

    target = int(source_appearance.target_view_index)
    corners = []
    for corner in source_appearance.corner_bindings:
        if corner.authority_class != "OBSERVED_LOCAL" or int(corner.donor_view_index) != target:
            raise QualificationError("APPEARANCE_ATLAS_REBIND_REQUIRES_LOCAL_OBSERVED_DONOR")
        x, y = map(float, corner.donor_raster_xy)
        if x < -1e-9 or x > float(source_width - 1) + 1e-9 or y < -1e-9 or y > float(source_height - 1) + 1e-9:
            raise QualificationError("APPEARANCE_ATLAS_SOURCE_DONOR_OUTSIDE_PANEL")
        ax, ay = float(panel_x_offset) + x, y
        corners.append(
            AppearanceCornerBinding(
                int(corner.face_index),
                int(corner.corner_index),
                _atlas_uv(ax, ay, atlas_width, atlas_height),
                target,
                (ax, ay),
                str(corner.source_observation_hash),
                "OBSERVED_LOCAL",
                str(corner.completion_id),
                float(corner.confidence),
            )
        )
    return build_appearance_binding(
        target_view_index=target,
        mesh_binding_hash=str(source_appearance.mesh_binding_hash),
        camera_binding_hash=str(source_appearance.camera_binding_hash),
        corner_bindings=tuple(corners),
        atlas_payload_hash=str(atlas_payload_hash),
        metadata={
            **dict(source_appearance.metadata or {}),
            "authority": "OBSERVATION_ONLY_ATLAS_REBIND",
            "source_appearance_lineage_hash": str(source_appearance.appearance_lineage_hash),
            "source_donor_selection_changed": False,
            "new_pixels_generated": False,
            "atlas_panel_x_offset": panel_x_offset,
            "atlas_panel_width": source_width,
            "atlas_panel_height": source_height,
            "atlas_width": atlas_width,
            "atlas_height": atlas_height,
            "material_uv_convention": "NATIVE_PIXEL_CENTER_TO_NORMALIZED_UV_V1",
        },
    )


def build_sprite_panel_appearance(
    mesh,
    *,
    target_view_index: int,
    source_observation_hash: str,
    source_width: int,
    source_height: int,
    atlas_width: int,
    atlas_height: int,
    panel_x_offset: int,
    atlas_payload_hash: str,
):
    source_width = _required_positive("SOURCE_WIDTH", source_width)
    source_height = _required_positive("SOURCE_HEIGHT", source_height)
    atlas_width = _required_positive("ATLAS_WIDTH", atlas_width)
    atlas_height = _required_positive("ATLAS_HEIGHT", atlas_height)
    panel_x_offset = int(panel_x_offset)
    if int(mesh.view_index) != int(target_view_index):
        raise QualificationError("APPEARANCE_ATLAS_SPRITE_VIEW_MISMATCH")
    if panel_x_offset < 0 or panel_x_offset + source_width > atlas_width or source_height > atlas_height:
        raise QualificationError("APPEARANCE_ATLAS_PANEL_OUTSIDE_TEXTURE")
    if not str(source_observation_hash or "").strip() or not str(atlas_payload_hash or "").strip():
        raise QualificationError("APPEARANCE_ATLAS_SPRITE_HASH_REQUIRED")

    vertices = {str(v.canonical_mesh_vertex_id): v for v in mesh.vertices}
    corners = []
    for face_index, face in enumerate(mesh.faces):
        for corner_index, vertex_id in enumerate(face):
            vertex = vertices.get(str(vertex_id))
            if vertex is None:
                raise QualificationError("APPEARANCE_ATLAS_SPRITE_UNKNOWN_VERTEX")
            xy = tuple(map(float, vertex.metadata.get("source_raster_xy", vertex.metadata.get("raster_xy", ()))))
            if len(xy) != 2:
                raise QualificationError("APPEARANCE_ATLAS_SPRITE_RASTER_WITNESS_REQUIRED")
            x, y = xy
            if x < -1e-9 or x > float(source_width - 1) + 1e-9 or y < -1e-9 or y > float(source_height - 1) + 1e-9:
                raise QualificationError("APPEARANCE_ATLAS_SPRITE_SOURCE_COORDINATE_OUTSIDE_PANEL")
            ax, ay = float(panel_x_offset) + x, y
            corners.append(
                AppearanceCornerBinding(
                    face_index,
                    corner_index,
                    _atlas_uv(ax, ay, atlas_width, atlas_height),
                    int(target_view_index),
                    (ax, ay),
                    str(source_observation_hash),
                    "OBSERVED_LOCAL",
                    "",
                    1.0,
                )
            )
    return build_appearance_binding(
        target_view_index=int(target_view_index),
        mesh_binding_hash=str(mesh.mesh_lineage_hash),
        camera_binding_hash=str(mesh.camera_binding_hash),
        corner_bindings=tuple(corners),
        atlas_payload_hash=str(atlas_payload_hash),
        metadata={
            "authority": "EXACT_SOURCE_OWNER_MASK_ATLAS_PANEL",
            "new_pixels_generated": False,
            "source_donor_selection_changed": False,
            "atlas_panel_x_offset": panel_x_offset,
            "atlas_panel_width": source_width,
            "atlas_panel_height": source_height,
            "atlas_width": atlas_width,
            "atlas_height": atlas_height,
            "material_uv_convention": "NATIVE_PIXEL_CENTER_TO_NORMALIZED_UV_V1",
        },
    )


__all__ = [
    "rebind_local_observed_appearance_to_atlas_panel",
    "build_sprite_panel_appearance",
]
