from __future__ import annotations

import math

from .hashing import content_sha256
from .types import QualificationError
from .v4 import build_appearance_binding
from .v4_types import AppearanceCornerBinding


_PIXEL_CENTER_COORDINATE_SYSTEM = "PIXEL_CENTER_XY"


def _node_raster(node, view_index: int):
    if int(view_index) not in set(map(int, node.support_views)):
        return None
    vals = [tuple(map(float, xy)) for v, xy in node.raster_bindings if int(v) == int(view_index)]
    return vals[0] if len(vals) == 1 else None


def _common_donor_view(surface_nodes, target_view_index: int):
    candidates = []
    for v in range(8):
        if all(_node_raster(n, v) is not None for n in surface_nodes):
            candidates.append(v)
    if int(target_view_index) in candidates:
        return int(target_view_index)
    return min(candidates) if candidates else None


def _pixel_center_contract(surface) -> int:
    metadata = dict(getattr(surface, "metadata", {}) or {})
    if metadata.get("raster_coordinate_system") != _PIXEL_CENTER_COORDINATE_SYSTEM:
        raise QualificationError("APPEARANCE_REQUIRES_PIXEL_CENTER_XY")
    try:
        resolution = int(metadata.get("resolution", 0))
    except (TypeError, ValueError):
        resolution = 0
    if resolution <= 0:
        raise QualificationError("APPEARANCE_REQUIRES_NATIVE_RASTER_RESOLUTION")
    return resolution


def _pixel_center_to_uv(xy, resolution: int):
    x, y = map(float, xy)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise QualificationError("APPEARANCE_NONFINITE_DONOR_RASTER")
    lo = -0.5
    hi = float(resolution) - 0.5
    eps = 1e-9
    if x < lo - eps or x > hi + eps or y < lo - eps or y > hi + eps:
        raise QualificationError("APPEARANCE_DONOR_RASTER_OUT_OF_NATIVE_FRAME")
    u = (x + 0.5) / float(resolution)
    v = 1.0 - (y + 0.5) / float(resolution)
    return (u, v)


def build_observed_appearance_binding(
    *,
    surface,
    mesh,
    target_view_index: int,
    camera_binding_hash: str,
    observation_hash_by_view: dict[int, str],
    atlas_payload_hash: str = "",
):
    if mesh.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("APPEARANCE_SURFACE_LINEAGE_MISMATCH")
    if mesh.camera_binding_hash != camera_binding_hash:
        raise QualificationError("APPEARANCE_CAMERA_LINEAGE_MISMATCH")
    resolution = _pixel_center_contract(surface)

    nodes = {n.surface_id: n for n in surface.surface_nodes}
    vertices = {v.canonical_mesh_vertex_id: v for v in mesh.vertices}
    corners = []
    unknown = []
    for fi, face in enumerate(mesh.faces):
        face_support = []
        face_nodes = []
        for ci, vid in enumerate(face):
            if vid not in vertices:
                raise QualificationError("APPEARANCE_UNKNOWN_MESH_VERTEX")
            support = []
            for sid, coeff in vertices[vid].support_binding.coefficients:
                if sid not in nodes:
                    raise QualificationError("APPEARANCE_UNKNOWN_SURFACE_SUPPORT")
                support.append((nodes[sid], float(coeff)))
            face_support.append((ci, support))
            face_nodes.extend(n for n, _ in support)

        # A raster triangle has one texture/source binding.  Per-corner UVs may differ,
        # but its donor observation may not.  Choose a donor that observes every
        # support node used by every corner of the face, preferring the target view.
        donor = _common_donor_view(face_nodes, int(target_view_index))
        if donor is None:
            for ci, support in face_support:
                unknown.append((fi, ci, tuple(n.surface_id for n, _ in support)))
            continue
        if donor not in observation_hash_by_view or not observation_hash_by_view[donor]:
            raise QualificationError("APPEARANCE_MISSING_DONOR_OBSERVATION_HASH")

        authority = "OBSERVED_LOCAL" if donor == int(target_view_index) else "OBSERVED_CROSS_VIEW"
        for ci, support in face_support:
            gx = gy = 0.0
            for node, coeff in support:
                xy = _node_raster(node, donor)
                gx += coeff * xy[0]
                gy += coeff * xy[1]
            donor_xy = (gx, gy)
            material_uv = _pixel_center_to_uv(donor_xy, resolution)
            source_hash = content_sha256(
                {
                    "observation": observation_hash_by_view[donor],
                    "donor_view": donor,
                    "surface_support": tuple((n.surface_id, c) for n, c in support),
                    "donor_raster_xy": donor_xy,
                    "raster_coordinate_system": _PIXEL_CENTER_COORDINATE_SYSTEM,
                    "resolution": resolution,
                }
            )
            corners.append(
                AppearanceCornerBinding(
                    fi,
                    ci,
                    material_uv,
                    donor,
                    donor_xy,
                    source_hash,
                    authority,
                    "",
                    1.0,
                )
            )
    if unknown:
        raise QualificationError(f"APPEARANCE_UNKNOWN_CORNERS:{unknown[:8]}")
    return build_appearance_binding(
        target_view_index=int(target_view_index),
        mesh_binding_hash=mesh.mesh_lineage_hash,
        camera_binding_hash=str(camera_binding_hash),
        corner_bindings=tuple(corners),
        atlas_payload_hash=str(atlas_payload_hash),
        metadata={
            "authority": "OBSERVATION_ONLY",
            "camera_refit": False,
            "source_mesh_uv_used": False,
            "unknown_completion_used": False,
            "raster_coordinate_system": _PIXEL_CENTER_COORDINATE_SYSTEM,
            "raster_resolution": resolution,
            "material_uv_convention": "NATIVE_PIXEL_CENTER_TO_NORMALIZED_UV_V1",
        },
    )
