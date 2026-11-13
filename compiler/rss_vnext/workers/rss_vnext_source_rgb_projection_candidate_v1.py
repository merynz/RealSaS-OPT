from __future__ import annotations

"""Original-RGB, source-alpha-gated *candidate* addressing on CanonicalMesh.

This is deliberately a weaker claim than authoritative texture ownership:
the camera is fitted from the silhouette and no source depth is observed.
Only pixels strictly inside original alpha, hit by exactly ONE projected face
under that hypothesis, receive a barycentric face candidate. Occluded,
overlapping, boundary, uncovered and opaque-scene pixels stay unassigned.
The output is a deterministic bounded CAS NPZ artifact. No GPU/model work.
"""

from hashlib import sha256
import io
import json
import math
import zlib
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR, canonical_mesh_lineage_hash_v1,
    validate_canonical_mesh_domain_v1,
)
from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    RSSvNextUncalibratedTwoImageSourceV1,
)
from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
    _observed_alpha, _project_fitted, _MAX_FACE_RASTER_VISITS, _MAX_VIEW_SIDE,
)

SCHEMA = "RealSaS.RSSvNext.SourceRGBImagePlaneCandidates.v1"
ARTIFACT_TYPE = "RealSaS.RSSvNextSourceRGBImagePlaneCandidates"
MAX_PER_VIEW = 65536
MAX_ARTIFACT_BYTES = 12 * 1024 * 1024

_KEYS = frozenset({
    "manifest_utf8", "view_index_u8", "pixel_xy_u16",
    "face_index_i32", "barycentric_f32", "source_rgba_u8",
})


def _orient(a: np.ndarray, b: np.ndarray, px, py):
    return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])


def _top_left(a: np.ndarray, b: np.ndarray) -> bool:
    dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
    return dy < 0. or (dy == 0. and dx > 0.)


def _inside_edge(value: np.ndarray, include_boundary: bool) -> np.ndarray:
    return (value > 1.e-12) | ((np.abs(value) <= 1.e-12) & include_boundary)


def _owner_map(
    points: np.ndarray, faces: np.ndarray, *, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    owner = np.full((height, width), -1, dtype=np.int32)
    visits = np.zeros((height, width), dtype=np.uint8)
    bary = np.zeros((height, width, 3), dtype=np.float32)
    total_visits = 0
    for face_index, indices in enumerate(faces):
        a, b, c = (points[int(i)] for i in indices)
        area = float(_orient(a, b, c[0], c[1]))
        if abs(area) <= 1.e-12:
            continue
        x0 = max(0, math.floor(float(min(a[0], b[0], c[0])) - .5))
        x1 = min(width - 1, math.ceil(float(max(a[0], b[0], c[0])) - .5))
        y0 = max(0, math.floor(float(min(a[1], b[1], c[1])) - .5))
        y1 = min(height - 1, math.ceil(float(max(a[1], b[1], c[1])) - .5))
        if x1 < x0 or y1 < y0:
            continue
        total_visits += (x1 - x0 + 1) * (y1 - y0 + 1)
        if total_visits > _MAX_FACE_RASTER_VISITS:
            raise RuntimeError("SOURCE_RGB_RASTER_VISIT_BUDGET")
        xx = np.arange(x0, x1 + 1, dtype=np.float64)[None, :] + .5
        yy = np.arange(y0, y1 + 1, dtype=np.float64)[:, None] + .5
        sign = 1. if area > 0 else -1.
        e0 = _orient(b, c, xx, yy)
        e1 = _orient(c, a, xx, yy)
        e2 = _orient(a, b, xx, yy)
        inside = (
            _inside_edge(sign * e0, _top_left(b, c) if sign > 0 else _top_left(c, b))
            & _inside_edge(sign * e1, _top_left(c, a) if sign > 0 else _top_left(a, c))
            & _inside_edge(sign * e2, _top_left(a, b) if sign > 0 else _top_left(b, a))
        )
        if not inside.any():
            continue
        seen = visits[y0:y1 + 1, x0:x1 + 1]
        slots = owner[y0:y1 + 1, x0:x1 + 1]
        first = inside & (seen == 0)
        repeated = inside & (seen != 0)
        slots[first] = face_index
        slots[repeated] = -2
        seen[inside] = np.minimum(seen[inside].astype(np.uint16) + 1, 2).astype(np.uint8)
        weights = bary[y0:y1 + 1, x0:x1 + 1]
        weights[:, :, 0][first] = (e0 / area)[first]
        weights[:, :, 1][first] = (e1 / area)[first]
        weights[:, :, 2][first] = (e2 / area)[first]
    return owner, bary


def _interior(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inside = np.zeros((h, w), dtype=np.bool_)
    if h <= 2 or w <= 2:
        return inside
    core = np.ones((h - 2, w - 2), dtype=np.bool_)
    for dy in range(3):
        for dx in range(3):
            core &= mask[dy:dy + h - 2, dx:dx + w - 2]
    inside[1:-1, 1:-1] = core
    return inside


def _encode_npz_deterministic(arrays: dict[str, np.ndarray]) -> bytes:
    out = io.BytesIO()
    with ZipFile(out, "w") as archive:
        for name, array in sorted(arrays.items()):
            if name not in _KEYS or array.dtype.hasobject:
                raise RuntimeError("SOURCE_RGB_ARTIFACT_ARRAY_SCHEMA_INVALID")
            buf = io.BytesIO()
            np.lib.format.write_array(buf, array, allow_pickle=False)
            record = ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            record.compress_type = ZIP_DEFLATED
            archive.writestr(record, buf.getvalue(), compress_type=ZIP_DEFLATED, compresslevel=6)
    packed = out.getvalue()
    if len(packed) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("SOURCE_RGB_ARTIFACT_CAS_BUDGET")
    return packed


def build_raw_source_rgb_image_plane_candidates_v1(
    *, source: RSSvNextUncalibratedTwoImageSourceV1,
    mesh: CanonicalMeshIR,
) -> tuple[bytes, dict[str, Any]]:
    """Replay real original pixels; forbid ambiguous face ownership.

    Every colored sample is a *fitted-camera image-plane correspondence*,
    NOT verified surface texture, UV, z-order, native color or product proof.
    """
    source.validate()
    validate_canonical_mesh_domain_v1(mesh)
    if mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh):
        raise RuntimeError("SOURCE_RGB_CANONICAL_MESH_LINEAGE_DRIFT")
    if source.view_claims != ("FRONT", "BACK"):
        raise RuntimeError("SOURCE_RGB_FRONT_BACK_CLAIMS_REQUIRED")
    if len(mesh.faces) > 32768 or len(mesh.vertices) > 32768:
        raise RuntimeError("SOURCE_RGB_CANONICAL_MESH_BUDGET")
    vertices = np.array([v.P for v in mesh.vertices], dtype=np.float64)
    index = {v.canonical_mesh_vertex_id: i for i, v in enumerate(mesh.vertices)}
    faces = np.array([
        [index[vertex_id] for vertex_id in face]
        for face in mesh.faces
    ], dtype=np.int32)
    records = []
    summaries = []
    for view in range(2):
        alpha, error = _observed_alpha(source, view)
        if error is not None:
            raise RuntimeError("SOURCE_RGB_ORIGINAL_ALPHA_UNAVAILABLE:" + error)
        height, width = alpha.shape
        with Image.open(io.BytesIO(source.original_files[view])) as image:
            rgba = image.convert("RGBA")
            if rgba.size != (width, height):
                rgba = rgba.resize((width, height), resample=Image.Resampling.NEAREST)
            original = np.asarray(rgba, dtype=np.uint8)
        projected, camera = _project_fitted(vertices, alpha, view)
        owners, bary = _owner_map(projected, faces, width=width, height=height)
        interior = _interior(alpha) & (original[:, :, 3] == 255)
        eligible = interior & (owners >= 0)
        pixels = np.argwhere(eligible)
        stride = max(1, math.ceil(len(pixels) / MAX_PER_VIEW))
        pixels = pixels[::stride]
        yy = pixels[:, 0]
        xx = pixels[:, 1]
        records.append({
            "view": np.full(len(pixels), view, dtype=np.uint8),
            "xy": np.column_stack((xx, yy)).astype(np.uint16),
            "face": owners[yy, xx].astype(np.int32),
            "bary": bary[yy, xx].astype(np.float32),
            "rgba": original[yy, xx].astype(np.uint8),
        })
        summaries.append({
            "view_index": view,
            "source_file_sha256": sha256(source.original_files[view]).hexdigest(),
            "derived_nearest_rgba_sha256": sha256(original.tobytes(order="C")).hexdigest(),
            "alpha_interior_opaque_pixel_count": int(interior.sum()),
            "fitted_unique_face_pixel_count": int(eligible.sum()),
            "fitted_overlap_ambiguous_pixel_count": int((interior & (owners == -2)).sum()),
            "fitted_uncovered_pixel_count": int((interior & (owners == -1)).sum()),
            "stored_sample_count": len(pixels),
            "deterministic_sample_stride": stride,
            "raster_width": width, "raster_height": height,
            "projection": camera,
            "source_camera_observed": False,
            "source_depth_observed": False,
            "source_visible_surface_id_known": False,
        })
    manifest = {
        "schema": SCHEMA,
        "source_binding_sha256": source.source_binding_sha256,
        "mesh_lineage_hash": mesh.mesh_lineage_hash,
        "mesh_face_count": len(mesh.faces),
        "view_summaries": summaries,
        "source_original_rgb_observation": True,
        "rgb_values_are_nearest_resampled_original_pixels": True,
        "face_correspondence_is_hypothesis_conditioned": True,
        "overlapping_face_pixel_ownership_unqualified": True,
        "unobserved_hidden_surfaces_color_unqualified": True,
        "source_rgb_surface_ownership_qualified": False,
        "source_camera_observed": False,
        "source_depth_observed": False,
        "pose_conditioned_source_visibility_qualified": False,
        "native_source_rgb_render_parity_qualified": False,
        "product_revision_minted": False,
        "sample_count": sum(len(x["view"]) for x in records),
        "max_samples_per_original_view": MAX_PER_VIEW,
    }
    encoded_manifest = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    arrays = {
        "manifest_utf8": np.frombuffer(encoded_manifest, dtype=np.uint8).copy(),
        "view_index_u8": np.concatenate([r["view"] for r in records]),
        "pixel_xy_u16": np.concatenate([r["xy"] for r in records], axis=0),
        "face_index_i32": np.concatenate([r["face"] for r in records]),
        "barycentric_f32": np.concatenate([r["bary"] for r in records], axis=0),
        "source_rgba_u8": np.concatenate([r["rgba"] for r in records], axis=0),
    }
    if arrays["barycentric_f32"].size and (
        not np.isfinite(arrays["barycentric_f32"]).all()
        or not np.allclose(arrays["barycentric_f32"].sum(axis=1), 1, atol=5.e-5)
    ):
        raise RuntimeError("SOURCE_RGB_BARYCENTRIC_NOT_FINITE_OR_SIMPLEX")
    return _encode_npz_deterministic(arrays), manifest


def decode_raw_source_rgb_image_plane_candidates_v1(
    data: bytes,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if type(data) is not bytes or len(data) > MAX_ARTIFACT_BYTES:
        raise RuntimeError("SOURCE_RGB_REOPEN_CAS_BUDGET")
    try:
        with ZipFile(io.BytesIO(data)) as check:
            records = check.infolist()
            if (
                len(records) != len(_KEYS)
                or {r.filename for r in records} != {k + ".npy" for k in _KEYS}
                or any(r.file_size > 8 * 1024 * 1024 or r.compress_type != ZIP_DEFLATED
                       for r in records)
                or sum(r.file_size for r in records) > 10 * 1024 * 1024
            ):
                raise RuntimeError("SOURCE_RGB_REOPEN_ZIP_RESOURCE_OR_KEYS")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != _KEYS:
                raise RuntimeError("SOURCE_RGB_REOPEN_ARRAY_KEYS")
            arrays = {key: archive[key] for key in sorted(_KEYS)}
        manifest = json.loads(arrays["manifest_utf8"].tobytes().decode("utf-8"))
    except (ValueError, OSError, KeyError, UnicodeDecodeError, BadZipFile, EOFError, zlib.error) as exc:
        raise RuntimeError("SOURCE_RGB_REOPEN_INVALID_ARCHIVE") from exc
    count = manifest.get("sample_count") if type(manifest) is dict else None
    if (
        type(manifest) is not dict
        or manifest.get("schema") != SCHEMA
        or type(count) is not int or not 0 <= count <= 2 * MAX_PER_VIEW
        or type(manifest.get("mesh_face_count")) is not int
        or not 1 <= manifest["mesh_face_count"] <= 32768
        or arrays["view_index_u8"].shape != (count,)
        or arrays["view_index_u8"].dtype != np.uint8
        or arrays["pixel_xy_u16"].shape != (count, 2)
        or arrays["pixel_xy_u16"].dtype != np.uint16
        or arrays["face_index_i32"].shape != (count,)
        or arrays["face_index_i32"].dtype != np.int32
        or not ((arrays["face_index_i32"] >= 0)
                & (arrays["face_index_i32"] < manifest["mesh_face_count"])).all()
        or arrays["barycentric_f32"].shape != (count, 3)
        or arrays["barycentric_f32"].dtype != np.float32
        or arrays["source_rgba_u8"].shape != (count, 4)
        or arrays["source_rgba_u8"].dtype != np.uint8
        or not np.isin(arrays["view_index_u8"], [0, 1]).all()
        or (arrays["source_rgba_u8"][:, 3] != 255).any()
        or type(manifest.get("view_summaries")) is not list
        or len(manifest["view_summaries"]) != 2
        or not np.isfinite(arrays["barycentric_f32"]).all()
        or not np.allclose(arrays["barycentric_f32"].sum(axis=1), 1., atol=5.e-5)
        or manifest.get("source_rgb_surface_ownership_qualified") is not False
        or manifest.get("native_source_rgb_render_parity_qualified") is not False
        or manifest.get("product_revision_minted") is not False
    ):
        raise RuntimeError("SOURCE_RGB_REOPEN_TYPED_OR_AUTHORITY_INVALID")
    return arrays, manifest


__all__ = [
    "SCHEMA", "ARTIFACT_TYPE", "MAX_ARTIFACT_BYTES",
    "build_raw_source_rgb_image_plane_candidates_v1",
    "decode_raw_source_rgb_image_plane_candidates_v1",
]
