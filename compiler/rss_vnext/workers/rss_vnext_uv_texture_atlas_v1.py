from __future__ import annotations

"""One stable mesh-addressed UV domain, two distinct original RGB layers.

This compiler operation starts ONLY from the original-source image-plane
candidate CAS, not a new model prediction or a second source projection. It
builds independent per-face UV charts and samples original RGBA into two
distinct view layers. Source colors that conflict in the same UV texel are
NOT blended. Uncovered texels remain UNKNOWN; the projection is based on a
fitted, uncalibrated camera hypothesis. Graphics controls presentation and
draw order independently of this texture carrier.
"""

from hashlib import sha256
import io
import json
import math
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR, canonical_mesh_lineage_hash_v1,
    validate_canonical_mesh_domain_v1,
)
from .rss_vnext_source_rgb_projection_candidate_v1 import (
    decode_raw_source_rgb_image_plane_candidates_v1,
)

SCHEMA = "RealSaS.RSSvNext.MeshUVTextureAtlasCandidate.v1"
ARTIFACT_TYPE = "RealSaS.RSSvNextMeshUVTextureAtlasCandidate"
MAX_DIMENSION = 2048
MAX_FACES = 4096
MAX_ARTIFACT_BYTES = 96 * 1024 * 1024
_TILE_OPTIONS = (16, 24, 32, 48, 64, 96)
_GUTTER = 2
_KEYS = frozenset({
    "manifest_utf8",
    "texture_rgba_u8",
    "texture_source_sample_mask_u8",
    "source_view_conflict_u8",
    "face_chart_xywh_i32",
    "face_vertex_uv_f32",
    "face_source_sample_counts_i32",
    "neural_rgb_candidate_rgba_u8",
    "neural_rgb_candidate_mask_u8",
    "neural_rgb_log_uncertainty_f16",
})


def _packing(number_of_faces: int, max_face_samples: int):
    if not 1 <= number_of_faces <= MAX_FACES:
        raise RuntimeError("TEXTURE_ATLAS_FACE_RESOURCE_LIMIT")
    preferred = min(_TILE_OPTIONS[-1], max(_TILE_OPTIONS[0], math.ceil(math.sqrt(max_face_samples))))
    # Prefer observed texel density, but large generic morphologies must
    # gracefully downshift chart size instead of failing solely because a
    # preferred 96px face tile cannot fit the 2048px resource budget.
    choices = [side for side in _TILE_OPTIONS if side >= preferred]
    choices += [side for side in reversed(_TILE_OPTIONS) if side < preferred]
    for side in choices:
        per_row = min(math.ceil(math.sqrt(number_of_faces)), MAX_DIMENSION // side)
        if per_row <= 0:
            continue
        rows = math.ceil(number_of_faces / per_row)
        width, height = per_row * side, rows * side
        if height <= MAX_DIMENSION:
            return side, per_row, width, height
    raise RuntimeError("TEXTURE_ATLAS_DIMENSION_RESOURCE_LIMIT")


def _archive(arrays: dict[str, np.ndarray]) -> bytes:
    out = io.BytesIO()
    with ZipFile(out, "w") as archive:
        if set(arrays) != _KEYS:
            raise RuntimeError("TEXTURE_ATLAS_ARRAY_SCHEMA")
        for name, array in sorted(arrays.items()):
            if array.dtype.hasobject:
                raise RuntimeError("TEXTURE_ATLAS_OBJECT_DTYPE_FORBIDDEN")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue(), compress_type=ZIP_DEFLATED, compresslevel=6)
    packed = out.getvalue()
    if not 1 <= len(packed) <= MAX_ARTIFACT_BYTES:
        raise RuntimeError("TEXTURE_ATLAS_CAS_BUDGET")
    return packed


def compile_rss_vnext_uv_texture_atlas_v1(
    *, source_rgb_candidates: bytes, mesh: CanonicalMeshIR,
    graphics_vertex_logits: np.ndarray | None = None,
    graphics_scene_content_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    validate_canonical_mesh_domain_v1(mesh)
    if mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh):
        raise RuntimeError("TEXTURE_ATLAS_CANONICAL_MESH_HASH_DRIFT")
    samples, original = decode_raw_source_rgb_image_plane_candidates_v1(
        source_rgb_candidates,
    )
    face_count = len(mesh.faces)
    if (
        original["mesh_lineage_hash"] != mesh.mesh_lineage_hash
        or original["mesh_face_count"] != face_count
    ):
        raise RuntimeError("TEXTURE_ATLAS_SOURCE_SAMPLE_MESH_DRIFT")
    has_neural_field = graphics_vertex_logits is not None
    if has_neural_field:
        graphics_vertex_logits = np.asarray(graphics_vertex_logits)
        if (
            graphics_vertex_logits.ndim != 2
            or graphics_vertex_logits.shape[0] != len(mesh.vertices)
            or graphics_vertex_logits.shape[1] < 5
            or not np.isfinite(graphics_vertex_logits).all()
            or type(graphics_scene_content_sha256) is not str
            or len(graphics_scene_content_sha256) != 64
            or any(x not in "0123456789abcdef"
                   for x in graphics_scene_content_sha256)
        ):
            raise RuntimeError("TEXTURE_ATLAS_NEURAL_GRAPHICS_FIELD_ABI")
        vertex_logits = graphics_vertex_logits.astype(np.float64)
    elif graphics_scene_content_sha256 is not None:
        raise RuntimeError("TEXTURE_ATLAS_NEURAL_GRAPHICS_UNBOUND_SCENE")
    index = {vertex.canonical_mesh_vertex_id: i
             for i, vertex in enumerate(mesh.vertices)}
    face_indices = np.array([
        [index[k] for k in face] for face in mesh.faces
    ], dtype=np.int64)
    sample_faces = samples["face_index_i32"]
    sample_views = samples["view_index_u8"]
    sample_rgba = samples["source_rgba_u8"]
    sample_bary = samples["barycentric_f32"].astype(np.float64)
    per_face = np.bincount(sample_faces, minlength=face_count)
    side, columns, width, height = _packing(face_count, int(per_face.max(initial=0)))
    charts = np.zeros((face_count, 4), dtype=np.int32)
    uv = np.zeros((face_count, 3, 2), dtype=np.float32)
    counts = np.zeros((face_count, 2), dtype=np.int32)
    inner = side - 2 * _GUTTER - 1
    neural_texture = np.zeros((height, width, 4), dtype=np.uint8)
    neural_mask = np.zeros((height, width), dtype=np.uint8)
    neural_log_uncertainty = np.zeros((height, width), dtype=np.float16)
    local_uv_axis = np.arange(inner + 1, dtype=np.float64) / inner
    uu, vv = np.meshgrid(local_uv_axis, local_uv_axis, indexing="xy")
    in_triangle = (uu + vv <= 1. + 1.e-12)
    texel_bary = np.stack((1. - uu - vv, uu, vv), axis=-1)
    for face in range(face_count):
        ox = (face % columns) * side
        oy = (face // columns) * side
        charts[face] = (ox, oy, side, side)
        x0, y0 = ox + _GUTTER, oy + _GUTTER
        uv[face, 0] = ((x0 + .5) / width, (y0 + .5) / height)
        uv[face, 1] = ((x0 + inner + .5) / width, (y0 + .5) / height)
        uv[face, 2] = ((x0 + .5) / width, (y0 + inner + .5) / height)
        if has_neural_field:
            # RGB logits 0:3, density logit 3, log uncertainty 4;
            # same unified H(x) that proposed TESSA/AXIS/MIRA topology.
            # These are inferred color candidates for UV completion,
            # never original observed pixels or source visibility.
            interpolated = np.einsum(
                "abc,cd->abd", texel_bary,
                vertex_logits[face_indices[face], :],
            )
            rgb = 1. / (1. + np.exp(-np.clip(interpolated[:, :, :3], -30., 30.)))
            rgba = np.concatenate(
                (np.rint(rgb * 255.).astype(np.uint8),
                 np.full((inner + 1, inner + 1, 1), 255, dtype=np.uint8)),
                axis=-1,
            )
            destination = neural_texture[y0:y0 + inner + 1, x0:x0 + inner + 1]
            destination[in_triangle] = rgba[in_triangle]
            neural_mask[y0:y0 + inner + 1, x0:x0 + inner + 1][in_triangle] = 1
            uncertainty = np.clip(interpolated[:, :, 4], -20., 20.).astype(np.float16)
            neural_log_uncertainty[
                y0:y0 + inner + 1, x0:x0 + inner + 1
            ][in_triangle] = uncertainty[in_triangle]

    texture = np.zeros((2, height, width, 4), dtype=np.uint8)
    has_color = np.zeros((2, height, width), dtype=np.uint8)
    conflicts = np.zeros((height, width), dtype=np.uint8)
    if len(sample_faces):
        ox = (sample_faces % columns) * side + _GUTTER
        oy = (sample_faces // columns) * side + _GUTTER
        u = sample_bary[:, 1].copy()
        v = sample_bary[:, 2].copy()
        if (
            not np.isfinite(u).all() or not np.isfinite(v).all()
            or np.any(u < -1.e-4) or np.any(v < -1.e-4)
            or np.any(u + v > 1.0001)
        ):
            raise RuntimeError("TEXTURE_ATLAS_BARYCENTRIC_DOMAIN_INVALID")
        x_local = np.rint(np.clip(u, 0., 1.) * inner).astype(np.int32)
        y_local = np.rint(np.clip(v, 0., 1.) * inner).astype(np.int32)
        x = ox + x_local
        y = oy + y_local
        target_bary_u = x_local / inner
        target_bary_v = y_local / inner
        sqdist = (u - target_bary_u)**2 + (v - target_bary_v)**2
        flat = ((sample_views.astype(np.int64) * height + y) * width + x)
        pixel_xy = samples["pixel_xy_u16"]
        # A texel may receive many original pixels. Use the nearest barycentric
        # pixel, breaking ties by original pixel Y then X, never mixing colors.
        order = np.lexsort((
            pixel_xy[:, 0], pixel_xy[:, 1], sqdist, flat,
        ))
        selected = order[np.concatenate((
            [True], flat[order][1:] != flat[order][:-1],
        ))]
        vy, yy, xx = sample_views[selected], y[selected], x[selected]
        texture[vy, yy, xx] = sample_rgba[selected]
        has_color[vy, yy, xx] = 1
        np.add.at(counts, (sample_faces, sample_views.astype(np.int32)), 1)
        both = (has_color[0] == 1) & (has_color[1] == 1)
        conflicts[both] = np.any(texture[0, both] != texture[1, both], axis=-1)
    n_known = [int(has_color[i].sum()) for i in range(2)]
    n_conflicts = int(conflicts.sum())
    manifest = {
        "schema": SCHEMA,
        "canonical_mesh_lineage_hash": mesh.mesh_lineage_hash,
        "original_rgb_candidate_sha256": sha256(source_rgb_candidates).hexdigest(),
        "original_source_binding_sha256": original["source_binding_sha256"],
        "face_count": face_count,
        "chart_policy": "FACE_LOCAL_BARYCENTRIC_DISCONTINUOUS_UV_CHART_V1",
        "atlas_size": [width, height],
        "face_tile_size": side,
        "requested_face_tile_size": min(
            _TILE_OPTIONS[-1],
            max(_TILE_OPTIONS[0], math.ceil(math.sqrt(int(per_face.max(initial=0))))),
        ),
        "texel_density_limited_by_atlas_budget": (
            side < min(_TILE_OPTIONS[-1],
                       max(_TILE_OPTIONS[0],
                           math.ceil(math.sqrt(int(per_face.max(initial=0))))))
        ),
        "per_face_chart_gutter_px": _GUTTER,
        "source_view_count": 2,
        "view_color_layer_provenance": "OBSERVED_RGBA_WITH_FITTED_CAMERA_FACE_ASSOCIATION",
        "source_view_color_texel_counts": n_known,
        "source_view_conflict_texel_count": n_conflicts,
        "source_rgb_values_from_original_files": True,
        "source_to_face_correspondence_camera_hypothesis": True,
        "inferred_surface_texture_completion_performed": bool(has_neural_field),
        "neural_graphics_field_source": (
            "SAME_UNIFIED_H_TESSA_VERTEX_GRAPHICS_LOGITS"
            if has_neural_field else "UNAVAILABLE_NO_NEURAL_COMPLETION"
        ),
        "neural_graphics_scene_content_sha256": graphics_scene_content_sha256,
        "neural_graphics_logits_sha256": (
            sha256(np.asarray(graphics_vertex_logits).tobytes(order="C")).hexdigest()
            if has_neural_field else None
        ),
        "neural_graphics_color_texel_count": int(neural_mask.sum()),
        "neural_graphics_rgb_is_inferred_not_observed": True,
        "neural_graphics_untrained_checkpoint_may_be_invalid": True,
        "neural_completion_never_overwrites_original_rgb_layers": True,
        "source_3d_surface_ownership_verified": False,
        "draw_order_is_not_texture_atlas_authority": True,
        "pose_conditioned_occlusion_qualified": False,
        "native_renderer_texture_parity_qualified": False,
        "unfilled_texture_texels_are_unknown": True,
        "canonical_uv_topology_is_mesh_face_addressed": True,
        "product_revision_minted": False,
    }
    manifest_buf = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    arrays = {
        "manifest_utf8": np.frombuffer(manifest_buf, dtype=np.uint8).copy(),
        "texture_rgba_u8": texture,
        "texture_source_sample_mask_u8": has_color,
        "source_view_conflict_u8": conflicts,
        "face_chart_xywh_i32": charts,
        "face_vertex_uv_f32": uv,
        "face_source_sample_counts_i32": counts,
        "neural_rgb_candidate_rgba_u8": neural_texture,
        "neural_rgb_candidate_mask_u8": neural_mask,
        "neural_rgb_log_uncertainty_f16": neural_log_uncertainty,
    }
    return _archive(arrays), manifest


def reopen_rss_vnext_uv_texture_atlas_v1(
    blob: bytes,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if type(blob) is not bytes or not 1 <= len(blob) <= MAX_ARTIFACT_BYTES:
        raise RuntimeError("TEXTURE_ATLAS_REOPEN_CAS_SIZE")
    try:
        with ZipFile(io.BytesIO(blob)) as archive:
            records = archive.infolist()
            if (
                len(records) != len(_KEYS)
                or {r.filename for r in records} != {key + ".npy" for key in _KEYS}
                or any(r.file_size > 96 * 1024 * 1024 or r.compress_type != ZIP_DEFLATED
                       for r in records)
                or sum(r.file_size for r in records) > 96 * 1024 * 1024
            ):
                raise RuntimeError("TEXTURE_ATLAS_REOPEN_ARCHIVE_DOMAIN")
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            if set(archive.files) != _KEYS:
                raise RuntimeError("TEXTURE_ATLAS_REOPEN_ARCHIVE_KEYS")
            arrays = {k: archive[k] for k in sorted(_KEYS)}
        meta = json.loads(arrays["manifest_utf8"].tobytes().decode("utf-8"))
    except (UnicodeDecodeError, KeyError, ValueError, OSError) as exc:
        raise RuntimeError("TEXTURE_ATLAS_REOPEN_PARSE_ERROR") from exc
    if type(meta) is not dict or meta.get("schema") != SCHEMA:
        raise RuntimeError("TEXTURE_ATLAS_REOPEN_MANIFEST_SCHEMA")
    n = meta.get("face_count")
    shape = meta.get("atlas_size")
    if (
        type(n) is not int or not 1 <= n <= MAX_FACES
        or type(shape) is not list or len(shape) != 2
        or any(type(x) is not int or not 1 <= x <= MAX_DIMENSION for x in shape)
    ):
        raise RuntimeError("TEXTURE_ATLAS_REOPEN_FACE_DIMENSION")
    w, h = shape
    t = arrays["texture_rgba_u8"]
    m = arrays["texture_source_sample_mask_u8"]
    c = arrays["source_view_conflict_u8"]
    uv = arrays["face_vertex_uv_f32"]
    charts = arrays["face_chart_xywh_i32"]
    counts = arrays["face_source_sample_counts_i32"]
    neural_texture = arrays["neural_rgb_candidate_rgba_u8"]
    neural_mask = arrays["neural_rgb_candidate_mask_u8"]
    neural_uncertainty = arrays["neural_rgb_log_uncertainty_f16"]
    if (
        t.shape != (2, h, w, 4) or t.dtype != np.uint8
        or m.shape != (2, h, w) or m.dtype != np.uint8
        or c.shape != (h, w) or c.dtype != np.uint8
        or charts.shape != (n, 4) or charts.dtype != np.int32
        or uv.shape != (n, 3, 2) or uv.dtype != np.float32
        or counts.shape != (n, 2) or counts.dtype != np.int32
        or neural_texture.shape != (h, w, 4) or neural_texture.dtype != np.uint8
        or neural_mask.shape != (h, w) or neural_mask.dtype != np.uint8
        or neural_uncertainty.shape != (h, w) or neural_uncertainty.dtype != np.float16
        or np.any((neural_mask != 0) & (neural_mask != 1))
        or not np.isfinite(neural_uncertainty).all()
        or np.any(neural_texture[neural_mask == 0] != 0)
        or type(meta.get("neural_graphics_color_texel_count")) is not int
        or meta["neural_graphics_color_texel_count"] != int(neural_mask.sum())
        or meta.get("neural_graphics_rgb_is_inferred_not_observed") is not True
        or meta.get("neural_completion_never_overwrites_original_rgb_layers") is not True
        or np.any((m != 0) & (m != 1))
        or np.any((c != 0) & (c != 1))
        or np.any(t[m == 0] != 0)
        or not np.isfinite(uv).all()
        or not ((uv >= 0.) & (uv <= 1.)).all()
        or meta.get("source_3d_surface_ownership_verified") is not False
        or meta.get("pose_conditioned_occlusion_qualified") is not False
        or meta.get("product_revision_minted") is not False
    ):
        raise RuntimeError("TEXTURE_ATLAS_REOPEN_DTYPE_OR_FALSE_AUTHORITY")
    return arrays, meta


__all__ = [
    "SCHEMA", "ARTIFACT_TYPE", "MAX_ARTIFACT_BYTES",
    "compile_rss_vnext_uv_texture_atlas_v1",
    "reopen_rss_vnext_uv_texture_atlas_v1",
]
