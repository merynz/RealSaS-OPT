from __future__ import annotations

"""Original-pixel RGB evidence mapped onto *hypothesized*, not observed, faces.

A reconstructed triangle and user-claimed FRONT/BACK view do NOT prove an
original camera or source-visible surface owner. This compiler measures actual
original decoded sRGB pixels, records exact immutable image hashes, excludes
multi-face projected overlap and anti-aliased/unknown alpha, and leaves every
face correspondence explicitly unqualified. No generated RGB view is a source
pixel, and no material/UV, hidden texture or ProductRevision is minted.
"""

from hashlib import sha256
import io
import math
from typing import Any

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    _covers_pixel_center,
)
from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    RSSvNextUncalibratedTwoImageSourceV1,
)
from .rss_vnext_raw_mesh_admission_v1 import (
    _MAX_FACE_RASTER_VISITS, _MAX_VIEW_SIDE, _observed_alpha,
    _project_fitted, admit_raw_two_view_mesh_v1,
)

SCHEMA = "RealSaS.RSSvNext.OriginalRGBProjectedCandidate.v1"
MAX_SAMPLES_PER_VIEW = 4096
MAX_SAMPLES_TOTAL = 2 * MAX_SAMPLES_PER_VIEW
MIN_INTERIOR_BARYCENTRIC = .06


def _orient(a, b, px: float, py: float) -> float:
    return (float(b[0]) - float(a[0])) * (py - float(a[1])) - (
        float(b[1]) - float(a[1])
    ) * (px - float(a[0]))


def _rgba_and_mask(source: RSSvNextUncalibratedTwoImageSourceV1, view: int):
    mask, blocker = _observed_alpha(source, view)
    if blocker is not None or mask is None:
        raise RuntimeError("RSS_VNEXT_SOURCE_RGB_OBSERVED_ALPHA_REQUIRED:" + str(blocker))
    with Image.open(io.BytesIO(source.original_files[view])) as image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        factor = min(1., _MAX_VIEW_SIDE / max(width, height))
        size = (max(16, round(width * factor)), max(16, round(height * factor)))
        if size != rgba.size:
            rgba = rgba.resize(size, resample=Image.Resampling.NEAREST)
        array = np.array(rgba, dtype=np.uint8, copy=True)
    if array.shape[:2] != mask.shape or not np.array_equal(
        array[:, :, 3] >= 128, mask,
    ):
        raise RuntimeError("RSS_VNEXT_SOURCE_RGB_ALPHA_REPLAY_DRIFT")
    return array, mask


def _sample_inferred_surface_pixels(
    rgba: np.ndarray, observed_mask: np.ndarray,
    projected_vertices: np.ndarray, faces: np.ndarray,
) -> dict[str, Any]:
    height, width = observed_mask.shape
    if (
        rgba.dtype != np.uint8 or rgba.shape != (height, width, 4)
        or projected_vertices.ndim != 2 or projected_vertices.shape[1] != 2
        or faces.ndim != 2 or faces.shape[1] != 3
        or faces.shape[0] > 32768 or not np.isfinite(projected_vertices).all()
        or not np.array_equal(rgba[:, :, 3] >= 128, observed_mask)
    ):
        raise RuntimeError("RSS_VNEXT_SOURCE_RGB_PROJECTION_AND_PIXELS_ABI")
    if faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(
        faces >= len(projected_vertices),
    ):
        raise RuntimeError("RSS_VNEXT_SOURCE_RGB_FACE_ADDRESS_DRIFT")
    # -1 = uncovered, -2 = multiple faces: without source depth/occlusion,
    # a multiply projected pixel cannot be assigned to any one face.
    owner = np.full((height, width), -1, dtype=np.int32)
    probes = 0
    for face_index, vertex_ids in enumerate(faces):
        tri = projected_vertices[vertex_ids]
        a, b, c = tri
        if abs(_orient(a, b, float(c[0]), float(c[1]))) <= 1.e-12:
            raise RuntimeError("RSS_VNEXT_SOURCE_RGB_DEGENERATE_TRIANGLE")
        x0 = max(0, math.floor(float(tri[:, 0].min()) - .5))
        x1 = min(width-1, math.ceil(float(tri[:, 0].max()) - .5))
        y0 = max(0, math.floor(float(tri[:, 1].min()) - .5))
        y1 = min(height-1, math.ceil(float(tri[:, 1].max()) - .5))
        probes += max(0, x1-x0+1) * max(0, y1-y0+1)
        if probes > _MAX_FACE_RASTER_VISITS:
            raise RuntimeError("RSS_VNEXT_SOURCE_RGB_RASTER_BUDGET")
        for y in range(y0, y1+1):
            for x in range(x0, x1+1):
                if _covers_pixel_center(a, b, c, x, y):
                    owner[y, x] = (
                        face_index if owner[y, x] == -1 else -2
                    )
    # Foreground coverage and overlap counts are measured over the ENTIRE
    # raster, not only the bounded sample subset.
    one_face = (owner >= 0) & observed_mask
    overlap = (owner == -2) & observed_mask
    none = (owner == -1) & observed_mask
    candidates = []
    for y in range(1, height-1):
        for x in range(1, width-1):
            if not one_face[y, x] or int(rgba[y, x, 3]) != 255:
                continue
            if not bool(observed_mask[y-1:y+2, x-1:x+2].all()):
                continue
            face_index = int(owner[y, x])
            face = faces[face_index]
            a, b, c = projected_vertices[face]
            px, py = x + .5, y + .5
            denom = _orient(a, b, float(c[0]), float(c[1]))
            lambdas = (
                _orient(b, c, px, py) / denom,
                _orient(c, a, px, py) / denom,
            )
            bary = (float(lambdas[0]), float(lambdas[1]),
                    float(1. - lambdas[0] - lambdas[1]))
            if min(bary) < MIN_INTERIOR_BARYCENTRIC:
                continue
            candidates.append({
                "pixel_xy": [x, y],
                "rgba8_observed": [int(v) for v in rgba[y, x]],
                "candidate_face_index": face_index,
                "hypothesized_face_barycentric": list(bary),
                "candidate_face_owner_is_source_truth": False,
            })
    # A stable scan-order stride preserves exact sample reproducibility.
    stride = max(1, math.ceil(len(candidates) / MAX_SAMPLES_PER_VIEW))
    samples = candidates[::stride][:MAX_SAMPLES_PER_VIEW]
    return {
        "sampled_source_opaque_interior_count": len(samples),
        "eligible_sample_count_before_budget": len(candidates),
        "deterministic_sampling_stride": stride,
        "raster_source_foreground_pixels": int(observed_mask.sum()),
        "raster_unique_projected_face_foreground_pixels": int(one_face.sum()),
        "raster_overlap_ambiguous_foreground_pixels": int(overlap.sum()),
        "raster_uncovered_foreground_pixels": int(none.sum()),
        "raster_owner_candidate_sha256": sha256(
            owner.astype("<i4", copy=False).tobytes(order="C"),
        ).hexdigest(),
        "samples": samples,
    }


def build_raw_original_rgb_candidate_v1(
    *, source: RSSvNextUncalibratedTwoImageSourceV1,
    scene: dict[str, np.ndarray], receipt: dict[str, Any],
    scene_digest: str, canonical_mesh,
) -> dict[str, Any]:
    if (
        source.source_binding_sha256 != receipt.get("source_binding_sha256")
        or canonical_mesh.mesh_lineage_hash is None
        or not (type(scene_digest) is str and len(scene_digest) == 64)
    ):
        raise RuntimeError("RSS_VNEXT_SOURCE_RGB_SOURCE_OR_SCENE_BINDING")
    expected, _, witness = admit_raw_two_view_mesh_v1(
        source=source, scene=scene, receipt=receipt, scene_digest=scene_digest,
    )
    if expected is None or expected != canonical_mesh or (
        not witness["source_static_silhouette_qualified"]
    ):
        raise RuntimeError("RSS_VNEXT_SOURCE_RGB_CANONICAL_STATIC_REPLAY_REQUIRED")
    xyz = scene["tessa_candidate_rest_vertices"][0].astype(np.float64)
    faces = scene["tessa_candidate_faces"]
    views = []
    for index in range(2):
        rgba, mask = _rgba_and_mask(source, index)
        projected, camera = _project_fitted(xyz, mask, index)
        statistics = _sample_inferred_surface_pixels(rgba, mask, projected, faces)
        views.append({
            "original_view_index": index,
            "original_view_claim": source.view_claims[index],
            "original_image_bytes_sha256": sha256(
                source.original_files[index],
            ).hexdigest(),
            "redecoded_resized_rgba8_sha256": sha256(
                rgba.tobytes(order="C"),
            ).hexdigest(),
            "observation_resolution_wh": [int(rgba.shape[1]), int(rgba.shape[0])],
            "camera_hypothesis": camera,
            "original_source_rgb_pixels_observed": True,
            "projected_face_correspondence_is_original_source_truth": False,
            "per_pixel_source_depth_observed": False,
            "dynamic_pose_visibility_qualified": False,
            **statistics,
        })
    payload = {
        "schema": SCHEMA,
        "source_binding_sha256": source.source_binding_sha256,
        "original_source_images_only": True,
        "scene_cas_sha256": scene_digest,
        "unified_checkpoint_sha256": receipt["checkpoint_sha256"],
        "canonical_mesh_lineage_hash": canonical_mesh.mesh_lineage_hash,
        "sampling_policy": "ORIGINAL_SRGB8_FULL_ALPHA_INTERIOR_UNIQUE_PROJECTED_FACE_V1",
        "geometry_policy": "FITTED_2D_CAMERA_HYPOTHESIS_NOT_SOURCE_OBSERVED_RAY",
        "source_rgb_pixels_observed": True,
        "source_rgb_to_canonical_face_ownership_qualified": False,
        "texture_uv_qualified": False,
        "source_hidden_surface_qualified": False,
        "source_appearance_release_qualified": False,
        "original_camera_observed": False,
        "product_revision_minted": False,
        "views": views,
        "sample_count": sum(
            v["sampled_source_opaque_interior_count"] for v in views
        ),
    }
    if payload["sample_count"] > MAX_SAMPLES_TOTAL:
        raise RuntimeError("RSS_VNEXT_SOURCE_RGB_TOTAL_SAMPLE_BUDGET")
    payload["candidate_hash"] = content_sha256(payload)
    return payload


__all__ = [
    "SCHEMA", "MAX_SAMPLES_PER_VIEW", "build_raw_original_rgb_candidate_v1",
]
