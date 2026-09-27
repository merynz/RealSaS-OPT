from __future__ import annotations

"""Reference CAA renderer: visibility and appearance remain separate authorities."""

from dataclasses import dataclass
import hashlib

import numpy as np

from .appearance_bake_v2 import (
    bilinear_premultiplied_rgba,
    conservative_bilinear_provenance,
)
from .appearance_color_v2 import premultiplied_linear_to_straight_srgb_u8
from .types import QualificationError
from .visibility_v2 import VISIBILITY_CONTRACT_V2_HASH, rasterize_visible_owner


RUNTIME_COVERAGE_SCALE = 2
RUNTIME_COVERAGE_SAMPLE_COUNT = RUNTIME_COVERAGE_SCALE * RUNTIME_COVERAGE_SCALE


@dataclass(frozen=True)
class ReferenceCAARender:
    premultiplied_rgba: np.ndarray
    straight_rgba_u8: np.ndarray
    geometry_visible: np.ndarray
    final_alpha: np.ndarray
    owner_face_index: np.ndarray
    second_owner_face_index: np.ndarray
    depth_margin: np.ndarray
    exact_depth_ambiguity: np.ndarray
    layer_overflow: np.ndarray
    contributing_layer_count: np.ndarray
    contributing_layer_mask: np.ndarray
    layer_owner_face_index: np.ndarray
    coverage_sample_owner_face_index: np.ndarray
    coverage_sample_count: int
    provenance_code: np.ndarray
    visibility_contract_hash: str = VISIBILITY_CONTRACT_V2_HASH


def _load_transport_texture(path) -> np.ndarray:
    from PIL import Image

    value = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 4:
        raise QualificationError("CAA_REFERENCE_TEXTURE_INVALID")
    return value


def load_face_uv(asset) -> np.ndarray:
    path = str(asset.uv_npz_path)
    with np.load(path, allow_pickle=False) as data:
        if "face_uv" not in data.files:
            raise QualificationError("CAA_REFERENCE_FACE_UV_MISSING")
        uv = np.asarray(data["face_uv"], dtype=np.float64)
    if uv.ndim != 3 or uv.shape[1:] != (3, 2):
        raise QualificationError("CAA_REFERENCE_FACE_UV_SHAPE_INVALID")
    return uv



def load_face_page_index(asset) -> np.ndarray:
    path = str(asset.uv_npz_path)
    with np.load(path, allow_pickle=False) as data:
        if "face_page_index" not in data.files:
            return np.zeros((len(load_face_uv(asset)),), dtype=np.int32)
        page = np.asarray(data["face_page_index"], dtype=np.int32)
    if page.ndim != 1 or len(page) != len(load_face_uv(asset)) or np.any(page < 0):
        raise QualificationError("CAA_REFERENCE_FACE_PAGE_INDEX_INVALID")
    return page

def load_provenance_atlas(asset) -> np.ndarray:
    with np.load(str(asset.provenance_npz_path), allow_pickle=False) as data:
        if "provenance" not in data.files:
            raise QualificationError("CAA_REFERENCE_PROVENANCE_MISSING")
        value = np.asarray(data["provenance"], dtype=np.uint8)
    if value.ndim not in (3, 4) or value.shape[0] != 8:
        raise QualificationError("CAA_REFERENCE_PROVENANCE_SHAPE_INVALID")
    return value


def _sample_nearest_scalar(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    points = np.asarray(uv, dtype=np.float64)
    h, w = source.shape[:2]
    x = np.rint(np.clip(points[:, 0], 0.0, 1.0) * float(w - 1)).astype(np.int64)
    y = np.rint(np.clip(points[:, 1], 0.0, 1.0) * float(h - 1)).astype(np.int64)
    return source[y, x]



def _sample_transport_pages(
    texture_rgba_u8: np.ndarray,
    provenance_atlas: np.ndarray,
    uv: np.ndarray,
    page_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    texture = np.asarray(texture_rgba_u8, dtype=np.uint8)
    provenance = np.asarray(provenance_atlas, dtype=np.uint8)
    points = np.asarray(uv, dtype=np.float64)
    pages = np.asarray(page_index, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 2 or pages.shape != (len(points),):
        raise QualificationError("CAA_REFERENCE_PAGED_SAMPLE_INPUT_INVALID")

    if texture.ndim == 3:
        if provenance.ndim != 2 or provenance.shape != texture.shape[:2]:
            raise QualificationError("CAA_REFERENCE_PROVENANCE_TEXTURE_DRIFT")
        if np.any(pages != 0):
            raise QualificationError("CAA_REFERENCE_SINGLE_PAGE_INDEX_DRIFT")
        return (
            bilinear_premultiplied_rgba(texture, points),
            conservative_bilinear_provenance(provenance, points).astype(np.uint8),
        )

    if (
        texture.ndim != 4
        or texture.shape[-1] != 4
        or provenance.ndim != 3
        or provenance.shape != texture.shape[:3]
    ):
        raise QualificationError("CAA_REFERENCE_PAGED_TEXTURE_SHAPE_INVALID")
    if np.any(pages < 0) or np.any(pages >= texture.shape[0]):
        raise QualificationError("CAA_REFERENCE_FACE_PAGE_OUT_OF_RANGE")

    sampled = np.zeros((len(points), 4), dtype=np.float64)
    sampled_provenance = np.full((len(points),), 255, dtype=np.uint8)
    for page in np.unique(pages):
        take = pages == int(page)
        sampled[take] = bilinear_premultiplied_rgba(
            texture[int(page)], points[take]
        )
        sampled_provenance[take] = conservative_bilinear_provenance(
            provenance[int(page)], points[take]
        ).astype(np.uint8)
    return sampled, sampled_provenance

def premultiplied_to_straight_u8(pm: np.ndarray) -> np.ndarray:
    """Compatibility wrapper: linear PM -> straight sRGB RGBA8."""
    value = np.asarray(pm, dtype=np.float64)
    if value.ndim != 3 or value.shape[2] != 4:
        raise QualificationError("CAA_REFERENCE_PM_RGBA_SHAPE_INVALID")
    return premultiplied_linear_to_straight_srgb_u8(value)

def _coverage_reshape(value: np.ndarray, *, channels: tuple[int, ...] = ()) -> np.ndarray:
    array = np.asarray(value)
    scale = int(RUNTIME_COVERAGE_SCALE)
    if array.shape[0] % scale or array.shape[1] % scale:
        raise QualificationError("CAA_REFERENCE_COVERAGE_GRID_SHAPE_INVALID")
    height = array.shape[0] // scale
    width = array.shape[1] // scale
    tail = tuple(array.shape[2:])
    reshaped = array.reshape(height, scale, width, scale, *tail)
    axes = (0, 2, 1, 3) + tuple(range(4, reshaped.ndim))
    return reshaped.transpose(axes)


def render_caa_reference(
    *,
    mesh,
    camera,
    face_uv: np.ndarray,
    texture_rgba_u8: np.ndarray,
    provenance_atlas: np.ndarray,
    face_page_index: np.ndarray | None = None,
    positions=None,
) -> ReferenceCAARender:
    visibility = rasterize_visible_owner(
        mesh,
        camera,
        positions=positions,
        coverage_scale=RUNTIME_COVERAGE_SCALE,
    )
    high_owner = visibility.owner_face_index
    if face_uv.shape != (len(mesh.faces), 3, 2):
        raise QualificationError("CAA_REFERENCE_FACE_UV_MESH_DRIFT")
    texture_value = np.asarray(texture_rgba_u8, dtype=np.uint8)
    provenance_value = np.asarray(provenance_atlas, dtype=np.uint8)
    if texture_value.ndim == 3:
        if provenance_value.ndim != 2 or provenance_value.shape != texture_value.shape[:2]:
            raise QualificationError("CAA_REFERENCE_PROVENANCE_TEXTURE_DRIFT")
        page_by_face = np.zeros((len(mesh.faces),), dtype=np.int32)
        if face_page_index is not None and np.any(
            np.asarray(face_page_index, dtype=np.int32) != 0
        ):
            raise QualificationError("CAA_REFERENCE_SINGLE_PAGE_INDEX_DRIFT")
    elif texture_value.ndim == 4:
        if (
            provenance_value.ndim != 3
            or provenance_value.shape != texture_value.shape[:3]
        ):
            raise QualificationError("CAA_REFERENCE_PROVENANCE_TEXTURE_DRIFT")
        if face_page_index is None:
            raise QualificationError("CAA_REFERENCE_PAGED_FACE_INDEX_REQUIRED")
        page_by_face = np.asarray(face_page_index, dtype=np.int32)
        if (
            page_by_face.shape != (len(mesh.faces),)
            or np.any(page_by_face < 0)
            or np.any(page_by_face >= texture_value.shape[0])
        ):
            raise QualificationError("CAA_REFERENCE_FACE_PAGE_INDEX_INVALID")
    else:
        raise QualificationError("CAA_REFERENCE_TEXTURE_INVALID")

    coverage_height, coverage_width = high_owner.shape
    sample_pm = np.zeros((coverage_height, coverage_width, 4), dtype=np.float64)
    sample_risk = np.zeros((coverage_height, coverage_width), dtype=np.uint8)
    sample_has_provenance = np.zeros(
        (coverage_height, coverage_width),
        dtype=bool,
    )
    sample_has_contribution = np.zeros(
        (coverage_height, coverage_width),
        dtype=bool,
    )
    sample_contribution_mask = np.zeros(
        (
            coverage_height,
            coverage_width,
            visibility.layer_owner_face_index.shape[2],
        ),
        dtype=bool,
    )
    high_layer_owner = visibility.layer_owner_face_index
    high_layer_bary = visibility.layer_barycentric
    layer_count = high_layer_owner.shape[2]

    for layer in range(layer_count):
        layer_mask = high_layer_owner[:, :, layer] >= 0
        ys, xs = np.nonzero(layer_mask)
        if not len(ys):
            continue
        face_index = high_layer_owner[ys, xs, layer].astype(np.int64)
        weights = high_layer_bary[ys, xs, layer].astype(np.float64)
        if not np.isfinite(weights).all():
            raise QualificationError("CAA_REFERENCE_BARYCENTRIC_NONFINITE")
        uv_tri = face_uv[face_index]
        uv = np.sum(uv_tri * weights[:, :, None], axis=1)
        sampled, sampled_provenance = _sample_transport_pages(
            texture_value,
            provenance_value,
            uv,
            page_by_face[face_index],
        )

        prior_provenance = sample_has_provenance[ys, xs]
        sample_risk[ys, xs] = np.where(
            prior_provenance,
            np.maximum(sample_risk[ys, xs], sampled_provenance),
            sampled_provenance,
        )
        sample_has_provenance[ys, xs] = True

        existing_alpha = sample_pm[ys, xs, 3]
        transmission = 1.0 - np.clip(existing_alpha, 0.0, 1.0)
        contribution = sampled * transmission[:, None]
        sample_pm[ys, xs] += contribution
        contributes = contribution[:, 3] > 1.0e-12
        if np.any(contributes):
            cy = ys[contributes]
            cx = xs[contributes]
            codes = sampled_provenance[contributes]
            prior = sample_has_contribution[cy, cx]
            sample_risk[cy, cx] = np.where(
                prior,
                np.maximum(sample_risk[cy, cx], codes),
                codes,
            )
            sample_has_contribution[cy, cx] = True
            sample_contribution_mask[cy, cx, layer] = True

    pm_grid = _coverage_reshape(sample_pm)
    pm = np.mean(pm_grid, axis=(2, 3))
    height, width = pm.shape[:2]

    provenance_grid = _coverage_reshape(sample_has_provenance)
    risk_grid = _coverage_reshape(sample_risk)
    any_provenance = np.any(provenance_grid, axis=(2, 3))
    conservative_risk = np.max(
        np.where(provenance_grid, risk_grid, 0),
        axis=(2, 3),
    )
    provenance = np.full((height, width), 255, dtype=np.uint8)
    provenance[any_provenance] = conservative_risk[any_provenance]

    sample_owner = _coverage_reshape(high_owner).reshape(
        height,
        width,
        RUNTIME_COVERAGE_SAMPLE_COUNT,
    )
    sample_depth = _coverage_reshape(visibility.depth).reshape(
        height,
        width,
        RUNTIME_COVERAGE_SAMPLE_COUNT,
    )
    sample_second_owner = _coverage_reshape(
        visibility.second_owner_face_index
    ).reshape(
        height,
        width,
        RUNTIME_COVERAGE_SAMPLE_COUNT,
    )
    sample_margin = _coverage_reshape(visibility.depth_margin).reshape(
        height,
        width,
        RUNTIME_COVERAGE_SAMPLE_COUNT,
    )
    representative_sample = np.argmin(sample_depth, axis=2)
    owner = np.take_along_axis(
        sample_owner,
        representative_sample[:, :, None],
        axis=2,
    )[:, :, 0]
    second_owner = np.take_along_axis(
        sample_second_owner,
        representative_sample[:, :, None],
        axis=2,
    )[:, :, 0]
    depth_margin = np.take_along_axis(
        sample_margin,
        representative_sample[:, :, None],
        axis=2,
    )[:, :, 0]

    layer_owner_grid = _coverage_reshape(high_layer_owner)
    contribution_grid = _coverage_reshape(sample_contribution_mask)
    flattened_layer_owner = layer_owner_grid.reshape(
        height,
        width,
        RUNTIME_COVERAGE_SAMPLE_COUNT * layer_count,
    )
    flattened_contribution = contribution_grid.reshape(
        height,
        width,
        RUNTIME_COVERAGE_SAMPLE_COUNT * layer_count,
    )
    contribution_count = np.count_nonzero(
        flattened_contribution,
        axis=2,
    ).astype(np.uint8)

    geometry_visible = np.any(sample_owner >= 0, axis=2)
    final_alpha = pm[..., 3] > 1.0e-8

    occupied = high_layer_owner >= 0
    adjacent = occupied[:, :, :-1] & occupied[:, :, 1:]
    depth_delta = np.full(
        visibility.layer_depth[:, :, 1:].shape,
        np.inf,
        dtype=np.float64,
    )
    if np.any(adjacent):
        front = visibility.layer_depth[:, :, :-1]
        back = visibility.layer_depth[:, :, 1:]
        depth_delta[adjacent] = np.abs(
            back[adjacent] - front[adjacent]
        )
    high_exact_depth_ambiguity = np.any(
        adjacent & (depth_delta <= 1.0e-12),
        axis=2,
    )
    exact_depth_ambiguity = np.any(
        _coverage_reshape(high_exact_depth_ambiguity),
        axis=(2, 3),
    )
    layer_overflow = np.any(
        _coverage_reshape(visibility.layer_overflow),
        axis=(2, 3),
    )

    straight = premultiplied_to_straight_u8(pm)
    return ReferenceCAARender(
        premultiplied_rgba=pm,
        straight_rgba_u8=straight,
        geometry_visible=geometry_visible,
        final_alpha=final_alpha,
        owner_face_index=owner,
        second_owner_face_index=second_owner,
        depth_margin=depth_margin,
        exact_depth_ambiguity=exact_depth_ambiguity,
        layer_overflow=layer_overflow,
        contributing_layer_count=contribution_count,
        contributing_layer_mask=flattened_contribution,
        layer_owner_face_index=flattened_layer_owner,
        coverage_sample_owner_face_index=sample_owner,
        coverage_sample_count=RUNTIME_COVERAGE_SAMPLE_COUNT,
        provenance_code=provenance,
    )

def rgba_bytes_sha256(rgba_u8: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(rgba_u8, dtype=np.uint8))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()
