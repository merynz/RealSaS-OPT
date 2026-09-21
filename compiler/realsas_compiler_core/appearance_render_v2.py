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


def load_provenance_atlas(asset) -> np.ndarray:
    with np.load(str(asset.provenance_npz_path), allow_pickle=False) as data:
        if "provenance" not in data.files:
            raise QualificationError("CAA_REFERENCE_PROVENANCE_MISSING")
        value = np.asarray(data["provenance"], dtype=np.uint8)
    if value.ndim != 3 or value.shape[0] != 8:
        raise QualificationError("CAA_REFERENCE_PROVENANCE_SHAPE_INVALID")
    return value


def _sample_nearest_scalar(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    points = np.asarray(uv, dtype=np.float64)
    h, w = source.shape[:2]
    x = np.rint(np.clip(points[:, 0], 0.0, 1.0) * float(w - 1)).astype(np.int64)
    y = np.rint(np.clip(points[:, 1], 0.0, 1.0) * float(h - 1)).astype(np.int64)
    return source[y, x]


def premultiplied_to_straight_u8(pm: np.ndarray) -> np.ndarray:
    """Compatibility wrapper: linear PM -> straight sRGB RGBA8."""
    value = np.asarray(pm, dtype=np.float64)
    if value.ndim != 3 or value.shape[2] != 4:
        raise QualificationError("CAA_REFERENCE_PM_RGBA_SHAPE_INVALID")
    return premultiplied_linear_to_straight_srgb_u8(value)

def render_caa_reference(
    *,
    mesh,
    camera,
    face_uv: np.ndarray,
    texture_rgba_u8: np.ndarray,
    provenance_atlas: np.ndarray,
    positions=None,
) -> ReferenceCAARender:
    visibility = rasterize_visible_owner(mesh, camera, positions=positions)
    owner = visibility.owner_face_index
    height, width = owner.shape
    if face_uv.shape != (len(mesh.faces), 3, 2):
        raise QualificationError("CAA_REFERENCE_FACE_UV_MESH_DRIFT")
    if provenance_atlas.shape[:2] != texture_rgba_u8.shape[:2]:
        raise QualificationError("CAA_REFERENCE_PROVENANCE_TEXTURE_DRIFT")

    pm = np.zeros((height, width, 4), dtype=np.float64)
    risk = np.zeros((height, width), dtype=np.uint8)
    has_contribution = np.zeros((height, width), dtype=bool)
    contribution_count = np.zeros((height, width), dtype=np.uint8)
    layer_owner = visibility.layer_owner_face_index
    layer_bary = visibility.layer_barycentric
    layer_count = layer_owner.shape[2]

    for layer in range(layer_count):
        layer_mask = layer_owner[:, :, layer] >= 0
        ys, xs = np.nonzero(layer_mask)
        if not len(ys):
            continue
        face_index = layer_owner[ys, xs, layer].astype(np.int64)
        weights = layer_bary[ys, xs, layer].astype(np.float64)
        if not np.isfinite(weights).all():
            raise QualificationError("CAA_REFERENCE_BARYCENTRIC_NONFINITE")
        uv_tri = face_uv[face_index]
        uv = np.sum(uv_tri * weights[:, :, None], axis=1)
        sampled = bilinear_premultiplied_rgba(texture_rgba_u8, uv)
        sampled_provenance = conservative_bilinear_provenance(
            provenance_atlas, uv
        ).astype(np.uint8)

        existing_alpha = pm[ys, xs, 3]
        transmission = 1.0 - np.clip(existing_alpha, 0.0, 1.0)
        contribution = sampled * transmission[:, None]
        pm[ys, xs] += contribution
        contributes = contribution[:, 3] > 1.0e-12
        if np.any(contributes):
            cy = ys[contributes]
            cx = xs[contributes]
            codes = sampled_provenance[contributes]
            prior = has_contribution[cy, cx]
            risk[cy, cx] = np.where(
                prior,
                np.maximum(risk[cy, cx], codes),
                codes,
            )
            has_contribution[cy, cx] = True
            contribution_count[cy, cx] = np.minimum(
                255,
                contribution_count[cy, cx].astype(np.uint16) + 1,
            ).astype(np.uint8)

    provenance = np.full((height, width), 255, dtype=np.uint8)
    provenance[has_contribution] = risk[has_contribution]
    straight = premultiplied_to_straight_u8(pm)
    geometry_visible = np.any(layer_owner >= 0, axis=2)
    final_alpha = pm[..., 3] > 1.0e-8

    occupied = layer_owner >= 0
    adjacent = occupied[:, :, :-1] & occupied[:, :, 1:]
    depth_delta = np.abs(
        visibility.layer_depth[:, :, 1:]
        - visibility.layer_depth[:, :, :-1]
    )
    exact_depth_ambiguity = np.any(
        adjacent & (depth_delta <= 1.0e-12),
        axis=2,
    )

    return ReferenceCAARender(
        premultiplied_rgba=pm,
        straight_rgba_u8=straight,
        geometry_visible=geometry_visible,
        final_alpha=final_alpha,
        owner_face_index=owner,
        second_owner_face_index=visibility.second_owner_face_index,
        depth_margin=visibility.depth_margin,
        exact_depth_ambiguity=exact_depth_ambiguity,
        layer_overflow=visibility.layer_overflow.copy(),
        contributing_layer_count=contribution_count,
        provenance_code=provenance,
    )

def rgba_bytes_sha256(rgba_u8: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(rgba_u8, dtype=np.uint8))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()
