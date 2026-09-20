from __future__ import annotations

"""Reference CAA renderer: visibility and appearance remain separate authorities."""

from dataclasses import dataclass
import hashlib

import numpy as np

from .appearance_bake_v2 import bilinear_premultiplied_rgba
from .types import QualificationError
from .visibility_v2 import VISIBILITY_CONTRACT_V2_HASH, rasterize_visible_owner


@dataclass(frozen=True)
class ReferenceCAARender:
    premultiplied_rgba: np.ndarray
    straight_rgba_u8: np.ndarray
    geometry_visible: np.ndarray
    final_alpha: np.ndarray
    owner_face_index: np.ndarray
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
    value = np.asarray(pm, dtype=np.float64)
    if value.ndim != 3 or value.shape[2] != 4:
        raise QualificationError("CAA_REFERENCE_PM_RGBA_SHAPE_INVALID")
    alpha = np.clip(value[..., 3:4], 0.0, 1.0)
    rgb = np.zeros_like(value[..., :3])
    nonzero = alpha[..., 0] > 1e-12
    if np.any(nonzero):
        rgb[nonzero] = np.clip(
            value[..., :3][nonzero] / alpha[nonzero],
            0.0,
            1.0,
        )
    straight = np.concatenate((rgb, alpha), axis=2)
    return np.clip(np.floor(straight * 255.0 + 0.5), 0, 255).astype(np.uint8)


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
    bary = visibility.barycentric
    height, width = owner.shape
    if face_uv.shape != (len(mesh.faces), 3, 2):
        raise QualificationError("CAA_REFERENCE_FACE_UV_MESH_DRIFT")
    if provenance_atlas.shape[:2] != texture_rgba_u8.shape[:2]:
        raise QualificationError("CAA_REFERENCE_PROVENANCE_TEXTURE_DRIFT")

    pm = np.zeros((height, width, 4), dtype=np.float32)
    provenance = np.full((height, width), 255, dtype=np.uint8)
    visible_y, visible_x = np.nonzero(owner >= 0)
    if len(visible_y):
        face_index = owner[visible_y, visible_x].astype(np.int64)
        weights = bary[visible_y, visible_x].astype(np.float64)
        if not np.isfinite(weights).all():
            raise QualificationError("CAA_REFERENCE_BARYCENTRIC_NONFINITE")
        uv_tri = face_uv[face_index]
        uv = np.sum(uv_tri * weights[:, :, None], axis=1)
        sampled = bilinear_premultiplied_rgba(texture_rgba_u8, uv)
        pm[visible_y, visible_x] = sampled.astype(np.float32)
        provenance[visible_y, visible_x] = _sample_nearest_scalar(
            provenance_atlas, uv
        ).astype(np.uint8)

    straight = premultiplied_to_straight_u8(pm)
    geometry_visible = owner >= 0
    final_alpha = pm[..., 3] > 1e-8
    return ReferenceCAARender(
        premultiplied_rgba=pm,
        straight_rgba_u8=straight,
        geometry_visible=geometry_visible,
        final_alpha=final_alpha,
        owner_face_index=owner,
        provenance_code=provenance,
    )


def rgba_bytes_sha256(rgba_u8: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(rgba_u8, dtype=np.uint8))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()
