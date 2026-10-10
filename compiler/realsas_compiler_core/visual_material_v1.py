from __future__ import annotations

"""Sealed source-raster CAA material, never a runtime appearance producer.

The support is the declared visual raster domain. It does not certify material
under occlusions or infer a part label from a mechanical skin weight.
"""

import hashlib
from pathlib import Path

import numpy as np

from .types import QualificationError

SAMPLING_CONTRACT = "CAA_RGBA8_LINEAR_PM_BILINEAR_VISUAL_V1"
MATERIAL_CONTRACT = "CAA_VISUAL_TEXEL_PROVENANCE_AND_SOURCE_VIEW_V1"
DEPTH_CONTRACT = "CANONICAL_CAMERA_DEPTH_ASCENDING__UNRESOLVED_TIES_FAIL_V1"
DEPTH_TIE_EPSILON = 1e-12
MAXIMUM_FRAGMENT_LAYERS = 4
PADDING_SOURCE_VIEW = np.iinfo(np.int16).min


def source_visual_material(rgba, foreground, view_index):
    image = np.asarray(rgba)
    mask = np.asarray(foreground, dtype=bool)
    if image.dtype != np.uint8 or image.shape != (*mask.shape, 4):
        raise QualificationError("VISUAL_MATERIAL_SOURCE_SHAPE_INVALID")
    # A foreground mask cannot turn an absent source sample into observed art.
    # Completion needs its own CAA compile/provenance/quality evidence.
    if np.any(mask & (image[..., 3] == 0)):
        raise QualificationError("VISUAL_MATERIAL_FOREGROUND_WITHOUT_APPEARANCE")
    result = image.copy()
    result[~mask] = 0
    provenance = np.where(mask, 0, 255).astype(np.uint8)
    source_view = np.where(mask, int(view_index), PADDING_SOURCE_VIEW).astype(np.int16)
    return result, provenance, source_view


def validate_visual_material(provenance, source_view, *, view_index, rgba=None):
    p, s = np.asarray(provenance), np.asarray(source_view)
    if p.ndim != 2 or s.shape != p.shape or p.dtype != np.uint8 or s.dtype != np.int16:
        raise QualificationError("VISUAL_MATERIAL_ARRAY_INVALID")
    valid = ((p == 0) & (s == int(view_index)))
    valid |= (p == 1) & (s >= 0) & (s < 8) & (s != int(view_index))
    valid |= (p == 2) & (s == -2)
    valid |= (p == 3) & (s == -4)
    valid |= (p == 4) & (s == -3)
    valid |= (p == 255) & (s == PADDING_SOURCE_VIEW)
    if not np.all(valid):
        raise QualificationError("VISUAL_MATERIAL_PROVENANCE_SOURCE_VIEW_DRIFT")
    if rgba is not None:
        image = np.asarray(rgba)
        if image.dtype != np.uint8 or image.shape != (*p.shape, 4):
            raise QualificationError("VISUAL_MATERIAL_TEXTURE_SHAPE_INVALID")
        if np.any(image[p == 255] != 0) or np.any(image[p == 3] != 0):
            raise QualificationError("VISUAL_MATERIAL_UNSUPPORTED_OR_PADDING_RGBA")


def load_visual_material(asset, *, view_index, rgba):
    path = Path(asset.provenance_npz_path).expanduser().resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != asset.provenance_npz_sha256:
        raise QualificationError("VISUAL_MATERIAL_PROVENANCE_BYTES_DRIFT")
    with np.load(path, allow_pickle=False) as data:
        pk, sk = f"view_{view_index}_provenance", f"view_{view_index}_source_view"
        if pk not in data.files or sk not in data.files:
            raise QualificationError("VISUAL_MATERIAL_PER_TEXEL_AUTHORITY_REQUIRED")
        p, s = data[pk].copy(), data[sk].copy()
    validate_visual_material(p, s, view_index=view_index, rgba=rgba)
    return p, s


def bilinear_material_samples(provenance, source_view, texture, *, x0, y0, x1, y1, tx, ty):
    """Conservative footprint provenance; zero-alpha padding is storage only.

    Unsupported support fails even when its transport alpha is zero. Mixing
    admitted classes retains the greatest provenance code and its donor identity;
    a tiny completion contribution cannot be relabelled as direct source.
    """
    shape = np.asarray(x0).shape
    code = np.full(shape, 255, dtype=np.uint8)
    donor = np.full(shape, PADDING_SOURCE_VIEW, dtype=np.int16)
    unsupported = np.zeros(shape, dtype=bool)
    for x, y, weight in ((x0, y0, (1-tx)*(1-ty)), (x1, y0, tx*(1-ty)),
                         (x0, y1, (1-tx)*ty), (x1, y1, tx*ty)):
        p, s = provenance[y, x], source_view[y, x]
        active = weight > 1e-12
        unsupported |= active & (p == 3)
        if np.any(active & (p == 255) & (texture[y, x, 3] > 0)):
            raise QualificationError("VISUAL_MATERIAL_VISIBLE_PADDING")
        take = active & (p != 255) & ((code == 255) | (p > code))
        code[take], donor[take] = p[take], s[take]
    code[unsupported], donor[unsupported] = 3, -4
    return code, donor
