from __future__ import annotations

"""First-class seam between rigging control substrate and drawable surface.

RiggingSurfaceIR remains the mechanical/control authority. DrawableSurfaceIR is the
conditioned render/deformation carrier derived from the same qualified dense surface
lineage. DrawableSupportBindingIR explicitly records how every drawable vertex is
supported by the rigging substrate.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError


DRAWABLE_SURFACE_SCHEMA = "RealSaS.DrawableSurfaceIR.v1"
DRAWABLE_SUPPORT_BINDING_SCHEMA = "RealSaS.DrawableSupportBindingIR.v1"


@dataclass(frozen=True)
class DrawableSurfaceIR:
    vertices: np.ndarray
    faces: np.ndarray
    source_dense_lineage_hash: str
    conditioning_contract_hash: str
    lineage_hash: str
    schema_version: str = DRAWABLE_SURFACE_SCHEMA


@dataclass(frozen=True)
class DrawableSupportBindingIR:
    drawable_lineage_hash: str
    rigging_lineage_hash: str
    support_ids: np.ndarray
    coeffs: np.ndarray
    locality_radius: float
    binding_hash: str
    schema_version: str = DRAWABLE_SUPPORT_BINDING_SCHEMA


def qualify_drawable_surface_v1(
    vertices,
    faces,
    *,
    source_dense_lineage_hash: str,
    conditioning_contract_hash: str,
) -> DrawableSurfaceIR:
    p = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise QualificationError("DRAWABLE_SURFACE_VERTICES_INVALID")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) < 1:
        raise QualificationError("DRAWABLE_SURFACE_FACES_INVALID")
    if np.any(f < 0) or np.any(f >= len(p)):
        raise QualificationError("DRAWABLE_SURFACE_FACE_INDEX_OOB")
    if np.any(f[:, 0] == f[:, 1]) or np.any(f[:, 1] == f[:, 2]) or np.any(f[:, 0] == f[:, 2]):
        raise QualificationError("DRAWABLE_SURFACE_DEGENERATE_FACE_INDEX")
    tri = p[f]
    double_area = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    if np.any(double_area <= 1.0e-12):
        raise QualificationError("DRAWABLE_SURFACE_ZERO_AREA_FACE")
    if len(source_dense_lineage_hash) != 64 or len(conditioning_contract_hash) != 64:
        raise QualificationError("DRAWABLE_SURFACE_HASH_IDENTITY_INVALID")
    payload = {
        "schema": DRAWABLE_SURFACE_SCHEMA,
        "source_dense_lineage_hash": source_dense_lineage_hash,
        "conditioning_contract_hash": conditioning_contract_hash,
        "vertex_count": int(len(p)),
        "face_count": int(len(f)),
        "vertices": p.tolist(),
        "faces": f.tolist(),
    }
    return DrawableSurfaceIR(
        vertices=p,
        faces=f,
        source_dense_lineage_hash=source_dense_lineage_hash,
        conditioning_contract_hash=conditioning_contract_hash,
        lineage_hash=content_sha256(payload),
    )


def qualify_drawable_support_binding_v1(
    *,
    drawable: DrawableSurfaceIR,
    rigging_lineage_hash: str,
    support_ids,
    coeffs,
    rigging_positions=None,
    locality_radius: float,
    tolerance: float = 1.0e-8,
) -> DrawableSupportBindingIR:
    ids = np.asarray(support_ids, dtype=np.int64)
    a = np.asarray(coeffs, dtype=np.float64)
    if len(rigging_lineage_hash) != 64:
        raise QualificationError("DRAWABLE_BINDING_RIGGING_HASH_INVALID")
    if ids.ndim != 2 or a.shape != ids.shape or ids.shape[0] != len(drawable.vertices):
        raise QualificationError("DRAWABLE_BINDING_SHAPE_INVALID")
    if ids.shape[1] < 1:
        raise QualificationError("DRAWABLE_BINDING_SUPPORT_SET_EMPTY")
    if not np.isfinite(a).all() or np.any(a < -float(tolerance)):
        raise QualificationError("DRAWABLE_BINDING_COEFF_INVALID")
    if not np.allclose(a.sum(axis=1), 1.0, atol=float(tolerance), rtol=0.0):
        raise QualificationError("DRAWABLE_BINDING_PARTITION_OF_UNITY_FAIL")
    if not np.isfinite(float(locality_radius)) or float(locality_radius) <= 0.0:
        raise QualificationError("DRAWABLE_BINDING_LOCALITY_RADIUS_INVALID")

    rigging_count = None
    if rigging_positions is not None:
        s = np.asarray(rigging_positions, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != 3 or not np.isfinite(s).all():
            raise QualificationError("DRAWABLE_BINDING_RIGGING_POSITIONS_INVALID")
        rigging_count = len(s)
        if np.any(ids < 0) or np.any(ids >= rigging_count):
            raise QualificationError("DRAWABLE_BINDING_SUPPORT_ID_OOB")
        support = s[ids]
        distance = np.linalg.norm(
            support - drawable.vertices[:, None, :],
            axis=2,
        )
        active = a > float(tolerance)
        if np.any(distance[active] > float(locality_radius) + float(tolerance)):
            raise QualificationError("DRAWABLE_BINDING_LOCALITY_FAIL")
    elif np.any(ids < 0):
        raise QualificationError("DRAWABLE_BINDING_SUPPORT_ID_NEGATIVE")

    payload = {
        "schema": DRAWABLE_SUPPORT_BINDING_SCHEMA,
        "drawable_lineage_hash": drawable.lineage_hash,
        "rigging_lineage_hash": rigging_lineage_hash,
        "support_ids": ids.tolist(),
        "coeffs": a.tolist(),
        "locality_radius": float(locality_radius),
        "rigging_count": rigging_count,
    }
    return DrawableSupportBindingIR(
        drawable_lineage_hash=drawable.lineage_hash,
        rigging_lineage_hash=rigging_lineage_hash,
        support_ids=ids,
        coeffs=a,
        locality_radius=float(locality_radius),
        binding_hash=content_sha256(payload),
    )


def transfer_skin_via_support_v1(
    rigging_weights,
    binding: DrawableSupportBindingIR,
    *,
    tolerance: float = 1.0e-8,
) -> np.ndarray:
    w = np.asarray(rigging_weights, dtype=np.float64)
    if w.ndim != 2 or not np.isfinite(w).all() or np.any(w < -float(tolerance)):
        raise QualificationError("DRAWABLE_SKIN_RIGGING_WEIGHTS_INVALID")
    if np.any(binding.support_ids >= len(w)):
        raise QualificationError("DRAWABLE_SKIN_SUPPORT_ID_OOB")
    if not np.allclose(w.sum(axis=1), 1.0, atol=float(tolerance), rtol=0.0):
        raise QualificationError("DRAWABLE_SKIN_RIGGING_ROWS_NOT_SIMPLEX")
    gathered = w[binding.support_ids]
    out = np.einsum("vk,vkj->vj", binding.coeffs, gathered, optimize=True)
    out = np.maximum(out, 0.0)
    row_sum = out.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 1.0e-12):
        raise QualificationError("DRAWABLE_SKIN_ZERO_ROW")
    out /= row_sum
    if not np.allclose(out.sum(axis=1), 1.0, atol=float(tolerance), rtol=0.0):
        raise QualificationError("DRAWABLE_SKIN_TRANSFER_SIMPLEX_FAIL")
    return out
