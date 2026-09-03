"""Compiler-owned deterministic substrate layer.

Canonical implementations for evidence-to-surface construction and local geometry
live here. The IRIS V2 adapter is intentionally imported from its submodule rather
than eagerly from this package to avoid a compatibility-path import cycle while
legacy flat module paths are retained.
"""

from .surface import build_surface_from_persistence, rigging_surface_from_d2_arrays
from .local_geometry import (
    DTB_ND1_HISTORICAL_OPERATOR_BLOB_SHA,
    DTB_ND1_OPERATOR_ID,
    attach_dtb_nd1_normals,
    dtb_nd1_operator_hash,
    dtb_nd1_operator_identity,
    orient_normal_against_ray,
    robust_local_plane_normals,
    robust_local_plane_normals_for_rows,
)

__all__ = [
    "build_surface_from_persistence",
    "rigging_surface_from_d2_arrays",
    "DTB_ND1_HISTORICAL_OPERATOR_BLOB_SHA",
    "DTB_ND1_OPERATOR_ID",
    "attach_dtb_nd1_normals",
    "dtb_nd1_operator_hash",
    "dtb_nd1_operator_identity",
    "orient_normal_against_ray",
    "robust_local_plane_normals",
    "robust_local_plane_normals_for_rows",
]
