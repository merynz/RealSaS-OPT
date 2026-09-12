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
from .scene_first_signed import (
    ZERO_SURFACE_NORMAL_OPERATOR_ID,
    ZERO_SURFACE_COMPACTOR_ID,
    ZERO_SURFACE_VISIBILITY_ID,
    ZERO_SURFACE_OBSERVATION_SUPPORT_ID,
    zero_surface_normal_operator_identity_v1,
    zero_surface_normal_operator_hash_v1,
    robust_zero_surface_normals_v1,
    rigging_surface_from_scene_first_zero_mesh_v1,
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
    "ZERO_SURFACE_NORMAL_OPERATOR_ID",
    "ZERO_SURFACE_COMPACTOR_ID",
    "ZERO_SURFACE_VISIBILITY_ID",
    "ZERO_SURFACE_OBSERVATION_SUPPORT_ID",
    "zero_surface_normal_operator_identity_v1",
    "zero_surface_normal_operator_hash_v1",
    "robust_zero_surface_normals_v1",
    "rigging_surface_from_scene_first_zero_mesh_v1",
]
