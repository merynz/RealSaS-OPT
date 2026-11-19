from __future__ import annotations

"""Immutable discrete address of the ONE seeded TESSA M*.

A supervision receipt must bind to the exact ordered indexed topology and
original-M0 parent provenance, not just face count or a source-rig name.
Live rest XYZ, IRIS H, predicted normals and per-step scene features cannot
be included here: their learnable values must change during optimization.

This is a *topology address*, not source-3D or semantic correspondence truth.
Independent original-source native-face + barycentric witnesses remain
mandatory for per-vertex skin or posed-surface supervision.
"""

from hashlib import sha256
import json

from .native_face_layer_truth_v1 import tensor_sha256_v1
from .tessa_seeded_surface_v1 import TESSASeededSurfaceV1

SCHEMA = "RealSaS.RSSvNext.SeededMStarDiscreteTopologyAddress.v1"


def seeded_mstar_teacher_address_sha256_v1(seed: TESSASeededSurfaceV1) -> str:
    if not isinstance(seed, TESSASeededSurfaceV1):
        raise ValueError("RSS_VNEXT_MSTAR_SOURCE_TOPOLOGY_REQUIRED")
    seed.validate()
    # The original seed face order and every TESSA midpoint's original M0
    # ancestry are part of the address. Excluding them would allow a new M*
    # with identical face counts but different vertex meaning to reuse truth.
    data = {
        "schema": SCHEMA,
        "source_authority": seed.source_authority,
        "original_vertex_count": seed.original_vertex_count,
        "original_face_count": seed.original_face_count,
        "final_vertex_count": int(seed.xyz.shape[1]),
        "final_face_count": int(seed.faces.shape[0]),
        "original_faces_sha256": tensor_sha256_v1(seed.original_faces),
        "faces_sha256": tensor_sha256_v1(seed.faces),
        "vertex_component_ids_sha256": tensor_sha256_v1(seed.vertex_component_ids),
        "face_component_ids_sha256": tensor_sha256_v1(seed.face_component_ids),
        "parent_seed_edges": [[int(a), int(b)] for a, b in seed.parent_seed_edges],
        "parent_seed_weights": [[float(a), float(b)] for a, b in seed.parent_seed_weights],
        "vertex_layer_ids_sha256": (
            tensor_sha256_v1(seed.vertex_layer_ids)
            if seed.vertex_layer_ids is not None else None
        ),
        "layer_names": list(seed.layer_names),
        "layer_roles": [list(x) for x in seed.layer_roles],
        "layer_namespace_sha256": seed.layer_namespace_sha256,
    }
    encoded = json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def require_qualified_mstar_teacher_address_v1(
    *,
    transport_address_sha256: str,
    active_mstar_address_sha256: str,
) -> None:
    """Hard no-fallback equality, before any source label reaches the loss."""
    valid = lambda x: (
        type(x) is str and len(x) == 64
        and all(ch in "0123456789abcdef" for ch in x)
    )
    if (
        not valid(transport_address_sha256)
        or not valid(active_mstar_address_sha256)
        or transport_address_sha256 != active_mstar_address_sha256
    ):
        raise ValueError("RSS_VNEXT_MSTAR_TEACHER_TOPOLOGY_ADDRESS_DRIFT")


__all__ = [
    "SCHEMA",
    "seeded_mstar_teacher_address_sha256_v1",
    "require_qualified_mstar_teacher_address_v1",
]
