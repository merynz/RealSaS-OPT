from __future__ import annotations

"""Transient numeric-compatibility view for sealed AXIS/Geppetto preprocessing.

The frozen AXIS V5.4.1 tensorizer predates the current product surface semantic
profile and derives its observed/completed masks from two historical validity
flag names. Those strings are *not* learned feature dimensions, and relation
kind strings are certificate-only. This adapter therefore adds only the legacy
mask aliases to a transient in-memory view while preserving every geometric,
view, graph and lineage value from the canonical product-surface evidence.

The compatibility view is never persisted and never acquires signed-zero or
canonical authority. The frozen model source stays byte-identical.
"""

from dataclasses import replace
from typing import Any

from .product_surface_court_v1 import (
    PRODUCT_SURFACE_COMPLETED_V1,
    PRODUCT_SURFACE_OBSERVED_V1,
    PRODUCT_SURFACE_PROFILE_V1,
    PRODUCT_SURFACE_RELATION_V1,
)
from .types import QualificationError, RiggingSurfaceIR


LEGACY_OBSERVED_ALIAS = "OBSERVED_SIGNED_ZERO_SURFACE"
LEGACY_COMPLETED_ALIAS = "MODEL_COMPLETED_SIGNED_ZERO_SURFACE"


def axis_numeric_compatibility_surface_v1(surface: RiggingSurfaceIR) -> RiggingSurfaceIR:
    if not isinstance(surface, RiggingSurfaceIR):
        raise TypeError("expected RiggingSurfaceIR")
    metadata = dict(surface.metadata or {})
    if metadata.get("surface_evidence_profile") != PRODUCT_SURFACE_PROFILE_V1:
        raise QualificationError("AXIS_COMPAT_PRODUCT_SURFACE_PROFILE_REQUIRED")
    if metadata.get("scene_first_signed_geometry") is not False:
        raise QualificationError("AXIS_COMPAT_SIGNED_GEOMETRY_AUTHORITY_FORBIDDEN")
    if metadata.get("signed_zero_surface_claimed") is not False:
        raise QualificationError("AXIS_COMPAT_SIGNED_ZERO_AUTHORITY_FORBIDDEN")
    if metadata.get("canonical_mesh_authority") is not False:
        raise QualificationError("AXIS_COMPAT_MESH_AUTHORITY_FORBIDDEN")
    if str(metadata.get("topology_neighborhood_authority") or "") != "DECODED_TESSA_FACE_INCIDENCE_ONLY":
        raise QualificationError("AXIS_COMPAT_TOPOLOGY_AUTHORITY_DRIFT")
    if str(metadata.get("normal_orientation_hint_role") or "") != "SIGN_ONLY_NOT_NEIGHBORHOOD_AUTHORITY":
        raise QualificationError("AXIS_COMPAT_NORMAL_HINT_ROLE_DRIFT")
    if not surface.geometry_lineage_hash:
        raise QualificationError("AXIS_COMPAT_SURFACE_LINEAGE_REQUIRED")

    nodes = []
    for node in surface.surface_nodes:
        flags = set(map(str, node.validity_flags))
        observed = PRODUCT_SURFACE_OBSERVED_V1 in flags
        completed = PRODUCT_SURFACE_COMPLETED_V1 in flags
        if observed == completed:
            raise QualificationError("AXIS_COMPAT_PRODUCT_OBSERVED_COMPLETED_INVALID")
        if observed != bool(node.support_views):
            raise QualificationError("AXIS_COMPAT_PRODUCT_SUPPORT_MASK_DRIFT")
        if LEGACY_OBSERVED_ALIAS in flags or LEGACY_COMPLETED_ALIAS in flags:
            raise QualificationError("AXIS_COMPAT_LEGACY_ALIAS_ALREADY_PERSISTED")
        flags.add(LEGACY_OBSERVED_ALIAS if observed else LEGACY_COMPLETED_ALIAS)
        nodes.append(
            replace(
                node,
                validity_flags=tuple(sorted(flags)),
                metadata={
                    **dict(node.metadata or {}),
                    "axis_numeric_compatibility_alias_added": True,
                    "axis_numeric_compatibility_alias_is_not_authority": True,
                },
            )
        )

    if not surface.local_relations:
        raise QualificationError("AXIS_COMPAT_LOCAL_RELATIONS_REQUIRED")
    if any(str(rel.relation_kind) != PRODUCT_SURFACE_RELATION_V1 for rel in surface.local_relations):
        raise QualificationError("AXIS_COMPAT_PRODUCT_RELATION_KIND_DRIFT")

    return replace(
        surface,
        surface_nodes=tuple(nodes),
        metadata={
            **metadata,
            "axis_numeric_compatibility_view": True,
            "axis_numeric_compatibility_view_persisted": False,
            "axis_numeric_compatibility_aliases": [
                LEGACY_OBSERVED_ALIAS,
                LEGACY_COMPLETED_ALIAS,
            ],
            "axis_numeric_compatibility_aliases_are_not_authority": True,
            "frozen_tensorizer_require_scene_first": False,
            "source_product_surface_geometry_lineage_hash": surface.geometry_lineage_hash,
        },
    )


def assert_axis_tensor_numeric_binding_v1(
    *,
    product_surface: RiggingSurfaceIR,
    compatibility_surface: RiggingSurfaceIR,
    tensor: Any,
) -> None:
    if compatibility_surface.geometry_lineage_hash != product_surface.geometry_lineage_hash:
        raise QualificationError("AXIS_COMPAT_SURFACE_LINEAGE_DRIFT")
    if str(tensor.source_surface_hash) != product_surface.geometry_lineage_hash:
        raise QualificationError("AXIS_COMPAT_TENSOR_SOURCE_BINDING_DRIFT")
    if tuple(tensor.surface_ids) != tuple(
        node.surface_id for node in sorted(product_surface.surface_nodes, key=lambda value: str(value.surface_id))
    ):
        raise QualificationError("AXIS_COMPAT_TENSOR_SURFACE_ID_DRIFT")
    observed_expected = tuple(
        PRODUCT_SURFACE_OBSERVED_V1 in set(map(str, node.validity_flags))
        for node in sorted(product_surface.surface_nodes, key=lambda value: str(value.surface_id))
    )
    completed_expected = tuple(not value for value in observed_expected)
    if tuple(bool(value) for value in tensor.observed.tolist()) != observed_expected:
        raise QualificationError("AXIS_COMPAT_TENSOR_OBSERVED_DRIFT")
    if tuple(bool(value) for value in tensor.completed.tolist()) != completed_expected:
        raise QualificationError("AXIS_COMPAT_TENSOR_COMPLETED_DRIFT")


__all__ = [
    "LEGACY_OBSERVED_ALIAS",
    "LEGACY_COMPLETED_ALIAS",
    "axis_numeric_compatibility_surface_v1",
    "assert_axis_tensor_numeric_binding_v1",
]
