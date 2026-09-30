from __future__ import annotations

"""Narrow geometry-domain JSON codec for the RealSaS V2 geometry lane.

This module intentionally excludes motion, appearance, product-state and
presentation imports. Geometry-stage implementation identities must not become
stale when semantically unrelated downstream domains change.
"""

from typing import Any

from .camera_authority_v1 import QualifiedCameraSetIR
from .camera_geometry_v2 import CameraProjectionV3
from .observation_authority_v1 import (
    QualifiedObservationSetIR,
    QualifiedObservationViewIR,
)
from .types import (
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)

Json = dict[str, Any]


def _schema(payload: Json, expected: str) -> None:
    actual = str(payload.get("schema_version") or payload.get("schema") or "")
    if actual != expected:
        raise ValueError(f"V2_ARTIFACT_SCHEMA_MISMATCH:{actual}!={expected}")


def qualified_camera_set_from_dict(payload: Json) -> QualifiedCameraSetIR:
    _schema(payload, "RealSaS.QualifiedCameraSetIR.v1")
    cameras = tuple(
        CameraProjectionV3(
            view_id=str(row["view_id"]),
            view_index=int(row["view_index"]),
            origin=tuple(map(float, row["origin"])),
            right=tuple(map(float, row["right"])),
            screen_up=tuple(map(float, row["screen_up"])),
            forward=tuple(map(float, row["forward"])),
            half_extent=float(row["half_extent"]),
            resolution=int(row["resolution"]),
            schema_version=str(
                row.get("schema_version")
                or "RealSaS.FullSurfaceCameraProjection.v3"
            ),
        )
        for row in (payload.get("cameras") or ())
    )
    return QualifiedCameraSetIR(
        cameras=cameras,
        camera_binding_hashes=tuple(
            map(str, payload.get("camera_binding_hashes") or ())
        ),
        source_bundle_sha256=str(payload["source_bundle_sha256"]),
        camera_set_hash=str(payload["camera_set_hash"]),
        schema_version=str(
            payload.get("schema_version") or "RealSaS.QualifiedCameraSetIR.v1"
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def qualified_observation_set_from_dict(payload: Json) -> QualifiedObservationSetIR:
    _schema(payload, "RealSaS.QualifiedObservationSetIR.v1")
    return QualifiedObservationSetIR(
        views=tuple(
            QualifiedObservationViewIR(
                view_index=int(row["view_index"]),
                width=int(row["width"]),
                height=int(row["height"]),
                source_observation_hash=str(row["source_observation_hash"]),
                source_raster_sha256=str(row["source_raster_sha256"]),
                foreground_mask_sha256=str(row["foreground_mask_sha256"]),
                camera_binding_hash=str(row["camera_binding_hash"]),
                qualification_state=str(row["qualification_state"]),
                evidence_refs=tuple(map(str, row.get("evidence_refs") or ())),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in (payload.get("views") or ())
        ),
        observation_set_hash=str(payload["observation_set_hash"]),
        schema_version=str(
            payload.get("schema_version")
            or "RealSaS.QualifiedObservationSetIR.v1"
        ),
        metadata=dict(payload.get("metadata") or {}),
    )


def rigging_surface_from_dict(payload: Json) -> RiggingSurfaceIR:
    _schema(payload, "RealSaS.RiggingSurfaceIR.v1")
    nodes = tuple(
        SurfaceNode(
            surface_id=str(row["surface_id"]),
            P=tuple(map(float, row["P"])),
            support_views=tuple(map(int, row.get("support_views") or ())),
            provenance_refs=tuple(map(str, row.get("provenance_refs") or ())),
            source_observation_ids=tuple(
                map(str, row.get("source_observation_ids") or ())
            ),
            raster_bindings=tuple(
                (int(vi), tuple(map(float, xy)))
                for vi, xy in (row.get("raster_bindings") or ())
            ),
            persistence_group_id=(
                None
                if row.get("persistence_group_id") is None
                else str(row["persistence_group_id"])
            ),
            derived_normal=(
                None
                if row.get("derived_normal") is None
                else tuple(map(float, row["derived_normal"]))
            ),
            validity_flags=tuple(map(str, row.get("validity_flags") or ())),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in (payload.get("surface_nodes") or ())
    )
    relations = tuple(
        SurfaceRelation(
            relation_id=str(row["relation_id"]),
            a_surface_id=str(row["a_surface_id"]),
            b_surface_id=str(row["b_surface_id"]),
            relation_kind=str(row["relation_kind"]),
            score=float(row.get("score", 1.0)),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in (payload.get("local_relations") or ())
    )
    return RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=relations,
        geometry_lineage_hash=str(payload.get("geometry_lineage_hash") or ""),
        builder_id=str(
            payload.get("builder_id")
            or "RealSaS.GeometricSubstrateAssembler.current"
        ),
        schema_version=str(
            payload.get("schema_version") or "RealSaS.RiggingSurfaceIR.v1"
        ),
        metadata=dict(payload.get("metadata") or {}),
    )
