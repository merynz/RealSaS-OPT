from __future__ import annotations

"""Independent typed reopening of immutable compiled Graphics in Proof.

The exact CompiledGraphics.v1 nested structure is reconstructed from CAS JSON,
then its graph hash and full face/fragment/runtime-address domain are checked.
Hash-consistent but invalid face/order/attachment addresses are still rejected.
"""

from typing import Any, Mapping

from .graphics_v1 import (
    CompiledGraphicsV1, GraphicsFragmentV1, GraphicsOrderEdgeV1,
    GraphicsViewStateV1, GraphicsAttachmentV1,
    compiled_graphics_hash_v1,
)
from .graphics_runtime_v1 import build_runtime_graphics_v1
from .types import QualificationError


def reopen_compiled_graphics_v1(
    payload: Mapping[str, Any], *, face_count: int,
):
    if (
        type(payload) is not dict
        or payload.get("schema_version") != "RealSaS.CompiledGraphics.v1"
        or type(payload.get("metadata")) is not dict
        or type(payload.get("fragments")) is not list
        or type(payload.get("views")) is not list
        or type(payload.get("attachments")) is not list
    ):
        raise QualificationError("GRAPHICS_REOPEN_SCHEMA_OR_COLLECTION_DRIFT")
    try:
        fragments = tuple(GraphicsFragmentV1(
            fragment_id=row["fragment_id"],
            face_indices=tuple(row["face_indices"]),
            support_band_pixels=row["support_band_pixels"],
            schema_version=row["schema_version"],
            metadata=dict(row["metadata"]),
        ) for row in payload["fragments"])
        views = tuple(GraphicsViewStateV1(
            view_index=row["view_index"],
            visible_fragment_ids=tuple(row["visible_fragment_ids"]),
            order_edges=tuple(GraphicsOrderEdgeV1(
                view_index=edge["view_index"],
                front_fragment_id=edge["front_fragment_id"],
                back_fragment_id=edge["back_fragment_id"],
                confidence=edge["confidence"],
                evidence_refs=tuple(edge["evidence_refs"]),
                schema_version=edge["schema_version"],
            ) for edge in row["order_edges"]),
            schema_version=row["schema_version"],
            metadata=dict(row["metadata"]),
        ) for row in payload["views"])
        attachments = tuple(GraphicsAttachmentV1(
            fragment_id=row["fragment_id"],
            parent_mechanical_domain_id=row["parent_mechanical_domain_id"],
            anchor_face_index=row["anchor_face_index"],
            barycentric_anchor=tuple(row["barycentric_anchor"]),
            confidence=row["confidence"],
            evidence_refs=tuple(row["evidence_refs"]),
            schema_version=row["schema_version"],
        ) for row in payload["attachments"])
        compiled = CompiledGraphicsV1(
            canonical_mesh_binding_hash=payload["canonical_mesh_binding_hash"],
            canonical_mesh_skin_binding_hash=payload["canonical_mesh_skin_binding_hash"],
            source_graphics_binding_hash=payload["source_graphics_binding_hash"],
            graphics_evidence_binding_hash=payload["graphics_evidence_binding_hash"],
            fragments=fragments, views=views, attachments=attachments,
            graph_hash=payload["graph_hash"],
            schema_version=payload["schema_version"],
            metadata=dict(payload["metadata"]),
        )
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise QualificationError("GRAPHICS_REOPEN_TYPED_DECODE_INVALID") from exc
    for fragment in fragments:
        if fragment.schema_version != "RealSaS.GraphicsFragment.v1":
            raise QualificationError("GRAPHICS_REOPEN_FRAGMENT_SCHEMA_DRIFT")
    for view in views:
        if view.schema_version != "RealSaS.GraphicsViewState.v1":
            raise QualificationError("GRAPHICS_REOPEN_VIEW_SCHEMA_DRIFT")
        for edge in view.order_edges:
            if edge.schema_version != "RealSaS.GraphicsOrderEdge.v1":
                raise QualificationError("GRAPHICS_REOPEN_EDGE_SCHEMA_DRIFT")
    for anchor in attachments:
        if anchor.schema_version != "RealSaS.GraphicsAttachment.v1":
            raise QualificationError("GRAPHICS_REOPEN_ATTACHMENT_SCHEMA_DRIFT")
    if compiled.graph_hash != compiled_graphics_hash_v1(compiled):
        raise QualificationError("GRAPHICS_REOPEN_GRAPH_HASH_DRIFT")
    runtime = build_runtime_graphics_v1(compiled, face_count=face_count)
    return compiled, runtime


__all__ = ["reopen_compiled_graphics_v1"]
