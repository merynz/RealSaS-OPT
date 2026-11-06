from __future__ import annotations

"""Canonical Graphics authority for the RealSaS product path.

Graphics owns all visual/presentation state. Complete Appearance Authority (CAA)
is one Graphics sublayer, not a sibling product stage. Role-free learned relation
evidence remains proposal/evidence; Compiler binds qualified presentation relations
and the source-preserving appearance asset onto the exact canonical mechanical
puppet without inventing missing perception.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .appearance_authority_v2 import (
    CompleteAppearanceAssetIR,
    complete_appearance_asset_from_dict,
    complete_appearance_asset_hash,
    validate_complete_appearance_asset,
)
from .canonical_mechanics_ir_v1 import (
    CanonicalMechanicsIRV1,
    canonical_mechanics_lineage_hash_v1,
)
from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_lineage_hash_v1
from .canonical_skin_ir_v2 import CanonicalSkinIRV2, canonical_skin_lineage_hash_v2
from .graphics_v1 import (
    CompiledGraphicsV1,
    GraphicsAttachmentV1,
    GraphicsFragmentV1,
    GraphicsOrderEdgeV1,
    GraphicsViewStateV1,
    compiled_graphics_hash_v1,
)
from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class CanonicalGraphicsIRV1:
    presentation: CompiledGraphicsV1
    appearance: CompleteAppearanceAssetIR
    mesh_lineage_hash: str
    skin_lineage_hash: str
    mechanics_lineage_hash: str
    presentation_graph_hash: str
    appearance_asset_hash: str
    graphics_lineage_hash: str
    schema_version: str = "RealSaS.CanonicalGraphicsIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_graphics_lineage_hash_v1(value: CanonicalGraphicsIRV1) -> str:
    payload = value.to_dict()
    payload.pop("graphics_lineage_hash", None)
    return content_sha256(payload)


def _validate_parent_lineage(
    mesh: CanonicalMeshIR,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
) -> None:
    if mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh):
        raise QualificationError("GRAPHICS_MESH_LINEAGE_DRIFT")
    if skin.skin_lineage_hash != canonical_skin_lineage_hash_v2(skin):
        raise QualificationError("GRAPHICS_SKIN_LINEAGE_DRIFT")
    if mechanics.mechanics_lineage_hash != canonical_mechanics_lineage_hash_v1(mechanics):
        raise QualificationError("GRAPHICS_MECHANICS_LINEAGE_DRIFT")
    if skin.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise QualificationError("GRAPHICS_SKIN_MESH_PARENT_DRIFT")
    if mechanics.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise QualificationError("GRAPHICS_MECHANICS_MESH_PARENT_DRIFT")
    if mechanics.skin_lineage_hash != skin.skin_lineage_hash:
        raise QualificationError("GRAPHICS_MECHANICS_SKIN_PARENT_DRIFT")


def mint_canonical_graphics_v1(
    mesh: CanonicalMeshIR,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    *,
    presentation: CompiledGraphicsV1,
    appearance: CompleteAppearanceAssetIR,
) -> CanonicalGraphicsIRV1:
    _validate_parent_lineage(mesh, skin, mechanics)
    if presentation.graph_hash != compiled_graphics_hash_v1(presentation):
        raise QualificationError("GRAPHICS_PRESENTATION_HASH_DRIFT")
    validate_complete_appearance_asset(appearance)
    if appearance.asset_hash != complete_appearance_asset_hash(appearance):
        raise QualificationError("GRAPHICS_APPEARANCE_HASH_DRIFT")
    if presentation.canonical_mesh_binding_hash != mesh.mesh_lineage_hash:
        raise QualificationError("GRAPHICS_PRESENTATION_MESH_BINDING_DRIFT")
    if presentation.canonical_mesh_skin_binding_hash != skin.skin_lineage_hash:
        raise QualificationError("GRAPHICS_PRESENTATION_SKIN_BINDING_DRIFT")
    if presentation.source_graphics_binding_hash != appearance.asset_hash:
        raise QualificationError("GRAPHICS_PRESENTATION_APPEARANCE_BINDING_DRIFT")
    metadata = dict(presentation.metadata or {})
    if metadata.get("single_canonical_mesh_authority") is not True:
        raise QualificationError("GRAPHICS_SINGLE_MESH_AUTHORITY_REQUIRED")
    if metadata.get("independent_visual_mesh_allowed") is not False:
        raise QualificationError("GRAPHICS_INDEPENDENT_VISUAL_MESH_FORBIDDEN")
    if metadata.get("source_pixels_win_where_observed") is not True:
        raise QualificationError("GRAPHICS_SOURCE_WINS_REQUIRED")

    value = CanonicalGraphicsIRV1(
        presentation=presentation,
        appearance=appearance,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
        presentation_graph_hash=presentation.graph_hash,
        appearance_asset_hash=appearance.asset_hash,
        graphics_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_GRAPHICS",
            "graphics_owns_all_visual_presentation": True,
            "caa_is_graphics_appearance_sublayer": True,
            "appearance_authority": "COMPLETE_APPEARANCE_AUTHORITY_V2",
            "presentation_relation_authority": "COMPILED_GRAPHICS_V1",
            "visibility_occlusion_owner": "GRAPHICS",
            "draw_order_owner": "GRAPHICS",
            "fragmentation_owner": "GRAPHICS",
            "presentation_attachment_owner": "GRAPHICS",
            "source_art_wins_where_observed": True,
            "mechanics_may_not_decide_draw_order": True,
            "motion_may_not_decide_draw_order": True,
            "compiler_perception_synthesis_performed": False,
        },
    )
    return replace(value, graphics_lineage_hash=canonical_graphics_lineage_hash_v1(value))


def _compiled_graphics_from_dict(payload: Mapping[str, Any]) -> CompiledGraphicsV1:
    fragments = tuple(
        GraphicsFragmentV1(
            fragment_id=str(row["fragment_id"]),
            face_indices=tuple(map(int, row.get("face_indices") or ())),
            support_band_pixels=int(row.get("support_band_pixels") or 0),
            schema_version=str(row.get("schema_version") or "RealSaS.GraphicsFragment.v1"),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("fragments") or ()
    )
    views = []
    for row in payload.get("views") or ():
        edges = tuple(
            GraphicsOrderEdgeV1(
                view_index=int(edge["view_index"]),
                front_fragment_id=str(edge["front_fragment_id"]),
                back_fragment_id=str(edge["back_fragment_id"]),
                confidence=float(edge["confidence"]),
                evidence_refs=tuple(map(str, edge.get("evidence_refs") or ())),
                schema_version=str(edge.get("schema_version") or "RealSaS.GraphicsOrderEdge.v1"),
            )
            for edge in row.get("order_edges") or ()
        )
        views.append(
            GraphicsViewStateV1(
                view_index=int(row["view_index"]),
                visible_fragment_ids=tuple(map(str, row.get("visible_fragment_ids") or ())),
                order_edges=edges,
                schema_version=str(row.get("schema_version") or "RealSaS.GraphicsViewState.v1"),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    attachments = tuple(
        GraphicsAttachmentV1(
            fragment_id=str(row["fragment_id"]),
            parent_mechanical_domain_id=str(row["parent_mechanical_domain_id"]),
            anchor_face_index=int(row["anchor_face_index"]),
            barycentric_anchor=tuple(map(float, row.get("barycentric_anchor") or ())),
            confidence=float(row["confidence"]),
            evidence_refs=tuple(map(str, row.get("evidence_refs") or ())),
            schema_version=str(row.get("schema_version") or "RealSaS.GraphicsAttachment.v1"),
        )
        for row in payload.get("attachments") or ()
    )
    value = CompiledGraphicsV1(
        canonical_mesh_binding_hash=str(payload["canonical_mesh_binding_hash"]),
        canonical_mesh_skin_binding_hash=str(payload["canonical_mesh_skin_binding_hash"]),
        source_graphics_binding_hash=str(payload["source_graphics_binding_hash"]),
        graphics_evidence_binding_hash=str(payload["graphics_evidence_binding_hash"]),
        fragments=fragments,
        views=tuple(views),
        attachments=attachments,
        graph_hash=str(payload["graph_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.CompiledGraphics.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.graph_hash != compiled_graphics_hash_v1(value):
        raise QualificationError("GRAPHICS_PRESENTATION_HASH_DRIFT")
    return value


def canonical_graphics_from_dict_v1(payload: Mapping[str, Any]) -> CanonicalGraphicsIRV1:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalGraphicsIR.v1":
        raise QualificationError("CANONICAL_GRAPHICS_SCHEMA_DRIFT")
    value = CanonicalGraphicsIRV1(
        presentation=_compiled_graphics_from_dict(dict(payload["presentation"])),
        appearance=complete_appearance_asset_from_dict(dict(payload["appearance"])),
        mesh_lineage_hash=str(payload["mesh_lineage_hash"]),
        skin_lineage_hash=str(payload["skin_lineage_hash"]),
        mechanics_lineage_hash=str(payload["mechanics_lineage_hash"]),
        presentation_graph_hash=str(payload["presentation_graph_hash"]),
        appearance_asset_hash=str(payload["appearance_asset_hash"]),
        graphics_lineage_hash=str(payload["graphics_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.presentation_graph_hash != value.presentation.graph_hash:
        raise QualificationError("CANONICAL_GRAPHICS_PRESENTATION_BINDING_DRIFT")
    if value.appearance_asset_hash != value.appearance.asset_hash:
        raise QualificationError("CANONICAL_GRAPHICS_APPEARANCE_BINDING_DRIFT")
    if value.graphics_lineage_hash != canonical_graphics_lineage_hash_v1(value):
        raise QualificationError("CANONICAL_GRAPHICS_LINEAGE_HASH_DRIFT")
    if value.metadata.get("authority") != "COMPILER_CANONICAL_GRAPHICS":
        raise QualificationError("CANONICAL_GRAPHICS_AUTHORITY_DRIFT")
    return value


__all__ = [
    "CanonicalGraphicsIRV1",
    "canonical_graphics_from_dict_v1",
    "canonical_graphics_lineage_hash_v1",
    "mint_canonical_graphics_v1",
]
