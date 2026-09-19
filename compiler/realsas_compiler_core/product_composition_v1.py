from __future__ import annotations

"""Qualified per-view composition policy for canonical-M presentation."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .hashing import content_sha256
from .observation_authority_v1 import QualifiedObservationSetIR, validate_qualified_observation_set
from .types import QualificationError

Json=dict[str,Any]
_PHYSICAL_OCCLUSION="CANONICAL_Z_BUFFER_VISIBLE_OWNER_V1"
_EQUAL_DEPTH="SOURCE_VISIBILITY_THEN_SLOT_ORDER_THEN_STABLE_FACE_KEY_V3"
_SLOT_ORDER_ROLE="UI_SETUP_ONLY__NOT_PHYSICAL_OCCLUSION"


@dataclass(frozen=True)
class QualifiedCompositionViewIR:
    view_index:int
    camera_binding_hash:str
    slot_order:tuple[str,...]
    physical_occlusion_rule:str
    equal_depth_tiebreak:str
    slot_order_role:str
    source_visible_owner_evidence_hash:str
    composition_binding_hash:str
    schema_version:str="RealSaS.QualifiedCompositionViewIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedCompositionSetIR:
    views:tuple[QualifiedCompositionViewIR,...]
    product_state_binding_hash:str
    mesh_binding_hash:str
    observation_set_binding_hash:str
    presentation_structure_binding_hash:str
    composition_set_hash:str
    schema_version:str="RealSaS.QualifiedCompositionSetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def composition_view_hash(value:QualifiedCompositionViewIR)->str:
    payload=value.to_dict(); payload.pop("composition_binding_hash",None)
    return content_sha256(payload)


def composition_set_hash(value:QualifiedCompositionSetIR)->str:
    payload=value.to_dict(); payload.pop("composition_set_hash",None)
    return content_sha256(payload)


def validate_composition_set(
    value:QualifiedCompositionSetIR,
    *,
    mesh,
    observation_set:QualifiedObservationSetIR,
    presentation_structure,
    product_state,
)->None:
    validate_qualified_observation_set(observation_set)
    expected={
        "product_state_binding_hash":product_state.product_state_hash,
        "mesh_binding_hash":mesh.mesh_lineage_hash,
        "observation_set_binding_hash":observation_set.observation_set_hash,
        "presentation_structure_binding_hash":presentation_structure.structure_lineage_hash,
    }
    for field_name,expected_hash in expected.items():
        if getattr(value,field_name)!=expected_hash:
            raise QualificationError(f"COMPOSITION_SET_BINDING_DRIFT:{field_name}")
    g5=str(mesh.qualification_report.get("g5_evidence_hash",""))
    if not g5:
        raise QualificationError("COMPOSITION_REQUIRES_QUALIFIED_G5_VISIBLE_OWNER_EVIDENCE")
    slots=tuple(sorted(presentation_structure.slots,key=lambda x:x.setup_order))
    if len({row.setup_order for row in slots})!=len(slots):
        raise QualificationError("COMPOSITION_SLOT_SETUP_ORDER_INVALID")
    expected_order=tuple(row.slot_id for row in slots)
    obs={int(row.view_index):row for row in observation_set.views}
    rows=tuple(sorted(value.views,key=lambda x:x.view_index))
    if len(rows)!=8 or tuple(row.view_index for row in rows)!=tuple(range(8)):
        raise QualificationError("COMPOSITION_REQUIRES_EXACT_8_VIEWS")
    for row in rows:
        if row.camera_binding_hash!=obs[row.view_index].camera_binding_hash:
            raise QualificationError("COMPOSITION_CAMERA_BINDING_DRIFT")
        if row.slot_order!=expected_order:
            raise QualificationError("COMPOSITION_SLOT_ORDER_DRIFT")
        if row.physical_occlusion_rule!=_PHYSICAL_OCCLUSION:
            raise QualificationError("COMPOSITION_PHYSICAL_OCCLUSION_RULE_DRIFT")
        if row.equal_depth_tiebreak!=_EQUAL_DEPTH:
            raise QualificationError("COMPOSITION_EQUAL_DEPTH_TIEBREAK_DRIFT")
        if row.slot_order_role!=_SLOT_ORDER_ROLE:
            raise QualificationError("COMPOSITION_SLOT_ORDER_ROLE_DRIFT")
        if row.source_visible_owner_evidence_hash!=g5:
            raise QualificationError("COMPOSITION_G5_EVIDENCE_DRIFT")
        if row.composition_binding_hash!=composition_view_hash(row):
            raise QualificationError("COMPOSITION_VIEW_HASH_MISMATCH")
    if value.composition_set_hash!=composition_set_hash(value):
        raise QualificationError("COMPOSITION_SET_HASH_MISMATCH")


def build_composition_set(
    *,
    mesh,
    observation_set:QualifiedObservationSetIR,
    presentation_structure,
    product_state,
)->QualifiedCompositionSetIR:
    validate_qualified_observation_set(observation_set)
    g5=str(mesh.qualification_report.get("g5_evidence_hash",""))
    if not g5:
        raise QualificationError("COMPOSITION_REQUIRES_QUALIFIED_G5_VISIBLE_OWNER_EVIDENCE")
    slots=tuple(sorted(presentation_structure.slots,key=lambda x:x.setup_order))
    slot_order=tuple(row.slot_id for row in slots)
    views=[]
    for observation in sorted(observation_set.views,key=lambda x:x.view_index):
        row=QualifiedCompositionViewIR(
            view_index=int(observation.view_index),
            camera_binding_hash=str(observation.camera_binding_hash),
            slot_order=slot_order,
            physical_occlusion_rule=_PHYSICAL_OCCLUSION,
            equal_depth_tiebreak=_EQUAL_DEPTH,
            slot_order_role=_SLOT_ORDER_ROLE,
            source_visible_owner_evidence_hash=g5,
            composition_binding_hash="",
            metadata={
                "physical_depth_owner":"QUALIFIED_CANONICAL_MESH",
                "semantic_slot_order_used_for_physical_occlusion":False,
                "motion_order_override_allowed_later":True,
                "clip_semantics_present":False,
            },
        )
        views.append(replace(row,composition_binding_hash=composition_view_hash(row)))
    value=QualifiedCompositionSetIR(
        views=tuple(views),
        product_state_binding_hash=product_state.product_state_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        observation_set_binding_hash=observation_set.observation_set_hash,
        presentation_structure_binding_hash=presentation_structure.structure_lineage_hash,
        composition_set_hash="",
        metadata={
            "physical_occlusion_rule":_PHYSICAL_OCCLUSION,
            "equal_depth_tiebreak":_EQUAL_DEPTH,
            "slot_order_role":_SLOT_ORDER_ROLE,
            "source_visible_owner_evidence_hash":g5,
        },
    )
    value=replace(value,composition_set_hash=composition_set_hash(value))
    validate_composition_set(
        value,mesh=mesh,observation_set=observation_set,
        presentation_structure=presentation_structure,product_state=product_state,
    )
    return value
