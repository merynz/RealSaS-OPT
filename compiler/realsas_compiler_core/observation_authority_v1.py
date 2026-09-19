from __future__ import annotations

"""Qualified eight-view source-observation authority for downstream product gates."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .hashing import content_sha256
from .types import QualificationError

Json=dict[str,Any]
_ALLOWED_STATES={"PASS","NORMALIZED_PASS"}


@dataclass(frozen=True)
class QualifiedObservationViewIR:
    view_index:int
    width:int
    height:int
    source_observation_hash:str
    source_raster_sha256:str
    foreground_mask_sha256:str
    camera_binding_hash:str
    qualification_state:str
    evidence_refs:tuple[str,...]
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedObservationSetIR:
    views:tuple[QualifiedObservationViewIR,...]
    observation_set_hash:str
    schema_version:str="RealSaS.QualifiedObservationSetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def qualified_observation_set_hash(value:QualifiedObservationSetIR)->str:
    payload=value.to_dict()
    payload.pop("observation_set_hash",None)
    return content_sha256(payload)


def validate_qualified_observation_set(value:QualifiedObservationSetIR)->None:
    if len(value.views)!=8:
        raise QualificationError("OBSERVATION_SET_REQUIRES_EXACT_8_VIEWS")
    rows=tuple(sorted(value.views,key=lambda row:row.view_index))
    if tuple(row.view_index for row in rows)!=tuple(range(8)):
        raise QualificationError("OBSERVATION_SET_VIEW_ORDER_INVALID")
    camera_hashes=set()
    for row in rows:
        if row.width<=0 or row.height<=0:
            raise QualificationError("OBSERVATION_VIEW_DIMENSION_INVALID")
        if (
            not row.source_observation_hash
            or len(row.source_raster_sha256)!=64
            or len(row.foreground_mask_sha256)!=64
            or not row.camera_binding_hash
        ):
            raise QualificationError("OBSERVATION_VIEW_AUTHORITY_BINDING_MISSING")
        if row.qualification_state not in _ALLOWED_STATES:
            raise QualificationError("OBSERVATION_VIEW_NOT_PRODUCT_QUALIFIED")
        if not row.evidence_refs:
            raise QualificationError("OBSERVATION_VIEW_EVIDENCE_MISSING")
        if row.camera_binding_hash in camera_hashes:
            raise QualificationError("OBSERVATION_SET_DUPLICATE_CAMERA_BINDING")
        camera_hashes.add(row.camera_binding_hash)
    if value.observation_set_hash!=qualified_observation_set_hash(value):
        raise QualificationError("OBSERVATION_SET_HASH_MISMATCH")


def build_qualified_observation_set(
    views:tuple[QualifiedObservationViewIR,...],
    *,
    metadata:Json|None=None,
)->QualifiedObservationSetIR:
    value=QualifiedObservationSetIR(tuple(views),"",metadata=dict(metadata or {}))
    value=replace(value,observation_set_hash=qualified_observation_set_hash(value))
    validate_qualified_observation_set(value)
    return value
