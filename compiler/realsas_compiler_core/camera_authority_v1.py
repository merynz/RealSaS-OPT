from __future__ import annotations

"""First-class exact eight-view camera authority for the product compiler."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable

from .hashing import content_sha256
from .playback_full_surface_v3 import CameraProjectionV3, qualify_camera_v3
from .types import QualificationError

Json=dict[str,Any]


def camera_projection_binding_hash(camera:CameraProjectionV3)->str:
    return content_sha256({
        "schema":camera.schema_version,
        "view_id":camera.view_id,
        "view_index":int(camera.view_index),
        "origin":tuple(map(float,camera.origin)),
        "right":tuple(map(float,camera.right)),
        "screen_up":tuple(map(float,camera.screen_up)),
        "forward":tuple(map(float,camera.forward)),
        "half_extent":float(camera.half_extent),
        "resolution":int(camera.resolution),
    })


@dataclass(frozen=True)
class QualifiedCameraSetIR:
    cameras:tuple[CameraProjectionV3,...]
    camera_binding_hashes:tuple[str,...]
    source_bundle_sha256:str
    camera_set_hash:str
    schema_version:str="RealSaS.QualifiedCameraSetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def qualified_camera_set_hash(value:QualifiedCameraSetIR)->str:
    payload=value.to_dict()
    payload.pop("camera_set_hash",None)
    return content_sha256(payload)


def validate_qualified_camera_set(value:QualifiedCameraSetIR)->None:
    rows=tuple(sorted(value.cameras,key=lambda c:int(c.view_index)))
    if len(rows)!=8 or tuple(int(c.view_index) for c in rows)!=tuple(range(8)):
        raise QualificationError("CAMERA_SET_REQUIRES_EXACT_DENSE_8_VIEWS")
    if tuple(str(c.view_id) for c in rows)!=tuple(f"V{i}" for i in range(8)):
        raise QualificationError("CAMERA_SET_VIEW_IDS_MUST_BE_V0_V7")
    expected=tuple(camera_projection_binding_hash(c) for c in rows)
    if tuple(value.camera_binding_hashes)!=expected:
        raise QualificationError("CAMERA_SET_BINDING_HASH_DRIFT")
    if len(set(expected))!=8:
        raise QualificationError("CAMERA_SET_REQUIRES_UNIQUE_CAMERA_BINDINGS")
    if len(str(value.source_bundle_sha256))!=64:
        raise QualificationError("CAMERA_SET_SOURCE_BUNDLE_SHA_INVALID")
    resolutions={int(c.resolution) for c in rows}
    half_extents={float(c.half_extent) for c in rows}
    if len(resolutions)!=1 or len(half_extents)!=1:
        raise QualificationError("CAMERA_SET_COMMON_FRAME_CONTRACT_VIOLATED")
    if value.camera_set_hash!=qualified_camera_set_hash(value):
        raise QualificationError("CAMERA_SET_HASH_MISMATCH")


def build_qualified_camera_set(
    rows:Iterable[dict|CameraProjectionV3],
    *,
    source_bundle_sha256:str,
    metadata:Json|None=None,
)->QualifiedCameraSetIR:
    raw=tuple(rows)
    if len(raw)!=8:
        raise QualificationError("CAMERA_SET_REQUIRES_EXACT_DENSE_8_VIEWS")
    cameras=[]
    for item in sorted(raw,key=lambda r:int(r.view_index if isinstance(r,CameraProjectionV3) else r["view_index"])):
        vi=int(item.view_index if isinstance(item,CameraProjectionV3) else item["view_index"])
        expected_id=f"V{vi}"
        if isinstance(item,CameraProjectionV3):
            if item.view_id!=expected_id:
                raise QualificationError("CAMERA_SET_VIEW_IDS_MUST_BE_V0_V7")
            camera=item
        else:
            camera=qualify_camera_v3(item,view_id=expected_id,view_index=vi)
        cameras.append(camera)
    value=QualifiedCameraSetIR(
        cameras=tuple(cameras),
        camera_binding_hashes=tuple(camera_projection_binding_hash(c) for c in cameras),
        source_bundle_sha256=str(source_bundle_sha256),
        camera_set_hash="",
        metadata={
            "authority":"EXACT_8_VIEW_CAMERA_CONTRACT",
            "camera_refit_downstream_forbidden":True,
            **dict(metadata or {}),
        },
    )
    value=replace(value,camera_set_hash=qualified_camera_set_hash(value))
    validate_qualified_camera_set(value)
    return value
