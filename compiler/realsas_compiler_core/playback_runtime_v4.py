from __future__ import annotations

"""Compact Runtime-v4 diagnostic contract.

Runtime-v4 keeps canonical attachment geometry once, stores per-view camera/UV/
appearance overlays, and stores canonical posed XYZ once per asset/frame. Dense
arrays remain contiguous NumPy buffers; identity composes their sealed SHA-256
values instead of recursively JSON-serializing millions of scalars.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Iterable
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,
    AttachmentKind,
    ReferenceRasterContractV1,
    RuntimeV3AppearancePatch,
    RuntimeV3FrameComposition,
    RuntimeV3Mesh,
    RuntimeV3PlaybackContract,
    RuntimeV3Slot,
    RuntimeV3Vertex,
    RuntimeV3VisibilityPolicy,
    TopologyClass,
    validate_frame_composition_v3,
)
from compiler.realsas_compiler_core.types import QualificationError

PLAYBACK_RUNTIME_V4_SCHEMA = "RealSaS.PlaybackRuntimeV4Contract.v1"
CAMERA_PROJECTION_V4_SCHEMA = "RealSaS.RuntimeV4CameraProjection.v1"

def _array_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = sha256()
    h.update(str(arr.dtype).encode("ascii")); h.update(b"|")
    h.update("x".join(map(str, arr.shape)).encode("ascii")); h.update(b"|")
    h.update(memoryview(arr).cast("B"))
    return h.hexdigest()

@dataclass(frozen=True)
class RuntimeV4Camera:
    view_id: str; view_index: int
    origin: tuple[float,float,float]; right: tuple[float,float,float]
    screen_up: tuple[float,float,float]; forward: tuple[float,float,float]
    half_extent: float; resolution: int
    schema_version: str = CAMERA_PROJECTION_V4_SCHEMA

@dataclass(frozen=True)
class RuntimeV4Asset:
    asset_id: str; slot_id: str; attachment_id: str
    attachment_kind: AttachmentKind; topology_class: TopologyClass
    canonical_xyz: np.ndarray
    triangles: np.ndarray

@dataclass(frozen=True)
class RuntimeV4ViewOverlay:
    view_id: str; asset_id: str
    uv: np.ndarray
    face_provenance: np.ndarray

@dataclass(frozen=True)
class RuntimeV4PlaybackContract:
    slots: tuple[RuntimeV3Slot,...]
    assets: tuple[RuntimeV4Asset,...]
    cameras: tuple[RuntimeV4Camera,...]
    overlays: tuple[RuntimeV4ViewOverlay,...]
    visibility: RuntimeV3VisibilityPolicy = RuntimeV3VisibilityPolicy()
    raster: ReferenceRasterContractV1 = ReferenceRasterContractV1()
    allow_completion: bool = False
    schema_version: str = PLAYBACK_RUNTIME_V4_SCHEMA
    @property
    def contract_hash(self) -> str:
        return content_sha256(_identity_payload(self))

def _identity_payload(contract: RuntimeV4PlaybackContract) -> dict:
    return {
        "schema_version": contract.schema_version,
        "slots": [asdict(x) for x in contract.slots],
        "assets": [{
            "asset_id":x.asset_id,"slot_id":x.slot_id,"attachment_id":x.attachment_id,
            "attachment_kind":x.attachment_kind.value,"topology_class":x.topology_class.value,
            "canonical_xyz_shape":list(np.asarray(x.canonical_xyz).shape),
            "canonical_xyz_sha256":_array_sha256(np.asarray(x.canonical_xyz)),
            "triangles_shape":list(np.asarray(x.triangles).shape),
            "triangles_sha256":_array_sha256(np.asarray(x.triangles)),
        } for x in contract.assets],
        "cameras":[asdict(x) for x in contract.cameras],
        "overlays":[{
            "view_id":x.view_id,"asset_id":x.asset_id,
            "uv_shape":list(np.asarray(x.uv).shape),"uv_sha256":_array_sha256(np.asarray(x.uv)),
            "face_provenance_shape":list(np.asarray(x.face_provenance).shape),
            "face_provenance_sha256":_array_sha256(np.asarray(x.face_provenance)),
        } for x in contract.overlays],
        "visibility":{**asdict(contract.visibility),"body_depth_write":contract.visibility.body_depth_write.value,"rigid_depth_write":contract.visibility.rigid_depth_write.value},
        "raster":asdict(contract.raster),"allow_completion":bool(contract.allow_completion),
    }

def contract_manifest_v4(contract: RuntimeV4PlaybackContract) -> dict:
    return _identity_payload(contract)

def _finite_axis(values,label):
    arr=np.asarray(values,dtype=np.float64)
    if arr.shape!=(3,) or not np.isfinite(arr).all(): raise QualificationError(label)
    return arr

def validate_runtime_v4_camera(camera: RuntimeV4Camera) -> None:
    if camera.schema_version!=CAMERA_PROJECTION_V4_SCHEMA: raise QualificationError("RUNTIME_V4_CAMERA_SCHEMA_MISMATCH")
    if not camera.view_id: raise QualificationError("RUNTIME_V4_CAMERA_VIEW_ID_REQUIRED")
    _finite_axis(camera.origin,"RUNTIME_V4_CAMERA_ORIGIN_INVALID")
    right=_finite_axis(camera.right,"RUNTIME_V4_CAMERA_RIGHT_INVALID"); up=_finite_axis(camera.screen_up,"RUNTIME_V4_CAMERA_UP_INVALID"); forward=_finite_axis(camera.forward,"RUNTIME_V4_CAMERA_FORWARD_INVALID")
    for name,axis in (("RIGHT",right),("UP",up),("FORWARD",forward)):
        if abs(float(np.linalg.norm(axis))-1.0)>1e-6: raise QualificationError(f"RUNTIME_V4_CAMERA_{name}_NOT_UNIT")
    if abs(float(right@up))>1e-6 or abs(float(right@forward))>1e-6 or abs(float(up@forward))>1e-6: raise QualificationError("RUNTIME_V4_CAMERA_BASIS_NOT_ORTHONORMAL")
    if not math.isfinite(float(camera.half_extent)) or camera.half_extent<=0: raise QualificationError("RUNTIME_V4_CAMERA_HALF_EXTENT_INVALID")
    if camera.resolution<=0: raise QualificationError("RUNTIME_V4_CAMERA_RESOLUTION_INVALID")

def validate_playback_runtime_v4_contract(contract: RuntimeV4PlaybackContract, *, required_view_ids: Iterable[str]=tuple(f"V{i}" for i in range(8))) -> str:
    if contract.schema_version!=PLAYBACK_RUNTIME_V4_SCHEMA: raise QualificationError("RUNTIME_V4_SCHEMA_MISMATCH")
    views=tuple(map(str,required_view_ids))
    if not views or len(views)!=len(set(views)): raise QualificationError("RUNTIME_V4_REQUIRED_VIEW_IDS_INVALID")
    vis=contract.visibility
    if not vis.full_surface_geometry_authority_required or not vis.visibility_by_face_deletion_forbidden: raise QualificationError("RUNTIME_V4_REQUIRES_FULL_SURFACE_AUTHORITY")
    if not vis.body_depth_test or vis.body_depth_write.value=="OFF": raise QualificationError("RUNTIME_V4_BODY_DEPTH_REQUIRED")
    if not contract.raster.per_frame_affine_refit_forbidden: raise QualificationError("RUNTIME_V4_PER_FRAME_AFFINE_REFIT_FORBIDDEN")
    slots=tuple(contract.slots); slot_ids=[s.slot_id for s in slots]
    if not slots or len(slot_ids)!=len(set(slot_ids)): raise QualificationError("RUNTIME_V4_SLOT_IDS_INVALID")
    if len({int(s.setup_order) for s in slots})!=len(slots): raise QualificationError("RUNTIME_V4_SLOT_SETUP_ORDER_MUST_BE_TOTAL")
    slot_set=set(slot_ids); assets=tuple(contract.assets); asset_ids=[a.asset_id for a in assets]
    if not assets or len(asset_ids)!=len(set(asset_ids)): raise QualificationError("RUNTIME_V4_ASSET_IDS_INVALID")
    attachment_ids=set(); asset_shapes={}
    for asset in assets:
        if asset.slot_id not in slot_set or not asset.attachment_id or not asset.asset_id: raise QualificationError("RUNTIME_V4_ASSET_IDENTITY_INVALID")
        if asset.attachment_id in attachment_ids: raise QualificationError("RUNTIME_V4_ATTACHMENT_ID_MUST_BE_UNIQUE_ASSET_IDENTITY")
        attachment_ids.add(asset.attachment_id); xyz=np.asarray(asset.canonical_xyz); tri=np.asarray(asset.triangles)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<3 or not np.isfinite(xyz).all(): raise QualificationError("RUNTIME_V4_CANONICAL_VERTEX_INVALID")
        if tri.ndim!=2 or tri.shape[1]!=3 or len(tri)==0 or not np.issubdtype(tri.dtype,np.integer): raise QualificationError("RUNTIME_V4_TRIANGLE_INVALID")
        if int(tri.min())<0 or int(tri.max())>=len(xyz): raise QualificationError("RUNTIME_V4_TRIANGLE_INDEX_OUT_OF_RANGE")
        asset_shapes[asset.asset_id]=(len(xyz),len(tri))
    camera_by_view={c.view_id:c for c in contract.cameras}
    if set(camera_by_view)!=set(views) or len(contract.cameras)!=len(views): raise QualificationError("RUNTIME_V4_CAMERA_VIEW_SET_MISMATCH")
    for i,v in enumerate(views):
        if camera_by_view[v].view_index!=i: raise QualificationError("RUNTIME_V4_CAMERA_VIEW_INDEX_DRIFT")
        validate_runtime_v4_camera(camera_by_view[v])
    overlay_keys=set()
    for o in contract.overlays:
        key=(o.view_id,o.asset_id)
        if key in overlay_keys: raise QualificationError("RUNTIME_V4_DUPLICATE_VIEW_ASSET_OVERLAY")
        overlay_keys.add(key)
        if o.view_id not in camera_by_view or o.asset_id not in asset_shapes: raise QualificationError("RUNTIME_V4_OVERLAY_REFERENCE_INVALID")
        nvert,nface=asset_shapes[o.asset_id]; uv=np.asarray(o.uv); prov=np.asarray(o.face_provenance)
        if uv.shape!=(nvert,2) or not np.isfinite(uv).all(): raise QualificationError("RUNTIME_V4_OVERLAY_UV_CARDINALITY_MISMATCH")
        if np.any(uv<0) or np.any(uv>1): raise QualificationError("RUNTIME_V4_UV_OUTSIDE_UNIT_DOMAIN")
        if prov.shape!=(nface,) or not np.issubdtype(prov.dtype,np.integer): raise QualificationError("RUNTIME_V4_OVERLAY_PROVENANCE_CARDINALITY_MISMATCH")
        if len(prov) and (int(prov.min())<0 or int(prov.max())>4): raise QualificationError("RUNTIME_V4_PROVENANCE_CODE_INVALID")
        if not contract.allow_completion and np.any(prov==4): raise QualificationError("RUNTIME_V4_COMPLETION_NOT_ALLOWED_BY_PRODUCT_POLICY")
    if overlay_keys!={(v,a.asset_id) for v in views for a in assets}: raise QualificationError("RUNTIME_V4_OVERLAY_VIEW_ASSET_COVERAGE_INCOMPLETE")
    if not {s.default_attachment_id for s in slots if s.default_attachment_id}.issubset(attachment_ids): raise QualificationError("RUNTIME_V4_DEFAULT_ATTACHMENT_UNKNOWN")
    return contract.contract_hash

def validate_runtime_v4_frame_composition(contract: RuntimeV4PlaybackContract, frame: RuntimeV3FrameComposition) -> None:
    compat_meshes=[]; compat_patches=[]
    for asset in contract.assets:
        for camera in contract.cameras:
            mid=f"{camera.view_id}:{asset.asset_id}"
            compat_meshes.append(RuntimeV3Mesh(camera.view_id,mid,asset.slot_id,asset.attachment_id,asset.attachment_kind,asset.topology_class,(RuntimeV3Vertex(0,0,0,0,0),RuntimeV3Vertex(1,0,0,1,0),RuntimeV3Vertex(0,1,0,0,1)),((0,1,2),)))
            compat_patches.append(RuntimeV3AppearancePatch(f"{mid}:compat",mid,(0,),AppearanceProvenance.UNSEEN,None,None))
    compat=RuntimeV3PlaybackContract(contract.slots,tuple(compat_meshes),tuple(compat_patches),contract.visibility,contract.raster,contract.allow_completion)
    validate_frame_composition_v3(compat,frame)
