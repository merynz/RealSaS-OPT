from __future__ import annotations

"""Stage-36 projection from canonical product authorities to compact Runtime-v4.

This is a deployment projection, not a new geometry truth. Runtime vertices are
face-corner expansions that retain an exact source canonical vertex id, allowing UV
seams and per-face donor provenance without mutating canonical QualifiedMeshIR M.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping
import zlib

import numpy as np

from .camera_authority_v1 import QualifiedCameraSetIR, qualified_camera_set_hash
from .hashing import content_sha256
from .motion_dynamic_proof_v1 import QualifiedDynamicMotionIR, qualified_dynamic_motion_hash
from .playback_full_surface_v3 import CameraProjectionV3
from .playback_runtime_v3 import (
    AppearanceProvenance, AttachmentKind, RuntimeV3FrameComposition, RuntimeV3Slot,
    TopologyClass,
)
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload
from .playback_runtime_v4 import (
    RuntimeV4AttachmentAsset, RuntimeV4Clip, RuntimeV4Frame, RuntimeV4PlaybackContract,
    RuntimeV4ViewAssetOverlay, RuntimeV4ViewOverlay, provenance_code,
    validate_playback_runtime_v4_contract, validate_runtime_v4_clip,
)
from .product_appearance_v1 import QualifiedAppearanceSetIR, qualified_appearance_set_hash
from .product_composition_v1 import QualifiedCompositionSetIR, composition_set_hash
from .product_authority_v1 import (
    QualifiedMeshIR, QualifiedPresentationGraphIR, qualified_mesh_lineage_hash,
    qualified_presentation_lineage_hash,
)
from .types import QualificationError

Json=dict[str,Any]
Vec2=tuple[float,float]
Vec3=tuple[float,float,float]


@dataclass(frozen=True)
class RuntimeTextureBindingIR:
    view_index:int
    source_raster_sha256:str
    transport_png_path:str
    transport_png_sha256:str
    transport_png_crc32:int
    width:int
    height:int
    schema_version:str="RealSaS.RuntimeTextureBindingIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RuntimeProjectedAssetIR:
    asset_id:str
    slot_id:str
    attachment_id:str
    attachment_kind:str
    source_face_indices:tuple[int,...]
    source_vertex_ids:tuple[str,...]
    rest_xyz:tuple[Vec3,...]
    triangles:tuple[tuple[int,int,int],...]
    sealed_source_hash:str
    schema_version:str="RealSaS.RuntimeProjectedAssetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RuntimeProjectedViewAssetIR:
    asset_id:str
    uv:tuple[Vec2,...]
    provenance_codes:tuple[int,...]
    donor_view_indices:tuple[int,...]
    schema_version:str="RealSaS.RuntimeProjectedViewAssetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RuntimeProjectedViewIR:
    view_id:str
    view_index:int
    camera:Json
    assets:tuple[RuntimeProjectedViewAssetIR,...]
    draw_order_slot_ids:tuple[str,...]
    schema_version:str="RealSaS.RuntimeProjectedViewIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RuntimeProjectedFrameIR:
    time_seconds:float
    posed_xyz_by_asset:tuple[tuple[str,tuple[Vec3,...]],...]
    frame_hash:str
    schema_version:str="RealSaS.RuntimeProjectedFrameIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RuntimeProjectedClipIR:
    clip_id:str
    display_name:str
    intent:str
    classification:str
    duration_seconds:float
    fps:float
    loop:bool
    frames:tuple[RuntimeProjectedFrameIR,...]
    clip_projection_hash:str
    schema_version:str="RealSaS.RuntimeProjectedClipIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RuntimeV4ProjectionIR:
    dynamic_motion_binding_hash:str
    product_state_binding_hash:str
    mesh_binding_hash:str
    presentation_binding_hash:str
    appearance_binding_hash:str
    composition_binding_hash:str
    camera_set_binding_hash:str
    observation_set_binding_hash:str
    slots:tuple[Json,...]
    assets:tuple[RuntimeProjectedAssetIR,...]
    views:tuple[RuntimeProjectedViewIR,...]
    textures:tuple[RuntimeTextureBindingIR,...]
    clips:tuple[RuntimeProjectedClipIR,...]
    projection_hash:str
    schema_version:str="RealSaS.RuntimeV4ProjectionIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def _hash_without(value,field_name):
    payload=value.to_dict(); payload.pop(field_name,None)
    return content_sha256(payload)


def runtime_projected_frame_hash(value): return _hash_without(value,"frame_hash")
def runtime_projected_clip_hash(value): return _hash_without(value,"clip_projection_hash")
def runtime_v4_projection_hash(value): return _hash_without(value,"projection_hash")


def _camera_dict(camera:CameraProjectionV3)->Json:
    return {
        "view_id":camera.view_id,"view_index":int(camera.view_index),
        "origin":tuple(map(float,camera.origin)),"right":tuple(map(float,camera.right)),
        "screen_up":tuple(map(float,camera.screen_up)),"forward":tuple(map(float,camera.forward)),
        "half_extent":float(camera.half_extent),"resolution":int(camera.resolution),
        "schema_version":camera.schema_version,
    }


def _camera_from_dict(raw:Mapping[str,Any])->CameraProjectionV3:
    return CameraProjectionV3(
        str(raw["view_id"]),int(raw["view_index"]),tuple(map(float,raw["origin"])),
        tuple(map(float,raw["right"])),tuple(map(float,raw["screen_up"])),
        tuple(map(float,raw["forward"])),float(raw["half_extent"]),int(raw["resolution"]),
        str(raw.get("schema_version") or "RealSaS.FullSurfaceCameraProjection.v3"),
    )


def _asset_kind(mechanical_class:str)->str:
    if mechanical_class=="RIGID": return AttachmentKind.RIGID_COMPONENT.value
    if mechanical_class=="DEFORMABLE": return AttachmentKind.DEFORMABLE_BODY.value
    raise QualificationError("RUNTIME_PROJECTION_MECHANICAL_CLASS_UNSUPPORTED")


def _provenance(authority_class:str)->AppearanceProvenance:
    if authority_class=="OBSERVED_LOCAL": return AppearanceProvenance.DIRECT_SOURCE
    if authority_class=="OBSERVED_CROSS_VIEW": return AppearanceProvenance.OTHER_VIEW_SOURCE
    raise QualificationError("RUNTIME_PROJECTION_APPEARANCE_AUTHORITY_UNSUPPORTED")


def _project_assets(mesh:QualifiedMeshIR,presentation:QualifiedPresentationGraphIR):
    vertex={v.canonical_mesh_vertex_id:v for v in mesh.vertices}
    face_component=[]
    for face in mesh.faces:
        comps={vertex[vid].component_id for vid in face}
        if len(comps)!=1: raise QualificationError("RUNTIME_PROJECTION_FACE_COMPONENT_DRIFT")
        face_component.append(next(iter(comps)))
    claimed=set(); assets=[]; face_by_asset={}
    for attachment in sorted(presentation.attachments,key=lambda a:a.attachment_id):
        components=set(attachment.mechanical_component_ids)
        explicit=tuple(int(x) for x in tuple((attachment.metadata or {}).get("mesh_face_indices") or ()))
        if explicit:
            if len(explicit)!=len(set(explicit)) or any(i<0 or i>=len(mesh.faces) for i in explicit):
                raise QualificationError("RUNTIME_PROJECTION_ATTACHMENT_FACE_INDEX_INVALID")
            if any(face_component[i] not in components for i in explicit):
                raise QualificationError("RUNTIME_PROJECTION_ATTACHMENT_FACE_COMPONENT_DRIFT")
            face_indices=tuple(sorted(explicit))
        else:
            # Backward-compatible fixture path. Canonical Stage30 v2 always emits
            # explicit face groups so multiple slots may share one mechanical component.
            face_indices=tuple(i for i,c in enumerate(face_component) if c in components)
        if not face_indices: raise QualificationError("RUNTIME_PROJECTION_ATTACHMENT_WITHOUT_FACE")
        if claimed.intersection(face_indices): raise QualificationError("RUNTIME_PROJECTION_FACE_ATTACHMENT_OVERLAP")
        claimed.update(face_indices)
        source_ids=[]; rest=[]; tris=[]
        for local_face,fi in enumerate(face_indices):
            face=mesh.faces[fi]
            for vid in face:
                source_ids.append(str(vid)); rest.append(tuple(map(float,vertex[vid].P)))
            base=local_face*3; tris.append((base,base+1,base+2))
        sealed=content_sha256({
            "schema":"RealSaS.RuntimeProjectedAssetSource.v1","mesh":mesh.mesh_lineage_hash,
            "attachment_id":attachment.attachment_id,"source_face_indices":face_indices,
            "source_vertex_ids":source_ids,
        })
        asset_id="ASSET:"+content_sha256({"mesh":mesh.mesh_lineage_hash,"attachment":attachment.attachment_id})[:24]
        assets.append(RuntimeProjectedAssetIR(
            asset_id,attachment.slot_id,attachment.attachment_id,_asset_kind(attachment.mechanical_class),
            face_indices,tuple(source_ids),tuple(rest),tuple(tris),sealed,
            metadata={
                "projection_vertex_mode":"FACE_CORNER_EXPANSION",
                "canonical_geometry_mutated":False,
                "presentation_group_hash":str((attachment.metadata or {}).get("presentation_group_hash") or ""),
                "presentation_group_face_indices":face_indices,
            },
        ))
        face_by_asset[asset_id]=face_indices
    if claimed!=set(range(len(mesh.faces))):
        raise QualificationError("RUNTIME_PROJECTION_FACE_ACCOUNTING_INCOMPLETE")
    return tuple(assets),face_by_asset


def _project_views(*,assets,face_by_asset,appearance,composition,camera_set):
    binding_by_view={b.target_view_index:b for b in appearance.bindings}
    composition_by_view={v.view_index:v for v in composition.views}
    views=[]
    for camera in sorted(camera_set.cameras,key=lambda c:c.view_index):
        binding=binding_by_view[camera.view_index]
        corner={(c.face_index,c.corner_index):c for c in binding.corner_bindings}
        rows=[]
        for asset in assets:
            uv=[]; prov=[]; donors=[]
            for fi in face_by_asset[asset.asset_id]:
                face_corners=[corner[(fi,ci)] for ci in range(3)]
                face_donors={int(c.donor_view_index) for c in face_corners}
                face_auth={str(c.authority_class) for c in face_corners}
                if len(face_donors)!=1 or len(face_auth)!=1:
                    raise QualificationError("RUNTIME_PROJECTION_FACE_APPEARANCE_NOT_UNIFORM")
                uv.extend(tuple(map(float,c.material_uv)) for c in face_corners)
                provenance=_provenance(next(iter(face_auth)))
                prov.append(provenance_code(provenance)); donors.append(next(iter(face_donors)))
            rows.append(RuntimeProjectedViewAssetIR(asset.asset_id,tuple(uv),tuple(prov),tuple(donors)))
        comp=composition_by_view[camera.view_index]
        views.append(RuntimeProjectedViewIR(
            camera.view_id,camera.view_index,_camera_dict(camera),tuple(rows),tuple(comp.slot_order),
            metadata={"physical_occlusion_rule":comp.physical_occlusion_rule,"slot_order_role":comp.slot_order_role},
        ))
    return tuple(views)


def _project_clips(*,assets,dynamic):
    clips=[]
    for clip in dynamic.clips:
        frames=[]
        for frame in clip.frames:
            by_vid=dict(frame.posed_vertex_xyz)
            posed=[]
            for asset in assets:
                rows=[]
                for vid in asset.source_vertex_ids:
                    if vid not in by_vid: raise QualificationError("RUNTIME_PROJECTION_DYNAMIC_VERTEX_MISSING")
                    rows.append(tuple(map(float,by_vid[vid])))
                posed.append((asset.asset_id,tuple(rows)))
            out=RuntimeProjectedFrameIR(
                float(frame.time_seconds),tuple(posed),"",
                metadata={"source_dynamic_frame_hash":frame.frame_hash},
            )
            frames.append(replace(out,frame_hash=runtime_projected_frame_hash(out)))
        fps=max(1,len(frames)-1)/float(clip.duration_seconds)
        out=RuntimeProjectedClipIR(
            clip.clip_id,clip.clip_id,clip.clip_kind,clip.classification,float(clip.duration_seconds),
            float(fps),bool(clip.loop),tuple(frames),"",
            metadata={
                "source_dynamic_clip_hash":clip.clip_proof_hash,
                "professional_motion_evidence":clip.professional_motion_evidence,
                "runtime_qualified_from_stage35":True,
            },
        )
        clips.append(replace(out,clip_projection_hash=runtime_projected_clip_hash(out)))
    return tuple(clips)


def to_runtime_v4_objects(value:RuntimeV4ProjectionIR):
    slots=tuple(RuntimeV3Slot(
        str(row["slot_id"]),str(row["bone_id"]),int(row["setup_order"]),
        None if row.get("default_attachment_id") is None else str(row["default_attachment_id"])
    ) for row in value.slots)
    assets=tuple(RuntimeV4AttachmentAsset(
        asset_id=a.asset_id,slot_id=a.slot_id,attachment_id=a.attachment_id,
        attachment_kind=AttachmentKind(a.attachment_kind),topology_class=TopologyClass.STATIC,
        rest_xyz=np.asarray(a.rest_xyz,dtype=np.float32),triangles=np.asarray(a.triangles,dtype=np.uint32),
        sealed_source_hash=a.sealed_source_hash,
    ) for a in value.assets)
    views=tuple(RuntimeV4ViewOverlay(
        view_id=v.view_id,view_index=v.view_index,camera=_camera_from_dict(v.camera),
        assets=tuple(RuntimeV4ViewAssetOverlay(
            asset_id=a.asset_id,uv=np.asarray(a.uv,dtype=np.float32),
            provenance_codes=np.asarray(a.provenance_codes,dtype=np.uint8),
            donor_view_indices=np.asarray(a.donor_view_indices,dtype=np.int16),
        ) for a in v.assets),
    ) for v in value.views)
    contract=RuntimeV4PlaybackContract(slots=slots,assets=assets,views=views,allow_completion=False)
    default_active={slot.slot_id:slot.default_attachment_id for slot in slots}
    clips=[]
    view_order={v.view_id:v.draw_order_slot_ids for v in value.views}
    for c in value.clips:
        frames=tuple(RuntimeV4Frame(
            time_seconds=f.time_seconds,
            canonical_posed_xyz_by_asset={aid:xyz for aid,xyz in f.posed_xyz_by_asset},
            composition_by_view={view_id:RuntimeV3FrameComposition(
                view_id=view_id,draw_order_slot_ids=tuple(order),
                active_attachment_by_slot=dict(default_active),clip_intervals=(),
            ) for view_id,order in view_order.items()},
        ) for f in c.frames)
        clips.append(RuntimeV4Clip(
            c.clip_id,c.display_name,c.intent,c.duration_seconds,c.fps,c.loop,frames,True
        ))
    textures=tuple(RuntimeV3TexturePayload(
        f"V{t.view_index}",t.transport_png_path,t.transport_png_sha256,t.transport_png_crc32,t.width,t.height
    ) for t in sorted(value.textures,key=lambda x:x.view_index))
    return contract,tuple(textures),tuple(clips)


def build_runtime_v4_projection(
    *,mesh:QualifiedMeshIR,presentation:QualifiedPresentationGraphIR,
    appearance:QualifiedAppearanceSetIR,composition:QualifiedCompositionSetIR,
    camera_set:QualifiedCameraSetIR,dynamic:QualifiedDynamicMotionIR,
    observation_set_hash:str,textures:tuple[RuntimeTextureBindingIR,...],
)->RuntimeV4ProjectionIR:
    if mesh.mesh_lineage_hash!=qualified_mesh_lineage_hash(mesh):
        raise QualificationError("RUNTIME_PROJECTION_MESH_HASH_DRIFT")
    if presentation.presentation_lineage_hash!=qualified_presentation_lineage_hash(presentation):
        raise QualificationError("RUNTIME_PROJECTION_PRESENTATION_HASH_DRIFT")
    if appearance.appearance_set_hash!=qualified_appearance_set_hash(appearance):
        raise QualificationError("RUNTIME_PROJECTION_APPEARANCE_HASH_DRIFT")
    if composition.composition_set_hash!=composition_set_hash(composition):
        raise QualificationError("RUNTIME_PROJECTION_COMPOSITION_HASH_DRIFT")
    if camera_set.camera_set_hash!=qualified_camera_set_hash(camera_set):
        raise QualificationError("RUNTIME_PROJECTION_CAMERA_HASH_DRIFT")
    if dynamic.dynamic_motion_hash!=qualified_dynamic_motion_hash(dynamic):
        raise QualificationError("RUNTIME_PROJECTION_DYNAMIC_HASH_DRIFT")
    expected=(
        (presentation.mesh_binding_hash,mesh.mesh_lineage_hash,"PRESENTATION_MESH"),
        (appearance.mesh_binding_hash,mesh.mesh_lineage_hash,"APPEARANCE_MESH"),
        (composition.mesh_binding_hash,mesh.mesh_lineage_hash,"COMPOSITION_MESH"),
        (dynamic.mesh_binding_hash,mesh.mesh_lineage_hash,"DYNAMIC_MESH"),
        (appearance.observation_set_binding_hash,observation_set_hash,"APPEARANCE_OBSERVATION"),
        (composition.observation_set_binding_hash,observation_set_hash,"COMPOSITION_OBSERVATION"),
    )
    for a,b,label in expected:
        if a!=b: raise QualificationError(f"RUNTIME_PROJECTION_BINDING_DRIFT:{label}")
    if len(textures)!=8 or {t.view_index for t in textures}!=set(range(8)):
        raise QualificationError("RUNTIME_PROJECTION_TEXTURE_MATRIX_INCOMPLETE")
    slots=tuple({
        "slot_id":s.slot_id,"bone_id":s.bone_id,"setup_order":int(s.setup_order),
        "default_attachment_id":s.default_attachment_id,
    } for s in sorted(presentation.slots,key=lambda s:s.setup_order))
    assets,face_by_asset=_project_assets(mesh,presentation)
    views=_project_views(
        assets=assets,face_by_asset=face_by_asset,appearance=appearance,
        composition=composition,camera_set=camera_set,
    )
    clips=_project_clips(assets=assets,dynamic=dynamic)
    value=RuntimeV4ProjectionIR(
        dynamic.dynamic_motion_hash,dynamic.product_state_binding_hash,mesh.mesh_lineage_hash,presentation.presentation_lineage_hash,
        appearance.appearance_set_hash,composition.composition_set_hash,camera_set.camera_set_hash,
        str(observation_set_hash),slots,assets,views,tuple(sorted(textures,key=lambda x:x.view_index)),
        clips,"",
        metadata={
            "representation":"SHARED_CANONICAL_ASSET__VIEW_OVERLAY__CANONICAL_XYZ_ONCE_PER_FRAME",
            "canonical_geometry_mutated":False,
            "runtime_projection_vertex_expansion":"FACE_CORNER_ONLY_FOR_UV_SEAMS",
            "view_specific_motion_mesh_truth":False,
            "allow_completion":False,
        },
    )
    value=replace(value,projection_hash=runtime_v4_projection_hash(value))
    contract,_,runtime_clips=to_runtime_v4_objects(value)
    validate_playback_runtime_v4_contract(contract)
    for clip in runtime_clips: validate_runtime_v4_clip(contract,clip)
    return value


def runtime_v4_projection_from_dict(payload:Mapping[str,Any])->RuntimeV4ProjectionIR:
    if str(payload.get("schema_version") or payload.get("schema") or "")!="RealSaS.RuntimeV4ProjectionIR.v1":
        raise ValueError("RUNTIME_PROJECTION_SCHEMA_MISMATCH")
    assets=tuple(RuntimeProjectedAssetIR(
        str(a["asset_id"]),str(a["slot_id"]),str(a["attachment_id"]),str(a["attachment_kind"]),
        tuple(map(int,a.get("source_face_indices") or ())),tuple(map(str,a.get("source_vertex_ids") or ())),
        tuple(tuple(map(float,p)) for p in a.get("rest_xyz") or ()),
        tuple(tuple(map(int,t)) for t in a.get("triangles") or ()),str(a["sealed_source_hash"]),
        schema_version=str(a.get("schema_version") or "RealSaS.RuntimeProjectedAssetIR.v1"),
        metadata=dict(a.get("metadata") or {}),
    ) for a in payload.get("assets") or ())
    views=[]
    for v in payload.get("views") or ():
        va=tuple(RuntimeProjectedViewAssetIR(
            str(a["asset_id"]),tuple(tuple(map(float,p)) for p in a.get("uv") or ()),
            tuple(map(int,a.get("provenance_codes") or ())),tuple(map(int,a.get("donor_view_indices") or ())),
            schema_version=str(a.get("schema_version") or "RealSaS.RuntimeProjectedViewAssetIR.v1"),
            metadata=dict(a.get("metadata") or {}),
        ) for a in v.get("assets") or ())
        views.append(RuntimeProjectedViewIR(
            str(v["view_id"]),int(v["view_index"]),dict(v["camera"]),va,
            tuple(map(str,v.get("draw_order_slot_ids") or ())),
            schema_version=str(v.get("schema_version") or "RealSaS.RuntimeProjectedViewIR.v1"),
            metadata=dict(v.get("metadata") or {}),
        ))
    textures=tuple(RuntimeTextureBindingIR(
        int(t["view_index"]),str(t["source_raster_sha256"]),str(t["transport_png_path"]),
        str(t["transport_png_sha256"]),int(t["transport_png_crc32"]),int(t["width"]),int(t["height"]),
        schema_version=str(t.get("schema_version") or "RealSaS.RuntimeTextureBindingIR.v1"),
        metadata=dict(t.get("metadata") or {}),
    ) for t in payload.get("textures") or ())
    clips=[]
    for c in payload.get("clips") or ():
        frames=tuple(RuntimeProjectedFrameIR(
            float(f["time_seconds"]),
            tuple((str(aid),tuple(tuple(map(float,p)) for p in xyz)) for aid,xyz in f.get("posed_xyz_by_asset") or ()),
            str(f["frame_hash"]),
            schema_version=str(f.get("schema_version") or "RealSaS.RuntimeProjectedFrameIR.v1"),
            metadata=dict(f.get("metadata") or {}),
        ) for f in c.get("frames") or ())
        clips.append(RuntimeProjectedClipIR(
            str(c["clip_id"]),str(c["display_name"]),str(c["intent"]),str(c["classification"]),
            float(c["duration_seconds"]),float(c["fps"]),bool(c["loop"]),frames,str(c["clip_projection_hash"]),
            schema_version=str(c.get("schema_version") or "RealSaS.RuntimeProjectedClipIR.v1"),
            metadata=dict(c.get("metadata") or {}),
        ))
    value=RuntimeV4ProjectionIR(
        str(payload["dynamic_motion_binding_hash"]),str(payload["product_state_binding_hash"]),str(payload["mesh_binding_hash"]),
        str(payload["presentation_binding_hash"]),str(payload["appearance_binding_hash"]),
        str(payload["composition_binding_hash"]),str(payload["camera_set_binding_hash"]),
        str(payload["observation_set_binding_hash"]),tuple(dict(x) for x in payload.get("slots") or ()),
        assets,tuple(views),textures,tuple(clips),str(payload["projection_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.RuntimeV4ProjectionIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.projection_hash!=runtime_v4_projection_hash(value): raise ValueError("RUNTIME_PROJECTION_HASH_MISMATCH")
    for c in value.clips:
        if c.clip_projection_hash!=runtime_projected_clip_hash(c): raise ValueError("RUNTIME_PROJECTION_CLIP_HASH_MISMATCH")
        for f in c.frames:
            if f.frame_hash!=runtime_projected_frame_hash(f): raise ValueError("RUNTIME_PROJECTION_FRAME_HASH_MISMATCH")
    contract,_,runtime_clips=to_runtime_v4_objects(value)
    validate_playback_runtime_v4_contract(contract)
    for clip in runtime_clips: validate_runtime_v4_clip(contract,clip)
    return value
