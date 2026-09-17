from __future__ import annotations

"""Full-surface -> compact Runtime-v4 shared-asset bridge."""

from typing import Mapping, Sequence
import numpy as np

from compiler.realsas_compiler_core.playback_full_surface_v3 import qualify_camera_v3
from compiler.realsas_compiler_core.playback_runtime_v3 import AttachmentKind, TopologyClass
from compiler.realsas_compiler_core.playback_runtime_v4 import RuntimeV4Asset, RuntimeV4Camera, RuntimeV4ViewOverlay
from compiler.realsas_compiler_core.types import QualificationError


def build_full_surface_runtime_v4(
    canonical_vertices,
    canonical_faces,
    cameras: Mapping[str, Mapping],
    uv_by_view: Mapping[str, np.ndarray],
    face_provenance_by_view: Mapping[str, np.ndarray],
    *,
    asset_id: str,
    slot_id: str,
    attachment_id: str,
    attachment_kind: AttachmentKind = AttachmentKind.DEFORMABLE_BODY,
    topology_class: TopologyClass = TopologyClass.STATIC,
    required_view_ids: Sequence[str] = tuple(f"V{i}" for i in range(8)),
):
    p=np.ascontiguousarray(canonical_vertices,dtype=np.float32)
    f=np.ascontiguousarray(canonical_faces,dtype=np.uint32)
    views=tuple(map(str,required_view_ids))
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<3 or not np.isfinite(p).all(): raise QualificationError("FULL_SURFACE_V4_CANONICAL_VERTICES_INVALID")
    if f.ndim!=2 or f.shape[1]!=3 or len(f)==0 or int(f.max())>=len(p): raise QualificationError("FULL_SURFACE_V4_CANONICAL_FACES_INVALID")
    if set(cameras)!=set(views) or set(uv_by_view)!=set(views) or set(face_provenance_by_view)!=set(views): raise QualificationError("FULL_SURFACE_V4_VIEW_SET_MISMATCH")
    asset=RuntimeV4Asset(str(asset_id),str(slot_id),str(attachment_id),attachment_kind,topology_class,p,f)
    out_cameras=[]; overlays=[]
    for i,view_id in enumerate(views):
        c=qualify_camera_v3(cameras[view_id],view_id=view_id,view_index=i)
        out_cameras.append(RuntimeV4Camera(c.view_id,c.view_index,c.origin,c.right,c.screen_up,c.forward,c.half_extent,c.resolution))
        uv=np.ascontiguousarray(uv_by_view[view_id],dtype=np.float32)
        prov=np.ascontiguousarray(face_provenance_by_view[view_id],dtype=np.uint8)
        if uv.shape!=(len(p),2) or prov.shape!=(len(f),): raise QualificationError("FULL_SURFACE_V4_OVERLAY_CARDINALITY_MISMATCH")
        overlays.append(RuntimeV4ViewOverlay(view_id,str(asset_id),uv,prov))
    return asset,tuple(out_cameras),tuple(overlays)
