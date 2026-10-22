from __future__ import annotations
from dataclasses import asdict,dataclass,field,replace
from typing import Any
from .graphics_v1 import CompiledGraphicsV1,compiled_graphics_hash_v1
from .hashing import content_sha256
from .types import QualificationError
Json=dict[str,Any]

@dataclass(frozen=True)
class RuntimeGraphicsViewV1:
    view_index:int; visible_fragment_indices:tuple[int,...]; topological_fragment_order:tuple[int,...]; order_edges:tuple[tuple[int,int],...]; schema_version:str="RealSaS.RuntimeGraphicsView.v1"
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class RuntimeGraphicsV1:
    canonical_mesh_binding_hash:str; compiled_graphics_binding_hash:str; face_fragment_index:tuple[int,...]; fragment_ids:tuple[str,...]; fragment_support_band_pixels:tuple[int,...]; views:tuple[RuntimeGraphicsViewV1,...]; attachments:tuple[Json,...]; runtime_graphics_hash:str; schema_version:str="RealSaS.RuntimeGraphics.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self):return asdict(self)
def runtime_graphics_hash_v1(v):
    p=v.to_dict();p.pop("runtime_graphics_hash",None);return content_sha256(p)
def _topological_order(n,edges):
    out={i:[] for i in range(n)};deg=[0]*n
    for a,b in edges:
        if a==b or not 0<=a<n or not 0<=b<n:raise QualificationError("RUNTIME_GRAPHICS_ORDER_EDGE_INVALID")
        out[a].append(b);deg[b]+=1
    ready=sorted(i for i,d in enumerate(deg) if d==0);order=[]
    while ready:
        x=ready.pop(0);order.append(x)
        for y in sorted(out[x]):
            deg[y]-=1
            if deg[y]==0:ready.append(y);ready.sort()
    if len(order)!=n:raise QualificationError("RUNTIME_GRAPHICS_ORDER_CYCLE")
    return tuple(order)
def build_runtime_graphics_v1(compiled:CompiledGraphicsV1,*,face_count:int):
    """Materialize graph CONTENT, not merely its hash, for runtime consumption."""
    if compiled.graph_hash!=compiled_graphics_hash_v1(compiled):raise QualificationError("RUNTIME_GRAPHICS_COMPILED_HASH_DRIFT")
    if face_count<=0 or not compiled.fragments:raise QualificationError("RUNTIME_GRAPHICS_INPUT_INVALID")
    fragments=tuple(compiled.fragments);index={f.fragment_id:i for i,f in enumerate(fragments)}
    if len(index)!=len(fragments):raise QualificationError("RUNTIME_GRAPHICS_FRAGMENT_ID_DUPLICATE")
    owner=[-1]*face_count
    for i,f in enumerate(fragments):
        for face in f.face_indices:
            if not 0<=int(face)<face_count or owner[int(face)]!=-1:raise QualificationError("RUNTIME_GRAPHICS_FACE_OWNER_INVALID")
            owner[int(face)]=i
    if any(x<0 for x in owner):raise QualificationError("RUNTIME_GRAPHICS_FACE_OWNER_INCOMPLETE")
    views=[]
    for v in sorted(compiled.views,key=lambda r:r.view_index):
        edges=tuple((index[e.front_fragment_id],index[e.back_fragment_id]) for e in v.order_edges);views.append(RuntimeGraphicsViewV1(int(v.view_index),tuple(sorted(index[x] for x in v.visible_fragment_ids)),_topological_order(len(fragments),edges),edges))
    attachments=tuple({"fragment_index":index[a.fragment_id],"parent_mechanical_domain_id":a.parent_mechanical_domain_id,"anchor_face_index":int(a.anchor_face_index),"barycentric_anchor":tuple(map(float,a.barycentric_anchor)),"confidence":float(a.confidence),"evidence_refs":tuple(a.evidence_refs)} for a in compiled.attachments)
    value=RuntimeGraphicsV1(compiled.canonical_mesh_binding_hash,compiled.graph_hash,tuple(owner),tuple(f.fragment_id for f in fragments),tuple(int(f.support_band_pixels) for f in fragments),tuple(views),attachments,"",metadata={"behaviorally_consumable":True,"hash_only_binding_forbidden":True,"triangle_array_order_is_presentation_authority":False,"single_canonical_mesh_authority":True,"silhouette_support_is_2d_only":True});return replace(value,runtime_graphics_hash=runtime_graphics_hash_v1(value))
