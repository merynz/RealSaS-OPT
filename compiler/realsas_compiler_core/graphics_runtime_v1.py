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
    if type(face_count) is not int or face_count<=0 or not compiled.fragments:
        raise QualificationError("RUNTIME_GRAPHICS_INPUT_INVALID")
    fragments=tuple(compiled.fragments)
    index={f.fragment_id:i for i,f in enumerate(fragments)}
    if (len(index)!=len(fragments)
        or any(type(f.fragment_id) is not str or not f.fragment_id for f in fragments)):
        raise QualificationError("RUNTIME_GRAPHICS_FRAGMENT_ID_DUPLICATE")
    owner=[-1]*face_count
    for i,f in enumerate(fragments):
        if not f.face_indices:
            raise QualificationError("RUNTIME_GRAPHICS_FACE_OWNER_INVALID")
        if type(f.support_band_pixels) is not int or not 0<=f.support_band_pixels<=4:
            raise QualificationError("RUNTIME_GRAPHICS_SUPPORT_BAND_INVALID")
        for face in f.face_indices:
            # Never coerce a forged/rehashed 0.7, True, or "0" into canonical
            # face zero. Content hashes establish identity, not typed validity.
            if (type(face) is not int or not 0<=face<face_count or owner[face]!=-1):
                raise QualificationError("RUNTIME_GRAPHICS_FACE_OWNER_INVALID")
            owner[face]=i
    if any(x<0 for x in owner):
        raise QualificationError("RUNTIME_GRAPHICS_FACE_OWNER_INCOMPLETE")
    views=[]
    seen_views=set()
    for v in sorted(compiled.views,key=lambda r:str(r.view_index)):
        if (type(v.view_index) is not int or not 0<=v.view_index<8
            or v.view_index in seen_views):
            raise QualificationError("RUNTIME_GRAPHICS_VIEW_ADDRESS_INVALID")
        seen_views.add(v.view_index)
        if (len(set(v.visible_fragment_ids))!=len(v.visible_fragment_ids)
            or any(x not in index for x in v.visible_fragment_ids)):
            raise QualificationError("RUNTIME_GRAPHICS_VISIBLE_FRAGMENT_INVALID")
        edges=[]
        for e in v.order_edges:
            if (type(e.view_index) is not int or e.view_index!=v.view_index
                or e.front_fragment_id not in index or e.back_fragment_id not in index):
                raise QualificationError("RUNTIME_GRAPHICS_ORDER_EDGE_ADDRESS_INVALID")
            edges.append((index[e.front_fragment_id],index[e.back_fragment_id]))
        edges=tuple(edges)
        views.append(RuntimeGraphicsViewV1(
            v.view_index,tuple(sorted(index[x] for x in v.visible_fragment_ids)),
            _topological_order(len(fragments),edges),edges,
        ))
    attachments=[]
    for a in compiled.attachments:
        if (a.fragment_id not in index or type(a.anchor_face_index) is not int
            or not 0<=a.anchor_face_index<face_count
            or not a.parent_mechanical_domain_id):
            raise QualificationError("RUNTIME_GRAPHICS_ATTACHMENT_ADDRESS_INVALID")
        if (len(a.barycentric_anchor)!=3
            or any(type(x) not in (float,int) or not 0<=x<=1 for x in a.barycentric_anchor)
            or abs(sum(a.barycentric_anchor)-1.0)>1e-6):
            raise QualificationError("RUNTIME_GRAPHICS_ATTACHMENT_BARYCENTRIC_INVALID")
        attachments.append({
            "fragment_index":index[a.fragment_id],
            "parent_mechanical_domain_id":a.parent_mechanical_domain_id,
            "anchor_face_index":a.anchor_face_index,
            "barycentric_anchor":tuple(map(float,a.barycentric_anchor)),
            "confidence":float(a.confidence),
            "evidence_refs":tuple(a.evidence_refs),
        })
    attachments=tuple(attachments)
    value=RuntimeGraphicsV1(compiled.canonical_mesh_binding_hash,compiled.graph_hash,tuple(owner),tuple(f.fragment_id for f in fragments),tuple(int(f.support_band_pixels) for f in fragments),tuple(views),attachments,"",metadata={
        "behaviorally_consumable": True,
        "hash_only_binding_forbidden": True,
        "triangle_array_order_is_presentation_authority": False,
        "single_canonical_mesh_authority": True,
        "silhouette_support_is_2d_only": True,
        # A topological tie-break can make a drawable schedule executable,
        # but cannot prove the removed source-evidence relation was false.
        # Keep representation policy distinct from observed source truth.
        "unresolved_source_order_conflict": any(
            bool(view.metadata.get("cycle_edges_removed"))
            for view in compiled.views
        ),
        "representation_policy_arbitration_used": any(
            bool(view.metadata.get("cycle_edges_removed"))
            for view in compiled.views
        ),
        # View-index schedules are not a pose-conditioned occlusion solver.
        "pose_conditioned_order_execution": False,
        # Inference coverage cannot become source-ray truth merely by
        # materializing eight view schedules. A runtime depth test validates
        # pose-local geometry, not the missing independent source observation.
        "source_visibility_coverage_complete": False,
        "learned_visibility_candidate_coverage_complete": bool(compiled.views) and not any(
            bool(v.metadata.get("visibility_unknown_fragment_ids"))
            for v in compiled.views
        ),
        "source_visibility_requires_independent_ray_receipt": True,
        "view_relation_eight_direction_address_coverage": tuple(
            sorted(v.view_index for v in compiled.views)
        ) == tuple(range(8)),
        "unknown_visibility_fragment_ids_by_view": {
            str(v.view_index): tuple(v.metadata.get("visibility_unknown_fragment_ids") or ())
            for v in compiled.views if v.metadata.get("visibility_unknown_fragment_ids")
        },
        "unknown_visibility_raster_policy": "DRAW_GEOMETRY_AND_DEFER_OCCLUSION_TO_DEPTH",
    });return replace(value,runtime_graphics_hash=runtime_graphics_hash_v1(value))
