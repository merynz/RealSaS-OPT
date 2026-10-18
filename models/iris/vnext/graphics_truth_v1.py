from __future__ import annotations
from dataclasses import asdict,dataclass,field
from typing import Any,Iterable,Mapping
import numpy as np
Json=dict[str,Any]

@dataclass(frozen=True)
class GraphicsStaticTruthRowV1:
    face_a:int; face_b:int; same_drawable_affinity:float; cut_probability:float; attachment_affinity:float; alpha_boundary_probability:float; appearance_boundary_probability:float; schema_version:str="RealSaS.GraphicsStaticTruthRow.v1"
@dataclass(frozen=True)
class GraphicsViewTruthRowV1:
    view_index:int; face_a:int; face_b:int; front_probability:float; occlusion_probability:float; visibility_probability_a:float; visibility_probability_b:float; schema_version:str="RealSaS.GraphicsViewTruthRow.v1"
@dataclass(frozen=True)
class GraphicsRelationTruthV1:
    static_rows:tuple[GraphicsStaticTruthRowV1,...]; view_rows:tuple[GraphicsViewTruthRowV1,...]; face_count:int; view_count:int; schema_version:str="RealSaS.GraphicsRelationTruth.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)

def _adjacent(faces):
    edges={}; pairs=set()
    for fi,(a,b,c) in enumerate(faces.tolist()):
        for u,v in ((a,b),(b,c),(c,a)): edges.setdefault(tuple(sorted((int(u),int(v)))),[]).append(fi)
    for rows in edges.values():
        rows=sorted(set(rows))
        for i,a in enumerate(rows):
            for b in rows[i+1:]: pairs.add((a,b))
    return tuple(sorted(pairs))

def build_graphics_relation_truth_v1(*,faces,anonymous_drawable_ids:Iterable[str],visible_face_mask,face_depth,occluding_face_pairs_by_view:Mapping[int,Iterable[tuple[int,int]]],alpha_boundary_by_adjacent_pair=None,appearance_boundary_by_adjacent_pair=None,attachment_affinity_by_pair=None):
    """Role-free supervision from anonymous drawable equivalence + raster overlap/depth."""
    tri=np.asarray(faces,dtype=np.int64); ids=tuple(map(str,anonymous_drawable_ids)); visible=np.asarray(visible_face_mask,dtype=bool); depth=np.asarray(face_depth,dtype=np.float64)
    if tri.ndim!=2 or tri.shape[1]!=3 or len(tri)==0: raise ValueError("GRAPHICS_TRUTH_FACES_INVALID")
    if len(ids)!=len(tri) or any(not x for x in ids): raise ValueError("GRAPHICS_TRUTH_DRAWABLE_ID_INVALID")
    if visible.ndim!=2 or visible.shape[1]!=len(tri) or depth.shape!=visible.shape: raise ValueError("GRAPHICS_TRUTH_VIEW_SHAPE_INVALID")
    if np.any(visible & ~np.isfinite(depth)): raise ValueError("GRAPHICS_TRUTH_VISIBLE_DEPTH_NONFINITE")
    alpha=dict(alpha_boundary_by_adjacent_pair or {}); appearance=dict(appearance_boundary_by_adjacent_pair or {}); attachment=dict(attachment_affinity_by_pair or {}); static=[]
    for a,b in _adjacent(tri):
        same=float(ids[a]==ids[b]); pair=(a,b); static.append(GraphicsStaticTruthRowV1(a,b,same,1.-same,float(attachment.get(pair,0.)),float(alpha.get(pair,0.)),float(appearance.get(pair,0.))))
    views=[]
    for view,pairs in sorted(occluding_face_pairs_by_view.items()):
        view=int(view)
        if not 0<=view<visible.shape[0]: raise ValueError("GRAPHICS_TRUTH_VIEW_INDEX_INVALID")
        for ra,rb in pairs:
            a,b=int(ra),int(rb)
            if a==b or min(a,b)<0 or max(a,b)>=len(tri): raise ValueError("GRAPHICS_TRUTH_OCCLUSION_PAIR_INVALID")
            va,vb=bool(visible[view,a]),bool(visible[view,b])
            if not (va or vb): continue
            if va and vb:
                da,db=float(depth[view,a]),float(depth[view,b])
                if abs(da-db)<=1e-12: continue
                front=float(da<db)
            else: front=float(va)
            views.append(GraphicsViewTruthRowV1(view,a,b,front,1.,float(va),float(vb)))
    return GraphicsRelationTruthV1(tuple(static),tuple(views),len(tri),visible.shape[0],metadata={"categorical_semantics_used":False,"anonymous_equivalence_ids_only":True,"teacher_drawable_ids_exposed_to_inference":False,"depth_ties_ignored_not_invented":True,"occlusion_rows_require_raster_overlap_teacher":True})
