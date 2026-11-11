from __future__ import annotations
from dataclasses import asdict,dataclass,field,replace
from typing import Any,Iterable,Mapping
from .hashing import content_sha256
from .types import QualificationError
Json=dict[str,Any]

@dataclass(frozen=True)
class GraphicsCompilePolicyV1:
    same_drawable_min:float=.65; cut_max:float=.35; boundary_max:float=.50; relation_confidence_min:float=.50; front_probability_min:float=.65; occlusion_probability_min:float=.50; visibility_probability_min:float=.50; attachment_affinity_min:float=.70; mechanical_contact_min:float=.75; support_band_alpha_boundary_min:float=.65; maximum_support_band_pixels:int=2; allow_unqualified_order_arbitration:bool=False
    def validate(self):
        vals=(self.same_drawable_min,self.cut_max,self.boundary_max,self.relation_confidence_min,self.front_probability_min,self.occlusion_probability_min,self.visibility_probability_min,self.attachment_affinity_min,self.mechanical_contact_min,self.support_band_alpha_boundary_min)
        if any(not 0<=float(v)<=1 for v in vals) or not 0<=self.maximum_support_band_pixels<=4: raise QualificationError("GRAPHICS_POLICY_INVALID")
@dataclass(frozen=True)
class GraphicsStaticRelationEvidenceV1:
    face_a:int; face_b:int; same_drawable_affinity:float; cut_probability:float; attachment_affinity:float; alpha_boundary_probability:float; appearance_boundary_probability:float; confidence:float; evidence_ref:str=""; schema_version:str="RealSaS.GraphicsStaticRelationEvidence.v1"
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class GraphicsViewRelationEvidenceV1:
    view_index:int; face_a:int; face_b:int; front_probability:float; occlusion_probability:float; visibility_probability_a:float; visibility_probability_b:float; confidence:float; evidence_ref:str=""; schema_version:str="RealSaS.GraphicsViewRelationEvidence.v1"
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class GraphicsEvidenceV1:
    mesh_binding_hash:str; static_relations:tuple[GraphicsStaticRelationEvidenceV1,...]; view_relations:tuple[GraphicsViewRelationEvidenceV1,...]; evidence_hash:str; schema_version:str="RealSaS.GraphicsEvidence.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class GraphicsFragmentV1:
    fragment_id:str; face_indices:tuple[int,...]; support_band_pixels:int; schema_version:str="RealSaS.GraphicsFragment.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class GraphicsOrderEdgeV1:
    view_index:int; front_fragment_id:str; back_fragment_id:str; confidence:float; evidence_refs:tuple[str,...]; schema_version:str="RealSaS.GraphicsOrderEdge.v1"
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class GraphicsAttachmentV1:
    fragment_id:str; parent_mechanical_domain_id:str; anchor_face_index:int; barycentric_anchor:tuple[float,float,float]; confidence:float; evidence_refs:tuple[str,...]; schema_version:str="RealSaS.GraphicsAttachment.v1"
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class GraphicsViewStateV1:
    view_index:int; visible_fragment_ids:tuple[str,...]; order_edges:tuple[GraphicsOrderEdgeV1,...]; schema_version:str="RealSaS.GraphicsViewState.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self):return asdict(self)
@dataclass(frozen=True)
class CompiledGraphicsV1:
    canonical_mesh_binding_hash:str; canonical_mesh_skin_binding_hash:str; source_graphics_binding_hash:str; graphics_evidence_binding_hash:str; fragments:tuple[GraphicsFragmentV1,...]; views:tuple[GraphicsViewStateV1,...]; attachments:tuple[GraphicsAttachmentV1,...]; graph_hash:str; schema_version:str="RealSaS.CompiledGraphics.v1"; metadata:Json=field(default_factory=dict)
    def to_dict(self):return asdict(self)

def _hash(payload,key): body=dict(payload);body.pop(key,None);return content_sha256(body)
def graphics_evidence_hash_v1(v):return _hash(v.to_dict(),"evidence_hash")
def compiled_graphics_hash_v1(v):return _hash(v.to_dict(),"graph_hash")
def build_graphics_evidence_v1(*,mesh_binding_hash,static_relations:Iterable[GraphicsStaticRelationEvidenceV1],view_relations:Iterable[GraphicsViewRelationEvidenceV1],metadata:Mapping[str,Any]|None=None):
    static,view=tuple(static_relations),tuple(view_relations)
    if not mesh_binding_hash or not static: raise QualificationError("GRAPHICS_EVIDENCE_INCOMPLETE")
    value=GraphicsEvidenceV1(str(mesh_binding_hash),static,view,"",metadata={"categorical_semantics_used":False,"authority_class":"LEARNED_RELATION_EVIDENCE_ONLY",**dict(metadata or {})});return replace(value,evidence_hash=graphics_evidence_hash_v1(value))

class _UF:
    """Sparse cannot-link constrained union for anonymous drawable candidates.

    A chain of locally high same-drawable affinities cannot transitively erase
    an explicit high-cut face relation. Sorting candidate edges makes the
    representation policy deterministic, independent of input row ordering.
    """
    def __init__(self,n):
        self.p=list(range(n))
        self.members={i:{i} for i in range(n)}
        self.forbidden={i:set() for i in range(n)}

    def find(self,x):
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x

    def cannot_link(self,a,b):
        self.forbidden[a].add(b)
        self.forbidden[b].add(a)

    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:
            return True
        # Keep union-by-size so each face's membership migrates O(log N)
        # times; the final fragment identity is sorted face IDs, not UF root.
        if len(self.members[a])<len(self.members[b]):
            a,b=b,a
        if any(self.forbidden[x].intersection(self.members[a])
               for x in self.members[b]):
            return False
        self.p[b]=a
        self.members[a].update(self.members.pop(b))
        return True

def _adj(faces):
    e={};out=set()
    for fi,(a,b,c) in enumerate(faces):
        for u,v in ((a,b),(b,c),(c,a)):e.setdefault(tuple(sorted((u,v))),[]).append(fi)
    for rows in e.values():
        rows=sorted(set(rows))
        for i,a in enumerate(rows):
            for b in rows[i+1:]:out.add((a,b))
    return out

def _prob(v,label):
    try:
        if isinstance(v, bool):
            raise ValueError("boolean is not a probability")
        value=float(v)
    except (ValueError, TypeError, OverflowError):
        raise QualificationError("GRAPHICS_PROBABILITY_INVALID:"+label) from None
    if not 0<=value<=1:
        raise QualificationError("GRAPHICS_PROBABILITY_INVALID:"+label)
    return value

def _cycle(edges):
    graph={};nodes=set()
    for e in edges:graph.setdefault(e.front_fragment_id,[]).append(e);nodes|={e.front_fragment_id,e.back_fragment_id}
    visiting=set();done=set();nodes_stack=[];edges_stack=[]
    def dfs(n):
        visiting.add(n);nodes_stack.append(n)
        for e in graph.get(n,()):
            x=e.back_fragment_id
            if x in visiting:return edges_stack[nodes_stack.index(x):]+[e]
            if x not in done:
                edges_stack.append(e);r=dfs(x)
                if r:return r
                edges_stack.pop()
        nodes_stack.pop();visiting.remove(n);done.add(n);return []
    for n in sorted(nodes):
        if n not in done:
            r=dfs(n)
            if r:return r
    return []
def _dag(edges):
    kept=list(edges);removed=[]
    while True:
        c=_cycle(kept)
        if not c:break
        weak=min(c,key=lambda e:(e.confidence,e.front_fragment_id,e.back_fragment_id));kept.remove(weak);removed.append(weak)
    return tuple(sorted(kept,key=lambda e:(e.view_index,e.front_fragment_id,e.back_fragment_id))),tuple(removed)

def compile_graphics_v1(*,canonical_mesh_binding_hash,canonical_mesh_skin_binding_hash,source_graphics_binding_hash,faces,evidence,mechanical_attachment_evidence=None,mechanical_domain_ids=(),policy=GraphicsCompilePolicyV1()):
    """Compile role-free relations onto the sole canonical M; no visual mesh exists."""
    policy.validate()
    if evidence.evidence_hash!=graphics_evidence_hash_v1(evidence):raise QualificationError("GRAPHICS_EVIDENCE_HASH_DRIFT")
    if evidence.mesh_binding_hash!=canonical_mesh_binding_hash:raise QualificationError("GRAPHICS_CANONICAL_MESH_BINDING_DRIFT")
    if not canonical_mesh_skin_binding_hash or not source_graphics_binding_hash:raise QualificationError("GRAPHICS_REQUIRED_BINDING_EMPTY")
    tri=tuple(tuple(map(int,f)) for f in faces);n=len(tri)
    if not n or any(len(f)!=3 for f in tri):raise QualificationError("GRAPHICS_CANONICAL_MESH_INVALID")
    uf=_UF(n);adj=_adj(tri);relations={};alpha=[0.]*n
    merge_candidates=[]
    cut_constraints=[]
    for r in evidence.static_relations:
        a,b=int(r.face_a),int(r.face_b)
        if a==b or min(a,b)<0 or max(a,b)>=n:raise QualificationError("GRAPHICS_STATIC_RELATION_FACE_INVALID")
        pair=tuple(sorted((a,b)))
        if pair in relations:raise QualificationError("GRAPHICS_STATIC_RELATION_DUPLICATE")
        # Validate ALL source scores, even if this pair will not merge.
        # Unused NaN/invalid evidence must not silently survive a court.
        for label, score in (
            ("same", r.same_drawable_affinity), ("cut", r.cut_probability),
            ("attachment", r.attachment_affinity),
            ("alpha", r.alpha_boundary_probability),
            ("appearance", r.appearance_boundary_probability),
            ("confidence", r.confidence),
        ):
            _prob(score, label)
        relations[pair]=r
        for f in pair:alpha[f]=max(alpha[f],_prob(r.alpha_boundary_probability,"alpha"))
        if r.confidence>=policy.relation_confidence_min:
            if r.cut_probability >= 1.0-policy.cut_max:
                cut_constraints.append(pair)
            if (pair in adj and r.same_drawable_affinity>=policy.same_drawable_min
                    and r.cut_probability<=policy.cut_max
                    and max(r.alpha_boundary_probability,r.appearance_boundary_probability)<=policy.boundary_max):
                merge_candidates.append((
                    -float(r.confidence*r.same_drawable_affinity),pair,
                ))
    for a,b in sorted(cut_constraints):
        uf.cannot_link(a,b)
    blocked_merges=[]
    for _,(a,b) in sorted(merge_candidates):
        if not uf.union(a,b):
            blocked_merges.append((a,b))
    groups={}
    for i in range(n):groups.setdefault(uf.find(i),[]).append(i)
    fragments=[];face_fragment={}
    for rows in sorted(groups.values(),key=min):
        rows=tuple(sorted(rows));fid="GFX:"+content_sha256({"mesh":canonical_mesh_binding_hash,"faces":rows})[:24];support=policy.maximum_support_band_pixels if max(alpha[i] for i in rows)>=policy.support_band_alpha_boundary_min else 0;fragments.append(GraphicsFragmentV1(fid,rows,int(support),metadata={"canonical_mesh_only":True,"categorical_identity":None,"support_band_changes_mechanical_geometry":False}))
        for i in rows:face_fragment[i]=fid
    vis={};pair_rows={};views_present=set()
    for r in evidence.view_relations:
        v=int(r.view_index);views_present.add(v);a,b=int(r.face_a),int(r.face_b)
        if v<0 or a==b or min(a,b)<0 or max(a,b)>=n:raise QualificationError("GRAPHICS_VIEW_RELATION_INVALID")
        for label, score in (
            ("front", r.front_probability), ("occlusion", r.occlusion_probability),
            ("vis_a", r.visibility_probability_a),
            ("vis_b", r.visibility_probability_b), ("view_conf", r.confidence),
        ):
            _prob(score, label)
        fa,fb=face_fragment[a],face_fragment[b];vis.setdefault((v,fa),[]).append(_prob(r.visibility_probability_a,"vis_a"));vis.setdefault((v,fb),[]).append(_prob(r.visibility_probability_b,"vis_b"))
        if fa==fb:continue
        conf=_prob(r.confidence,"view_conf");occ=_prob(r.occlusion_probability,"occ");front=_prob(r.front_probability,"front")
        if conf<policy.relation_confidence_min or occ<policy.occlusion_probability_min:continue
        lo,hi=sorted((fa,fb));score=front if fa==lo else 1-front;pair_rows.setdefault((v,lo,hi),[]).append((score,conf*occ,r.evidence_ref))
    view_states=[]
    had_order_conflict=False
    for v in sorted(views_present):
        raw=[]
        for (vv,lo,hi),rows in sorted(pair_rows.items()):
            if vv!=v:continue
            weight=sum(x[1] for x in rows);score=sum(x[0]*x[1] for x in rows)/weight
            if score>=policy.front_probability_min:front,back,p=lo,hi,score
            elif score<=1-policy.front_probability_min:front,back,p=hi,lo,1-score
            else:continue
            raw.append(GraphicsOrderEdgeV1(v,front,back,float(p),tuple(sorted({x[2] for x in rows if x[2]}))))
        kept,removed=_dag(raw)
        had_order_conflict=had_order_conflict or bool(removed)
        if removed and not policy.allow_unqualified_order_arbitration:
            raise QualificationError("GRAPHICS_SOURCE_ORDER_CONFLICT_UNQUALIFIED")
        visible=[]
        visibility_unknown=[]
        for f in fragments:
            values=vis.get((v,f.fragment_id),())
            if not values:
                # Canonical body triangles remain drawable for exact 3D
                # depth/ray evaluation; absent source view evidence is
                # UNKNOWN, not learned-visible or learned-hidden.
                visibility_unknown.append(f.fragment_id)
                visible.append(f.fragment_id)
                continue
            p=sum(values)/len(values)
            if p>=policy.visibility_probability_min:
                visible.append(f.fragment_id)
        view_states.append(GraphicsViewStateV1(
            v,tuple(sorted(visible)),kept,metadata={
                "cycle_edges_removed":[x.to_dict() for x in removed],
                "triangle_array_order_is_presentation_authority":False,
                "visibility_unknown_fragment_ids":tuple(sorted(visibility_unknown)),
                "unobserved_visibility_raster_policy":"DRAW_GEOMETRY_AND_DEFER_OCCLUSION_TO_DEPTH",
                "missing_source_view_visibility_is_not_positive_label":True,
            },
        ))
    attachments=[]
    mechanical=dict(mechanical_attachment_evidence or {})
    qualified_domains=frozenset(str(x) for x in mechanical_domain_ids)
    seen_attached_fragments=set()
    for pair,r in relations.items():
        if r.attachment_affinity<policy.attachment_affinity_min or r.confidence<policy.relation_confidence_min:continue
        m=dict(mechanical.get(pair) or {})
        if not m:
            continue
        contact=_prob(m.get("persistent_contact",0),"mechanical_contact")
        if contact<policy.mechanical_contact_min:
            continue
        fa,fb=face_fragment[pair[0]],face_fragment[pair[1]]
        if fa==fb:continue
        try:
            bary=tuple(float(x) for x in (m.get("barycentric_anchor") or ()))
        except (TypeError, ValueError, OverflowError):
            raise QualificationError("GRAPHICS_ATTACHMENT_BARYCENTRIC_INVALID") from None
        try:
            anchor=int(m.get("anchor_face_index",-1))
            child=int(m.get("child_face_index",-1))
        except (ValueError, TypeError, OverflowError):
            raise QualificationError("GRAPHICS_ATTACHMENT_PARENT_CHILD_FACE_BINDING_INVALID") from None
        parent=str(m.get("parent_mechanical_domain_id") or "")
        if (len(bary)!=3 or any(not 0<=x<=1 for x in bary)
                or abs(sum(bary)-1)>1e-6):
            raise QualificationError("GRAPHICS_ATTACHMENT_BARYCENTRIC_INVALID")
        # The attachment is an explicit directional relation: child drawable
        # follows a SOURCE-BOUND parent face with an admitted mechanical domain.
        # Picking the numerically first face here would attach an arbitrary
        # drawable, including the actual parent to itself.
        if (anchor not in pair or child not in pair or anchor==child
                or face_fragment[anchor]==face_fragment[child]):
            raise QualificationError("GRAPHICS_ATTACHMENT_PARENT_CHILD_FACE_BINDING_INVALID")
        if not parent or parent not in qualified_domains:
            raise QualificationError("GRAPHICS_ATTACHMENT_MECHANICAL_DOMAIN_UNQUALIFIED")
        child_fragment=face_fragment[child]
        if child_fragment in seen_attached_fragments:
            raise QualificationError("GRAPHICS_ATTACHMENT_CHILD_MULTIPLE_PARENTS")
        seen_attached_fragments.add(child_fragment)
        attachments.append(GraphicsAttachmentV1(
            child_fragment,parent,anchor,bary,
            min(float(r.attachment_affinity),contact),
            tuple(x for x in (r.evidence_ref,str(m.get("evidence_ref") or "")) if x),
        ))
    value=CompiledGraphicsV1(str(canonical_mesh_binding_hash),str(canonical_mesh_skin_binding_hash),str(source_graphics_binding_hash),evidence.evidence_hash,tuple(fragments),tuple(view_states),tuple(attachments),"",metadata={"authority_class":"COMPILED_GRAPHICS","single_canonical_mesh_authority":True,"independent_visual_mesh_allowed":False,"categorical_semantics_used":False,"source_pixels_win_where_observed":True,"runtime_consumption_required":True,"triangle_array_order_is_presentation_authority":False,"silhouette_support_is_2d_only":True,"representation_policy_arbitration_used":had_order_conflict,
"unresolved_source_order_conflict":had_order_conflict,
"source_graphics_qualification_status":"UNQUALIFIED_CONFLICT_POLICY" if had_order_conflict else "CANDIDATE_SCHEDULE_ONLY",
"pose_conditioned_visibility_proven":False,
"source_visibility_coverage_complete":not any(
    v.metadata.get("visibility_unknown_fragment_ids") for v in view_states
),
"transitive_cut_constraints_enforced":True,
"candidate_merge_blocked_by_cut":tuple(blocked_merges),
"order_arbitration_not_source_truth":True})
    return replace(value,graph_hash=compiled_graphics_hash_v1(value))
