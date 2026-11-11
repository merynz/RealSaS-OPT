from __future__ import annotations
from dataclasses import asdict,dataclass,field,replace
from typing import Any,Iterable,Mapping
from .hashing import content_sha256
from .types import QualificationError
from .canonical_mesh_ir_v1 import (
    CanonicalMeshIR, canonical_mesh_lineage_hash_v1,
    validate_canonical_mesh_domain_v1,
)
from .canonical_rig_ir_v2 import (
    CanonicalRigIRV2, canonical_rig_lineage_hash_v2,
    validate_canonical_rig_domain_v2,
)
from .canonical_skin_ir_v2 import (
    CanonicalSkinIRV2, canonical_skin_lineage_hash_v2,
    validate_canonical_skin_domain_v2,
)
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
    if not mesh_binding_hash:
        raise QualificationError("GRAPHICS_EVIDENCE_MESH_BINDING_REQUIRED")
    # A single-face canonical mesh has no possible pair relation. An empty
    # graph is admissible *as empty evidence*, never a visible-source claim.
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
"source_visibility_coverage_complete":bool(view_states) and not any(
    v.metadata.get("visibility_unknown_fragment_ids") for v in view_states
),
"view_relation_eight_direction_address_coverage":tuple(sorted(v.view_index for v in view_states))==tuple(range(8)),
"transitive_cut_constraints_enforced":True,
"candidate_merge_blocked_by_cut":tuple(blocked_merges),
"order_arbitration_not_source_truth":True})
    return replace(value,graph_hash=compiled_graphics_hash_v1(value))


def compile_graphics_on_canonical_puppet_v1(
    *,
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    source_graphics_binding_hash: str,
    evidence: GraphicsEvidenceV1,
    mechanical_attachment_evidence: Mapping[tuple[int,int], Mapping[str,Any]] | None = None,
    policy: GraphicsCompilePolicyV1 = GraphicsCompilePolicyV1(),
) -> CompiledGraphicsV1:
    """Single-mesh-safe entrypoint for Graphics after Mesh/Rig/Skin qualification.

    The historic face-indexed compiler takes only caller-supplied hashes and
    face arrays; it cannot itself prove those identities belong to the same
    canonical puppet. This wrapper binds exact canonical bytes and derives
    face indices and mechanical domains from those real qualified parents.
    No neural candidate index is accepted as a canonical address without a
    separately qualified correspondence.
    """
    if mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh):
        raise QualificationError("GRAPHICS_CANONICAL_MESH_HASH_DRIFT")
    if rig.rig_lineage_hash != canonical_rig_lineage_hash_v2(rig):
        raise QualificationError("GRAPHICS_CANONICAL_RIG_HASH_DRIFT")
    if skin.skin_lineage_hash != canonical_skin_lineage_hash_v2(skin):
        raise QualificationError("GRAPHICS_CANONICAL_SKIN_HASH_DRIFT")
    validate_canonical_mesh_domain_v1(mesh)
    validate_canonical_rig_domain_v2(rig)
    validate_canonical_skin_domain_v2(skin)
    if (
        mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH"
        or rig.metadata.get("authority") != "COMPILER_CANONICAL_RIG"
        or skin.metadata.get("authority") != "COMPILER_CANONICAL_SKIN"
    ):
        raise QualificationError("GRAPHICS_CANONICAL_PUPPET_AUTHORITY_REQUIRED")
    if (
        rig.mesh_lineage_hash != mesh.mesh_lineage_hash
        or skin.mesh_lineage_hash != mesh.mesh_lineage_hash
        or skin.deform_structure_hash != rig.deform_structure_hash
        or skin.mesh_carrier_evidence_hash != mesh.carrier_evidence_hash
        or skin.mesh_carrier_topology_hash != mesh.carrier_topology_hash
        or skin.mesh_carrier_geometry_hash != mesh.carrier_geometry_hash
    ):
        raise QualificationError("GRAPHICS_CANONICAL_PUPPET_PARENT_DRIFT")
    vertex_ids = tuple(row.canonical_mesh_vertex_id for row in mesh.vertices)
    if {row.canonical_mesh_vertex_id for row in skin.rows} != set(vertex_ids):
        raise QualificationError("GRAPHICS_CANONICAL_SKIN_VERTEX_COVERAGE_DRIFT")
    index = {vertex_id:i for i,vertex_id in enumerate(vertex_ids)}
    faces = tuple(tuple(index[vertex_id] for vertex_id in face) for face in mesh.faces)
    mechanical_domains = tuple(
        [row.canonical_joint_id for row in rig.joints]
        + [row.canonical_control_id for row in rig.controls]
    )
    return compile_graphics_v1(
        canonical_mesh_binding_hash=mesh.mesh_lineage_hash,
        canonical_mesh_skin_binding_hash=skin.skin_lineage_hash,
        source_graphics_binding_hash=source_graphics_binding_hash,
        faces=faces,
        evidence=evidence,
        mechanical_attachment_evidence=mechanical_attachment_evidence,
        mechanical_domain_ids=mechanical_domains,
        policy=policy,
    )

def compile_neural_region_graphics_on_canonical_puppet_v1(
    *,
    evidence: "RSSvNextInferenceEvidenceIRV1",
    expected_checkpoint_sha256: str,
    candidate_vertex_ids_by_index: tuple[str, ...],
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    max_candidate_pairs: int = 4096,
    policy: GraphicsCompilePolicyV1 = GraphicsCompilePolicyV1(),
) -> CompiledGraphicsV1:
    """Lower *unqualified* learned region affinities onto an actual puppet.

    Source-to-canonical static qualification MUST have already minted the
    canonical Mesh with this exact neural evidence hash in its provenance.
    The address correspondence is an explicit upstream compiler input, and
    every vertex, component partition and oriented face is verified here.
    This operation NEVER converts same-region probability into source
    drawable/cut/occlusion truth, invents a visual mesh or promotes a
    ProductRevision. It only gives the current Graphics representation policy
    meaningful model-backed candidates on the one qualified mesh.
    """
    from math import exp, isfinite
    from .rss_vnext_inference_evidence_v1 import (
        candidate_face_region_readout_v1, validate_inference_evidence_v1,
    )

    validate_inference_evidence_v1(
        evidence, expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    if evidence.mesh_status != "UNQUALIFIED_MESH_CANDIDATE" or evidence.mesh is None:
        raise QualificationError("GRAPHICS_NEURAL_CANDIDATE_MESH_REQUIRED")
    if evidence.evidence_hash not in mesh.proposal_evidence_refs:
        raise QualificationError("GRAPHICS_NEURAL_STATIC_QUALIFICATION_PARENT_MISSING")
    if (
        mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh)
        or mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH"
    ):
        raise QualificationError("GRAPHICS_NEURAL_CANONICAL_MESH_REQUIRED")
    validate_canonical_mesh_domain_v1(mesh)

    candidate = evidence.mesh
    rest = candidate["rest_xyz"]
    mapping = candidate_vertex_ids_by_index
    if (
        len(mapping) != len(rest) or len(mapping) != len(mesh.vertices)
        or any(type(k) is not str or not k for k in mapping)
        or len(set(mapping)) != len(mapping)
    ):
        raise QualificationError("GRAPHICS_NEURAL_VERTEX_CORRESPONDENCE_INVALID")
    vertex_by_source = {v.source_candidate_vertex_id: v for v in mesh.vertices}
    if set(mapping) != set(vertex_by_source):
        raise QualificationError("GRAPHICS_NEURAL_VERTEX_PARENT_ID_DRIFT")
    component_to_canonical: dict[int, str] = {}
    canonical_to_component: dict[str, int] = {}
    for i, source_id in enumerate(mapping):
        v = vertex_by_source[source_id]
        if not all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(rest[i], v.P)):
            raise QualificationError("GRAPHICS_NEURAL_VERTEX_POSITION_DRIFT")
        key = candidate["vertex_components"][i]
        if (key in component_to_canonical and component_to_canonical[key] != v.component_id
            or v.component_id in canonical_to_component and canonical_to_component[v.component_id] != key):
            raise QualificationError("GRAPHICS_NEURAL_COMPONENT_PARTITION_DRIFT")
        component_to_canonical[key] = v.component_id
        canonical_to_component[v.component_id] = key

    # Face orientation is essential to ray normals, visibility and sampling.
    # Accept cyclic index rotations, NEVER a silently flipped winding.
    def cyclic(face: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        return (face, face[1:] + face[:1], face[2:] + face[:2])

    canonical_face_lookup = {}
    for i, face in enumerate(mesh.faces):
        for rotated in cyclic(face):
            if rotated in canonical_face_lookup:
                raise QualificationError("GRAPHICS_NEURAL_CANONICAL_FACE_DUPLICATE")
            canonical_face_lookup[rotated] = i
    face_map = []
    for face in candidate["faces"]:
        named = tuple(
            vertex_by_source[mapping[vertex_index]].canonical_mesh_vertex_id
            for vertex_index in face
        )
        if named not in canonical_face_lookup:
            raise QualificationError("GRAPHICS_NEURAL_FACE_TOPOLOGY_OR_WINDING_DRIFT")
        face_map.append(canonical_face_lookup[named])
    if (len(face_map) != len(mesh.faces)
        or len(set(face_map)) != len(mesh.faces)):
        raise QualificationError("GRAPHICS_NEURAL_FACE_COVERAGE_DRIFT")

    rows = candidate_face_region_readout_v1(evidence)
    canonical_probs: dict[int, tuple[tuple[float, ...], float]] = {}
    for row in rows:
        k = face_map[row["candidate_face_index"]]
        raw = row["candidate_region_probabilities"]
        unknown = float(row["unknown_probability"])
        mass = sum(raw)
        # UNKNOWN never contributes to a positive relation. A zero known
        # mass makes this pair unknown rather than artificially different.
        probs = tuple(float(x) / mass for x in raw) if mass > 1e-9 else tuple(0.0 for _ in raw)
        canonical_probs[k] = (probs, 1.0 - unknown)
    ordered_face_vertices = tuple(mesh.faces)
    adjacency: set[tuple[int, int]] = set()
    edge_faces: dict[tuple[str, str], list[int]] = {}
    for face_index, face in enumerate(ordered_face_vertices):
        for a, b in ((face[0],face[1]), (face[1],face[2]), (face[2],face[0])):
            edge_faces.setdefault(tuple(sorted((a, b))), []).append(face_index)
    for indices in edge_faces.values():
        for i, a in enumerate(indices):
            for b in indices[i + 1:]:
                adjacency.add(tuple(sorted((a, b))))
    if max_candidate_pairs < 0 or len(adjacency) > max_candidate_pairs:
        raise QualificationError("GRAPHICS_NEURAL_PAIR_RESOURCE_LIMIT")

    relations = []
    for a, b in sorted(adjacency):
        left, confidence_a = canonical_probs[a]
        right, confidence_b = canonical_probs[b]
        confidence = min(confidence_a, confidence_b)
        # A learned categorical co-region probability is NOT a proven source
        # drawable boundary. Compiler may choose a conservative partition,
        # but no cut or stable independent-edit authority is certified.
        same = max(0.0, min(1.0, sum(x * y for x, y in zip(left, right))))
        if confidence < 1e-8:
            same = 0.0
        relations.append(GraphicsStaticRelationEvidenceV1(
            a, b, same, 1.0 - same if confidence >= 1e-8 else 0.5,
            0.0, 0.5, 0.5, confidence,
            evidence_ref=evidence.evidence_hash,
        ))

    compiler_input = build_graphics_evidence_v1(
        mesh_binding_hash=mesh.mesh_lineage_hash,
        static_relations=relations,
        view_relations=(),  # 3D region evidence is not source view/occlusion.
        metadata={
            "source_neural_evidence_hash": evidence.evidence_hash,
            "candidate_mesh_only": True,
            "qualified_mesh_face_rebinding": True,
            "cut_semantics": "REGION_DISAGREEMENT_POLICY_NOT_SOURCE_CUT_TRUTH",
            "attachment_evidence_available": False,
            "source_visibility_evidence_available": False,
        },
    )
    compiled = compile_graphics_on_canonical_puppet_v1(
        mesh=mesh, rig=rig, skin=skin,
        source_graphics_binding_hash=evidence.evidence_hash,
        evidence=compiler_input, policy=policy,
    )
    metadata = dict(compiled.metadata)
    metadata.update({
        "origin": "NEURAL_ANONYMOUS_REGION_CANDIDATE",
        "exact_candidate_to_canonical_face_map": tuple(face_map),
        "qualified_drawable_source_truth": False,
        "qualified_source_occlusion": False,
        "qualified_source_attachments": False,
        "independent_editability_proven": False,
        "learned_region_is_not_drawable_truth": True,
        "source_graphics_qualification_status": "UNQUALIFIED_NEURAL_POLICY_ONLY",
    })
    result = replace(compiled, metadata=metadata, graph_hash="")
    return replace(result, graph_hash=compiled_graphics_hash_v1(result))
