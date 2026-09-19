from __future__ import annotations

"""Stage-35 canonical 3D dynamic motion proof.

The authority in this module is the single canonical QualifiedMeshIR. Eight views are
not eight motion meshes. We execute the Stage-34 canonical joint tracks in object
space, skin the exact Stage-28 mesh, measure contacts and conditioning, and seal the
sampled canonical 3D frames. Runtime/view projection remains downstream.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

import numpy as np

from .canonical_puppet_state_v1 import CanonicalPuppetStateIR, canonical_puppet_state_hash
from .hashing import content_sha256
from .motion_compile_v1 import (
    MotionCompileConstraintSetIR,
    QualifiedMotionIR,
    motion_constraint_set_hash,
    qualified_motion_hash,
)
from .product_authority_v1 import (
    MeshQualificationPolicyIR,
    QualifiedMeshIR,
    QualifiedPresentationGraphIR,
    qualified_mesh_lineage_hash,
    qualified_presentation_lineage_hash,
)
from .types import QualifiedMeshSkinIR, QualifiedSkeletonIR, QualificationError

Json=dict[str,Any]
Vec2=tuple[float,float]
Vec3=tuple[float,float,float]

DYNAMIC_EVALUATOR_SEMANTIC_VERSION="RealSaS.CanonicalDynamicMotionEvaluator.v1"
_UNIFORM_SAMPLE_COUNT=17
_CONTACT_REL_TOL=1e-5
_MOTION_REL_EPS=1e-8


@dataclass(frozen=True)
class DynamicContactProofIR:
    contact_id:str
    clip_id:str
    canonical_joint_id:str
    anchor_xy:Vec2
    max_drift:float
    tolerance:float
    sample_count:int
    status:str
    proof_hash:str
    schema_version:str="RealSaS.DynamicContactProofIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class CanonicalDynamicFrameIR:
    time_seconds:float
    joint_world_positions:tuple[tuple[str,Vec3],...]
    posed_vertex_xyz:tuple[tuple[str,Vec3],...]
    max_vertex_displacement:float
    min_triangle_area_ratio:float
    max_triangle_area_ratio:float
    max_triangle_condition_number:float
    frame_hash:str
    schema_version:str="RealSaS.CanonicalDynamicFrameIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class DynamicMotionClipProofIR:
    clip_id:str
    clip_kind:str
    classification:str
    duration_seconds:float
    loop:bool
    frames:tuple[CanonicalDynamicFrameIR,...]
    contact_proofs:tuple[DynamicContactProofIR,...]
    max_vertex_displacement:float
    professional_motion_evidence:bool
    clip_proof_hash:str
    schema_version:str="RealSaS.DynamicMotionClipProofIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedDynamicMotionIR:
    qualified_motion_binding_hash:str
    constraint_set_binding_hash:str
    product_state_binding_hash:str
    skeleton_binding_hash:str
    mesh_binding_hash:str
    mesh_skin_binding_hash:str
    presentation_binding_hash:str
    mesh_policy_binding_hash:str
    evaluator_semantic_version:str
    clips:tuple[DynamicMotionClipProofIR,...]
    qualification_report:Json
    dynamic_motion_hash:str
    schema_version:str="RealSaS.QualifiedDynamicMotionIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def _hash_without(value,field_name:str)->str:
    payload=value.to_dict(); payload.pop(field_name,None)
    return content_sha256(payload)


def dynamic_contact_proof_hash(value:DynamicContactProofIR)->str:
    return _hash_without(value,"proof_hash")


def canonical_dynamic_frame_hash(value:CanonicalDynamicFrameIR)->str:
    return _hash_without(value,"frame_hash")


def dynamic_motion_clip_proof_hash(value:DynamicMotionClipProofIR)->str:
    return _hash_without(value,"clip_proof_hash")


def qualified_dynamic_motion_hash(value:QualifiedDynamicMotionIR)->str:
    return _hash_without(value,"dynamic_motion_hash")


def _mesh_skin_hash(value:QualifiedMeshSkinIR)->str:
    payload=value.to_dict(); payload.pop("mesh_skin_lineage_hash",None)
    return content_sha256(payload)


def _policy_hash(value:MeshQualificationPolicyIR)->str:
    payload=value.to_dict(); payload.pop("qualification_policy_lineage_hash",None)
    return content_sha256(payload)


def _scale(skeleton:QualifiedSkeletonIR)->float:
    pts=np.asarray([j.position for j in skeleton.joints],dtype=np.float64)
    if pts.ndim!=2 or pts.shape[1]!=3 or not np.isfinite(pts).all():
        raise QualificationError("MOTION_DYNAMIC_SKELETON_NONFINITE")
    span=float(np.linalg.norm(pts.max(axis=0)-pts.min(axis=0))) if len(pts)>1 else 0.0
    return max(1.0,span)


def _topological_skeleton(skeleton:QualifiedSkeletonIR):
    by={j.canonical_joint_id:j for j in skeleton.joints}
    if not by or len(by)!=len(skeleton.joints) or skeleton.root_id not in by:
        raise QualificationError("MOTION_DYNAMIC_SKELETON_INVALID")
    visiting=set(); done=set(); order=[]
    def visit(jid):
        if jid in done: return
        if jid in visiting: raise QualificationError("MOTION_DYNAMIC_SKELETON_CYCLE")
        visiting.add(jid)
        parent=by[jid].parent_canonical_id
        if parent is not None:
            if parent not in by: raise QualificationError("MOTION_DYNAMIC_SKELETON_PARENT_UNKNOWN")
            visit(parent)
        visiting.remove(jid); done.add(jid); order.append(jid)
    for jid in sorted(by): visit(jid)
    return tuple(order),by


def _interp(keys,time_seconds:float,attr:str,default):
    if not keys: return default
    def value(key): return getattr(key,attr)
    if time_seconds<=float(keys[0].time_seconds): return value(keys[0])
    if time_seconds>=float(keys[-1].time_seconds): return value(keys[-1])
    for a,b in zip(keys,keys[1:]):
        ta,tb=float(a.time_seconds),float(b.time_seconds)
        if ta<=time_seconds<=tb:
            u=(time_seconds-ta)/(tb-ta)
            va,vb=value(a),value(b)
            if isinstance(va,(tuple,list)):
                return tuple((1.0-u)*float(x)+u*float(y) for x,y in zip(va,vb))
            return (1.0-u)*float(va)+u*float(vb)
    return value(keys[-1])


def _delta(track,time_seconds:float)->np.ndarray:
    if track is None:
        rotation=0.0; translation=(0.0,0.0); scale=(1.0,1.0)
    else:
        keys=track.keyframes
        rotation=float(_interp(keys,time_seconds,"rotation_deg",0.0))
        translation=tuple(map(float,_interp(keys,time_seconds,"translation_xy",(0.0,0.0))))
        scale=tuple(map(float,_interp(keys,time_seconds,"scale_xy",(1.0,1.0))))
    if not all(math.isfinite(x) for x in (rotation,*translation,*scale)):
        raise QualificationError("MOTION_DYNAMIC_NONFINITE_TRACK_SAMPLE")
    angle=math.radians(rotation); c=math.cos(angle); s=math.sin(angle)
    T=np.eye(4,dtype=np.float64); T[0,3]=translation[0]; T[1,3]=translation[1]
    R=np.eye(4,dtype=np.float64); R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c
    S=np.eye(4,dtype=np.float64); S[0,0]=scale[0]; S[1,1]=scale[1]
    return T@R@S


def _T(p)->np.ndarray:
    out=np.eye(4,dtype=np.float64)
    out[:3,3]=np.asarray(p,dtype=np.float64)
    return out


def _joint_pose(*,skeleton,tracks,time_seconds,order,by_id):
    posed={}; bind={}
    for jid in order:
        joint=by_id[jid]; rest=np.asarray(joint.position,dtype=np.float64)
        parent=joint.parent_canonical_id
        bind[jid]=_T(rest)
        if parent is None:
            posed[jid]=_T(rest)@_delta(tracks.get(jid),time_seconds)
        else:
            parent_rest=np.asarray(by_id[parent].position,dtype=np.float64)
            posed[jid]=posed[parent]@_T(rest-parent_rest)@_delta(tracks.get(jid),time_seconds)
    skin={jid:posed[jid]@np.linalg.inv(bind[jid]) for jid in order}
    positions={jid:tuple(map(float,(posed[jid]@np.asarray((0.0,0.0,0.0,1.0)))[:3])) for jid in order}
    return skin,positions


def _weights(mesh:QualifiedMeshIR,mesh_skin:QualifiedMeshSkinIR,skeleton:QualifiedSkeletonIR):
    vids=tuple(v.canonical_mesh_vertex_id for v in mesh.vertices)
    if len(set(vids))!=len(vids): raise QualificationError("MOTION_DYNAMIC_DUPLICATE_MESH_VERTEX")
    jids=tuple(sorted(j.canonical_joint_id for j in skeleton.joints)); ji={j:i for i,j in enumerate(jids)}
    rows={r.canonical_mesh_vertex_id:r for r in mesh_skin.rows}
    if set(rows)!=set(vids): raise QualificationError("MOTION_DYNAMIC_MESH_SKIN_ACCOUNTING_DRIFT")
    W=np.zeros((len(vids),len(jids)),dtype=np.float64)
    for i,vid in enumerate(vids):
        for jid,w in rows[vid].influences:
            if jid not in ji: raise QualificationError("MOTION_DYNAMIC_SKIN_JOINT_UNKNOWN")
            W[i,ji[jid]]=float(w)
    if not np.isfinite(W).all() or np.any(W<-1e-10) or not np.allclose(W.sum(axis=1),1.0,atol=1e-8,rtol=0.0):
        raise QualificationError("MOTION_DYNAMIC_SKIN_SIMPLEX_INVALID")
    return vids,jids,W


def _skin_vertices(rest_xyz,weights,jids,skin_matrices):
    hom=np.concatenate([rest_xyz,np.ones((len(rest_xyz),1),dtype=np.float64)],axis=1)
    per=np.stack([(hom@m.T)[:,:3] for m in (skin_matrices[j] for j in jids)],axis=1)
    out=np.sum(per*weights[:,:,None],axis=1)
    if not np.isfinite(out).all(): raise QualificationError("MOTION_DYNAMIC_NONFINITE_DEFORMATION")
    return out


def _condition(points)->tuple[float,float]:
    p0,p1,p2=(np.asarray(x,dtype=np.float64) for x in points)
    e1=p1-p0; e2=p2-p0
    area=float(np.linalg.norm(np.cross(e1,e2)))
    a=float(e1@e1); b=float(e1@e2); c=float(e2@e2)
    disc=max(0.0,(a-c)*(a-c)+4.0*b*b)
    lmax=0.5*(a+c+math.sqrt(disc)); lmin=0.5*(a+c-math.sqrt(disc))
    if area<=1e-15 or lmin<=1e-15: return area,float("inf")
    return area,math.sqrt(lmax/lmin)


def _frame_metrics(mesh,rest,posed,policy):
    index={v.canonical_mesh_vertex_id:i for i,v in enumerate(mesh.vertices)}
    min_ratio=float("inf"); max_ratio=0.0; max_cond=0.0
    for face in mesh.faces:
        ids=[index[x] for x in face]
        ra,_=_condition(rest[ids]); pa,cond=_condition(posed[ids])
        if ra<=1e-15 or pa<=1e-15 or not math.isfinite(cond):
            raise QualificationError("MOTION_DYNAMIC_TRIANGLE_DEGENERATE")
        ratio=pa/ra
        min_ratio=min(min_ratio,ratio); max_ratio=max(max_ratio,ratio); max_cond=max(max_cond,cond)
        if ratio<float(policy.g3_min_dynamic_area_ratio)-1e-9 or ratio>float(policy.g3_max_dynamic_area_ratio)+1e-9:
            raise QualificationError("MOTION_DYNAMIC_TRIANGLE_AREA_RATIO_FAIL")
        if cond>float(policy.g3_max_dynamic_condition_number)+1e-9:
            raise QualificationError("MOTION_DYNAMIC_TRIANGLE_CONDITION_FAIL")
    return min_ratio,max_ratio,max_cond


def _sample_times(clip,contact_rows):
    duration=float(clip.duration_seconds)
    times={float(x) for x in np.linspace(0.0,duration,_UNIFORM_SAMPLE_COUNT)}
    for track in clip.tracks:
        for key in track.keyframes: times.add(float(key.time_seconds))
    for row in contact_rows:
        times.add(float(row.start_time_seconds)); times.add(float(row.end_time_seconds))
    return tuple(sorted(times))


def _validate_bindings(*,motion,constraints,product_state,skeleton,mesh,mesh_skin,presentation,mesh_policy):
    if motion.motion_lineage_hash!=qualified_motion_hash(motion):
        raise QualificationError("MOTION_DYNAMIC_SOURCE_MOTION_HASH_DRIFT")
    if constraints.constraint_set_hash!=motion_constraint_set_hash(constraints):
        raise QualificationError("MOTION_DYNAMIC_CONSTRAINT_HASH_DRIFT")
    if product_state.product_state_hash!=canonical_puppet_state_hash(product_state):
        raise QualificationError("MOTION_DYNAMIC_PRODUCT_STATE_HASH_DRIFT")
    if mesh.mesh_lineage_hash!=qualified_mesh_lineage_hash(mesh):
        raise QualificationError("MOTION_DYNAMIC_MESH_HASH_DRIFT")
    if mesh_skin.mesh_skin_lineage_hash!=_mesh_skin_hash(mesh_skin):
        raise QualificationError("MOTION_DYNAMIC_MESH_SKIN_HASH_DRIFT")
    if presentation.presentation_lineage_hash!=qualified_presentation_lineage_hash(presentation):
        raise QualificationError("MOTION_DYNAMIC_PRESENTATION_HASH_DRIFT")
    if mesh_policy.qualification_policy_lineage_hash!=_policy_hash(mesh_policy):
        raise QualificationError("MOTION_DYNAMIC_MESH_POLICY_HASH_DRIFT")
    expected=(
        (motion.product_state_binding_hash,product_state.product_state_hash,"PRODUCT"),
        (motion.skeleton_binding_hash,skeleton.skeleton_lineage_hash,"SKELETON"),
        (motion.presentation_binding_hash,presentation.presentation_lineage_hash,"PRESENTATION"),
        (motion.constraint_set_binding_hash,constraints.constraint_set_hash,"CONSTRAINT"),
        (mesh_skin.mesh_binding_hash,mesh.mesh_lineage_hash,"MESH_SKIN_MESH"),
        (mesh_skin.skeleton_binding_hash,skeleton.skeleton_lineage_hash,"MESH_SKIN_SKELETON"),
        (mesh.qualification_policy_hash,mesh_policy.qualification_policy_lineage_hash,"MESH_POLICY"),
    )
    for actual,wanted,label in expected:
        if actual!=wanted: raise QualificationError(f"MOTION_DYNAMIC_BINDING_DRIFT:{label}")
    if str(mesh.qualification_report.get("g3_stress_probe_status"))!="PASS":
        raise QualificationError("MOTION_DYNAMIC_REQUIRES_STAGE27_G3_PASS")
    if str(mesh.qualification_report.get("g3_envelope_binding_hash"))!=motion.envelope_binding_hash:
        raise QualificationError("MOTION_DYNAMIC_G3_ENVELOPE_DRIFT")
    components={v.component_id for v in mesh.vertices}
    owners={}
    for a in presentation.attachments:
        for cid in a.mechanical_component_ids:
            if cid in owners: raise QualificationError("MOTION_DYNAMIC_COMPONENT_ATTACHMENT_OVERLAP")
            owners[cid]=a.attachment_id
    if set(owners)!=components:
        raise QualificationError("MOTION_DYNAMIC_COMPONENT_ATTACHMENT_ACCOUNTING_DRIFT")


def build_qualified_dynamic_motion(
    *,
    motion:QualifiedMotionIR,
    constraints:MotionCompileConstraintSetIR,
    product_state:CanonicalPuppetStateIR,
    skeleton:QualifiedSkeletonIR,
    mesh:QualifiedMeshIR,
    mesh_skin:QualifiedMeshSkinIR,
    presentation:QualifiedPresentationGraphIR,
    mesh_policy:MeshQualificationPolicyIR,
)->QualifiedDynamicMotionIR:
    _validate_bindings(
        motion=motion,constraints=constraints,product_state=product_state,skeleton=skeleton,
        mesh=mesh,mesh_skin=mesh_skin,presentation=presentation,mesh_policy=mesh_policy,
    )
    order,by_id=_topological_skeleton(skeleton)
    vids,jids,W=_weights(mesh,mesh_skin,skeleton)
    rest=np.asarray([v.P for v in mesh.vertices],dtype=np.float64)
    if not np.isfinite(rest).all(): raise QualificationError("MOTION_DYNAMIC_REST_NONFINITE")
    scale=_scale(skeleton); contact_tol=_CONTACT_REL_TOL*scale; motion_eps=_MOTION_REL_EPS*scale
    contacts_by_clip={}
    for row in constraints.contact_constraints:
        contacts_by_clip.setdefault(row.clip_id,[]).append(row)
    clips=[]
    any_nonzero=False
    for clip in motion.clips:
        tracks={t.canonical_joint_id:t for t in clip.tracks}
        if len(tracks)!=len(clip.tracks): raise QualificationError("MOTION_DYNAMIC_DUPLICATE_TRACK")
        contact_rows=tuple(contacts_by_clip.get(clip.clip_id,()))
        times=_sample_times(clip,contact_rows)
        frames=[]; poses={}
        clip_max=0.0
        for t in times:
            skin,joint_positions=_joint_pose(
                skeleton=skeleton,tracks=tracks,time_seconds=t,order=order,by_id=by_id
            )
            posed=_skin_vertices(rest,W,jids,skin)
            displacement=np.linalg.norm(posed-rest,axis=1)
            max_disp=float(displacement.max(initial=0.0)); clip_max=max(clip_max,max_disp)
            min_area,max_area,max_cond=_frame_metrics(mesh,rest,posed,mesh_policy)
            frame=CanonicalDynamicFrameIR(
                time_seconds=float(t),
                joint_world_positions=tuple((jid,tuple(map(float,joint_positions[jid]))) for jid in sorted(joint_positions)),
                posed_vertex_xyz=tuple((vid,tuple(map(float,posed[i]))) for i,vid in enumerate(vids)),
                max_vertex_displacement=max_disp,
                min_triangle_area_ratio=float(min_area),
                max_triangle_area_ratio=float(max_area),
                max_triangle_condition_number=float(max_cond),
                frame_hash="",
                metadata={"canonical_3d_authority":True,"view_projection_performed":False},
            )
            frame=replace(frame,frame_hash=canonical_dynamic_frame_hash(frame))
            frames.append(frame); poses[float(t)]=joint_positions
        contact_proofs=[]
        for row in contact_rows:
            anchor=np.asarray(poses[float(row.start_time_seconds)][row.canonical_joint_id][:2],dtype=np.float64)
            relevant=[t for t in times if float(row.start_time_seconds)-1e-12<=t<=float(row.end_time_seconds)+1e-12]
            drifts=[float(np.linalg.norm(np.asarray(poses[t][row.canonical_joint_id][:2])-anchor)) for t in relevant]
            max_drift=max(drifts,default=0.0)
            status="PASS" if max_drift<=contact_tol else "FAIL"
            proof=DynamicContactProofIR(
                row.contact_id,row.clip_id,row.canonical_joint_id,tuple(map(float,anchor)),
                max_drift,contact_tol,len(relevant),status,"",
                metadata={"mode":row.mode,"anchor_time_seconds":float(row.start_time_seconds)},
            )
            proof=replace(proof,proof_hash=dynamic_contact_proof_hash(proof))
            if status!="PASS": raise QualificationError(f"MOTION_DYNAMIC_CONTACT_FAIL:{row.contact_id}")
            contact_proofs.append(proof)
        nonzero=clip_max>motion_eps
        any_nonzero=any_nonzero or nonzero
        clip_proof=DynamicMotionClipProofIR(
            clip_id=clip.clip_id,clip_kind=clip.clip_kind,classification=clip.classification,
            duration_seconds=clip.duration_seconds,loop=clip.loop,frames=tuple(frames),
            contact_proofs=tuple(contact_proofs),max_vertex_displacement=clip_max,
            professional_motion_evidence=bool(clip.classification=="ARTIST_SOURCE" and nonzero),
            clip_proof_hash="",
            metadata={
                "canonical_motion_nonzero":nonzero,
                "motion_nonzero_epsilon":motion_eps,
                "frame_count":len(frames),
                "continuous_envelope_authority":"STAGE27_G3_FULL_ENVELOPE_PASS",
                "attachment_policy":"STATIC_QUALIFIED_PRESENTATION_ONLY_V1",
            },
        )
        clips.append(replace(clip_proof,clip_proof_hash=dynamic_motion_clip_proof_hash(clip_proof)))
    if not clips: raise QualificationError("MOTION_DYNAMIC_CLIP_SET_EMPTY")
    if not any_nonzero: raise QualificationError("MOTION_DYNAMIC_ALL_CLIPS_STATIC")
    value=QualifiedDynamicMotionIR(
        qualified_motion_binding_hash=motion.motion_lineage_hash,
        constraint_set_binding_hash=constraints.constraint_set_hash,
        product_state_binding_hash=product_state.product_state_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        mesh_skin_binding_hash=mesh_skin.mesh_skin_lineage_hash,
        presentation_binding_hash=presentation.presentation_lineage_hash,
        mesh_policy_binding_hash=mesh_policy.qualification_policy_lineage_hash,
        evaluator_semantic_version=DYNAMIC_EVALUATOR_SEMANTIC_VERSION,
        clips=tuple(sorted(clips,key=lambda x:x.clip_id)),
        qualification_report={
            "status":"PASS_DYNAMIC_CANONICAL_3D",
            "dynamic_proof_passed":True,
            "canonical_3d_single_mesh_truth":True,
            "view_specific_motion_mesh_truth":False,
            "all_contacts_satisfied":True,
            "all_sampled_frames_conditioned":True,
            "stage27_full_envelope_g3_bound":True,
            "attachment_invariants_preserved":True,
            "any_clip_nonzero":True,
            "professional_motion_clip_count":sum(c.professional_motion_evidence for c in clips),
            "artist_source_clip_count":sum(c.classification=="ARTIST_SOURCE" for c in clips),
            "mechanical_probe_clip_count":sum(c.classification=="MECHANICAL_PROBE_ONLY" for c in clips),
        },
        dynamic_motion_hash="",
        metadata={
            "runtime_export_authorized":True,
            "runtime_must_project_these_exact_canonical_frames":True,
            "uniform_sample_count":_UNIFORM_SAMPLE_COUNT,
            "contact_relative_tolerance":_CONTACT_REL_TOL,
            "evaluator_semantic_version":DYNAMIC_EVALUATOR_SEMANTIC_VERSION,
        },
    )
    return replace(value,dynamic_motion_hash=qualified_dynamic_motion_hash(value))


def qualified_dynamic_motion_from_dict(payload:Mapping[str,Any])->QualifiedDynamicMotionIR:
    if str(payload.get("schema_version") or payload.get("schema") or "")!="RealSaS.QualifiedDynamicMotionIR.v1":
        raise ValueError("MOTION_DYNAMIC_SCHEMA_MISMATCH")
    clips=[]
    for raw_clip in payload.get("clips") or ():
        frames=[]
        for raw in raw_clip.get("frames") or ():
            frames.append(CanonicalDynamicFrameIR(
                float(raw["time_seconds"]),
                tuple((str(j),tuple(map(float,p))) for j,p in raw.get("joint_world_positions") or ()),
                tuple((str(v),tuple(map(float,p))) for v,p in raw.get("posed_vertex_xyz") or ()),
                float(raw["max_vertex_displacement"]),float(raw["min_triangle_area_ratio"]),
                float(raw["max_triangle_area_ratio"]),float(raw["max_triangle_condition_number"]),
                str(raw["frame_hash"]),
                schema_version=str(raw.get("schema_version") or "RealSaS.CanonicalDynamicFrameIR.v1"),
                metadata=dict(raw.get("metadata") or {}),
            ))
        contacts=[]
        for raw in raw_clip.get("contact_proofs") or ():
            contacts.append(DynamicContactProofIR(
                str(raw["contact_id"]),str(raw["clip_id"]),str(raw["canonical_joint_id"]),
                tuple(map(float,raw["anchor_xy"])),float(raw["max_drift"]),float(raw["tolerance"]),
                int(raw["sample_count"]),str(raw["status"]),str(raw["proof_hash"]),
                schema_version=str(raw.get("schema_version") or "RealSaS.DynamicContactProofIR.v1"),
                metadata=dict(raw.get("metadata") or {}),
            ))
        clips.append(DynamicMotionClipProofIR(
            str(raw_clip["clip_id"]),str(raw_clip["clip_kind"]),str(raw_clip["classification"]),
            float(raw_clip["duration_seconds"]),bool(raw_clip["loop"]),tuple(frames),tuple(contacts),
            float(raw_clip["max_vertex_displacement"]),bool(raw_clip["professional_motion_evidence"]),
            str(raw_clip["clip_proof_hash"]),
            schema_version=str(raw_clip.get("schema_version") or "RealSaS.DynamicMotionClipProofIR.v1"),
            metadata=dict(raw_clip.get("metadata") or {}),
        ))
    value=QualifiedDynamicMotionIR(
        str(payload["qualified_motion_binding_hash"]),str(payload["constraint_set_binding_hash"]),
        str(payload["product_state_binding_hash"]),str(payload["skeleton_binding_hash"]),
        str(payload["mesh_binding_hash"]),str(payload["mesh_skin_binding_hash"]),
        str(payload["presentation_binding_hash"]),str(payload["mesh_policy_binding_hash"]),
        str(payload["evaluator_semantic_version"]),tuple(clips),dict(payload.get("qualification_report") or {}),
        str(payload["dynamic_motion_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.QualifiedDynamicMotionIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.dynamic_motion_hash!=qualified_dynamic_motion_hash(value):
        raise ValueError("MOTION_DYNAMIC_HASH_MISMATCH")
    for clip in value.clips:
        if clip.clip_proof_hash!=dynamic_motion_clip_proof_hash(clip): raise ValueError("MOTION_DYNAMIC_CLIP_HASH_MISMATCH")
        for frame in clip.frames:
            if frame.frame_hash!=canonical_dynamic_frame_hash(frame): raise ValueError("MOTION_DYNAMIC_FRAME_HASH_MISMATCH")
        for proof in clip.contact_proofs:
            if proof.proof_hash!=dynamic_contact_proof_hash(proof): raise ValueError("MOTION_DYNAMIC_CONTACT_HASH_MISMATCH")
    return value
