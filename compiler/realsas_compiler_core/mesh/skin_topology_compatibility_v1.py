from __future__ import annotations

"""Deterministic topology x skin compatibility proof and bounded seam-cut repair.

This module does not alter QualifiedSkinIR weights. It asks whether the frozen
mechanical connectivity remains numerically admissible when the exact skin field
is stressed in canonical joint frames. Failure produces a new candidate lineage
by removing only faces that bridge mechanically incompatible skin regions.

The repaired candidate is a repair proposal, not product authority. The
orchestrator must restart qualification from Stage18/19 and rebuild all
candidate-bound downstream artifacts.
"""

from dataclasses import replace
import math
from typing import Any

import numpy as np

from .deformation_stress_v1 import _candidate_skin_matrix
from .deformation_stress_v2 import _pose_skin_matrices
from ..hashing import content_sha256
from ..joint_frames_v1 import derive_joint_frames_from_skeleton
from ..product_authority_v1 import (
    canonical_mesh_candidate_lineage_hash,
    validate_deformation_capability_envelope,
    validate_mesh_qualification_policy,
)
from ..types import QualificationError

Json = dict[str, Any]

SKIN_TOPOLOGY_COMPAT_SCHEMA = "RealSaS.SkinTopologyCompatibilityReport.v1"
SKIN_TOPOLOGY_REPAIR_SCHEMA = "RealSaS.SkinTopologyRepairDirective.v1"
DEFAULT_RISK_L1_MIN = 0.5
DEFAULT_MAX_EDGE_RATIO = 4.0
DEFAULT_STRESS_ANGLE_DEG = 120.0
DEFAULT_MAX_REPAIR_ITERATIONS = 4


def _face_indices(candidate):
    vi = {str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)}
    if len(vi) != len(candidate.vertices):
        raise QualificationError("SKIN_TOPOLOGY_VERTEX_ID_DUPLICATE")
    out=[]
    for face in candidate.faces:
        if len(face) != 3 or any(str(v) not in vi for v in face):
            raise QualificationError("SKIN_TOPOLOGY_FACE_INVALID")
        out.append(tuple(vi[str(v)] for v in face))
    if not out:
        raise QualificationError("SKIN_TOPOLOGY_FACE_SET_EMPTY")
    return np.asarray(out,dtype=np.int64)


def _skin_l1_per_face(weights,faces):
    face_array=np.asarray(faces,dtype=np.int64)
    if face_array.ndim!=2 or face_array.shape[1]!=3:
        raise QualificationError("SKIN_TOPOLOGY_FACE_INDEX_ARRAY_INVALID")
    wf=weights[face_array]
    d01=np.abs(wf[:,0]-wf[:,1]).sum(axis=1)
    d12=np.abs(wf[:,1]-wf[:,2]).sum(axis=1)
    d20=np.abs(wf[:,2]-wf[:,0]).sum(axis=1)
    return np.maximum(np.maximum(d01,d12),d20)


def _triangle_metrics_batch(rest,posed,faces):
    r=rest[faces]
    p=posed[faces]

    re01=np.linalg.norm(r[:,1]-r[:,0],axis=1)
    re12=np.linalg.norm(r[:,2]-r[:,1],axis=1)
    re20=np.linalg.norm(r[:,0]-r[:,2],axis=1)
    pe01=np.linalg.norm(p[:,1]-p[:,0],axis=1)
    pe12=np.linalg.norm(p[:,2]-p[:,1],axis=1)
    pe20=np.linalg.norm(p[:,0]-p[:,2],axis=1)
    ratios=np.stack((
        pe01/np.maximum(re01,1e-12),
        pe12/np.maximum(re12,1e-12),
        pe20/np.maximum(re20,1e-12),
    ),axis=1)
    edge_min=ratios.min(axis=1)
    edge_max=ratios.max(axis=1)

    r1=r[:,1]-r[:,0]
    r2=r[:,2]-r[:,0]
    l1=np.linalg.norm(r1,axis=1)
    if np.any(l1<=1e-12):
        raise QualificationError("SKIN_TOPOLOGY_REST_EDGE_DEGENERATE")
    u=r1/l1[:,None]
    x2=np.sum(r2*u,axis=1)
    perp=r2-x2[:,None]*u
    y2=np.linalg.norm(perp,axis=1)
    if np.any(y2<=1e-12):
        raise QualificationError("SKIN_TOPOLOGY_REST_TRIANGLE_DEGENERATE")

    inv=np.zeros((len(faces),2,2),dtype=np.float64)
    inv[:,0,0]=1.0/l1
    inv[:,0,1]=-x2/(l1*y2)
    inv[:,1,1]=1.0/y2

    pedges=np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=2)
    F=np.einsum("nij,njk->nik",pedges,inv)
    singular=np.linalg.svd(F,compute_uv=False)
    smax=singular[:,0]
    smin=singular[:,1]
    area=smax*smin
    condition=smax/np.maximum(smin,1e-15)
    return area,condition,edge_min,edge_max


def _stress_angle(envelope):
    meta=dict(envelope.metadata or {})
    value=meta.get("skin_topology_compatibility_stress_angle_deg",DEFAULT_STRESS_ANGLE_DEG)
    value=float(value)
    if not math.isfinite(value) or value<=0.0 or value>180.0:
        raise QualificationError("SKIN_TOPOLOGY_STRESS_ANGLE_INVALID")
    return value


def run_skin_topology_compatibility_v1(
    candidate,
    *,
    surface,
    skeleton,
    skin,
    envelope,
    cameras,
    policy,
    risk_l1_min:float=DEFAULT_RISK_L1_MIN,
    max_edge_ratio:float=DEFAULT_MAX_EDGE_RATIO,
):
    validate_mesh_qualification_policy(policy)
    validate_deformation_capability_envelope(
        envelope,known_joint_ids={j.canonical_joint_id for j in skeleton.joints}
    )
    if envelope.skeleton_lineage_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("SKIN_TOPOLOGY_ENVELOPE_SKELETON_DRIFT")
    if risk_l1_min < 0.0 or risk_l1_min > 2.0:
        raise QualificationError("SKIN_TOPOLOGY_RISK_L1_INVALID")
    if max_edge_ratio <= 1.0 or not math.isfinite(max_edge_ratio):
        raise QualificationError("SKIN_TOPOLOGY_EDGE_RATIO_INVALID")

    rest,weights,faces=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=skin
    )
    faces=np.asarray(faces,dtype=np.int64)
    if faces.ndim!=2 or faces.shape[1]!=3:
        raise QualificationError("SKIN_TOPOLOGY_FACE_INDEX_ARRAY_INVALID")
    skin_l1=_skin_l1_per_face(weights,faces)
    risky=np.nonzero(skin_l1>float(risk_l1_min))[0]

    stress_angle=_stress_angle(envelope)
    frames=derive_joint_frames_from_skeleton(skeleton,cameras=cameras)
    joint_ids=tuple(j.canonical_joint_id for j in skeleton.joints)
    max_area=np.ones((len(candidate.faces),),dtype=np.float64)
    min_area=np.ones((len(candidate.faces),),dtype=np.float64)
    max_condition=np.ones((len(candidate.faces),),dtype=np.float64)
    max_edge=np.ones((len(candidate.faces),),dtype=np.float64)
    min_edge=np.ones((len(candidate.faces),),dtype=np.float64)
    worst_probe=["REST"]*len(candidate.faces)

    probe_count=1
    if len(risky):
        for jid in sorted(joint_ids):
            for axis_index,axis_name in enumerate(("X","Y","Z")):
                for sign in (-1.0,1.0):
                    deg=sign*stress_angle
                    skin_by_id=_pose_skin_matrices(
                        skeleton,frames,joint_id=jid,local_axis_index=axis_index,degrees=deg
                    )
                    matrices=np.stack([skin_by_id[x] for x in joint_ids],axis=0)
                    hom=np.concatenate(
                        [rest,np.ones((len(rest),1),dtype=np.float64)],axis=1
                    )
                    per=np.stack(
                        [(hom@matrices[k].T)[:,:3] for k in range(len(joint_ids))],
                        axis=1,
                    )
                    posed=np.sum(per*weights[:,:,None],axis=1)
                    area,cond,emin,emax=_triangle_metrics_batch(rest,posed,faces[risky])
                    probe_id=f"{jid}:LOCAL_{axis_name}:{deg:+g}"
                    for local,fi in enumerate(risky.tolist()):
                        values=(
                            float(area[local]),
                            float(cond[local]),
                            float(emin[local]),
                            float(emax[local]),
                        )
                        if not all(math.isfinite(x) for x in values):
                            max_condition[fi]=float("inf")
                            worst_probe[fi]=probe_id
                            continue
                        severity=max(
                            values[1]/max(float(policy.g3_max_dynamic_condition_number),1e-12),
                            values[3]/max(float(max_edge_ratio),1e-12),
                            values[0]/max(float(policy.g3_max_dynamic_area_ratio),1e-12),
                            float(policy.g3_min_dynamic_area_ratio)/max(values[0],1e-12),
                        )
                        previous=max(
                            max_condition[fi]/max(float(policy.g3_max_dynamic_condition_number),1e-12),
                            max_edge[fi]/max(float(max_edge_ratio),1e-12),
                            max_area[fi]/max(float(policy.g3_max_dynamic_area_ratio),1e-12),
                            float(policy.g3_min_dynamic_area_ratio)/max(min_area[fi],1e-12),
                        )
                        if severity>previous:
                            worst_probe[fi]=probe_id
                        max_area[fi]=max(max_area[fi],values[0])
                        min_area[fi]=min(min_area[fi],values[0])
                        max_condition[fi]=max(max_condition[fi],values[1])
                        max_edge[fi]=max(max_edge[fi],values[3])
                        min_edge[fi]=min(min_edge[fi],values[2])
                    probe_count+=1

    unsafe=(
        (skin_l1>float(risk_l1_min))
        & (
            (max_area>float(policy.g3_max_dynamic_area_ratio))
            | (min_area<float(policy.g3_min_dynamic_area_ratio))
            | (max_condition>float(policy.g3_max_dynamic_condition_number))
            | (max_edge>float(max_edge_ratio))
        )
    )
    unsafe_ids=np.nonzero(unsafe)[0]
    top=sorted(
        unsafe_ids.tolist(),
        key=lambda fi:(
            max_edge[fi],
            max_area[fi],
            max_condition[fi],
            skin_l1[fi],
        ),
        reverse=True,
    )[:256]

    report={
        "schema":SKIN_TOPOLOGY_COMPAT_SCHEMA,
        "candidate_lineage_hash":candidate.candidate_lineage_hash,
        "surface_lineage_hash":surface.geometry_lineage_hash,
        "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "skin_lineage_hash":skin.skin_lineage_hash,
        "envelope_lineage_hash":envelope.envelope_lineage_hash,
        "qualification_policy_hash":policy.qualification_policy_lineage_hash,
        "stress_angle_deg":stress_angle,
        "probe_count":int(probe_count),
        "face_count":len(candidate.faces),
        "risky_face_count":int(len(risky)),
        "unsafe_face_count":int(len(unsafe_ids)),
        "risk_l1_min":float(risk_l1_min),
        "max_edge_ratio_limit":float(max_edge_ratio),
        "max_area_ratio_limit":float(policy.g3_max_dynamic_area_ratio),
        "min_area_ratio_limit":float(policy.g3_min_dynamic_area_ratio),
        "max_condition_limit":float(policy.g3_max_dynamic_condition_number),
        "weight_mutation":False,
        "passed":bool(len(unsafe_ids)==0),
        "unsafe_face_indices":tuple(map(int,unsafe_ids.tolist())),
        "top_unsafe_faces":tuple({
            "face_index":int(fi),
            "vertex_ids":tuple(map(str,candidate.faces[fi])),
            "skin_l1_max":float(skin_l1[fi]),
            "max_edge_ratio":float(max_edge[fi]),
            "max_area_ratio":float(max_area[fi]),
            "min_area_ratio":float(min_area[fi]),
            "max_condition_number":float(max_condition[fi]),
            "worst_probe_id":str(worst_probe[fi]),
        } for fi in top),
        "report_hash":"",
        "metadata":{
            "role":"TOPOLOGY_X_SKIN_MECHANICAL_COMPATIBILITY",
            "actual_motion_capability_claimed":False,
            "repair_owner":"STAGE35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
            "skin_weights_are_immutable":True,
        },
    }
    report["report_hash"]=content_sha256({k:v for k,v in report.items() if k!="report_hash"})
    return report


def seam_cut_candidate_v1(candidate,unsafe_face_indices,*,report_hash:str,max_iterations:int=DEFAULT_MAX_REPAIR_ITERATIONS):
    unsafe=set(map(int,unsafe_face_indices))
    if not unsafe:
        return candidate,{
            "schema":SKIN_TOPOLOGY_REPAIR_SCHEMA,
            "status":"NO_REPAIR_REQUIRED",
            "source_candidate_lineage_hash":candidate.candidate_lineage_hash,
            "repaired_candidate_lineage_hash":candidate.candidate_lineage_hash,
            "removed_face_count":0,
            "weight_mutation":False,
            "max_iterations":int(max_iterations),
            "iterations_used":0,
        }
    if max_iterations<1:
        raise QualificationError("SKIN_TOPOLOGY_REPAIR_BUDGET_INVALID")
    if min(unsafe)<0 or max(unsafe)>=len(candidate.faces):
        raise QualificationError("SKIN_TOPOLOGY_UNSAFE_FACE_INDEX_INVALID")

    previous_iteration=int(dict(candidate.metadata or {}).get("skin_topology_repair_iteration",0))
    next_iteration=previous_iteration+1
    if next_iteration>int(max_iterations):
        raise QualificationError("SKIN_TOPOLOGY_REPAIR_BUDGET_EXHAUSTED")
    root_lineage=str(
        dict(candidate.metadata or {}).get("skin_topology_repair_root_candidate_lineage_hash")
        or candidate.candidate_lineage_hash
    )
    keep=[i for i in range(len(candidate.faces)) if i not in unsafe]
    if not keep:
        raise QualificationError("SKIN_TOPOLOGY_REPAIR_WOULD_REMOVE_ALL_FACES")
    faces=tuple(candidate.faces[i] for i in keep)
    edges=set()
    for a,b,c in faces:
        edges.add(tuple(sorted((str(a),str(b)))))
        edges.add(tuple(sorted((str(b),str(c)))))
        edges.add(tuple(sorted((str(c),str(a)))))
    provisional=replace(
        candidate,
        faces=faces,
        edges=tuple(sorted(edges)),
        producer_id="RealSaS.Stage35SkinTopologySeamCutRepair.v1",
        candidate_lineage_hash="",
        metadata={
            **dict(candidate.metadata or {}),
            "repair_kind":"SKIN_TOPOLOGY_SEAM_CUT",
            "source_candidate_lineage_hash":candidate.candidate_lineage_hash,
            "skin_topology_repair_root_candidate_lineage_hash":root_lineage,
            "skin_topology_repair_iteration":next_iteration,
            "skin_topology_repair_max_iterations":int(max_iterations),
            "compatibility_report_hash":str(report_hash),
            "removed_face_count":len(unsafe),
            "weight_mutation":False,
            "vertex_position_mutation":False,
            "repair_requires_stage19_plus_requalification":True,
        },
    )
    repaired=replace(
        provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional),
    )
    directive={
        "schema":SKIN_TOPOLOGY_REPAIR_SCHEMA,
        "status":"REPAIR_PROPOSED__RESTART_FROM_STAGE18_LINEAGE",
        "source_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "repaired_candidate_lineage_hash":repaired.candidate_lineage_hash,
        "compatibility_report_hash":str(report_hash),
        "repair_root_candidate_lineage_hash":root_lineage,
        "repair_parent_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "repair_iteration":next_iteration,
        "removed_face_count":len(unsafe),
        "face_count_before":len(candidate.faces),
        "face_count_after":len(repaired.faces),
        "repair_operation":"SEAM_CUT_BY_UNSAFE_FACE_REMOVAL",
        "weight_mutation":False,
        "vertex_position_mutation":False,
        "local_cdt_required":False,
        "max_iterations":int(max_iterations),
        "iterations_used":next_iteration,
        "downstream_requalification_start":"19_STATIC_CANONICAL_MESH_QUALIFIED",
        "fail_closed_if_requalification_not_completed":True,
    }
    directive["directive_hash"]=content_sha256(directive)
    return repaired,directive
