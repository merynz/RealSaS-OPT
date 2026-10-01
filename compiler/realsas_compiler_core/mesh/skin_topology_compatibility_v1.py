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
from ..motion_3d_v1 import apply_lbs_matrix_v1
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
    """Derive G3B stress from the sealed deformation capability envelope.

    G3B is a compatibility proof inside the admitted mechanical capability; it
    must not silently widen that capability with an unrelated hard-coded angle.
    """
    ranges=tuple(envelope.joint_ranges or ())
    if not ranges:
        raise QualificationError("SKIN_TOPOLOGY_STRESS_RANGE_MISSING")
    values=[]
    for row in ranges:
        lo=float(row.min_rotation_deg)
        hi=float(row.max_rotation_deg)
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise QualificationError("SKIN_TOPOLOGY_STRESS_RANGE_NONFINITE")
        values.append(max(abs(lo),abs(hi)))
    value=max(values)
    if not math.isfinite(value) or value<0.0 or value>180.0:
        raise QualificationError("SKIN_TOPOLOGY_STRESS_ANGLE_INVALID")
    return float(value)


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
    stress_all_faces:bool=False,
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
    risk_mask=(
        np.ones((len(candidate.faces),),dtype=bool)
        if bool(stress_all_faces)
        else (skin_l1>float(risk_l1_min))
    )
    risky=np.nonzero(risk_mask)[0]

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
        risk_mask
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
        "stress_all_faces":bool(stress_all_faces),
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
            "risk_prefilter_is_safety_authority":not bool(stress_all_faces),
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


def _mechanical_owner_surface_id(vertex, *, partition_owner:dict[str,str])->str|None:
    """Resolve the surface node that owns Stage35 mechanical skin transfer.

    Identity vertices own their sole surface node.  Holeless seam vertices use
    the separately sealed skin_support_coefficients, never their cross-seam
    geometry interpolation coefficients.
    """
    mode=str(vertex.support_binding.mode)
    if mode=="IDENTITY_SURFACE_NODE":
        coeffs=tuple(vertex.support_binding.coefficients)
        if len(coeffs)!=1 or abs(float(coeffs[0][1])-1.0)>1e-12:
            raise QualificationError("SKIN_TOPOLOGY_IDENTITY_SUPPORT_INVALID")
        sid=str(coeffs[0][0])
    elif mode=="SEAM_GEOMETRY_INTERPOLATION":
        md=dict(vertex.support_binding.metadata or {})
        support=tuple(md.get("skin_support_coefficients") or ())
        if len(support)!=1 or abs(float(support[0][1])-1.0)>1e-12:
            raise QualificationError("SKIN_TOPOLOGY_SEAM_SKIN_OWNER_INVALID")
        sid=str(support[0][0])
        declared=str(md.get("mechanical_component_id") or "")
        if not declared or declared!=str(vertex.component_id):
            raise QualificationError("SKIN_TOPOLOGY_SEAM_COMPONENT_BINDING_INVALID")
    else:
        return None

    owner=str(partition_owner.get(sid) or "")
    if not owner:
        raise QualificationError("SKIN_TOPOLOGY_MECHANICAL_OWNER_SURFACE_UNKNOWN")
    if owner!=str(vertex.component_id):
        raise QualificationError("SKIN_TOPOLOGY_MECHANICAL_OWNER_COMPONENT_DRIFT")
    return sid


def _source_surface_skin_matrix(surface, skeleton, skin):
    if skin.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("SOURCE_EDGE_PROBE_SKIN_SURFACE_DRIFT")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("SOURCE_EDGE_PROBE_SKIN_SKELETON_DRIFT")

    joint_ids = tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    if len(set(joint_ids)) != len(joint_ids):
        raise QualificationError("SOURCE_EDGE_PROBE_JOINT_ID_DUPLICATE")
    joint_index = {jid: i for i, jid in enumerate(joint_ids)}

    rows = {str(row.surface_id): row for row in skin.rows}
    surface_ids = tuple(str(node.surface_id) for node in surface.surface_nodes)
    if set(rows) != set(surface_ids):
        raise QualificationError("SOURCE_EDGE_PROBE_SKIN_ROWS_INCOMPLETE")

    rest = np.asarray(
        [tuple(map(float, node.P)) for node in surface.surface_nodes],
        dtype=np.float64,
    )
    if rest.shape != (len(surface_ids), 3) or not np.isfinite(rest).all():
        raise QualificationError("SOURCE_EDGE_PROBE_REST_INVALID")

    weights = np.zeros((len(surface_ids), len(joint_ids)), dtype=np.float64)
    for i, sid in enumerate(surface_ids):
        seen = set()
        for jid, weight in rows[sid].influences:
            jid = str(jid)
            if jid in seen or jid not in joint_index:
                raise QualificationError("SOURCE_EDGE_PROBE_SKIN_INFLUENCE_INVALID")
            seen.add(jid)
            value = float(weight)
            if not math.isfinite(value) or value < 0.0:
                raise QualificationError("SOURCE_EDGE_PROBE_SKIN_WEIGHT_INVALID")
            weights[i, joint_index[jid]] = value
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise QualificationError("SOURCE_EDGE_PROBE_SKIN_SIMPLEX_INVALID")
    return surface_ids, joint_ids, rest, weights


def _run_source_edge_probe_ratio_v1(
    *,
    surface,
    skeleton,
    skin,
    envelope,
    cameras,
    max_edge_ratio: float = DEFAULT_MAX_EDGE_RATIO,
) -> Json:
    validate_deformation_capability_envelope(
        envelope,
        known_joint_ids={j.canonical_joint_id for j in skeleton.joints},
    )
    if envelope.skeleton_lineage_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("SOURCE_EDGE_PROBE_ENVELOPE_SKELETON_DRIFT")
    if not math.isfinite(float(max_edge_ratio)) or float(max_edge_ratio) <= 0.0:
        raise QualificationError("SOURCE_EDGE_PROBE_RATIO_LIMIT_INVALID")

    surface_ids, joint_ids, rest, weights = _source_surface_skin_matrix(
        surface, skeleton, skin
    )
    index = {sid: i for i, sid in enumerate(surface_ids)}

    edge_pairs = []
    seen = set()
    for relation in surface.local_relations:
        a = str(relation.a_surface_id)
        b = str(relation.b_surface_id)
        if a == b or a not in index or b not in index:
            raise QualificationError("SOURCE_EDGE_PROBE_RELATION_ENDPOINT_INVALID")
        pair = (a, b) if a < b else (b, a)
        if pair in seen:
            continue
        seen.add(pair)
        edge_pairs.append(pair)
    edge_pairs = tuple(sorted(edge_pairs))
    if not edge_pairs:
        raise QualificationError("SOURCE_EDGE_PROBE_RELATION_SET_EMPTY")

    ia = np.asarray([index[a] for a, _ in edge_pairs], dtype=np.int64)
    ib = np.asarray([index[b] for _, b in edge_pairs], dtype=np.int64)
    rest_length = np.linalg.norm(rest[ib] - rest[ia], axis=1)
    if np.any(~np.isfinite(rest_length)) or np.any(rest_length <= 1e-12):
        raise QualificationError("SOURCE_EDGE_PROBE_REST_EDGE_DEGENERATE")

    static_l1 = np.sum(np.abs(weights[ia] - weights[ib]), axis=1)
    max_ratio = np.ones(len(edge_pairs), dtype=np.float64)

    frames = derive_joint_frames_from_skeleton(skeleton, cameras=cameras)
    stress_angle = float(_stress_angle(envelope))
    probe_count = 0
    for jid in sorted(joint_ids):
        for axis_index in range(3):
            for sign in (-1.0, 1.0):
                skin_by_id = _pose_skin_matrices(
                    skeleton,
                    frames,
                    joint_id=jid,
                    local_axis_index=axis_index,
                    degrees=sign * stress_angle,
                )
                matrices = np.stack(
                    [skin_by_id[x] for x in joint_ids],
                    axis=0,
                )
                posed = apply_lbs_matrix_v1(rest, weights, matrices)
                ratio = (
                    np.linalg.norm(posed[ib] - posed[ia], axis=1)
                    / rest_length
                )
                if not np.isfinite(ratio).all():
                    raise QualificationError("SOURCE_EDGE_PROBE_NONFINITE_RATIO")
                max_ratio = np.maximum(max_ratio, ratio)
                probe_count += 1

    unsafe = max_ratio > float(max_edge_ratio)
    rows = tuple(
        {
            "a_surface_id": edge_pairs[i][0],
            "b_surface_id": edge_pairs[i][1],
            "max_edge_ratio": float(max_ratio[i]),
            "pairwise_skin_l1": float(static_l1[i]),
            "unsafe": bool(unsafe[i]),
        }
        for i in range(len(edge_pairs))
    )
    return {
        "schema": "RealSaS.SourceEdgeProbeCompatibility.v1",
        "stress_angle_deg": stress_angle,
        "max_edge_ratio_limit": float(max_edge_ratio),
        "probe_count": int(probe_count),
        "source_edge_count": len(edge_pairs),
        "unsafe_source_edge_count": int(np.count_nonzero(unsafe)),
        "rows": rows,
        "report_hash": content_sha256({
            "schema": "RealSaS.SourceEdgeProbeCompatibility.v1",
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "envelope_lineage_hash": envelope.envelope_lineage_hash,
            "stress_angle_deg": stress_angle,
            "max_edge_ratio_limit": float(max_edge_ratio),
            "probe_count": int(probe_count),
            "rows": rows,
        }),
    }


def _propose_source_edge_probe_repartition_directive_v2(
    candidate,
    *,
    surface,
    skeleton,
    skin,
    partition,
    compatibility_report: Json,
    envelope,
    cameras,
) -> Json:
    report_hash = str(compatibility_report.get("report_hash") or "")
    if not report_hash:
        raise QualificationError("SKIN_TOPOLOGY_REPARTITION_REPORT_HASH_MISSING")

    probe = _run_source_edge_probe_ratio_v1(
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        max_edge_ratio=DEFAULT_MAX_EDGE_RATIO,
    )
    existing = {
        tuple(sorted((str(row.a_surface_id), str(row.b_surface_id)))): str(row.decision)
        for row in partition.boundary_constraints
    }

    ordered = []
    for row in probe["rows"]:
        if not bool(row["unsafe"]):
            continue
        pair = tuple(sorted((
            str(row["a_surface_id"]),
            str(row["b_surface_id"]),
        )))
        if pair not in existing:
            raise QualificationError("SOURCE_EDGE_PROBE_PAIR_NOT_IN_PARENT_PARTITION")
        if existing[pair] == "SEPARATE":
            continue
        ratio = float(row["max_edge_ratio"])
        skin_l1 = float(row["pairwise_skin_l1"])
        ordered.append({
            "constraint_id": "DYNPROBE:" + content_sha256({
                "compatibility_report_hash": report_hash,
                "source_edge_probe_hash": probe["report_hash"],
                "pair": pair,
                "max_edge_ratio": ratio,
            })[:20],
            "a_surface_id": pair[0],
            "b_surface_id": pair[1],
            "decision": "SEPARATE",
            "evidence_refs": (
                f"{report_hash}:SOURCE_EDGE_PROBE:{pair[0]}:{pair[1]}",
                f"{probe['report_hash']}:MAX_EDGE_RATIO:{ratio:.17g}",
            ),
            "max_pairwise_skin_l1": skin_l1,
            "unsafe_face_indices": (),
            "confidence": min(
                1.0,
                max(
                    0.0,
                    ratio / max(float(DEFAULT_MAX_EDGE_RATIO), 1e-12) - 1.0,
                ),
            ),
            "metadata": {
                "evidence_class": "STAGE35_SOURCE_EDGE_PROBE_CANNOT_LINK",
                "automatic": True,
                "manual_authoring_used": False,
                "source_edge_probe_max_ratio": ratio,
                "source_edge_probe_ratio_limit": float(DEFAULT_MAX_EDGE_RATIO),
                "source_edge_probe_stress_angle_deg": float(
                    probe["stress_angle_deg"]
                ),
                "pairwise_skin_l1": skin_l1,
                "direct_probe_seed": True,
            },
        })
    ordered = tuple(sorted(
        ordered,
        key=lambda x: (str(x["a_surface_id"]), str(x["b_surface_id"])),
    ))

    unsafe_face_count = int(compatibility_report.get("unsafe_face_count") or 0)
    directive = {
        "schema": "RealSaS.MechanicalRepartitionDirective.v2",
        "status": (
            "REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY"
            if ordered else
            "ABSTAIN__NO_MECHANICAL_OWNER_BOUNDARY_PROPOSAL"
        ),
        "source_candidate_lineage_hash": candidate.candidate_lineage_hash,
        "source_surface_lineage_hash": surface.geometry_lineage_hash,
        "source_partition_lineage_hash": partition.partition_lineage_hash,
        "source_skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "source_skin_lineage_hash": skin.skin_lineage_hash,
        "compatibility_report_hash": report_hash,
        "unsafe_face_count": unsafe_face_count,
        "candidate_separate_pair_count": len(ordered),
        "unresolved_unsafe_face_count": 0,
        "proposed_boundary_overrides": ordered,
        "seed_strategy": "SOURCE_EDGE_PROBE_RATIO_V1",
        "source_edge_probe_hash": str(probe["report_hash"]),
        "source_edge_probe_count": int(probe["source_edge_count"]),
        "unsafe_source_edge_count": int(probe["unsafe_source_edge_count"]),
        "source_edge_probe_stress_angle_deg": float(probe["stress_angle_deg"]),
        "source_edge_probe_max_edge_ratio_limit": float(
            probe["max_edge_ratio_limit"]
        ),
        "repair_operation": "STAGE17_REPARTITION_THEN_STAGE18_HOLELESS_DENSE_SUBDIVISION",
        "face_deletion_count": 0,
        "weight_mutation": False,
        "vertex_position_mutation_at_stage35": False,
        "restart_from": "17_MECHANICAL_PARTITION_QUALIFIED",
        "mandatory_requalification_through": "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
        "auto_apply_allowed": False,
        "requires_trustworthy_skin_reliability_authority": True,
        "fail_closed_if_not_repartitioned": True,
        "directive_hash": "",
    }
    directive["directive_hash"] = content_sha256({
        k: v for k, v in directive.items() if k != "directive_hash"
    })
    return directive


def propose_mechanical_repartition_directive_v2(
    candidate,
    *,
    surface,
    skeleton,
    skin,
    partition,
    compatibility_report: Json,
    envelope=None,
    cameras=None,
    seed_strategy: str = "UNSAFE_FACE_LOCAL_L1_V1",
) -> Json:
    """Convert Stage35 unsafe-face evidence into a compiler-owned repartition proposal.

    This function never deletes faces and never mutates weights.  It only proposes
    candidate SEPARATE boundary pairs from mechanically owned unsafe face edges
    with the strongest admissible skin discontinuity.  Identity vertices use
    their surface node; holeless seam vertices use skin_support_coefficients,
    never cross-seam geometry coefficients.  Automatic application is deliberately
    forbidden until an explicit trustworthy-skin/reliability authority exists.
    """
    if seed_strategy == "SOURCE_EDGE_PROBE_RATIO_V1":
        if envelope is None or cameras is None:
            raise QualificationError("SOURCE_EDGE_PROBE_CONTEXT_REQUIRED")
        return _propose_source_edge_probe_repartition_directive_v2(
            candidate,
            surface=surface,
            skeleton=skeleton,
            skin=skin,
            partition=partition,
            compatibility_report=compatibility_report,
            envelope=envelope,
            cameras=cameras,
        )
    if seed_strategy != "UNSAFE_FACE_LOCAL_L1_V1":
        raise QualificationError("MECHANICAL_REPARTITION_SEED_STRATEGY_UNSUPPORTED")

    unsafe=tuple(map(int,compatibility_report.get("unsafe_face_indices") or ()))
    report_hash=str(compatibility_report.get("report_hash") or "")
    if not report_hash:
        raise QualificationError("SKIN_TOPOLOGY_REPARTITION_REPORT_HASH_MISSING")
    if any(i<0 or i>=len(candidate.faces) for i in unsafe):
        raise QualificationError("SKIN_TOPOLOGY_UNSAFE_FACE_INDEX_INVALID")

    rest,weights,faces=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=skin
    )
    del rest
    faces=np.asarray(faces,dtype=np.int64)
    vertices=tuple(candidate.vertices)

    existing={
        tuple(sorted((str(row.a_surface_id),str(row.b_surface_id)))):str(row.decision)
        for row in partition.boundary_constraints
    }
    partition_owner={
        str(sid):str(component.component_id)
        for component in partition.components
        for sid in component.surface_ids
    }

    proposals={}
    unresolved=0
    for fi in unsafe:
        idx=tuple(map(int,faces[fi].tolist()))
        edge_rows=[]
        for ia,ib in ((0,1),(1,2),(2,0)):
            a,b=idx[ia],idx[ib]
            l1=float(np.abs(weights[a]-weights[b]).sum())
            edge_rows.append((l1,a,b))
        edge_rows.sort(key=lambda x:(x[0],-min(x[1],x[2]),-max(x[1],x[2])),reverse=True)

        chosen=None
        for l1,a,b in edge_rows:
            sa=_mechanical_owner_surface_id(vertices[a],partition_owner=partition_owner)
            sb=_mechanical_owner_surface_id(vertices[b],partition_owner=partition_owner)
            if sa is None or sb is None or sa==sb:
                continue
            pair=tuple(sorted((sa,sb)))
            if pair not in existing:
                continue
            if existing.get(pair)=="SEPARATE":
                continue
            chosen=(l1,a,b,sa,sb,pair)
            break
        if chosen is None:
            unresolved+=1
            continue
        l1,a,b,sa,sb,pair=chosen
        row=proposals.setdefault(pair,{
            "a_surface_id":pair[0],
            "b_surface_id":pair[1],
            "decision":"SEPARATE",
            "evidence_refs":[],
            "max_pairwise_skin_l1":0.0,
            "unsafe_face_indices":[],
            "confidence":0.0,
            "metadata":{
                "evidence_class":"STAGE35_DYNAMIC_SKIN_TOPOLOGY_MECHANICAL",
                "automatic":True,
                "manual_authoring_used":False,
                "mechanical_owner_surface_support_required":True,
                "identity_or_holeless_seam_owner_supported":True,
            },
        })
        row["evidence_refs"].append(f"{report_hash}:FACE:{fi}")
        row["unsafe_face_indices"].append(int(fi))
        row["max_pairwise_skin_l1"]=max(float(row["max_pairwise_skin_l1"]),l1)
        row["confidence"]=max(float(row["confidence"]),min(1.0,max(0.0,l1/2.0)))

    ordered=[]
    for pair in sorted(proposals):
        row=proposals[pair]
        row["evidence_refs"]=tuple(sorted(set(row["evidence_refs"])))
        row["unsafe_face_indices"]=tuple(sorted(set(row["unsafe_face_indices"])))
        row["constraint_id"]="DYNSEP:"+content_sha256({
            "report_hash":report_hash,
            "pair":pair,
            "faces":row["unsafe_face_indices"],
        })[:20]
        ordered.append(row)

    directive={
        "schema":"RealSaS.MechanicalRepartitionDirective.v2",
        "status":(
            "REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY"
            if ordered else
            "ABSTAIN__NO_MECHANICAL_OWNER_BOUNDARY_PROPOSAL"
        ),
        "source_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "source_surface_lineage_hash":surface.geometry_lineage_hash,
        "source_partition_lineage_hash":partition.partition_lineage_hash,
        "source_skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "source_skin_lineage_hash":skin.skin_lineage_hash,
        "compatibility_report_hash":report_hash,
        "unsafe_face_count":len(unsafe),
        "candidate_separate_pair_count":len(ordered),
        "unresolved_unsafe_face_count":int(unresolved),
        "proposed_boundary_overrides":tuple(ordered),
        "repair_operation":"STAGE17_REPARTITION_THEN_STAGE18_HOLELESS_DENSE_SUBDIVISION",
        "face_deletion_count":0,
        "weight_mutation":False,
        "vertex_position_mutation_at_stage35":False,
        "restart_from":"17_MECHANICAL_PARTITION_QUALIFIED",
        "mandatory_requalification_through":"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
        "auto_apply_allowed":False,
        "requires_trustworthy_skin_reliability_authority":True,
        "fail_closed_if_not_repartitioned":True,
        "directive_hash":"",
    }
    directive["directive_hash"]=content_sha256({k:v for k,v in directive.items() if k!="directive_hash"})
    return directive
