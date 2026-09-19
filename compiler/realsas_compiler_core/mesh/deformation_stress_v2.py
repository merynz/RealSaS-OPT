from __future__ import annotations

"""Full-3D local-frame G3 numerical conditioning probe.

This is not a motion capability envelope and it does not predict animation quality.
It checks that candidate M + qualified W remain numerically well conditioned under
small independent rotations about each derived local X/Y/Z joint-frame axis.
Actual Idle/Run/Slash capability is proved later by Stage35 exact clip execution.
"""

from dataclasses import replace
import math

import numpy as np

from .deformation_stress_v1 import (
    G3DeformationStressReportIR,
    _candidate_skin_matrix,
    _triangle_metrics,
)
from ..hashing import content_sha256
from ..joint_frames_v1 import derive_joint_frames_from_skeleton, frame_set_hash
from ..motion_3d_v1 import apply_lbs_matrix_v1
from ..product_authority_v1 import (
    validate_deformation_capability_envelope,
    validate_mesh_qualification_policy,
)
from ..types import QualificationError

G3_LOCAL_MICRO_STRESS_ANGLE_DEG=10.0
G3_LOCAL_MICRO_STRESS_SCHEMA="RealSaS.G3LocalFrameMicroStress.v2"


def _axis_angle(axis,degrees):
    a=np.asarray(axis,dtype=np.float64)
    n=float(np.linalg.norm(a))
    if n<=1e-12 or not np.isfinite(a).all():
        raise QualificationError("G3_V2_AXIS_INVALID")
    x,y,z=(a/n).tolist()
    angle=math.radians(float(degrees))
    c=math.cos(angle); s=math.sin(angle); C=1.0-c
    return np.asarray([
        [x*x*C+c,x*y*C-z*s,x*z*C+y*s],
        [y*x*C+z*s,y*y*C+c,y*z*C-x*s],
        [z*x*C-y*s,z*y*C+x*s,z*z*C+c],
    ],dtype=np.float64)


def _mat4(R=None,t=(0.0,0.0,0.0)):
    out=np.eye(4,dtype=np.float64)
    if R is not None:
        out[:3,:3]=np.asarray(R,dtype=np.float64)
    out[:3,3]=np.asarray(t,dtype=np.float64)
    return out


def _topological(skeleton):
    by={j.canonical_joint_id:j for j in skeleton.joints}
    if len(by)!=len(skeleton.joints) or skeleton.root_id not in by:
        raise QualificationError("G3_V2_SKELETON_INVALID")
    order=[]; visiting=set(); done=set()
    def visit(jid):
        if jid in done:return
        if jid in visiting:raise QualificationError("G3_V2_SKELETON_CYCLE")
        visiting.add(jid)
        p=by[jid].parent_canonical_id
        if p is not None:
            if p not in by:raise QualificationError("G3_V2_PARENT_UNKNOWN")
            visit(p)
        visiting.remove(jid); done.add(jid); order.append(jid)
    for jid in sorted(by):visit(jid)
    return tuple(order),by


def _rest_global(skeleton,frames):
    return {
        jid:_mat4(np.asarray(frame.rotation_matrix,dtype=np.float64),frame.rest_position)
        for jid,frame in frames.items()
    }


def _pose_skin_matrices(skeleton,frames,*,joint_id=None,local_axis_index=None,degrees=0.0):
    order,by=_topological(skeleton)
    rest=_rest_global(skeleton,frames)
    posed={}
    for jid in order:
        parent=by[jid].parent_canonical_id
        if parent is None:
            local_rest=rest[jid]
        else:
            local_rest=np.linalg.inv(rest[parent])@rest[jid]
        D=np.eye(4,dtype=np.float64)
        if jid==joint_id:
            axis=np.zeros(3,dtype=np.float64); axis[int(local_axis_index)]=1.0
            D[:3,:3]=_axis_angle(axis,degrees)
        posed[jid]=local_rest@D if parent is None else posed[parent]@local_rest@D
    return {
        jid:posed[jid]@np.linalg.inv(rest[jid])
        for jid in order
    }


def run_g3_local_frame_micro_stress_v2(
    candidate,
    *,
    surface,
    skeleton,
    skin,
    envelope,
    cameras,
    policy,
):
    validate_mesh_qualification_policy(policy)
    validate_deformation_capability_envelope(
        envelope,known_joint_ids={j.canonical_joint_id for j in skeleton.joints}
    )
    if envelope.skeleton_lineage_hash!=skeleton.skeleton_lineage_hash:
        raise QualificationError("G3_V2_ENVELOPE_SKELETON_DRIFT")
    frames=derive_joint_frames_from_skeleton(skeleton,cameras=cameras)
    frames_hash=frame_set_hash(frames)
    rest_positions,weights,faces=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=skin
    )
    joint_ids=tuple(j.canonical_joint_id for j in skeleton.joints)
    joint_index={jid:i for i,jid in enumerate(joint_ids)}
    if len(joint_index)!=len(joint_ids):
        raise QualificationError("G3_V2_JOINT_ID_DUPLICATE")

    probes=[("REST",None,None,0.0)]
    for jid in sorted(joint_ids):
        for axis_index,axis_name in enumerate(("X","Y","Z")):
            for sign in (-1.0,1.0):
                degrees=sign*G3_LOCAL_MICRO_STRESS_ANGLE_DEG
                probes.append((f"{jid}:LOCAL_{axis_name}:{degrees:+g}",jid,axis_index,degrees))

    per_probe=[]
    failures=set()
    global_min_area=float("inf")
    global_max_area=0.0
    global_max_condition=0.0
    global_min_edge=float("inf")
    global_max_edge=0.0

    for probe_id,jid,axis_index,degrees in probes:
        skin_by_id=_pose_skin_matrices(
            skeleton,frames,joint_id=jid,local_axis_index=axis_index,degrees=degrees
        )
        matrices=np.stack([skin_by_id[x] for x in joint_ids],axis=0)
        posed=apply_lbs_matrix_v1(rest_positions,weights,matrices)
        min_area=float("inf"); max_area=0.0
        max_condition=0.0; min_edge=float("inf"); max_edge=0.0
        for face in faces:
            metric=_triangle_metrics(rest_positions[list(face)],posed[list(face)])
            area_ratio,condition,edge_min,edge_max,smin=metric
            if not all(math.isfinite(x) for x in metric):
                failures.add("NONFINITE_DEFORMATION_METRIC")
                continue
            min_area=min(min_area,area_ratio); max_area=max(max_area,area_ratio)
            max_condition=max(max_condition,condition)
            min_edge=min(min_edge,edge_min); max_edge=max(max_edge,edge_max)
            if area_ratio<policy.g3_min_dynamic_area_ratio:
                failures.add("DYNAMIC_AREA_RATIO_BELOW_MIN")
            if area_ratio>policy.g3_max_dynamic_area_ratio:
                failures.add("DYNAMIC_AREA_RATIO_ABOVE_MAX")
            if condition>policy.g3_max_dynamic_condition_number:
                failures.add("DYNAMIC_CONDITION_NUMBER_ABOVE_MAX")
        global_min_area=min(global_min_area,min_area)
        global_max_area=max(global_max_area,max_area)
        global_max_condition=max(global_max_condition,max_condition)
        global_min_edge=min(global_min_edge,min_edge)
        global_max_edge=max(global_max_edge,max_edge)
        per_probe.append({
            "probe_id":probe_id,
            "joint_id":jid,
            "local_axis_index":axis_index,
            "rotation_degrees":degrees,
            "derived_joint_frame_set_hash":frames_hash,
            "minimum_area_ratio":min_area,
            "maximum_area_ratio":max_area,
            "maximum_condition_number":max_condition,
            "minimum_edge_ratio":min_edge,
            "maximum_edge_ratio":max_edge,
        })

    probe_plan_hash=content_sha256({
        "schema":G3_LOCAL_MICRO_STRESS_SCHEMA,
        "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "derived_joint_frame_set_hash":frames_hash,
        "angle_deg":G3_LOCAL_MICRO_STRESS_ANGLE_DEG,
        "probe_ids":tuple(row["probe_id"] for row in per_probe),
        "actual_motion_capability_claimed":False,
    })
    passed=not failures
    value=G3DeformationStressReportIR(
        candidate_lineage_hash=candidate.candidate_lineage_hash,
        surface_lineage_hash=surface.geometry_lineage_hash,
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        envelope_lineage_hash=envelope.envelope_lineage_hash,
        axis_contract_hash=frames_hash,
        qualification_policy_hash=policy.qualification_policy_lineage_hash,
        probe_plan_hash=probe_plan_hash,
        probe_count=len(per_probe),
        face_count=len(faces),
        minimum_area_ratio=float(global_min_area),
        maximum_area_ratio=float(global_max_area),
        maximum_condition_number=float(global_max_condition),
        minimum_edge_ratio=float(global_min_edge),
        maximum_edge_ratio=float(global_max_edge),
        failure_invariants=tuple(sorted(failures)),
        passed=passed,
        per_probe=tuple(per_probe),
        report_hash="",
        metadata={
            "measurement_role":"STAGE27_LOCAL_3D_NUMERICAL_CONDITIONING_ONLY",
            "probe_schema":G3_LOCAL_MICRO_STRESS_SCHEMA,
            "micro_stress_angle_deg":G3_LOCAL_MICRO_STRESS_ANGLE_DEG,
            "derived_joint_frame_set_hash":frames_hash,
            "actual_motion_capability_claimed":False,
            "actual_clip_capability_authority":"STAGE35_EXACT_QUATERNION_CLIP_EXECUTION",
            "mesh_skin_product_authority_minted":False,
        },
    )
    payload=value.to_dict(); payload.pop("report_hash",None)
    return replace(value,report_hash=content_sha256(payload))
