from __future__ import annotations

"""G3B topology x skin compatibility directly on exact carrier-native W_M."""

import math
import numpy as np

from .carrier_mechanics_v1 import carrier_skin_matrix_v1
from .hashing import content_sha256
from .joint_frames_v2 import derive_joint_frames_post_bind_v2
from .mesh.deformation_stress_v2 import _pose_skin_matrices
from .mesh.skin_topology_compatibility_v1 import (
    SKIN_TOPOLOGY_COMPAT_SCHEMA,
    DEFAULT_RISK_L1_MIN,
    DEFAULT_MAX_EDGE_RATIO,
    _skin_l1_per_face,
    _triangle_metrics_batch,
    _stress_angle,
)
from .product_authority_v1 import validate_deformation_capability_envelope, validate_mesh_qualification_policy
from .types import QualificationError


def run_skin_topology_compatibility_carrier_v1(
    carrier, *, skeleton, skin, envelope, cameras, policy,
    risk_l1_min:float=DEFAULT_RISK_L1_MIN,
    max_edge_ratio:float=DEFAULT_MAX_EDGE_RATIO,
    stress_all_faces:bool=False,
):
    validate_mesh_qualification_policy(policy)
    validate_deformation_capability_envelope(envelope,known_joint_ids={j.canonical_joint_id for j in skeleton.joints})
    if envelope.skeleton_lineage_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("CARRIER_SKIN_TOPOLOGY_ENVELOPE_SKELETON_DRIFT")
    if risk_l1_min<0.0 or risk_l1_min>2.0: raise QualificationError("CARRIER_SKIN_TOPOLOGY_RISK_L1_INVALID")
    if max_edge_ratio<=1.0 or not math.isfinite(max_edge_ratio): raise QualificationError("CARRIER_SKIN_TOPOLOGY_EDGE_RATIO_INVALID")

    rest,weights,faces,joint_ids=carrier_skin_matrix_v1(carrier,skeleton,skin)
    skin_l1=_skin_l1_per_face(weights,faces)
    risk_mask=np.ones((len(faces),),bool) if stress_all_faces else (skin_l1>float(risk_l1_min))
    risky=np.nonzero(risk_mask)[0]
    stress_angle=_stress_angle(envelope)
    frames,frame_report=derive_joint_frames_post_bind_v2(skeleton,carrier_skin=skin,cameras=cameras)
    max_area=np.ones((len(faces),),np.float64); min_area=np.ones((len(faces),),np.float64)
    max_condition=np.ones((len(faces),),np.float64); max_edge=np.ones((len(faces),),np.float64); min_edge=np.ones((len(faces),),np.float64)
    worst_probe=["REST"]*len(faces); probe_count=1
    if len(risky):
        for jid in sorted(joint_ids):
            for axis_index,axis_name in enumerate(("X","Y","Z")):
                for sign in (-1.0,1.0):
                    deg=sign*stress_angle
                    skin_by_id=_pose_skin_matrices(skeleton,frames,joint_id=jid,local_axis_index=axis_index,degrees=deg)
                    matrices=np.stack([skin_by_id[x] for x in joint_ids],axis=0)
                    hom=np.concatenate([rest,np.ones((len(rest),1),np.float64)],axis=1)
                    per=np.stack([(hom@matrices[k].T)[:,:3] for k in range(len(joint_ids))],axis=1)
                    posed=np.sum(per*weights[:,:,None],axis=1)
                    area,cond,emin,emax=_triangle_metrics_batch(rest,posed,faces[risky])
                    probe_id=f"{jid}:LOCAL_{axis_name}:{deg:+g}"
                    for local,fi in enumerate(risky.tolist()):
                        vals=(float(area[local]),float(cond[local]),float(emin[local]),float(emax[local]))
                        if not all(math.isfinite(x) for x in vals):
                            max_condition[fi]=float("inf"); worst_probe[fi]=probe_id; continue
                        severity=max(vals[1]/max(float(policy.g3_max_dynamic_condition_number),1e-12), vals[3]/max(float(max_edge_ratio),1e-12), vals[0]/max(float(policy.g3_max_dynamic_area_ratio),1e-12), float(policy.g3_min_dynamic_area_ratio)/max(vals[0],1e-12))
                        previous=max(max_condition[fi]/max(float(policy.g3_max_dynamic_condition_number),1e-12), max_edge[fi]/max(float(max_edge_ratio),1e-12), max_area[fi]/max(float(policy.g3_max_dynamic_area_ratio),1e-12), float(policy.g3_min_dynamic_area_ratio)/max(min_area[fi],1e-12))
                        if severity>previous: worst_probe[fi]=probe_id
                        max_area[fi]=max(max_area[fi],vals[0]); min_area[fi]=min(min_area[fi],vals[0])
                        max_condition[fi]=max(max_condition[fi],vals[1]); max_edge[fi]=max(max_edge[fi],vals[3]); min_edge[fi]=min(min_edge[fi],vals[2])
                    probe_count+=1
    unsafe=risk_mask & ((max_area>float(policy.g3_max_dynamic_area_ratio)) | (min_area<float(policy.g3_min_dynamic_area_ratio)) | (max_condition>float(policy.g3_max_dynamic_condition_number)) | (max_edge>float(max_edge_ratio)))
    unsafe_ids=np.nonzero(unsafe)[0]
    top=sorted(unsafe_ids.tolist(),key=lambda fi:(max_edge[fi],max_area[fi],max_condition[fi],skin_l1[fi]),reverse=True)[:256]
    vertex_ids=tuple(map(str,carrier.ordered_vertex_ids))
    report={
        "schema":SKIN_TOPOLOGY_COMPAT_SCHEMA,
        "candidate_lineage_hash":carrier.candidate_mesh_binding_hash,
        "surface_lineage_hash":carrier.source_surface_binding_hash,
        "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "skin_lineage_hash":skin.skin_lineage_hash,
        "carrier_evidence_hash":carrier.carrier_evidence_hash,
        "carrier_topology_hash":carrier.topology_hash,
        "carrier_geometry_hash":carrier.geometry_hash,
        "envelope_lineage_hash":envelope.envelope_lineage_hash,
        "qualification_policy_hash":policy.qualification_policy_lineage_hash,
        "stress_angle_deg":stress_angle,"probe_count":int(probe_count),"face_count":len(faces),
        "risky_face_count":int(len(risky)),"unsafe_face_count":int(len(unsafe_ids)),"risk_l1_min":float(risk_l1_min),
        "stress_all_faces":bool(stress_all_faces),"max_edge_ratio_limit":float(max_edge_ratio),
        "max_area_ratio_limit":float(policy.g3_max_dynamic_area_ratio),"min_area_ratio_limit":float(policy.g3_min_dynamic_area_ratio),
        "max_condition_limit":float(policy.g3_max_dynamic_condition_number),"weight_mutation":False,"passed":bool(len(unsafe_ids)==0),
        "unsafe_face_indices":tuple(map(int,unsafe_ids.tolist())),
        "top_unsafe_faces":tuple({
            "face_index":int(fi),"vertex_ids":tuple(vertex_ids[int(v)] for v in faces[fi]),"skin_l1_max":float(skin_l1[fi]),
            "max_edge_ratio":float(max_edge[fi]),"max_area_ratio":float(max_area[fi]),"min_area_ratio":float(min_area[fi]),
            "max_condition_number":float(max_condition[fi]),"worst_probe_id":str(worst_probe[fi]),
        } for fi in top),
        "report_hash":"",
        "metadata":{
            "role":"TOPOLOGY_X_CARRIER_NATIVE_SKIN_MECHANICAL_COMPATIBILITY",
            "mechanical_topology_host":"STAGE19_EXACT_CARRIER_M",
            "semantic_skin_transfer_performed":False,"actual_motion_capability_claimed":False,
            "repair_semantics":"NEW_M_INVALIDATES_W_M__RERUN_MIRA_REQUIRED",
            "skin_weights_are_immutable":True,"risk_prefilter_is_safety_authority":not bool(stress_all_faces),
            "joint_frame_qualification":frame_report,
        },
    }
    report["report_hash"]=content_sha256({k:v for k,v in report.items() if k!="report_hash"})
    return report

__all__=["run_skin_topology_compatibility_carrier_v1"]
