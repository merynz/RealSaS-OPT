from __future__ import annotations

import math
import numpy as np

from .types import QualificationError


def mesh_triangle_area_report(mesh) -> dict:
    by_id={v.canonical_mesh_vertex_id:np.asarray(v.P,dtype=np.float64) for v in mesh.vertices}; areas=[]
    for face in mesh.faces:
        if len(face)!=3: raise QualificationError("DEFORMATION_PROOF_REQUIRES_TRIANGULATED_MESH")
        a,b,c=(by_id[x] for x in face); area=.5*float(np.linalg.norm(np.cross(b-a,c-a))); areas.append(area)
    if not areas: return {"face_count":0,"min_area":0.0,"mean_area":0.0,"degenerate_faces":0}
    return {"face_count":len(areas),"min_area":float(min(areas)),"mean_area":float(np.mean(areas)),"degenerate_faces":int(sum(a<=1e-12 for a in areas))}


def mesh_skin_dense_weights(mesh, mesh_skin, skeleton):
    jids=tuple(sorted(j.canonical_joint_id for j in skeleton.joints)); ji={j:i for i,j in enumerate(jids)}; rows={r.canonical_mesh_vertex_id:r for r in mesh_skin.rows}; w=np.zeros((len(mesh.vertices),len(jids)),np.float64); p=np.zeros((len(mesh.vertices),3),np.float64)
    for i,v in enumerate(mesh.vertices):
        p[i]=v.P
        if v.canonical_mesh_vertex_id not in rows: raise QualificationError("DEFORMATION_PROOF_MISSING_MESH_SKIN_ROW")
        for jid,value in rows[v.canonical_mesh_vertex_id].influences:
            if jid not in ji: raise QualificationError("DEFORMATION_PROOF_ILLEGAL_JOINT")
            w[i,ji[jid]]=float(value)
    if (w<0).any() or not np.allclose(w.sum(1),1.0,atol=1e-7,rtol=0): raise QualificationError("DEFORMATION_PROOF_WEIGHT_SIMPLEX")
    return p,w,jids


def verified_mesh_lbs_measurement(mesh, mesh_skin, skeleton, transforms, expected):
    from experiments.geppetto_arachne_r6_20260901.verified_lbs_v1 import apply_verified_lbs_v1, verified_lbs_report_v1
    p,w,jids=mesh_skin_dense_weights(mesh,mesh_skin,skeleton); pred=apply_verified_lbs_v1(p,w,np.asarray(transforms,dtype=np.float64)); report=verified_lbs_report_v1(pred,np.asarray(expected,dtype=np.float64))
    return {"rms":report.rms,"p95":report.p95,"max_error":report.max_error,"point_count":report.point_count,"probe_count":report.probe_count,"joint_order":jids}


def effective_motion_track_count(motion_state) -> int:
    count=0
    for t in motion_state.joint_tracks:
        if len(t.keys)<2: continue
        first=(t.keys[0].translation_xy,t.keys[0].rotation_deg,t.keys[0].scale_xy,t.keys[0].depth_offset)
        if any((k.translation_xy,k.rotation_deg,k.scale_xy,k.depth_offset)!=first for k in t.keys[1:]): count+=1
    return count
