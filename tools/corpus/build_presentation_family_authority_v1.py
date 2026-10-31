#!/usr/bin/env python3
from __future__ import annotations

"""Extract graphics truth-family candidates from one source-equivalent Stage48 package.

This is a measured-domain extractor, not a renderer and not a training admission.
It keeps three domains distinct:
- APPEARANCE_CAA: only directly identifiable source-equivalent RGBA hits on rays
  with exactly one measurable source-over contributor;
- VISIBILITY_OCCLUSION: measured hit transmittance/contribution and jointly
  contributing same-ray pair attenuation;
- RAY_COMPOSITING: complete PASS ray stacks retained as compiler-oracle evidence.

Hidden/ambiguous rows are absent, never encoded as zero targets.  Surface locus
(face+barycentric) is provenance/addressing evidence, not a student input claim.
"""

import argparse, hashlib, json
from pathlib import Path
import numpy as np

SCHEMA="RealSaS.PresentationFamilyAuthority.v1"
STATUS="PASS_PRESENTATION_FAMILY_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
PASS_RAY="PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()


def ref(path:Path)->dict:
    path=Path(path).resolve(); return {"path":str(path),"sha256":sha256(path)}


def digest(value)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def load(path:Path)->dict:
    v=json.loads(Path(path).read_text())
    if not isinstance(v,dict): raise ValueError("PRESENTATION_FAMILY_JSON_OBJECT")
    return v


def packed_core(value):
    return {k:value.get(k) for k in ("schema","subject_id","teacher_scene_sha256","geometry_npz_sha256","candidate_sha256","camera_bundle_sha256","packed_source_scene_blend_sha256","sampling_contract","rgb_carrier_calibration","material_policy_replay","selected_ray_count","pass_ray_count","fail_ray_count","abstain_ray_count","rays","authority","court_status")}


def source_core(value):
    return {k:value.get(k) for k in ("schema","subject_id","source_sha256","candidate_sha256","camera_bundle_sha256","appearance_sha256","geometry_npz_sha256","source_material_transform","rgb_carrier_calibration","equivalence_scope","hit_count","hits","authority")}


def hit_key(ray,hit_index,face_index):
    return (int(ray["ray_index"]),int(ray["view_index"]),int(ray["pixel_x"]),int(ray["pixel_y"]),int(hit_index),int(face_index))


def packed_hits(value):
    out={}
    for ray in value.get("rays") or []:
        m=dict(ray.get("measurement") or {})
        if m.get("status")!=PASS_RAY or m.get("valid") is not True: continue
        hits=list(ray.get("canonical_hits") or []); alpha=np.asarray(m.get("effective_alpha"),np.float64); rgb=np.asarray(m.get("straight_linear_rgb"),np.float64)
        if alpha.shape!=(len(hits),) or rgb.shape!=(len(hits),3): raise ValueError("PRESENTATION_FAMILY_PACKED_LAYOUT")
        for i,h in enumerate(hits):
            k=hit_key(ray,i,int(h["face_index"])); out[k]=(float(alpha[i]),rgb[i]*alpha[i])
    return out


def source_hits(value):
    out={}
    for row in value.get("hits") or []:
        pseudo={"ray_index":row["ray_index"],"view_index":row["view_index"],"pixel_x":row["pixel_x"],"pixel_y":row["pixel_y"]}; k=hit_key(pseudo,int(row["hit_index"]),int(row["face_index"])); out[k]=(float(row["effective_alpha"]),np.asarray(row["premultiplied_linear_rgb"],np.float64),float(row["footprint_convergence_delta"]))
    return out


def source_equivalence(packed_a,packed_b,source_a,source_b,tol=5e-3):
    if packed_core(packed_a)!=packed_core(packed_b) or source_core(source_a)!=source_core(source_b): raise ValueError("PRESENTATION_FAMILY_REPLAY_NOT_EXACT")
    ph=packed_hits(packed_a); sh=source_hits(source_a)
    if not ph or set(ph)!=set(sh): raise ValueError("PRESENTATION_FAMILY_SOURCE_DOMAIN")
    max_a=max_rgb=max_conv=0.0
    for k in ph:
        pa,prgb=ph[k]; sa,srgb,conv=sh[k]; max_a=max(max_a,abs(pa-sa)); max_rgb=max(max_rgb,float(np.max(np.abs(prgb-srgb)))); max_conv=max(max_conv,conv)
    if max(max_a,max_rgb,max_conv)>tol: raise ValueError("PRESENTATION_FAMILY_SOURCE_EQUIVALENCE")
    return {"hit_count":len(ph),"maximum_alpha_error":max_a,"maximum_premultiplied_rgb_error":max_rgb,"maximum_footprint_convergence_delta":max_conv}


def write_npz(path:Path,**arrays):
    for name,value in arrays.items():
        a=np.asarray(value)
        if a.dtype==object or (np.issubdtype(a.dtype,np.floating) and not np.isfinite(a).all()): raise ValueError("PRESENTATION_FAMILY_INVALID_ARRAY:"+name)
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".partial.npz"); np.savez_compressed(tmp,**arrays); tmp.replace(path)


def build(subject_id:str,summary_path:Path,arrays_path:Path,packed_a_path:Path,packed_b_path:Path,source_a_path:Path,source_b_path:Path,output_dir:Path)->dict:
    paths=[Path(x).resolve() for x in (summary_path,arrays_path,packed_a_path,packed_b_path,source_a_path,source_b_path)]; summary,packed_a,packed_b,source_a,source_b=map(load,(paths[0],paths[2],paths[3],paths[4],paths[5]))
    if summary.get("schema")!="RealSaS.PresentationResponseTruthCandidate.v1" or summary.get("subject_id")!=subject_id: raise ValueError("PRESENTATION_FAMILY_SUMMARY")
    if str(summary.get("array_sha256"))!=sha256(paths[1]): raise ValueError("PRESENTATION_FAMILY_ARRAY_BINDING")
    q=dict(summary.get("renderer_subject_qualification") or {})
    if q.get("status")!="PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE" or q.get("source_equivalent") is not True or q.get("packed_replay_exact") is not True or q.get("source_reference_replay_exact") is not True: raise ValueError("PRESENTATION_FAMILY_RENDERER_QUALIFICATION")
    if any(str(x.get("subject_id") or "")!=subject_id for x in (packed_a,packed_b,source_a,source_b)): raise ValueError("PRESENTATION_FAMILY_SUBJECT")
    eq=source_equivalence(packed_a,packed_b,source_a,source_b)
    floor=float((summary.get("policy") or {}).get("student_observable_contribution_floor",1e-6))
    if not np.isfinite(floor) or floor<0: raise ValueError("PRESENTATION_FAMILY_CONTRIBUTION_FLOOR")

    app_key=[]; app_face=[]; app_bary=[]; app_view=[]; app_rgba=[]
    hit_ray=[]; hit_face=[]; hit_depth=[]; hit_alpha=[]; hit_trans=[]; hit_contrib=[]
    pair_ray=[]; pair_a=[]; pair_b=[]; pair_front=[]; pair_occ=[]
    ray_key=[]; ray_offsets=[0]; ray_face=[]; ray_depth=[]; ray_alpha=[]; ray_trans=[]; ray_contrib=[]; ray_full_alpha=[]; ray_full_rgb=[]
    pass_ray_count=0; pass_hit_count=0
    for ray in packed_a.get("rays") or []:
        m=dict(ray.get("measurement") or {})
        if m.get("status")!=PASS_RAY or m.get("valid") is not True: continue
        hits=list(ray.get("canonical_hits") or []); n=len(hits)
        alpha=np.asarray(m.get("effective_alpha"),np.float64); rgb=np.asarray(m.get("straight_linear_rgb"),np.float64); trans=np.asarray(m.get("incoming_transmittance"),np.float64); contribution=np.asarray(m.get("contribution"),np.float64)
        if alpha.shape!=(n,) or rgb.shape!=(n,3) or trans.shape!=(n,) or contribution.shape!=(n,) or not all(np.isfinite(x).all() for x in (alpha,rgb,trans,contribution)): raise ValueError("PRESENTATION_FAMILY_RAY_LAYOUT")
        pass_ray_count+=1; pass_hit_count+=n; contributing=contribution>floor; directly=int(np.count_nonzero(contributing))==1
        rk=[int(ray["ray_index"]),int(ray["view_index"]),int(ray["pixel_x"]),int(ray["pixel_y"])]; ray_key.append(rk)
        for i,h in enumerate(hits):
            face=int(h["face_index"]); bary=np.asarray(h["barycentric"],np.float64); depth=float(h["depth"])
            if bary.shape!=(3,) or not np.isfinite(bary).all() or not np.isclose(bary.sum(),1,atol=1e-7,rtol=0): raise ValueError("PRESENTATION_FAMILY_HIT_ADDRESS")
            hit_ray.append(int(ray["ray_index"])); hit_face.append(face); hit_depth.append(depth); hit_alpha.append(alpha[i]); hit_trans.append(trans[i]); hit_contrib.append(contribution[i])
            ray_face.append(face); ray_depth.append(depth); ray_alpha.append(alpha[i]); ray_trans.append(trans[i]); ray_contrib.append(contribution[i])
            if directly and bool(contributing[i]):
                app_key.append(rk+[i]); app_face.append(face); app_bary.append(bary); app_view.append(int(ray["view_index"])); app_rgba.append([rgb[i,0],rgb[i,1],rgb[i,2],alpha[i]])
        ray_offsets.append(len(ray_face))
        pred_alpha=float(m.get("predicted_full_alpha",np.sum(contribution)))
        pred_rgb=np.asarray(m.get("predicted_full_premultiplied_linear_rgb",np.sum(rgb*contribution[:,None],axis=0)),np.float64)
        if not np.isfinite(pred_alpha) or pred_rgb.shape!=(3,) or not np.isfinite(pred_rgb).all(): raise ValueError("PRESENTATION_FAMILY_FULL_RESULT")
        ray_full_alpha.append(pred_alpha); ray_full_rgb.append(pred_rgb)
        for i in range(n):
            for j in range(i+1,n):
                if not (contributing[i] and contributing[j]): continue
                for a,b,front,occ in ((i,j,1.0,float(alpha[i])),(j,i,0.0,0.0)):
                    pair_ray.append(int(ray["ray_index"])); pair_a.append(int(hits[a]["face_index"])); pair_b.append(int(hits[b]["face_index"])); pair_front.append(front); pair_occ.append(occ)
    if pass_ray_count<=0: raise ValueError("PRESENTATION_FAMILY_NO_PASS_RAY")
    output_dir=Path(output_dir).resolve(); output_dir.mkdir(parents=True,exist_ok=True); out=output_dir/"PRESENTATION_FAMILY_AUTHORITY_V1.npz"
    write_npz(out,
        appearance_key=np.asarray(app_key,np.int32).reshape(-1,5),appearance_face=np.asarray(app_face,np.int32),appearance_barycentric=np.asarray(app_bary,np.float32).reshape(-1,3),appearance_view=np.asarray(app_view,np.int32),appearance_rgba=np.asarray(app_rgba,np.float32).reshape(-1,4),
        visibility_hit_ray=np.asarray(hit_ray,np.int32),visibility_hit_face=np.asarray(hit_face,np.int32),visibility_hit_depth=np.asarray(hit_depth,np.float32),visibility_hit_alpha=np.asarray(hit_alpha,np.float32),visibility_hit_incoming_transmittance=np.asarray(hit_trans,np.float32),visibility_hit_contribution=np.asarray(hit_contrib,np.float32),
        visibility_pair_ray=np.asarray(pair_ray,np.int32),visibility_pair_face_a=np.asarray(pair_a,np.int32),visibility_pair_face_b=np.asarray(pair_b,np.int32),visibility_pair_front_probability=np.asarray(pair_front,np.float32),visibility_pair_occlusion_probability=np.asarray(pair_occ,np.float32),
        ray_key=np.asarray(ray_key,np.int32).reshape(-1,4),ray_stack_offsets=np.asarray(ray_offsets,np.int32),ray_stack_face=np.asarray(ray_face,np.int32),ray_stack_depth=np.asarray(ray_depth,np.float32),ray_stack_alpha=np.asarray(ray_alpha,np.float32),ray_stack_transmittance=np.asarray(ray_trans,np.float32),ray_stack_contribution=np.asarray(ray_contrib,np.float32),ray_full_alpha=np.asarray(ray_full_alpha,np.float32),ray_full_premultiplied_linear_rgb=np.asarray(ray_full_rgb,np.float32).reshape(-1,3))
    manifest={"schema":SCHEMA,"status":STATUS,"subject_id":subject_id,"parents":{"stage48_summary":ref(paths[0]),"stage48_arrays":ref(paths[1]),"packed_a":ref(paths[2]),"packed_b":ref(paths[3]),"source_reference_a":ref(paths[4]),"source_reference_b":ref(paths[5])},"output":{"path":str(out),"sha256":sha256(out)},"source_equivalence":eq,"counts":{"pass_ray_count":pass_ray_count,"pass_hit_count":pass_hit_count,"directly_identifiable_appearance_hit_count":len(app_face),"measured_visibility_hit_count":len(hit_face),"student_observable_pair_direction_count":len(pair_a),"oracle_ray_count":len(ray_key)},"family_candidate_outcomes":{"APPEARANCE_CAA":"QUALIFIED_MASKED_MEASURED_CANDIDATE__STUDENT_ADMISSION_REQUIRED","VISIBILITY_OCCLUSION":"QUALIFIED_MASKED_MEASURED_CANDIDATE__STUDENT_ADMISSION_REQUIRED","RAY_COMPOSITING":"QUALIFIED_MASKED_ORACLE_CANDIDATE__STUDENT_ADMISSION_REQUIRED"},"observability":{"APPEARANCE_CAA":"OBSERVATION_CONDITIONED__EXACTLY_ONE_MEASURABLE_CONTRIBUTOR_ONLY","VISIBILITY_OCCLUSION":"MECHANISM_CONDITIONAL__JOINTLY_CONTRIBUTING_SAME_RAY_ONLY","RAY_COMPOSITING":"COMPILER_ORACLE_COMPLETE_PASS_RAY_DOMAIN__NO_AUTOMATIC_STUDENT_SUPERVISION"},"policy":{"unknown_or_ambiguous_rows":"ABSENT_NOT_ZERO","geometric_front_order_student_target":False,"compiler_oracle_retains_complete_ray_order":True,"appearance_provenance":"OBSERVED_SOURCE_EQUIVALENT","material_or_owner_ids_as_student_targets":False,"global_sampler_qualified":False},"authority":{"measurement_candidate_only":True,"student_supervision_authorized":False,"final_family_authority_minted":False,"training_authorized":False}}
    manifest["identity_sha256"]=digest(manifest); mp=output_dir/"PRESENTATION_FAMILY_AUTHORITY_MANIFEST_V1.json"; mp.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n"); return manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument("--subject-id",required=True)
    for flag in ("summary","arrays","packed-a","packed-b","source-a","source-b","output-dir"): p.add_argument("--"+flag,type=Path,required=True)
    a=p.parse_args(); r=build(a.subject_id,a.summary,a.arrays,a.packed_a,a.packed_b,a.source_a,a.source_b,a.output_dir); print(json.dumps({"status":r["status"],"subject_id":r["subject_id"],"counts":r["counts"]},sort_keys=True))


if __name__=="__main__": main()
