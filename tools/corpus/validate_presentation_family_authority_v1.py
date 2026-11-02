#!/usr/bin/env python3
from __future__ import annotations

"""Independent court for PresentationFamilyAuthority.v1.

The court does not import the extractor or Stage48 qualification helper. It
re-opens both renderer replays and both raw-source references, proves exact replay
and source-equivalence on the same hit keys, verifies scalar source-over algebra,
and reconstructs the compact appearance/visibility/oracle domains from PACKED_A.
"""

import argparse, hashlib, json
from pathlib import Path
import numpy as np

PARENT_SCHEMA="RealSaS.PresentationFamilyAuthority.v1"
PARENT_STATUS="PASS_PRESENTATION_FAMILY_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA="RealSaS.PresentationFamilyAuthorityCourt.v1"
STATUS="PASS_PRESENTATION_FAMILY_INDEPENDENT_COURT__STUDENT_ADMISSION_SEPARATE"
PASS_RAY="PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""): h.update(c)
    return h.hexdigest()


def bound(row,code,json_required=False):
    row=dict(row or {}); p=Path(str(row.get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=str(row.get("sha256") or ""): raise ValueError(code)
    return (p,json.loads(p.read_text())) if json_required else p


def core_packed(v):
    return {k:v.get(k) for k in ("schema","subject_id","teacher_scene_sha256","geometry_npz_sha256","candidate_sha256","camera_bundle_sha256","packed_source_scene_blend_sha256","sampling_contract","rgb_carrier_calibration","material_policy_replay","selected_ray_count","pass_ray_count","fail_ray_count","abstain_ray_count","rays","authority","court_status")}


def core_source(v):
    return {k:v.get(k) for k in ("schema","subject_id","source_sha256","candidate_sha256","camera_bundle_sha256","appearance_sha256","geometry_npz_sha256","source_material_transform","rgb_carrier_calibration","equivalence_scope","hit_count","hits","authority")}


def key(ray,i,face): return (int(ray["ray_index"]),int(ray["view_index"]),int(ray["pixel_x"]),int(ray["pixel_y"]),int(i),int(face))


def packed_map(v):
    out={}
    for ray in v.get("rays") or []:
        m=dict(ray.get("measurement") or {})
        if m.get("status")!=PASS_RAY or m.get("valid") is not True: continue
        hits=list(ray.get("canonical_hits") or []); a=np.asarray(m.get("effective_alpha"),np.float64); rgb=np.asarray(m.get("straight_linear_rgb"),np.float64)
        if a.shape!=(len(hits),) or rgb.shape!=(len(hits),3): raise ValueError("PRESENTATION_COURT_PACKED_LAYOUT")
        for i,h in enumerate(hits): out[key(ray,i,h["face_index"])]=(float(a[i]),rgb[i]*a[i])
    return out


def source_map(v):
    out={}
    for row in v.get("hits") or []:
        pseudo={"ray_index":row["ray_index"],"view_index":row["view_index"],"pixel_x":row["pixel_x"],"pixel_y":row["pixel_y"]}
        out[key(pseudo,row["hit_index"],row["face_index"])]=(float(row["effective_alpha"]),np.asarray(row["premultiplied_linear_rgb"],np.float64),float(row["footprint_convergence_delta"]))
    return out


def partition_rows(packed,floor):
    app_key=[]; app_face=[]; app_bary=[]; app_view=[]; app_rgba=[]
    hit_ray=[]; hit_face=[]; hit_depth=[]; hit_alpha=[]; hit_trans=[]; hit_contrib=[]
    pair_ray=[]; pair_a=[]; pair_b=[]; pair_front=[]; pair_occ=[]
    ray_key=[]; offsets=[0]; stack_face=[]; stack_depth=[]; stack_alpha=[]; stack_trans=[]; stack_contrib=[]; full_alpha=[]; full_rgb=[]
    pass_rays=0; pass_hits=0
    for ray in packed.get("rays") or []:
        m=dict(ray.get("measurement") or {})
        if m.get("status")!=PASS_RAY or m.get("valid") is not True: continue
        hits=list(ray.get("canonical_hits") or []); n=len(hits); a=np.asarray(m.get("effective_alpha"),np.float64); rgb=np.asarray(m.get("straight_linear_rgb"),np.float64); trans=np.asarray(m.get("incoming_transmittance"),np.float64); contrib=np.asarray(m.get("contribution"),np.float64)
        if a.shape!=(n,) or rgb.shape!=(n,3) or trans.shape!=(n,) or contrib.shape!=(n,): raise ValueError("PRESENTATION_COURT_RAY_LAYOUT")
        if not all(np.isfinite(x).all() for x in (a,rgb,trans,contrib)) or np.any((a<0)|(a>1)) or np.any((trans<0)|(trans>1)) or np.any((contrib<0)|(contrib>1)): raise ValueError("PRESENTATION_COURT_RAY_VALUES")
        expected_trans=np.ones(n,np.float64)
        for i in range(1,n): expected_trans[i]=expected_trans[i-1]*(1-a[i-1])
        expected_contrib=expected_trans*a
        if not np.allclose(trans,expected_trans,atol=1e-7,rtol=0) or not np.allclose(contrib,expected_contrib,atol=1e-7,rtol=0): raise ValueError("PRESENTATION_COURT_SOURCE_OVER_ALGEBRA")
        expected_alpha=float(np.sum(expected_contrib)); expected_rgb=np.sum(rgb*expected_contrib[:,None],axis=0)
        declared_alpha=float(m.get("predicted_full_alpha",expected_alpha)); declared_rgb=np.asarray(m.get("predicted_full_premultiplied_linear_rgb",expected_rgb),np.float64)
        if abs(declared_alpha-expected_alpha)>1e-7 or declared_rgb.shape!=(3,) or not np.allclose(declared_rgb,expected_rgb,atol=1e-7,rtol=0): raise ValueError("PRESENTATION_COURT_FULL_REPLAY")
        pass_rays+=1; pass_hits+=n; contributing=contrib>floor; direct=int(np.count_nonzero(contributing))==1; rk=[int(ray["ray_index"]),int(ray["view_index"]),int(ray["pixel_x"]),int(ray["pixel_y"])]; ray_key.append(rk)
        for i,h in enumerate(hits):
            face=int(h["face_index"]); bary=np.asarray(h["barycentric"],np.float64); depth=float(h["depth"])
            if bary.shape!=(3,) or not np.isclose(bary.sum(),1,atol=1e-7,rtol=0): raise ValueError("PRESENTATION_COURT_ADDRESS")
            hit_ray.append(rk[0]); hit_face.append(face); hit_depth.append(depth); hit_alpha.append(a[i]); hit_trans.append(trans[i]); hit_contrib.append(contrib[i]); stack_face.append(face); stack_depth.append(depth); stack_alpha.append(a[i]); stack_trans.append(trans[i]); stack_contrib.append(contrib[i])
            if direct and contributing[i]: app_key.append(rk+[i]); app_face.append(face); app_bary.append(bary); app_view.append(rk[1]); app_rgba.append([*rgb[i],a[i]])
        offsets.append(len(stack_face)); full_alpha.append(expected_alpha); full_rgb.append(expected_rgb)
        for i in range(n):
            for j in range(i+1,n):
                if not (contributing[i] and contributing[j]): continue
                for x,y,front,occ in ((i,j,1.,float(a[i])),(j,i,0.,0.)):
                    pair_ray.append(rk[0]); pair_a.append(int(hits[x]["face_index"])); pair_b.append(int(hits[y]["face_index"])); pair_front.append(front); pair_occ.append(occ)
    return {
        "appearance_key":np.asarray(app_key,np.int32).reshape(-1,5),"appearance_face":np.asarray(app_face,np.int32),"appearance_barycentric":np.asarray(app_bary,np.float32).reshape(-1,3),"appearance_view":np.asarray(app_view,np.int32),"appearance_rgba":np.asarray(app_rgba,np.float32).reshape(-1,4),
        "visibility_hit_ray":np.asarray(hit_ray,np.int32),"visibility_hit_face":np.asarray(hit_face,np.int32),"visibility_hit_depth":np.asarray(hit_depth,np.float32),"visibility_hit_alpha":np.asarray(hit_alpha,np.float32),"visibility_hit_incoming_transmittance":np.asarray(hit_trans,np.float32),"visibility_hit_contribution":np.asarray(hit_contrib,np.float32),
        "visibility_pair_ray":np.asarray(pair_ray,np.int32),"visibility_pair_face_a":np.asarray(pair_a,np.int32),"visibility_pair_face_b":np.asarray(pair_b,np.int32),"visibility_pair_front_probability":np.asarray(pair_front,np.float32),"visibility_pair_occlusion_probability":np.asarray(pair_occ,np.float32),
        "ray_key":np.asarray(ray_key,np.int32).reshape(-1,4),"ray_stack_offsets":np.asarray(offsets,np.int32),"ray_stack_face":np.asarray(stack_face,np.int32),"ray_stack_depth":np.asarray(stack_depth,np.float32),"ray_stack_alpha":np.asarray(stack_alpha,np.float32),"ray_stack_transmittance":np.asarray(stack_trans,np.float32),"ray_stack_contribution":np.asarray(stack_contrib,np.float32),"ray_full_alpha":np.asarray(full_alpha,np.float32),"ray_full_premultiplied_linear_rgb":np.asarray(full_rgb,np.float32).reshape(-1,3),
        "pass_rays":pass_rays,"pass_hits":pass_hits,
    }


def validate(manifest_path:Path)->dict:
    manifest_path=manifest_path.resolve(); m=json.loads(manifest_path.read_text())
    if m.get("schema")!=PARENT_SCHEMA or m.get("status")!=PARENT_STATUS: raise ValueError("PRESENTATION_COURT_PARENT")
    parents=dict(m.get("parents") or {}); summary_path,summary=bound(parents.get("stage48_summary"),"PRESENTATION_COURT_SUMMARY_DRIFT",True); arrays_path=bound(parents.get("stage48_arrays"),"PRESENTATION_COURT_ARRAYS_DRIFT"); pa_path,pa=bound(parents.get("packed_a"),"PRESENTATION_COURT_PACKED_A_DRIFT",True); pb_path,pb=bound(parents.get("packed_b"),"PRESENTATION_COURT_PACKED_B_DRIFT",True); sa_path,sa=bound(parents.get("source_reference_a"),"PRESENTATION_COURT_SOURCE_A_DRIFT",True); sb_path,sb=bound(parents.get("source_reference_b"),"PRESENTATION_COURT_SOURCE_B_DRIFT",True); output=bound(m.get("output"),"PRESENTATION_COURT_OUTPUT_DRIFT")
    subject=str(m.get("subject_id") or "")
    if summary.get("subject_id")!=subject or any(str(x.get("subject_id") or "")!=subject for x in (pa,pb,sa,sb)): raise ValueError("PRESENTATION_COURT_SUBJECT")
    if str(summary.get("array_sha256"))!=sha256(arrays_path): raise ValueError("PRESENTATION_COURT_ARRAY_BINDING")
    q=dict(summary.get("renderer_subject_qualification") or {})
    if q.get("status")!="PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE" or q.get("source_equivalent") is not True: raise ValueError("PRESENTATION_COURT_RENDERER_QUALIFICATION")
    if core_packed(pa)!=core_packed(pb) or core_source(sa)!=core_source(sb): raise ValueError("PRESENTATION_COURT_REPLAY_EXACT")
    pm,sm=packed_map(pa),source_map(sa)
    if not pm or set(pm)!=set(sm): raise ValueError("PRESENTATION_COURT_SOURCE_DOMAIN")
    max_a=max_rgb=max_conv=0.0
    for k in pm:
        aa,argb=pm[k]; ba,brgb,conv=sm[k]; max_a=max(max_a,abs(aa-ba)); max_rgb=max(max_rgb,float(np.max(np.abs(argb-brgb)))); max_conv=max(max_conv,conv)
    if max(max_a,max_rgb,max_conv)>5e-3: raise ValueError("PRESENTATION_COURT_SOURCE_EQUIVALENCE")
    floor=float((summary.get("policy") or {}).get("student_observable_contribution_floor",1e-6)); expected=partition_rows(pa,floor)
    with np.load(output,allow_pickle=False) as z:
        names=[k for k in expected if not k.startswith("pass_")]
        if not set(names).issubset(z.files): raise ValueError("PRESENTATION_COURT_OUTPUT_ARRAYS")
        for name in names:
            actual=np.asarray(z[name]); wanted=np.asarray(expected[name])
            if actual.shape!=wanted.shape: raise ValueError("PRESENTATION_COURT_SHAPE:"+name)
            if np.issubdtype(wanted.dtype,np.floating):
                if not np.allclose(actual,wanted,atol=2e-6,rtol=2e-6): raise ValueError("PRESENTATION_COURT_VALUE:"+name)
            elif not np.array_equal(actual,wanted): raise ValueError("PRESENTATION_COURT_VALUE:"+name)
    counts=dict(m.get("counts") or {})
    expected_counts=(expected["pass_rays"],expected["pass_hits"],len(expected["appearance_face"]),len(expected["visibility_hit_face"]),len(expected["visibility_pair_face_a"]),len(expected["ray_key"]))
    actual_counts=(counts.get("pass_ray_count"),counts.get("pass_hit_count"),counts.get("directly_identifiable_appearance_hit_count"),counts.get("measured_visibility_hit_count"),counts.get("student_observable_pair_direction_count"),counts.get("oracle_ray_count"))
    if actual_counts!=expected_counts: raise ValueError("PRESENTATION_COURT_COUNTS")
    if (m.get("authority") or {}).get("student_supervision_authorized") is not False or (m.get("authority") or {}).get("training_authorized") is not False: raise ValueError("PRESENTATION_COURT_PREMATURE_TRAINING_AUTHORITY")
    return {"schema":SCHEMA,"status":STATUS,"subject_id":subject,"counts":{"appearance_valid_rows":len(expected["appearance_face"]),"visibility_hit_rows":len(expected["visibility_hit_face"]),"visibility_pair_rows":len(expected["visibility_pair_face_a"]),"ray_oracle_rows":len(expected["ray_key"])},"source_equivalence":{"maximum_alpha_error":max_a,"maximum_premultiplied_rgb_error":max_rgb,"maximum_footprint_convergence_delta":max_conv},"family_candidate_outcomes":{"APPEARANCE_CAA":"QUALIFIED_MASKED_MEASURED_CANDIDATE__STUDENT_ADMISSION_REQUIRED" if len(expected["appearance_face"]) else "UNAVAILABLE_NO_DIRECTLY_IDENTIFIABLE_ROWS","VISIBILITY_OCCLUSION":"QUALIFIED_MASKED_MEASURED_CANDIDATE__STUDENT_ADMISSION_REQUIRED" if len(expected["visibility_hit_face"]) else "UNAVAILABLE_NO_MEASURED_ROWS","RAY_COMPOSITING":"QUALIFIED_MASKED_ORACLE_CANDIDATE__STUDENT_ADMISSION_REQUIRED" if len(expected["ray_key"]) else "UNAVAILABLE_NO_PASS_RAYS"},"authority":{"independent_court_passed":True,"student_supervision_authorized":False,"final_family_authority_minted":False,"training_authorized":False},"parent_manifest":{"path":str(manifest_path),"sha256":sha256(manifest_path)}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); r=validate(a.manifest); a.output.resolve().parent.mkdir(parents=True,exist_ok=True); a.output.resolve().write_text(json.dumps(r,sort_keys=True,indent=2)+"\n"); print(json.dumps({"status":r["status"],"subject_id":r["subject_id"],"counts":r["counts"]},sort_keys=True))


if __name__=="__main__": main()
