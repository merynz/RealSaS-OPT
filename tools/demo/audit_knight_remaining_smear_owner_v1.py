from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict,CAA_PROVENANCE_BY_CODE
from compiler.realsas_compiler_core.appearance_render_v2 import load_face_page_index,load_face_uv,load_provenance_atlas,render_caa_reference
from compiler.realsas_compiler_core.artifact_codec_v2 import canonical_mesh_candidate_from_dict,qualified_camera_set_from_dict,qualified_skeleton_from_dict,qualified_skin_from_dict
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_texture_pages
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights,_ctx,_skin,_tracks_for_clip
from tools.demo.render_knight_graph_seam_cut_preview_v1 import _faces,_edge_l1,_metrics,_graph_seam_cut,RISK_L1_MIN,MAX_EDGE_STRETCH,MAX_AREA_RATIO,MIN_AREA_RATIO,MAX_CONDITION,CLIPS

CASES=(("idle_v2_f0","demo_idle_v1",2,0),("run_v2_f0","demo_run_v1",2,0),("slash_v0_f0","demo_slash_v1",0,0))

def _final_counts(render,n):
    owner=np.asarray(render.owner_face_index,dtype=np.int64)
    alpha=np.asarray(render.final_alpha,dtype=bool)
    vals=owner[alpha]
    vals=vals[vals>=0]
    return np.bincount(vals,minlength=n)[:n] if len(vals) else np.zeros(n,dtype=np.int64)

def _prov_for_face(render,fi):
    owner=np.asarray(render.owner_face_index,dtype=np.int64)
    alpha=np.asarray(render.final_alpha,dtype=bool)
    prov=np.asarray(render.provenance_code,dtype=np.uint8)
    m=alpha&(owner==int(fi))
    vals=prov[m]
    out={}
    for code in np.unique(vals):
        out[CAA_PROVENANCE_BY_CODE.get(int(code),str(int(code)))]=int(np.sum(vals==code))
    return out

def run(authority_root,run_id,out_dir):
    ctx=_ctx(authority_root,run_id)
    cand=canonical_mesh_candidate_from_dict(stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    skel=qualified_skeleton_from_dict(stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    skin=qualified_skin_from_dict(stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))
    cams=tuple(sorted(qualified_camera_set_from_dict(stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")).cameras,key=lambda c:int(c.view_index)))
    app=complete_appearance_asset_from_dict(stage_output_payload(ctx,"23_COMPLETE_APPEARANCE_ASSET_BAKED","RealSaS.CompleteAppearanceAssetIR.v2"))
    uv=load_face_uv(app);page=load_face_page_index(app);prov=load_provenance_atlas(app);tex={int(x.direction_index):x for x in app.textures}
    sr=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    jids,W=_candidate_skin_weights(cand,skin,skel);rest=np.asarray([v.P for v in cand.vertices],float);faces=_faces(cand);el1=_edge_l1(W,faces);fl1=el1.max(axis=1)
    risky=np.nonzero(fl1>RISK_L1_MIN)[0]
    me=np.ones(len(cand.faces));ma=np.ones(len(cand.faces));mia=np.ones(len(cand.faces));mc=np.ones(len(cand.faces))
    cache={}
    for clip,short in CLIPS:
        payload=json.loads((ctx["run_root"]/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip}.motion.json").read_text())
        tracks,mapping=_tracks_for_clip(payload,skel,cams,sr);times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")));cache[clip]=(tracks,times)
        for t in times:
            mats,_,_=_joint_pose_v2(skeleton=skel,tracks=tracks,time_seconds=float(t),cameras=cams);posed=_skin(rest,W,jids,mats)
            a,c,emin,emax=_metrics(rest,posed,faces[risky]);me[risky]=np.maximum(me[risky],emax);ma[risky]=np.maximum(ma[risky],a);mia[risky]=np.minimum(mia[risky],a);mc[risky]=np.maximum(mc[risky],c)
    unsafe=set(map(int,np.nonzero((fl1>RISK_L1_MIN)&((me>MAX_EDGE_STRETCH)|(ma>MAX_AREA_RATIO)|(mia<MIN_AREA_RATIO)|(mc>MAX_CONDITION)))[0]))
    repaired,keep,forbidden=_graph_seam_cut(cand,unsafe,el1);rf=_faces(repaired)
    inverse_original={ri:int(oi) for ri,oi in enumerate(keep.tolist())}
    out={"schema":"RealSaS.KnightRemainingSmearOwnerAudit.v1","repaired_lineage":repaired.candidate_lineage_hash,"forbidden_edge_count":len(forbidden),"removed_face_count":len(cand.faces)-len(repaired.faces),"cases":[]}
    for label,clip,view,frame_i in CASES:
        tracks,times=cache[clip];t=float(times[frame_i]);mats,_,_=_joint_pose_v2(skeleton=skel,tracks=tracks,time_seconds=t,cameras=cams);posed=_skin(rest,W,jids,mats)
        texture=_load_texture_pages(tex[view])
        dyn=render_caa_reference(mesh=repaired,camera=cams[view],face_uv=uv[keep],texture_rgba_u8=texture,provenance_atlas=prov[view],face_page_index=page[keep],positions=posed)
        rst=render_caa_reference(mesh=repaired,camera=cams[view],face_uv=uv[keep],texture_rgba_u8=texture,provenance_atlas=prov[view],face_page_index=page[keep],positions=rest)
        dc=_final_counts(dyn,len(repaired.faces));rc=_final_counts(rst,len(repaired.faces))
        a,c,emin,emax=_metrics(rest,posed,rf)
        growth=dc-rc
        order=np.argsort(-growth)[:60]
        rows=[]
        for ri in order:
            if growth[ri]<=0: continue
            oi=inverse_original[int(ri)]
            rows.append({"repaired_face_index":int(ri),"original_face_index":oi,"vertex_ids":list(map(str,repaired.faces[int(ri)])),"dynamic_pixels":int(dc[ri]),"rest_pixels":int(rc[ri]),"pixel_growth":int(growth[ri]),"pixel_ratio":float(dc[ri]/max(int(rc[ri]),1)),"skin_l1_max":float(fl1[oi]),"edge_ratio":float(emax[ri]),"area_ratio":float(a[ri]),"condition":float(c[ri]),"provenance":_prov_for_face(dyn,int(ri))})
        prov_all=np.asarray(dyn.provenance_code,dtype=np.uint8)[np.asarray(dyn.final_alpha,dtype=bool)]
        dist={CAA_PROVENANCE_BY_CODE.get(int(code),str(int(code))):int(np.sum(prov_all==code)) for code in np.unique(prov_all)}
        out["cases"].append({"label":label,"clip_id":clip,"view":view,"frame_index":frame_i,"time_seconds":t,"final_alpha_pixels":int(np.count_nonzero(dyn.final_alpha)),"rest_alpha_pixels":int(np.count_nonzero(rst.final_alpha)),"provenance_distribution":dist,"top_screen_growth_faces":rows})
    out_dir.mkdir(parents=True,exist_ok=True);(out_dir/"REPORT.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print("REMAINING_SMEAR_OWNER_AUDIT_PASS",json.dumps({"cases":len(out["cases"])},sort_keys=True))

def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out-dir",type=Path,required=True);a=p.parse_args();run(a.authority_root,a.run_id,a.out_dir)
if __name__=="__main__":main()
