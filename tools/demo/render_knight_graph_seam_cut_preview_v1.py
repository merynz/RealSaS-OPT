from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_page_index, load_face_uv, load_provenance_atlas, render_caa_reference,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, qualified_camera_set_from_dict,
    qualified_skeleton_from_dict, qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.product_authority_v1 import canonical_mesh_candidate_lineage_hash
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_texture_pages
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights, _composite_sheet, _ctx, _skin, _tracks_for_clip,
)

RISK_L1_MIN=1e-4
MAX_EDGE_STRETCH=4.0
MAX_AREA_RATIO=20.0
MIN_AREA_RATIO=0.05
MAX_CONDITION=16.0
CLIPS=(("demo_idle_v1","idle"),("demo_run_v1","run"),("demo_slash_v1","slash"))


def _faces(candidate):
    vi={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    return np.asarray([[vi[str(x)] for x in face] for face in candidate.faces],dtype=np.int64)


def _edge_l1(W,faces):
    wf=W[faces]
    return np.stack((
        np.abs(wf[:,0]-wf[:,1]).sum(axis=1),
        np.abs(wf[:,1]-wf[:,2]).sum(axis=1),
        np.abs(wf[:,2]-wf[:,0]).sum(axis=1),
    ),axis=1)


def _metrics(rest,posed,faces):
    r=rest[faces]; p=posed[faces]
    re=np.stack((
        np.linalg.norm(r[:,1]-r[:,0],axis=1),
        np.linalg.norm(r[:,2]-r[:,1],axis=1),
        np.linalg.norm(r[:,0]-r[:,2],axis=1),
    ),axis=1)
    pe=np.stack((
        np.linalg.norm(p[:,1]-p[:,0],axis=1),
        np.linalg.norm(p[:,2]-p[:,1],axis=1),
        np.linalg.norm(p[:,0]-p[:,2],axis=1),
    ),axis=1)
    edge=pe/np.maximum(re,1e-12)
    r1=r[:,1]-r[:,0]; r2=r[:,2]-r[:,0]
    l1=np.linalg.norm(r1,axis=1)
    u=r1/np.maximum(l1[:,None],1e-12)
    x2=np.sum(r2*u,axis=1)
    perp=r2-x2[:,None]*u
    y2=np.linalg.norm(perp,axis=1)
    inv=np.zeros((len(faces),2,2),dtype=np.float64)
    inv[:,0,0]=1/np.maximum(l1,1e-12)
    inv[:,0,1]=-x2/np.maximum(l1*y2,1e-12)
    inv[:,1,1]=1/np.maximum(y2,1e-12)
    P=np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=2)
    F=np.einsum("nij,njk->nik",P,inv)
    s=np.linalg.svd(F,compute_uv=False)
    area=s[:,0]*s[:,1]
    cond=s[:,0]/np.maximum(s[:,1],1e-15)
    return area,cond,edge.min(axis=1),edge.max(axis=1)


def _pair(a,b): return tuple(sorted((str(a),str(b))))


def _graph_seam_cut(candidate,unsafe,edge_l1):
    face_edges=[]
    forbidden=set()
    for fi,face in enumerate(candidate.faces):
        a,b,c=face
        es=(_pair(a,b),_pair(b,c),_pair(c,a))
        face_edges.append(es)
        if fi in unsafe:
            k=int(np.argmax(edge_l1[fi]))
            forbidden.add(es[k])
    keep=[fi for fi,es in enumerate(face_edges) if not any(e in forbidden for e in es)]
    faces=tuple(candidate.faces[i] for i in keep)
    edges=sorted({e for fi in keep for e in face_edges[fi]})
    provisional=replace(
        candidate,faces=faces,edges=tuple(edges),candidate_lineage_hash="",
        producer_id="RealSaS.Stage35SkinTopologyGraphSeamCut.v1",
        metadata={
            **dict(candidate.metadata or {}),
            "repair_kind":"GRAPH_EDGE_SEAM_CUT",
            "source_candidate_lineage_hash":candidate.candidate_lineage_hash,
            "forbidden_edge_count":len(forbidden),
            "removed_face_count":len(candidate.faces)-len(faces),
            "weight_mutation":False,
            "vertex_position_mutation":False,
            "demo_visual_validation_only":True,
        },
    )
    repaired=replace(provisional,candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional))
    return repaired,np.asarray(keep,dtype=np.int64),forbidden


def _gif(sheet_path,out_path,duration):
    sheet=Image.open(sheet_path).convert("RGBA")
    cw=sheet.width//4; rh=sheet.height//2; header=28; ch=rh-header
    frames=[]
    for i in range(4):
        a=sheet.crop((i*cw,header,(i+1)*cw,header+ch))
        b=sheet.crop((i*cw,rh+header,(i+1)*cw,rh+header+ch))
        out=Image.new("RGBA",(2*cw,ch),(0,0,0,0));out.alpha_composite(a,(0,0));out.alpha_composite(b,(cw,0))
        th=min(640,ch); tw=max(1,round(out.width*th/out.height))
        if out.size!=(tw,th): out=out.resize((tw,th),Image.Resampling.LANCZOS)
        frames.append(out.convert("P",palette=Image.Palette.ADAPTIVE,colors=255))
    frames[0].save(out_path,save_all=True,append_images=frames[1:],duration=duration,loop=0,disposal=2,optimize=False,transparency=0)


def run(authority_root,run_id,out_dir):
    ctx=_ctx(authority_root,run_id)
    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    skeleton=qualified_skeleton_from_dict(stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    skin=qualified_skin_from_dict(stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))
    camera_set=qualified_camera_set_from_dict(stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    cameras=tuple(sorted(camera_set.cameras,key=lambda c:int(c.view_index)))
    appearance=complete_appearance_asset_from_dict(stage_output_payload(ctx,"23_COMPLETE_APPEARANCE_ASSET_BAKED","RealSaS.CompleteAppearanceAssetIR.v2"))
    uv=load_face_uv(appearance); page=load_face_page_index(appearance); prov=load_provenance_atlas(appearance)
    texture_by_view={int(x.direction_index):x for x in appearance.textures}
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    joint_ids,W=_candidate_skin_weights(candidate,skin,skeleton)
    rest=np.asarray([v.P for v in candidate.vertices],dtype=np.float64)
    faces=_faces(candidate)
    edge_l1=_edge_l1(W,faces)
    face_l1=edge_l1.max(axis=1)
    risky=np.nonzero(face_l1>RISK_L1_MIN)[0]
    max_edge=np.ones(len(candidate.faces)); max_area=np.ones(len(candidate.faces)); min_area=np.ones(len(candidate.faces)); max_cond=np.ones(len(candidate.faces))
    cache={}
    for clip_id,short in CLIPS:
        payload=json.loads((ctx["run_root"]/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json").read_text())
        tracks,mapping=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        cache[clip_id]=(payload,tracks,mapping,times)
        for t in times:
            mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,W,joint_ids,mats)
            a,c,emin,emax=_metrics(rest,posed,faces[risky])
            max_edge[risky]=np.maximum(max_edge[risky],emax);max_area[risky]=np.maximum(max_area[risky],a);min_area[risky]=np.minimum(min_area[risky],a);max_cond[risky]=np.maximum(max_cond[risky],c)
    unsafe=set(map(int,np.nonzero((face_l1>RISK_L1_MIN)&((max_edge>MAX_EDGE_STRETCH)|(max_area>MAX_AREA_RATIO)|(min_area<MIN_AREA_RATIO)|(max_cond>MAX_CONDITION)))[0]))
    repaired,keep,forbidden=_graph_seam_cut(candidate,unsafe,edge_l1)

    # Exact retained-face recheck on the same actual clip samples.
    repaired_faces=_faces(repaired)
    retained_max_edge=1.0;retained_max_area=1.0;retained_max_cond=1.0;retained_min_area=1.0
    for clip_id,short in CLIPS:
        _,tracks,_,times=cache[clip_id]
        for t in times:
            mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,W,joint_ids,mats)
            a,c,emin,emax=_metrics(rest,posed,repaired_faces)
            retained_max_edge=max(retained_max_edge,float(emax.max()));retained_max_area=max(retained_max_area,float(a.max()));retained_max_cond=max(retained_max_cond,float(c.max()));retained_min_area=min(retained_min_area,float(a.min()))

    out_dir.mkdir(parents=True,exist_ok=True)
    report={
        "schema":"RealSaS.KnightGraphSeamCutPreview.v1","status":"DIAGNOSTIC_ONLY",
        "face_count_before":len(candidate.faces),"face_count_after":len(repaired.faces),
        "removed_face_count":len(candidate.faces)-len(repaired.faces),"forbidden_edge_count":len(forbidden),
        "risky_face_count":len(risky),"unsafe_face_count":len(unsafe),
        "thresholds":{"risk_l1_min":RISK_L1_MIN,"max_edge":MAX_EDGE_STRETCH,"max_area":MAX_AREA_RATIO,"min_area":MIN_AREA_RATIO,"max_condition":MAX_CONDITION},
        "retained_actual_clip_metrics":{"max_edge":retained_max_edge,"max_area":retained_max_area,"min_area":retained_min_area,"max_condition":retained_max_cond},
        "source_candidate_lineage_hash":candidate.candidate_lineage_hash,"repaired_candidate_lineage_hash":repaired.candidate_lineage_hash,
        "weight_mutation":False,"vertex_position_mutation":False,"product_authority_claimed":False,
        "clips":[]
    }
    durations={"idle":833,"run":208,"slash":278}
    for clip_id,short in CLIPS:
        _,tracks,mapping,times=cache[clip_id]
        imgs=[]
        for view in (0,2):
            texture=_load_texture_pages(texture_by_view[view])
            for i,t in enumerate(times):
                mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
                posed=_skin(rest,W,joint_ids,mats)
                rr=render_caa_reference(mesh=repaired,camera=cameras[view],face_uv=uv[keep],texture_rgba_u8=texture,provenance_atlas=prov[view],face_page_index=page[keep],positions=posed)
                imgs.append((f"{short} V{view} t={float(t):.2f}",Image.fromarray(np.asarray(rr.straight_rgba_u8,dtype=np.uint8),"RGBA")))
        sheet=_composite_sheet(imgs,columns=4,label=short)
        sp=out_dir/f"KNIGHT_{short.upper()}_GRAPH_SEAM_CUT_PREVIEW_V1.png"; gp=out_dir/f"KNIGHT_{short.upper()}_GRAPH_SEAM_CUT_PREVIEW_V1.gif"
        sheet.save(sp);_gif(sp,gp,durations[short])
        report["clips"].append({"clip_id":clip_id,"sheet_path":str(sp),"gif_path":str(gp),"mapping":mapping})
    (out_dir/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_GRAPH_SEAM_CUT_PREVIEW_PASS",json.dumps({"unsafe_face_count":len(unsafe),"forbidden_edge_count":len(forbidden),"removed_face_count":len(candidate.faces)-len(repaired.faces),"retained":report["retained_actual_clip_metrics"]},sort_keys=True))


def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out-dir",type=Path,required=True);a=p.parse_args();run(a.authority_root,a.run_id,a.out_dir)
if __name__=="__main__":main()
