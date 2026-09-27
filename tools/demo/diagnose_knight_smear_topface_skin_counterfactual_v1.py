from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_page_index,
    load_face_uv,
    load_provenance_atlas,
    render_caa_reference,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_texture_pages
from tools.demo.diagnose_knight_smear_face_causality_v1 import _counts, _metric_rows, _submesh
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
    _skin,
    _tracks_for_clip,
)

CASES = (
    ("idle_v2_f0", "demo_idle_v1", 2, 0),
    ("run_v2_f0", "demo_run_v1", 2, 0),
    ("slash_v0_f0", "demo_slash_v1", 0, 0),
)


def _rgba(render):
    return np.asarray(render.straight_rgba_u8, dtype=np.uint8)


def _face_geom(rest, posed, idx):
    rp = rest[idx]
    pp = posed[idx]
    re = np.asarray([
        np.linalg.norm(rp[1]-rp[0]),
        np.linalg.norm(rp[2]-rp[1]),
        np.linalg.norm(rp[0]-rp[2]),
    ])
    pe = np.asarray([
        np.linalg.norm(pp[1]-pp[0]),
        np.linalg.norm(pp[2]-pp[1]),
        np.linalg.norm(pp[0]-pp[2]),
    ])
    ra = float(0.5*np.linalg.norm(np.cross(rp[1]-rp[0], rp[2]-rp[0])))
    pa = float(0.5*np.linalg.norm(np.cross(pp[1]-pp[0], pp[2]-pp[0])))
    return {
        "max_edge_stretch": float(np.max(pe/np.maximum(re,1e-12))),
        "area_ratio_3d": float(pa/max(ra,1e-18)),
        "max_vertex_displacement": float(np.max(np.linalg.norm(pp-rp,axis=1))),
    }


def _top_weights(joint_ids, row, n=6):
    order=np.argsort(-row)
    return [
        {"joint_id": str(joint_ids[int(i)]), "weight": float(row[int(i)])}
        for i in order[:n] if float(row[int(i)]) > 1e-8
    ]


def run(authority_root: Path, run_id: str, out_dir: Path):
    ctx=_ctx(authority_root,run_id)
    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"
    ))
    skeleton=qualified_skeleton_from_dict(stage_output_payload(
        ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"
    ))
    skin=qualified_skin_from_dict(stage_output_payload(
        ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"
    ))
    camera_set=qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"
    ))
    cameras=tuple(sorted(camera_set.cameras,key=lambda c:int(c.view_index)))
    appearance=complete_appearance_asset_from_dict(stage_output_payload(
        ctx,"23_COMPLETE_APPEARANCE_ASSET_BAKED","RealSaS.CompleteAppearanceAssetIR.v2"
    ))
    face_uv=load_face_uv(appearance)
    face_page=load_face_page_index(appearance)
    provenance=load_provenance_atlas(appearance)
    texture_by_view={int(row.direction_index):row for row in appearance.textures}
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    joint_ids,W=_candidate_skin_weights(candidate,skin,skeleton)
    rest=np.asarray([v.P for v in candidate.vertices],dtype=np.float64)
    vid_to_index={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    out_dir.mkdir(parents=True,exist_ok=True)

    report={
        "schema":"RealSaS.KnightSmearTopFaceSkinCounterfactual.v1",
        "status":"DIAGNOSTIC_ONLY",
        "run_id":run_id,
        "candidate_lineage_hash":candidate.candidate_lineage_hash,
        "cases":[],
        "repair_attempted":False,
        "product_authority_claimed":False,
    }

    for label,clip_id,view,frame_index in CASES:
        payload=json.loads((
            ctx["run_root"]/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json"
        ).read_text())
        tracks,_=_tracks_for_clip(payload,skeleton,cameras,source_report)
        sample_times=np.linspace(
            0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop"))
        )
        t=float(sample_times[frame_index])
        skin_mats,_,_=_joint_pose_v2(
            skeleton=skeleton,tracks=tracks,time_seconds=t,cameras=cameras
        )
        posed=_skin(rest,W,joint_ids,skin_mats)
        texture=_load_texture_pages(texture_by_view[view])

        full=render_caa_reference(
            mesh=candidate,camera=cameras[view],face_uv=face_uv,
            texture_rgba_u8=texture,provenance_atlas=provenance[view],
            face_page_index=face_page,positions=posed,
        )
        rest_render=render_caa_reference(
            mesh=candidate,camera=cameras[view],face_uv=face_uv,
            texture_rgba_u8=texture,provenance_atlas=provenance[view],
            face_page_index=face_page,positions=rest,
        )
        rows=_metric_rows(candidate,rest,posed,W)
        dyn=_counts(full,len(candidate.faces))
        rst=_counts(rest_render,len(candidate.faces))
        for row in rows:
            fi=row["face_index"]
            row["dynamic_contribution_samples"]=int(dyn[fi])
            row["rest_contribution_samples"]=int(rst[fi])
            row["growth"]=int(dyn[fi]-rst[fi])
        offender=max(rows,key=lambda r:(r["growth"],r["dynamic_contribution_samples"]))
        fi=int(offender["face_index"])
        face=candidate.faces[fi]
        idx=np.asarray([vid_to_index[str(v)] for v in face],dtype=np.int64)

        Wcf=W.copy()
        mean_w=np.mean(Wcf[idx],axis=0)
        mean_w/=max(float(mean_w.sum()),1e-12)
        Wcf[idx]=mean_w[None,:]
        posed_cf=_skin(rest,Wcf,joint_ids,skin_mats)

        one=_submesh(candidate,[fi])
        original_face=render_caa_reference(
            mesh=one,camera=cameras[view],face_uv=face_uv[[fi]],
            texture_rgba_u8=texture,provenance_atlas=provenance[view],
            face_page_index=face_page[[fi]],positions=posed,
        )
        homogenized_face=render_caa_reference(
            mesh=one,camera=cameras[view],face_uv=face_uv[[fi]],
            texture_rgba_u8=texture,provenance_atlas=provenance[view],
            face_page_index=face_page[[fi]],positions=posed_cf,
        )
        keep=[i for i in range(len(candidate.faces)) if i!=fi]
        minus=render_caa_reference(
            mesh=_submesh(candidate,keep),camera=cameras[view],face_uv=face_uv[keep],
            texture_rgba_u8=texture,provenance_atlas=provenance[view],
            face_page_index=face_page[keep],positions=posed,
        )

        images=[
            ("FULL",Image.fromarray(_rgba(full),"RGBA")),
            ("TOP_FACE_ORIGINAL",Image.fromarray(_rgba(original_face),"RGBA")),
            ("TOP_FACE_HOMOGENIZED_SKIN",Image.fromarray(_rgba(homogenized_face),"RGBA")),
            ("FULL_MINUS_TOP_FACE",Image.fromarray(_rgba(minus),"RGBA")),
        ]
        w,h=images[0][1].size
        panel=Image.new("RGBA",(w*4,h+30),(0,0,0,0))
        draw=ImageDraw.Draw(panel)
        for i,(name,img) in enumerate(images):
            panel.alpha_composite(img,(i*w,30))
            draw.text((i*w+6,7),name,fill=(255,255,255,255))
        panel_path=out_dir/f"{label}_topface_skin_counterfactual.png"
        panel.save(panel_path)

        vertex_details=[]
        for local_i,vi in enumerate(idx):
            vertex=candidate.vertices[int(vi)]
            vertex_details.append({
                "vertex_id":str(vertex.candidate_vertex_id),
                "original_top_weights":_top_weights(joint_ids,W[int(vi)]),
                "support_binding": [
                    [str(sid),float(coeff)]
                    for sid,coeff in vertex.support_binding.coefficients
                ],
            })

        original_geom=_face_geom(rest,posed,idx)
        cf_geom=_face_geom(rest,posed_cf,idx)
        full_rgba=_rgba(full)
        minus_rgba=_rgba(minus)
        report["cases"].append({
            "label":label,
            "clip_id":clip_id,
            "view_index":view,
            "frame_index":frame_index,
            "time_seconds":t,
            "offender_face_index":fi,
            "offender_vertex_ids":list(map(str,face)),
            "rest_min_angle_deg":float(offender["min_angle_deg"]),
            "rest_aspect":float(offender["aspect"]),
            "policy_violating":bool(offender["policy_violating"]),
            "skin_weight_pair_l1_max":float(offender["skin_weight_pair_l1_max"]),
            "dynamic_contribution_samples":int(offender["dynamic_contribution_samples"]),
            "rest_contribution_samples":int(offender["rest_contribution_samples"]),
            "contribution_growth_samples":int(offender["growth"]),
            "original_geometry":original_geom,
            "homogenized_skin_geometry":cf_geom,
            "original_face_alpha_pixels":int(np.count_nonzero(original_face.final_alpha)),
            "homogenized_face_alpha_pixels":int(np.count_nonzero(homogenized_face.final_alpha)),
            "full_minus_top_face_changed_pixels":int(np.count_nonzero(np.any(full_rgba!=minus_rgba,axis=2))),
            "vertex_details":vertex_details,
            "homogenized_top_weights":_top_weights(joint_ids,mean_w),
            "panel_path":str(panel_path),
        })

    (out_dir/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_SMEAR_TOPFACE_SKIN_COUNTERFACTUAL_PASS",json.dumps({
        "cases":len(report["cases"]),"repair_attempted":False
    },sort_keys=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    run(a.authority_root,a.run_id,a.out_dir)


if __name__=="__main__":
    main()
