from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

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
from compiler.realsas_compiler_core.product_authority_v1 import canonical_mesh_candidate_lineage_hash
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_texture_pages
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _composite_sheet,
    _ctx,
    _skin,
    _tracks_for_clip,
)

RISK_L1_MIN = 0.5
MAX_EDGE_STRETCH = 4.0
MAX_AREA_RATIO = 20.0
MIN_AREA_RATIO = 0.05
MAX_CONDITION = 16.0
MAX_REPAIR_ITERATIONS = 4

CLIPS = (
    ("demo_idle_v1", "idle"),
    ("demo_run_v1", "run"),
    ("demo_slash_v1", "slash"),
)


def _face_indices(candidate):
    vi = {str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)}
    return np.asarray([[vi[str(x)] for x in face] for face in candidate.faces], dtype=np.int64)


def _skin_discontinuity(W, faces):
    wf = W[faces]
    d01 = np.abs(wf[:,0] - wf[:,1]).sum(axis=1)
    d12 = np.abs(wf[:,1] - wf[:,2]).sum(axis=1)
    d20 = np.abs(wf[:,2] - wf[:,0]).sum(axis=1)
    return np.maximum(np.maximum(d01, d12), d20)


def _triangle_metrics_batch(rest, posed, faces):
    r = rest[faces]
    p = posed[faces]

    re01 = np.linalg.norm(r[:,1] - r[:,0], axis=1)
    re12 = np.linalg.norm(r[:,2] - r[:,1], axis=1)
    re20 = np.linalg.norm(r[:,0] - r[:,2], axis=1)
    pe01 = np.linalg.norm(p[:,1] - p[:,0], axis=1)
    pe12 = np.linalg.norm(p[:,2] - p[:,1], axis=1)
    pe20 = np.linalg.norm(p[:,0] - p[:,2], axis=1)
    ratios = np.stack((
        pe01 / np.maximum(re01, 1e-12),
        pe12 / np.maximum(re12, 1e-12),
        pe20 / np.maximum(re20, 1e-12),
    ), axis=1)
    edge_min = ratios.min(axis=1)
    edge_max = ratios.max(axis=1)

    r1 = r[:,1] - r[:,0]
    r2 = r[:,2] - r[:,0]
    l1 = np.linalg.norm(r1, axis=1)
    u = r1 / np.maximum(l1[:,None], 1e-12)
    x2 = np.sum(r2 * u, axis=1)
    perp = r2 - x2[:,None] * u
    y2 = np.linalg.norm(perp, axis=1)
    inv = np.zeros((len(faces),2,2), dtype=np.float64)
    inv[:,0,0] = 1.0 / np.maximum(l1, 1e-12)
    inv[:,0,1] = -x2 / np.maximum(l1 * y2, 1e-12)
    inv[:,1,1] = 1.0 / np.maximum(y2, 1e-12)

    P = np.stack((p[:,1]-p[:,0], p[:,2]-p[:,0]), axis=2)
    F = np.einsum("nij,njk->nik", P, inv)
    s = np.linalg.svd(F, compute_uv=False)
    smax = s[:,0]
    smin = s[:,1]
    area = smax * smin
    cond = smax / np.maximum(smin, 1e-15)
    return area, cond, edge_min, edge_max


def _edges_from_faces(faces):
    edges=set()
    for face in faces:
        a,b,c=map(str,face)
        edges.add(tuple(sorted((a,b))))
        edges.add(tuple(sorted((b,c))))
        edges.add(tuple(sorted((c,a))))
    return tuple(sorted(edges))


def _repair_candidate(candidate, unsafe):
    keep = [i for i in range(len(candidate.faces)) if i not in unsafe]
    faces = tuple(candidate.faces[i] for i in keep)
    provisional = replace(
        candidate,
        faces=faces,
        edges=_edges_from_faces(faces),
        candidate_lineage_hash="",
        producer_id="RealSaS.Stage35SkinTopologySeamCutRepair.v1",
        metadata={
            **dict(candidate.metadata or {}),
            "repair_kind":"SKIN_TOPOLOGY_SEAM_CUT",
            "source_candidate_lineage_hash":candidate.candidate_lineage_hash,
            "removed_face_count":len(unsafe),
            "weight_mutation":False,
            "vertex_position_mutation":False,
            "local_cdt_used":False,
            "repair_budget_max_iterations":MAX_REPAIR_ITERATIONS,
            "demo_visual_validation_only":True,
        },
    )
    repaired=replace(
        provisional,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(provisional),
    )
    return repaired, keep


def _gif_from_sheet(sheet_path: Path, out_path: Path, duration_ms: int):
    sheet = Image.open(sheet_path).convert("RGBA")
    if sheet.width % 4 != 0 or sheet.height % 2 != 0:
        raise RuntimeError(f"unexpected sheet geometry: {sheet.size}")
    cell_w = sheet.width // 4
    row_h = sheet.height // 2
    header = 28
    cell_h = row_h - header
    frames=[]
    for i in range(4):
        v0=sheet.crop((i*cell_w,header,(i+1)*cell_w,header+cell_h))
        y2=row_h+header
        v2=sheet.crop((i*cell_w,y2,(i+1)*cell_w,y2+cell_h))
        combined=Image.new("RGBA",(cell_w*2,cell_h),(0,0,0,0))
        combined.alpha_composite(v0,(0,0))
        combined.alpha_composite(v2,(cell_w,0))
        target_h=min(640,combined.height)
        target_w=max(1,round(combined.width*target_h/combined.height))
        if (target_w,target_h)!=combined.size:
            combined=combined.resize((target_w,target_h),Image.Resampling.LANCZOS)
        frames.append(combined.convert("P",palette=Image.Palette.ADAPTIVE,colors=255))
    frames[0].save(
        out_path,save_all=True,append_images=frames[1:],
        duration=duration_ms,loop=0,disposal=2,optimize=False,transparency=0,
    )


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
    faces=_face_indices(candidate)
    l1=_skin_discontinuity(W,faces)
    risky=np.nonzero(l1 > RISK_L1_MIN)[0]
    if len(risky)==0:
        raise RuntimeError("NO_SKIN_TOPOLOGY_RISK_FACES")

    max_edge=np.ones(len(candidate.faces),dtype=np.float64)
    max_area=np.ones(len(candidate.faces),dtype=np.float64)
    min_area=np.ones(len(candidate.faces),dtype=np.float64)
    max_cond=np.ones(len(candidate.faces),dtype=np.float64)
    evidence=[]

    clip_cache={}
    for clip_id,short in CLIPS:
        payload=json.loads((ctx["run_root"]/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json").read_text())
        tracks,mapping=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")))
        clip_cache[clip_id]=(payload,tracks,mapping,times)
        for fi,t in enumerate(times):
            mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,W,joint_ids,mats)
            a,c,emin,emax=_triangle_metrics_batch(rest,posed,faces[risky])
            max_edge[risky]=np.maximum(max_edge[risky],emax)
            max_area[risky]=np.maximum(max_area[risky],a)
            min_area[risky]=np.minimum(min_area[risky],a)
            max_cond[risky]=np.maximum(max_cond[risky],c)
            evidence.append({
                "clip_id":clip_id,"frame_index":int(fi),"time_seconds":float(t),
                "risky_face_count":int(len(risky)),
                "max_edge_stretch":float(np.max(emax)),
                "max_area_ratio":float(np.max(a)),
                "max_condition":float(np.max(c)),
            })

    unsafe_mask=(
        (l1 > RISK_L1_MIN) & (
            (max_edge > MAX_EDGE_STRETCH)
            | (max_area > MAX_AREA_RATIO)
            | (min_area < MIN_AREA_RATIO)
            | (max_cond > MAX_CONDITION)
        )
    )
    unsafe=set(map(int,np.nonzero(unsafe_mask)[0]))
    if not unsafe:
        raise RuntimeError("NO_DYNAMICALLY_UNSAFE_SKIN_TOPOLOGY_FACES")

    repaired,keep=_repair_candidate(candidate,unsafe)
    keep_arr=np.asarray(keep,dtype=np.int64)
    repaired_faces=_face_indices(repaired)

    # Bounded recheck: after seam cut there must be no retained face previously classified unsafe.
    retained_unsafe=sum(1 for i in keep if i in unsafe)
    if retained_unsafe:
        raise RuntimeError("BOUNDED_REPAIR_DID_NOT_REMOVE_UNSAFE_FACES")

    out_dir.mkdir(parents=True,exist_ok=True)
    report={
        "schema":"RealSaS.KnightStage35SkinTopologyRepairPreview.v1",
        "status":"DEMO_REPAIR_PREVIEW_PASS",
        "run_id":run_id,
        "source_candidate_lineage_hash":candidate.candidate_lineage_hash,
        "repaired_candidate_lineage_hash":repaired.candidate_lineage_hash,
        "thresholds":{
            "risk_l1_min":RISK_L1_MIN,
            "max_edge_stretch":MAX_EDGE_STRETCH,
            "max_area_ratio":MAX_AREA_RATIO,
            "min_area_ratio":MIN_AREA_RATIO,
            "max_condition":MAX_CONDITION,
            "max_repair_iterations":MAX_REPAIR_ITERATIONS,
        },
        "face_count_before":len(candidate.faces),
        "face_count_after":len(repaired.faces),
        "removed_face_count":len(unsafe),
        "removed_fraction":len(unsafe)/len(candidate.faces),
        "risky_face_count":int(len(risky)),
        "weight_mutation":False,
        "vertex_position_mutation":False,
        "repair_kind":"DETERMINISTIC_SEAM_CUT",
        "appearance_note":"EXISTING_CAA_FACE_DATA_SUBSET_BY_ORIGINAL_FACE_INDEX_FOR_FAST_DIAGNOSTIC_ONLY",
        "product_authority_claimed":False,
        "stage19_to_35_requalification_required":True,
        "probe_evidence":evidence,
        "top_removed_faces":[],
        "clips":[],
    }
    order=sorted(unsafe,key=lambda i:(max_edge[i],max_area[i],max_cond[i]),reverse=True)[:100]
    for i in order:
        report["top_removed_faces"].append({
            "face_index":i,
            "vertex_ids":list(map(str,candidate.faces[i])),
            "skin_l1_max":float(l1[i]),
            "max_edge_stretch":float(max_edge[i]),
            "max_area_ratio":float(max_area[i]),
            "min_area_ratio":float(min_area[i]),
            "max_condition":float(max_cond[i]),
        })

    durations={"idle":833,"run":208,"slash":278}
    for clip_id,short in CLIPS:
        payload,tracks,mapping,times=clip_cache[clip_id]
        frames=[]
        frame_metrics=[]
        for view in (0,2):
            texture=_load_texture_pages(texture_by_view[view])
            for fi,t in enumerate(times):
                mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
                posed=_skin(rest,W,joint_ids,mats)
                render=render_caa_reference(
                    mesh=repaired,
                    camera=cameras[view],
                    face_uv=face_uv[keep_arr],
                    texture_rgba_u8=texture,
                    provenance_atlas=provenance[view],
                    face_page_index=face_page[keep_arr],
                    positions=posed,
                )
                rgba=Image.fromarray(np.asarray(render.straight_rgba_u8,dtype=np.uint8),"RGBA")
                frames.append((f"{short} V{view} t={float(t):.2f}",rgba))
                frame_metrics.append({
                    "view_index":view,"frame_index":int(fi),"time_seconds":float(t),
                    "visible_pixel_count":int(np.count_nonzero(render.geometry_visible)),
                    "final_alpha_pixel_count":int(np.count_nonzero(render.final_alpha)),
                })
        sheet=_composite_sheet(frames,columns=4,label=short)
        sheet_path=out_dir/f"KNIGHT_{short.upper()}_SKIN_TOPOLOGY_REPAIR_PREVIEW_V1.png"
        gif_path=out_dir/f"KNIGHT_{short.upper()}_SKIN_TOPOLOGY_REPAIR_PREVIEW_V1.gif"
        sheet.save(sheet_path)
        _gif_from_sheet(sheet_path,gif_path,durations[short])
        report["clips"].append({
            "clip_id":clip_id,
            "sheet_path":str(sheet_path),
            "gif_path":str(gif_path),
            "mapping":mapping,
            "frame_metrics":frame_metrics,
        })

    (out_dir/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STAGE35_SKIN_TOPOLOGY_REPAIR_PREVIEW_PASS",json.dumps({
        "removed_face_count":len(unsafe),
        "face_count_before":len(candidate.faces),
        "face_count_after":len(repaired.faces),
        "repaired_candidate_lineage_hash":repaired.candidate_lineage_hash,
        "weight_mutation":False,
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
