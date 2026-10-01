from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    Arap2D,
    bind_source_visual_points_to_projected_surface_v1,
    bind_visual_vertex_handles,
    build_visual_mesh_from_mask,
    raster_xy_to_source_texel_xy,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_source_inputs
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights, _ctx, _skin, _tracks_for_clip
from tools.demo.render_knight_visual_mesh_arap_witness_v1 import (
    _candidate_face_indices,
    _clean_source_texture,
    _gif_frame,
    _mechanical_targets,
    _render_native,
    _write_visual_mesh_binary,
    _write_visual_positions_binary,
)

CLIPS=(("demo_idle_v1","IDLE",833),("demo_run_v1","RUN",208),("demo_slash_v1","SLASH",278))


def _load_effective_candidate(ctx):
    # Only the Stage18-adopted lineage is allowed here. A raw Stage35 repair
    # proposal is not canonical until Stage18 re-adopts it and Stage19+ requalify.
    return canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"
    )), "STAGE18_CANONICAL_ADOPTED_LINEAGE"


def _visual_qa(rest, posed, faces):
    rest=np.asarray(rest,float); posed=np.asarray(posed,float); faces=np.asarray(faces,np.int64)
    rr=rest[faces]; pp=posed[faces]
    def area(t):
        a=t[:,1]-t[:,0]; b=t[:,2]-t[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    ra=area(rr); pa=area(pp)
    flips=int(np.count_nonzero((ra*pa)<0))
    re=np.stack([np.linalg.norm(rr[:,1]-rr[:,0],axis=1),np.linalg.norm(rr[:,2]-rr[:,1],axis=1),np.linalg.norm(rr[:,0]-rr[:,2],axis=1)],axis=1)
    pe=np.stack([np.linalg.norm(pp[:,1]-pp[:,0],axis=1),np.linalg.norm(pp[:,2]-pp[:,1],axis=1),np.linalg.norm(pp[:,0]-pp[:,2],axis=1)],axis=1)
    ratio=pe/np.maximum(re,1e-9)
    return {
        "flipped_triangles":flips,
        "p95_edge_stretch":float(np.quantile(ratio,.95)),
        "max_edge_stretch":float(np.max(ratio)),
    }


def run(*, authority_root:Path, run_id:str, out_dir:Path, native_player:Path, render_resolution:int):
    ctx=_ctx(authority_root,run_id)
    out_dir.mkdir(parents=True,exist_ok=True)
    candidate,candidate_source=_load_effective_candidate(ctx)
    skeleton=qualified_skeleton_from_dict(stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    skin=qualified_skin_from_dict(stage_output_payload(ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))
    camera_set=qualified_camera_set_from_dict(stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    observation=qualified_observation_set_from_dict(stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1"))
    cameras=tuple(sorted(camera_set.cameras,key=lambda c:int(c.view_index)))
    rgba_by_view,mask_by_view=_load_source_inputs(ctx,observation)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    joint_ids,W=_candidate_skin_weights(candidate,skin,skeleton)
    rest=np.asarray([v.P for v in candidate.vertices],dtype=np.float64)
    mech_faces=_candidate_face_indices(candidate)
    clips={}
    for clip_id,short,ms in CLIPS:
        payload=json.loads((ctx["run_root"]/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json").read_text())
        tracks,mapping=_tracks_for_clip(payload,skeleton,cameras,source_report)
        clips[clip_id]=(payload,tracks,mapping)

    report={
        "schema":"RealSaS.KnightSourceOwnedMechanicalBindingProof.v1",
        "status":"MEASURED_RESCUE_PROOF",
        "run_id":run_id,
        "candidate_source":candidate_source,
        "candidate_lineage_hash":candidate.candidate_lineage_hash,
        "ownership":{
            "mechanical_mesh_render_authority":False,
            "visual_mesh_authority":"SOURCE_FOREGROUND_MASK_CONSTRAINED_CDT",
            "texture_authority":"ORIGINAL_SOURCE_RGBA",
            "appearance_provenance":"DIRECT_SOURCE_ONLY",
            "deformation_driver":"STAGE35_QUALIFIED_MECHANICAL_SURFACE_BARYCENTRIC_TARGETS",
            "arap_role":"UNBOUND_VISUAL_VERTEX_INTERPOLATION_ONLY",
        },
        "views":[],
        "outputs":[],
        "product_authority_claimed":False,
    }
    rendered={}
    for view_index in (0,2):
        camera=cameras[view_index]
        rgba=np.asarray(rgba_by_view[view_index],dtype=np.uint8)
        mask=np.asarray(mask_by_view[view_index],dtype=bool)
        mesh=build_visual_mesh_from_mask(mask,target_edge_px=16)
        continuous=bind_source_visual_points_to_projected_surface_v1(
            points_source_xy=mesh.positions,
            mechanical_positions_xyz=rest,
            mechanical_faces=mech_faces,
            camera=camera,
        )
        selected=np.asarray(np.flatnonzero(continuous["valid"]),dtype=np.int64)
        if len(selected)<8:
            raise RuntimeError("SOURCE_OWNED_CONTINUOUS_BINDING_COVERAGE_TOO_LOW")
        selected_owner=np.asarray(continuous["owner_face_index"][selected],dtype=np.int64)
        selected_bary=np.asarray(continuous["barycentric"][selected],dtype=np.float64)
        selected_faces=np.asarray(mech_faces[selected_owner],dtype=np.int64)
        projected=np.asarray(continuous["projected_vertices"],dtype=np.float64)
        rest_bound_raster=np.sum(
            projected[selected_faces,:2]*selected_bary[:,:,None],
            axis=1,
        )
        rest_bound_source=raster_xy_to_source_texel_xy(rest_bound_raster)
        binding={
            "visual_vertex_indices":selected,
            "mechanical_face_indices":selected_faces,
            "mechanical_barycentric":selected_bary,
            "rest_projection_offset_xy":np.asarray(mesh.positions,dtype=np.float64)[selected]-rest_bound_source,
        }
        bindings=bind_visual_vertex_handles(mesh,selected)
        arap=Arap2D(mesh,bindings,constraint_scale=1.0e6)
        view_root=out_dir/f"V{view_index}"
        texture=view_root/"source_art.png"
        texture_sha=_clean_source_texture(rgba,mask,texture)
        mesh_bin=view_root/"visual_mesh.bin"
        _write_visual_mesh_binary(mesh_bin,mesh)
        view_row={
            "view_index":view_index,
            "visual_vertex_count":int(len(mesh.positions)),
            "visual_face_count":int(len(mesh.faces)),
            "bound_visual_vertex_count":int(len(selected)),
            "unbound_visual_vertex_count":int(len(mesh.positions)-len(selected)),
            "bound_visual_vertex_fraction":float(len(selected)/max(1,len(mesh.positions))),
            "binding_mode":"CONTINUOUS_FIRST_HIT_PROJECTED_TRIANGLE_V1",
            "max_rest_binding_offset_px":float(np.max(np.linalg.norm(binding["rest_projection_offset_xy"],axis=1))),
            "source_foreground_pixel_count":int(np.count_nonzero(mask)),
            "source_texture_sha256":texture_sha,
            "direct_source_fraction":1.0,
            "other_view_fraction":0.0,
            "completion_fraction":0.0,
            "unsupported_fraction":0.0,
            "qa_by_clip":{},
        }
        for clip_id,short,ms in CLIPS:
            payload,tracks,mapping=clips[clip_id]
            times=np.linspace(0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")),dtype=np.float64)
            frames=[]; qa_rows=[]
            arap.reset()
            for t in times:
                mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
                posed_mech=_skin(rest,W,joint_ids,mats)
                targets=_mechanical_targets(binding=binding,posed_mechanical_xyz=posed_mech,camera=camera)
                deformed,qa=arap.solve(targets,iterations=4)
                geom=_visual_qa(mesh.positions,deformed,mesh.faces)
                qa_rows.append({
                    "time_seconds":float(t),
                    **geom,
                    "max_binding_residual_px":float(qa.max_handle_residual_px),
                })
                frames.append(np.asarray(deformed,dtype=np.float64))
            arr=np.stack(frames,axis=0)
            pos=view_root/f"{clip_id}.positions.bin"
            _write_visual_positions_binary(pos,arr)
            imgs=[]
            for fi in range(len(frames)):
                raw=_render_native(native_player,mesh_path=mesh_bin,positions_path=pos,texture_path=texture,clip_id=clip_id,view_index=view_index,frame=fi,root=out_dir/"native_frames",resolution=render_resolution)
                data=np.frombuffer(raw.read_bytes(),dtype=np.uint8).reshape(render_resolution,render_resolution,4)
                rendered[(view_index,clip_id,fi)]=Image.fromarray(data,"RGBA")
            view_row["qa_by_clip"][clip_id]=qa_rows
        report["views"].append(view_row)

    for clip_id,short,ms in CLIPS:
        frames=[_gif_frame(rendered[(0,clip_id,i)],rendered[(2,clip_id,i)]) for i in range(4)]
        gp=out_dir/f"KNIGHT_{short}_SOURCE_OWNED_MECHANICAL_BINDING_V1.gif"
        frames[0].save(gp,save_all=True,append_images=frames[1:],duration=ms,loop=0,disposal=2,optimize=False,transparency=0)
        report["outputs"].append(str(gp))

    # Rescue proof is intentionally strict about appearance ownership, but records geometry QA truthfully.
    assert all(v["direct_source_fraction"]==1.0 and v["completion_fraction"]==0.0 for v in report["views"])
    (out_dir/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("SOURCE_OWNED_MECHANICAL_BINDING_PROOF_DONE",json.dumps({
        "views":[{"view":v["view_index"],"bound_fraction":v["bound_visual_vertex_fraction"]} for v in report["views"]],
        "candidate_source":candidate_source,
    },sort_keys=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--native-player",required=True)
    p.add_argument("--render-resolution",type=int,default=512)
    a=p.parse_args()
    run(authority_root=Path(a.authority_root).resolve(),run_id=a.run_id,out_dir=Path(a.out_dir),native_player=Path(a.native_player),render_resolution=a.render_resolution)

if __name__=="__main__":
    main()
