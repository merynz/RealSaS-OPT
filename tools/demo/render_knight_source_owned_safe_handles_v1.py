from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    Arap2D,
    bind_source_visual_points_to_projected_surface_v1,
    bind_visual_vertex_handles,
    build_visual_mesh_from_mask,
    raster_xy_to_source_texel_xy,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_source_inputs
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights, _ctx, _skin, _tracks_for_clip
from tools.demo.render_knight_visual_mesh_arap_witness_v1 import (
    _candidate_face_indices, _clean_source_texture, _gif_frame, _mechanical_targets,
    _render_native, _write_visual_mesh_binary, _write_visual_positions_binary,
)

CLIPS=(("demo_idle_v1","IDLE",833),("demo_run_v1","RUN",208),("demo_slash_v1","SLASH",278))


def _read(root:Path, stage:str, name:str):
    path=root/"artifacts"/stage/name
    if not path.is_file():
        raise RuntimeError(f"RESCUE_PARENT_ARTIFACT_MISSING:{path}")
    return json.loads(path.read_text()), path


def _visual_qa(rest, posed, faces):
    rest=np.asarray(rest,float); posed=np.asarray(posed,float); faces=np.asarray(faces,np.int64)
    rr=rest[faces]; pp=posed[faces]
    def signed_area(t):
        a=t[:,1]-t[:,0]; b=t[:,2]-t[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    ra=signed_area(rr); pa=signed_area(pp)
    flips=int(np.count_nonzero((ra*pa)<0))
    re=np.stack((np.linalg.norm(rr[:,1]-rr[:,0],axis=1),np.linalg.norm(rr[:,2]-rr[:,1],axis=1),np.linalg.norm(rr[:,0]-rr[:,2],axis=1)),axis=1)
    pe=np.stack((np.linalg.norm(pp[:,1]-pp[:,0],axis=1),np.linalg.norm(pp[:,2]-pp[:,1],axis=1),np.linalg.norm(pp[:,0]-pp[:,2],axis=1)),axis=1)
    ratio=pe/np.maximum(re,1e-9)
    return {
        "flipped_triangles":flips,
        "p95_edge_stretch":float(np.quantile(ratio,.95)),
        "max_edge_stretch":float(np.max(ratio)),
    }


def run(*, authority_root:Path, parent_run_id:str, out_dir:Path, native_player:Path, render_resolution:int):
    root=authority_root/"runs"/parent_run_id
    ctx=_ctx(authority_root,parent_run_id)
    candidate_raw,candidate_path=_read(root,"18_CANONICAL_MESH_ADDRESSING_BUILD","canonical_mesh_candidate.json")
    policy_raw,policy_path=_read(root,"18_CANONICAL_MESH_ADDRESSING_BUILD","mesh_qualification_policy.json")
    surface_raw,surface_path=_read(root,"15_RIGGING_SURFACE_QUALIFIED","qualified_rigging_surface.json")
    skeleton_raw,skeleton_path=_read(root,"28_SKELETON_QUALIFIED","qualified_skeleton.json")
    skin_raw,skin_path=_read(root,"32_SKIN_QUALIFIED","qualified_skin.json")
    envelope_raw,envelope_path=_read(root,"34_DEFORMATION_CAPABILITY_ENVELOPE","deformation_envelope.json")
    cameras_raw,cameras_path=_read(root,"05_CAMERA_CONTRACT_SOLVED","qualified_camera_set.json")
    obs_raw,obs_path=_read(root,"07_OBSERVATION_CONTRACT_QUALIFIED","qualified_observation_set.json")

    candidate=canonical_mesh_candidate_from_dict(candidate_raw)
    policy=mesh_policy_from_dict(policy_raw)
    surface=rigging_surface_from_dict(surface_raw)
    skeleton=qualified_skeleton_from_dict(skeleton_raw)
    skin=qualified_skin_from_dict(skin_raw)
    envelope=deformation_envelope_from_dict(envelope_raw)
    camera_set=qualified_camera_set_from_dict(cameras_raw)
    observation=qualified_observation_set_from_dict(obs_raw)
    cameras=tuple(sorted(camera_set.cameras,key=lambda c:int(c.view_index)))

    compatibility=run_skin_topology_compatibility_v1(
        candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,
        cameras=cameras,policy=policy,
    )
    unsafe=set(map(int,compatibility["unsafe_face_indices"]))
    joint_ids,W=_candidate_skin_weights(candidate,skin,skeleton)
    rest=np.asarray([v.P for v in candidate.vertices],dtype=np.float64)
    mech_faces=_candidate_face_indices(candidate)
    rgba_by_view,mask_by_view=_load_source_inputs(ctx,observation)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    clips={}
    for clip_id,short,ms in CLIPS:
        payload=json.loads((root/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json").read_text())
        tracks,mapping=_tracks_for_clip(payload,skeleton,cameras,source_report)
        clips[clip_id]=(payload,tracks,mapping)

    import hashlib
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    out_dir.mkdir(parents=True,exist_ok=True)
    report={
        "schema":"RealSaS.KnightSourceOwnedSafeMechanicalHandlesProof.v1",
        "status":"MEASURED_RESCUE_PROOF",
        "parent_run_id":parent_run_id,
        "parent_run_ledger_used_as_authority":False,
        "input_artifacts":[{
            "path":str(p),"sha256":sha(p)
        } for p in (candidate_path,policy_path,surface_path,skeleton_path,skin_path,envelope_path,cameras_path,obs_path)],
        "mechanical_compatibility":{
            "report_hash":compatibility["report_hash"],
            "unsafe_face_count":compatibility["unsafe_face_count"],
            "risky_face_count":compatibility["risky_face_count"],
            "stress_angle_deg":compatibility["stress_angle_deg"],
            "passed_before_driver_filter":compatibility["passed"],
            "unsafe_faces_forbidden_as_visual_drivers":True,
        },
        "ownership":{
            "mechanical_mesh_render_authority":False,
            "visual_mesh_authority":"SOURCE_FOREGROUND_MASK_CONSTRAINED_CDT",
            "texture_authority":"ORIGINAL_SOURCE_RGBA",
            "appearance_provenance":"DIRECT_SOURCE_ONLY",
            "driver_authority":"G3B_SAFE_FIRST_HIT_MECHANICAL_SURFACE_ONLY",
            "unsafe_mechanical_faces":"FORBIDDEN_AS_VISUAL_HANDLES",
            "arap_role":"SOURCE_VISUAL_MESH_REGULARIZATION_AND_UNDRIVEN_VERTEX_INTERPOLATION",
        },
        "views":[],"outputs":[],"product_authority_claimed":False,
    }
    rendered={}
    for vi in (0,2):
        camera=cameras[vi]
        rgba=np.asarray(rgba_by_view[vi],dtype=np.uint8); mask=np.asarray(mask_by_view[vi],dtype=bool)
        mesh=build_visual_mesh_from_mask(mask,target_edge_px=16)
        continuous=bind_source_visual_points_to_projected_surface_v1(
            points_source_xy=mesh.positions,mechanical_positions_xyz=rest,
            mechanical_faces=mech_faces,camera=camera,
        )
        owner=np.asarray(continuous["owner_face_index"],dtype=np.int64)
        valid=np.asarray(continuous["valid"],dtype=bool)
        safe=valid & np.asarray([int(x) not in unsafe if int(x)>=0 else False for x in owner],dtype=bool)
        selected=np.flatnonzero(safe).astype(np.int64)
        if len(selected)<32:
            raise RuntimeError("RESCUE_SAFE_VISUAL_HANDLE_COVERAGE_TOO_LOW")
        selected_owner=owner[selected]
        selected_bary=np.asarray(continuous["barycentric"][selected],dtype=np.float64)
        selected_faces=mech_faces[selected_owner]
        projected=np.asarray(continuous["projected_vertices"],dtype=np.float64)
        rest_bound_raster=np.sum(projected[selected_faces,:2]*selected_bary[:,:,None],axis=1)
        rest_bound_source=raster_xy_to_source_texel_xy(rest_bound_raster)
        binding={
            "visual_vertex_indices":selected,
            "mechanical_face_indices":selected_faces,
            "mechanical_barycentric":selected_bary,
            "rest_projection_offset_xy":np.asarray(mesh.positions,dtype=np.float64)[selected]-rest_bound_source,
        }
        bindings=bind_visual_vertex_handles(mesh,selected)
        arap=Arap2D(mesh,bindings,constraint_scale=1.0e3)
        view_root=out_dir/f"V{vi}"; texture=view_root/"source_art.png"
        texture_sha=_clean_source_texture(rgba,mask,texture)
        mesh_bin=view_root/"visual_mesh.bin"; _write_visual_mesh_binary(mesh_bin,mesh)
        row={
            "view_index":vi,"visual_vertex_count":len(mesh.positions),"visual_face_count":len(mesh.faces),
            "continuous_bound_count":int(valid.sum()),"continuous_bound_fraction":float(valid.mean()),
            "safe_handle_count":len(selected),"safe_handle_fraction":float(len(selected)/len(mesh.positions)),
            "unsafe_bound_vertex_count":int(np.count_nonzero(valid & ~safe)),
            "max_rest_binding_offset_px":float(np.max(np.linalg.norm(binding["rest_projection_offset_xy"],axis=1))),
            "source_texture_sha256":texture_sha,
            "direct_source_fraction":1.0,"other_view_fraction":0.0,"completion_fraction":0.0,"unsupported_fraction":0.0,
            "qa_by_clip":{},
        }
        for clip_id,short,ms in CLIPS:
            payload,tracks,mapping=clips[clip_id]
            times=np.linspace(0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")),dtype=np.float64)
            arap.reset(); frames=[]; qa_rows=[]
            for t in times:
                mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
                posed=_skin(rest,W,joint_ids,mats)
                targets=_mechanical_targets(binding=binding,posed_mechanical_xyz=posed,camera=camera)
                deformed,qa=arap.solve(targets,iterations=4)
                geom=_visual_qa(mesh.positions,deformed,mesh.faces)
                qa_rows.append({"time_seconds":float(t),**geom,"max_handle_residual_px":float(qa.max_handle_residual_px)})
                frames.append(np.asarray(deformed,dtype=np.float64))
            pos=view_root/f"{clip_id}.positions.bin"; _write_visual_positions_binary(pos,np.stack(frames))
            for fi in range(len(frames)):
                raw=_render_native(native_player,mesh_path=mesh_bin,positions_path=pos,texture_path=texture,clip_id=clip_id,view_index=vi,frame=fi,root=out_dir/"native_frames",resolution=render_resolution)
                data=np.frombuffer(raw.read_bytes(),dtype=np.uint8).reshape(render_resolution,render_resolution,4)
                rendered[(vi,clip_id,fi)]=Image.fromarray(data,"RGBA")
            row["qa_by_clip"][clip_id]=qa_rows
        report["views"].append(row)

    for clip_id,short,ms in CLIPS:
        frames=[_gif_frame(rendered[(0,clip_id,i)],rendered[(2,clip_id,i)]) for i in range(4)]
        gp=out_dir/f"KNIGHT_{short}_SOURCE_OWNED_SAFE_HANDLES_V1.gif"
        frames[0].save(gp,save_all=True,append_images=frames[1:],duration=ms,loop=0,disposal=2,optimize=False,transparency=0)
        report["outputs"].append(str(gp))
    (out_dir/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("SAFE_HANDLE_PROOF_DONE",json.dumps({
        "unsafe_faces":compatibility["unsafe_face_count"],
        "views":[{"view":v["view_index"],"continuous":v["continuous_bound_fraction"],"safe_handles":v["safe_handle_fraction"]} for v in report["views"]],
    },sort_keys=True))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--authority-root",required=True); p.add_argument("--parent-run-id",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--native-player",required=True); p.add_argument("--render-resolution",type=int,default=512)
    a=p.parse_args(); run(authority_root=Path(a.authority_root).resolve(),parent_run_id=a.parent_run_id,out_dir=Path(a.out_dir),native_player=Path(a.native_player),render_resolution=a.render_resolution)
if __name__=="__main__": main()
