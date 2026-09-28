from __future__ import annotations

import argparse, json, hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

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
    VisualMesh2D,
    build_visual_mesh_from_mask,
    build_visual_mesh_from_region_labels_v1,
    partition_source_mask_by_safe_face_adjacency_v1,
    raster_xy_to_source_texel_xy,
)
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_source_inputs
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights, _ctx, _skin, _tracks_for_clip
from tools.demo.render_knight_visual_mesh_arap_witness_v1 import (
    _candidate_face_indices, _clean_source_texture, _gif_frame, _mechanical_targets,
    _render_native, _write_visual_mesh_binary, _write_visual_positions_binary,
)

CLIPS=(("demo_idle_v1","IDLE",833),("demo_run_v1","RUN",208),("demo_slash_v1","SLASH",278))


def _read(root:Path,stage:str,name:str):
    p=root/"artifacts"/stage/name
    if not p.is_file(): raise RuntimeError(f"RESCUE_PARENT_ARTIFACT_MISSING:{p}")
    return json.loads(p.read_text()),p


def _safe_face_components(faces:np.ndarray, unsafe:set[int]):
    edge_faces=defaultdict(list)
    for fi,(a,b,c) in enumerate(np.asarray(faces,dtype=np.int64).tolist()):
        for u,v in ((a,b),(b,c),(c,a)):
            edge_faces[(min(u,v),max(u,v))].append(fi)
    adj=[[] for _ in range(len(faces))]
    for rows in edge_faces.values():
        if len(rows)==2:
            a,b=rows
            if a not in unsafe and b not in unsafe:
                adj[a].append(b); adj[b].append(a)
    labels=np.full(len(faces),-1,dtype=np.int32)
    sizes=[]; cid=0
    for fi in range(len(faces)):
        if fi in unsafe or labels[fi]>=0: continue
        q=[fi]; labels[fi]=cid; n=0
        for x in q:
            n+=1
            for y in adj[x]:
                if labels[y]<0:
                    labels[y]=cid; q.append(y)
        sizes.append(n); cid+=1
    return labels,np.asarray(sizes,dtype=np.int64)


def _partition_source_mask(mask,owner,face_component,min_seed_pixels=256):
    mask=np.asarray(mask,dtype=bool)
    owner=np.asarray(owner,dtype=np.int64)
    pixel_label=np.full(mask.shape,-1,dtype=np.int32)
    valid=mask & (owner>=0)
    idx=owner[valid]
    labs=face_component[idx]
    pixel_label[valid]=labs
    vals,counts=np.unique(pixel_label[(pixel_label>=0)&mask],return_counts=True)
    major=[int(v) for v,c in zip(vals.tolist(),counts.tolist()) if int(c)>=int(min_seed_pixels)]
    if not major:
        raise RuntimeError("VISUAL_REGION_NO_MAJOR_SAFE_COMPONENT")
    known=mask & np.isin(pixel_label,np.asarray(major,dtype=np.int32))
    if not np.any(known):
        raise RuntimeError("VISUAL_REGION_NO_SAFE_SEEDS")
    _dist,indices=ndimage.distance_transform_edt(~known,return_indices=True)
    ny,nx=indices
    final=np.full(mask.shape,-1,dtype=np.int32)
    final[mask]=pixel_label[ny[mask],nx[mask]]
    if np.any(final[mask]<0):
        raise RuntimeError("VISUAL_REGION_FILL_INCOMPLETE")
    regions=[]
    for rid in sorted(set(final[mask].tolist())):
        final_mask=mask & (final==rid)
        seed_mask=mask & (pixel_label==rid)
        if not np.any(seed_mask):
            raise RuntimeError(f"VISUAL_REGION_SEED_EMPTY:{rid}")
        regions.append((int(rid),final_mask,seed_mask))
    return final,pixel_label,regions


def _combine_region_meshes(regions,width,height,target_edge_px=16):
    positions=[];faces=[];uv=[];vertex_region=[];face_region=[];offset=0
    rows=[]
    for rid,region_mask,seed_mask in regions:
        mesh=build_visual_mesh_from_mask(region_mask,target_edge_px=target_edge_px)
        positions.append(np.asarray(mesh.positions,dtype=np.float64))
        faces.append(np.asarray(mesh.faces,dtype=np.uint32)+offset)
        uv.append(np.asarray(mesh.uv,dtype=np.float64))
        vertex_region.extend([rid]*len(mesh.positions))
        face_region.extend([rid]*len(mesh.faces))
        rows.append({
            "region_id":rid,
            "pixel_count":int(np.count_nonzero(region_mask)),
            "seed_pixel_count":int(np.count_nonzero(seed_mask)),
            "vertex_count":int(len(mesh.positions)),
            "face_count":int(len(mesh.faces)),
        })
        offset+=len(mesh.positions)
    out=VisualMesh2D(
        positions=np.concatenate(positions,axis=0),
        faces=np.concatenate(faces,axis=0).astype(np.uint32),
        uv=np.concatenate(uv,axis=0),
        width=int(width),height=int(height),
    )
    return out,np.asarray(vertex_region,dtype=np.int32),np.asarray(face_region,dtype=np.int32),rows


def _region_bindings(mesh,vertex_region,regions,visibility,mech_faces,seed_labels):
    pos=np.asarray(mesh.positions,dtype=np.float64)
    owner=np.asarray(visibility.owner_face_index,dtype=np.int64)
    bary=np.asarray(visibility.barycentric,dtype=np.float64)
    projected=np.asarray(visibility.projected_vertices,dtype=np.float64)
    out_faces=np.empty((len(pos),3),dtype=np.int64)
    out_bary=np.empty((len(pos),3),dtype=np.float64)
    out_offset=np.empty((len(pos),2),dtype=np.float64)
    seed_distance=np.empty((len(pos),),dtype=np.float64)

    region_map={rid:(rm,sm) for rid,rm,sm in regions}
    for rid in sorted(region_map):
        ids=np.flatnonzero(vertex_region==rid)
        _rm,seed_mask=region_map[rid]
        ys,xs=np.nonzero(seed_mask)
        seed_owner=owner[ys,xs]
        good=(seed_owner>=0)
        good &= (seed_labels[ys,xs] == int(rid))
        good &= np.isfinite(bary[ys,xs]).all(axis=1)
        ys=ys[good]; xs=xs[good]; seed_owner=seed_owner[good]
        if len(xs)<3:
            raise RuntimeError(f"VISUAL_REGION_BIND_SEEDS_TOO_LOW:{rid}:{len(xs)}")
        seed_xy=np.column_stack((xs.astype(np.float64),ys.astype(np.float64)))
        tree=cKDTree(seed_xy)
        dist,nn=tree.query(pos[ids],k=1)
        sy=ys[np.asarray(nn,dtype=np.int64)]
        sx=xs[np.asarray(nn,dtype=np.int64)]
        own=owner[sy,sx]
        bb=bary[sy,sx]
        mf=mech_faces[own]
        rest_bound_raster=np.sum(projected[mf,:2]*bb[:,:,None],axis=1)
        rest_bound_source=raster_xy_to_source_texel_xy(rest_bound_raster)
        out_faces[ids]=mf
        out_bary[ids]=bb
        out_offset[ids]=pos[ids]-rest_bound_source
        seed_distance[ids]=np.asarray(dist,dtype=np.float64)
    return {
        "visual_vertex_indices":np.arange(len(pos),dtype=np.int64),
        "mechanical_face_indices":out_faces,
        "mechanical_barycentric":out_bary,
        "rest_projection_offset_xy":out_offset,
        "nearest_safe_seed_distance_px":seed_distance,
    }


def _visual_qa(rest,posed,faces):
    rest=np.asarray(rest,float);posed=np.asarray(posed,float);faces=np.asarray(faces,np.int64)
    rr=rest[faces];pp=posed[faces]
    def area(t):
        a=t[:,1]-t[:,0];b=t[:,2]-t[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    ra=area(rr);pa=area(pp)
    flips=int(np.count_nonzero(ra*pa<0))
    re=np.stack((np.linalg.norm(rr[:,1]-rr[:,0],axis=1),np.linalg.norm(rr[:,2]-rr[:,1],axis=1),np.linalg.norm(rr[:,0]-rr[:,2],axis=1)),axis=1)
    pe=np.stack((np.linalg.norm(pp[:,1]-pp[:,0],axis=1),np.linalg.norm(pp[:,2]-pp[:,1],axis=1),np.linalg.norm(pp[:,0]-pp[:,2],axis=1)),axis=1)
    ratio=pe/np.maximum(re,1e-9)
    return {
        "flipped_triangles":flips,
        "p95_edge_stretch":float(np.quantile(ratio,.95)),
        "p99_edge_stretch":float(np.quantile(ratio,.99)),
        "max_edge_stretch":float(np.max(ratio)),
    }


def run(*,authority_root:Path,parent_run_id:str,out_dir:Path,native_player:Path,render_resolution:int):
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

    compatibility=run_skin_topology_compatibility_v1(candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,cameras=cameras,policy=policy)
    unsafe=set(map(int,compatibility["unsafe_face_indices"]))
    joint_ids,W=_candidate_skin_weights(candidate,skin,skeleton)
    rest=np.asarray([v.P for v in candidate.vertices],dtype=np.float64)
    mech_faces=_candidate_face_indices(candidate)
    face_component,component_sizes=_safe_face_components(mech_faces,unsafe)
    rgba_by_view,mask_by_view=_load_source_inputs(ctx,observation)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    clips={}
    for clip_id,short,ms in CLIPS:
        payload=json.loads((root/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json").read_text())
        tracks,mapping=_tracks_for_clip(payload,skeleton,cameras,source_report)
        clips[clip_id]=(payload,tracks,mapping)

    out_dir.mkdir(parents=True,exist_ok=True)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    report={
        "schema":"RealSaS.KnightSourceOwnedDeformationRegionProof.v1",
        "status":"MEASURED_RESCUE_PROOF",
        "parent_run_id":parent_run_id,
        "product_authority_claimed":False,
        "ownership":{
            "mechanical_mesh_render_authority":False,
            "visual_mesh_authority":"SOURCE_FOREGROUND_MASK_X_G3B_SAFE_LOCAL_ADJACENCY_CHART",
            "texture_authority":"ORIGINAL_SOURCE_RGBA",
            "appearance_provenance":"DIRECT_SOURCE_ONLY",
            "visual_cross_region_faces":0,
            "unsafe_mechanical_faces_rendered":False,
            "motion_driver":"REGION_LOCAL_SAFE_MECHANICAL_BARYCENTRIC_BINDING",
            "arap_used":False,
        },
        "mechanical_compatibility":{
            "report_hash":compatibility["report_hash"],
            "unsafe_face_count":compatibility["unsafe_face_count"],
            "safe_component_count":int(len(component_sizes)),
            "largest_safe_components":sorted(component_sizes.tolist(),reverse=True)[:30],
        },
        "input_artifacts":[{"path":str(p),"sha256":sha(p)} for p in (candidate_path,policy_path,surface_path,skeleton_path,skin_path,envelope_path,cameras_path,obs_path)],
        "views":[],"outputs":[],
    }
    rendered={}
    for vi in (0,2):
        camera=cameras[vi]
        rgba=np.asarray(rgba_by_view[vi],dtype=np.uint8)
        mask=np.asarray(mask_by_view[vi],dtype=bool)
        h,w=mask.shape
        visibility=rasterize_visible_owner(candidate,camera,positions=rest,width=w,height=h,max_layers=4)
        final_labels,seed_labels,chart_rows=partition_source_mask_by_safe_face_adjacency_v1(
            mask,
            visibility.owner_face_index,
            mech_faces,
            unsafe,
            minimum_seed_pixels=64,
        )
        region_value=build_visual_mesh_from_region_labels_v1(
            mask,
            final_labels,
            target_edge_px=16,
        )
        mesh=region_value.mesh
        vertex_region=np.asarray(region_value.vertex_region_id,dtype=np.int32)
        face_region=np.asarray(region_value.face_region_id,dtype=np.int32)
        region_rows=[]
        seed_count_by_id={int(r["region_id"]):int(r["safe_seed_pixel_count"]) for r in chart_rows}
        regions=[]
        for row0 in region_value.region_rows:
            rid=int(row0["region_id"])
            rm=mask & (final_labels==rid)
            sm=mask & (seed_labels==rid)
            regions.append((rid,rm,sm))
            region_rows.append({**dict(row0),"safe_seed_pixel_count":seed_count_by_id[rid]})
        binding=_region_bindings(mesh,vertex_region,regions,visibility,mech_faces,seed_labels)
        rest_reconstructed=_mechanical_targets(binding=binding,posed_mechanical_xyz=rest,camera=camera)
        rest_error=np.linalg.norm(rest_reconstructed-np.asarray(mesh.positions,float),axis=1)
        view_root=out_dir/f"V{vi}"
        texture=view_root/"source_art.png"; tex_sha=_clean_source_texture(rgba,mask,texture)
        mesh_bin=view_root/"visual_mesh.bin";_write_visual_mesh_binary(mesh_bin,mesh)
        row={
            "view_index":vi,
            "source_foreground_pixel_count":int(mask.sum()),
            "region_count":len(regions),
            "partition_contract":"SOURCE_RASTER_4N_X_STAGE35_SAFE_SHARED_EDGE_V1",
            "regions":region_rows,
            "visual_vertex_count":int(len(mesh.positions)),
            "visual_face_count":int(len(mesh.faces)),
            "max_rest_reconstruction_error_px":float(rest_error.max()),
            "max_nearest_safe_seed_distance_px":float(np.max(binding["nearest_safe_seed_distance_px"])),
            "p95_nearest_safe_seed_distance_px":float(np.quantile(binding["nearest_safe_seed_distance_px"],.95)),
            "source_texture_sha256":tex_sha,
            "direct_source_fraction":1.0,"other_view_fraction":0.0,"completion_fraction":0.0,"unsupported_fraction":0.0,
            "qa_by_clip":{},
        }
        for clip_id,short,ms in CLIPS:
            payload,tracks,mapping=clips[clip_id]
            times=np.linspace(0.0,float(payload["duration_seconds"]),4,endpoint=not bool(payload.get("loop")),dtype=np.float64)
            frames=[]; qa_rows=[]
            for t in times:
                mats,_,_=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
                posed=_skin(rest,W,joint_ids,mats)
                deformed=_mechanical_targets(binding=binding,posed_mechanical_xyz=posed,camera=camera)
                geom=_visual_qa(mesh.positions,deformed,mesh.faces)
                qa_rows.append({"time_seconds":float(t),**geom})
                frames.append(np.asarray(deformed,dtype=np.float64))
            pos=view_root/f"{clip_id}.positions.bin";_write_visual_positions_binary(pos,np.stack(frames))
            for fi in range(4):
                raw=_render_native(native_player,mesh_path=mesh_bin,positions_path=pos,texture_path=texture,clip_id=clip_id,view_index=vi,frame=fi,root=out_dir/"native_frames",resolution=render_resolution)
                arr=np.frombuffer(raw.read_bytes(),dtype=np.uint8).reshape(render_resolution,render_resolution,4)
                rendered[(vi,clip_id,fi)]=Image.fromarray(arr,"RGBA")
            row["qa_by_clip"][clip_id]=qa_rows
        report["views"].append(row)

    for clip_id,short,ms in CLIPS:
        frames=[_gif_frame(rendered[(0,clip_id,i)],rendered[(2,clip_id,i)]) for i in range(4)]
        gp=out_dir/f"KNIGHT_{short}_SOURCE_OWNED_DEFORMATION_REGIONS_V1.gif"
        frames[0].save(gp,save_all=True,append_images=frames[1:],duration=ms,loop=0,disposal=2,optimize=False,transparency=0)
        report["outputs"].append(str(gp))
    (out_dir/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("DEFORMATION_REGION_PROOF_DONE",json.dumps({
        "unsafe_faces":len(unsafe),
        "views":[{"view":v["view_index"],"regions":v["region_count"],"vertices":v["visual_vertex_count"],"faces":v["visual_face_count"]} for v in report["views"]],
    },sort_keys=True))


def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",required=True);p.add_argument("--parent-run-id",required=True);p.add_argument("--out-dir",required=True);p.add_argument("--native-player",required=True);p.add_argument("--render-resolution",type=int,default=512)
    a=p.parse_args();run(authority_root=Path(a.authority_root).resolve(),parent_run_id=a.parent_run_id,out_dir=Path(a.out_dir),native_player=Path(a.native_player),render_resolution=a.render_resolution)

if __name__=="__main__":main()
