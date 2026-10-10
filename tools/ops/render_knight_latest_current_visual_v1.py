from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    read_json,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import project_points_xyz_v3
from compiler.realsas_compiler_core.carrier_skin_v1 import qualified_carrier_skin_from_dict
from compiler.realsas_compiler_core.deformation_envelope_derivation_v2 import derive_deformation_envelope_v2
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import mechanical_carrier_evidence_from_dict
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.skin_topology_compatibility_carrier_v1 import run_skin_topology_compatibility_carrier_v1
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_core.visual_material_render_v1 import render_visual_material
from compiler.realsas_compiler_core.visual_material_v1 import load_visual_material
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    bind_region_visual_vertices_to_mechanical_affine_v1,
    build_visual_mesh_from_region_labels_v1,
    evaluate_region_visual_binding_v1,
    partition_source_mask_by_safe_face_adjacency_v1,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_source_inputs
from tools.demo.render_knight_motion_preview_v1 import _skin, _tracks_for_clip

CAA_RUN='KNIGHT_LATEST_CURRENT_CAA_VISUAL_20261010'
MAT_RUN='KNIGHT_LATEST_CARRIER_MATERIALIZATION_20261010'
SOURCE_REPORT=Path('canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json')


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()


def stage_row(ledger,sid):
    return next(r for r in ledger['stages'] if r['id']==sid)


def stage_payload(ledger,sid,schema):
    out=next(o for o in stage_row(ledger,sid).get('outputs',[]) if o.get('schema')==schema)
    p=Path(out['path'])
    if not p.is_file() or sha256(p)!=str(out['sha256']): raise RuntimeError(f'STAGE_BYTES_DRIFT::{sid}::{schema}')
    return json.loads(p.read_text()),p


def mechanical_faces(candidate):
    ids=[str(v.candidate_vertex_id) for v in candidate.vertices]
    idx={x:i for i,x in enumerate(ids)}
    if len(idx)!=len(ids): raise RuntimeError('DUPLICATE_CARRIER_VERTEX_ID')
    return np.asarray([[idx[str(x)] for x in f] for f in candidate.faces],dtype=np.int64)


def motion_payloads(input_root:Path):
    found={}
    for p in input_root.rglob('*.json'):
        try: x=json.loads(p.read_text())
        except Exception: continue
        if x.get('schema')!='RealSaS.MotionSourceClip.v2': continue
        cid=str(x.get('clip_id') or '')
        if cid in {'demo_idle_v1','demo_run_v1','demo_slash_v1'}:
            found[cid]=(x,p)
    if set(found)!={'demo_idle_v1','demo_run_v1','demo_slash_v1'}:
        raise RuntimeError(f'MOTION_SET_INCOMPLETE::{sorted(found)}')
    return found


def visual_metrics(rest,posed,faces):
    r=np.asarray(rest,float)[np.asarray(faces,np.int64)]
    p=np.asarray(posed,float)[np.asarray(faces,np.int64)]
    def area(t):
        a=t[:,1]-t[:,0]; b=t[:,2]-t[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    ra=area(r); pa=area(p)
    rl=np.stack((np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)),axis=1)
    pl=np.stack((np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)),axis=1)
    q=pl/np.maximum(rl,1e-9)
    return {'flipped_triangles':int(np.count_nonzero(ra*pa<0)),'p95_edge_ratio':float(np.quantile(q,.95)),'p99_edge_ratio':float(np.quantile(q,.99)),'max_edge_ratio':float(np.max(q))}


def compose_grid(images,labels):
    if len(images)!=8: raise ValueError('need 8 views')
    w,h=images[0].size
    canvas=Image.new('RGBA',(w*4,h*2),(22,22,22,255)); draw=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(images,labels)):
        x=(i%4)*w; y=(i//4)*h
        canvas.alpha_composite(im,(x,y)); draw.text((x+6,y+6),label,fill=(255,255,255,255),stroke_width=1,stroke_fill=(0,0,0,255))
    return canvas


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--authority-root',type=Path,required=True); ap.add_argument('--input-root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--resolution',type=int,default=320); ap.add_argument('--frames',type=int,default=10)
    a=ap.parse_args(); auth=a.authority_root.resolve(); caa=auth/'runs'/CAA_RUN; mat=auth/'runs'/MAT_RUN; out=a.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    ledger=json.loads((caa/'ACTIVE_RUN_V2.json').read_text()); manifest=json.loads((caa/'run_manifest.json').read_text())
    candidate=canonical_mesh_candidate_from_dict(json.loads((mat/'rebound_candidate.json').read_text()))
    carrier=mechanical_carrier_evidence_from_dict(json.loads((mat/'mechanical_carrier_evidence.json').read_text()))
    skin=qualified_carrier_skin_from_dict(json.loads((mat/'qualified_carrier_skin.json').read_text()))
    skeleton=qualified_skeleton_from_dict(json.loads((a.input_root/'AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_QUALIFIED_SKELETON_V541.json').read_text()))
    cam_raw,cam_path=stage_payload(ledger,'05_CAMERA_CONTRACT_SOLVED','RealSaS.QualifiedCameraSetIR.v1'); camera_set=qualified_camera_set_from_dict(cam_raw); cameras=tuple(sorted(camera_set.cameras,key=lambda x:int(x.view_index)))
    obs_raw,obs_path=stage_payload(ledger,'07_OBSERVATION_CONTRACT_QUALIFIED','RealSaS.QualifiedObservationSetIR.v1'); observation=qualified_observation_set_from_dict(obs_raw)
    policy_raw,policy_path=stage_payload(ledger,'18_CANONICAL_MESH_ADDRESSING_BUILD','RealSaS.MeshQualificationPolicyIR.v1'); policy=mesh_policy_from_dict(policy_raw)
    asset_raw,asset_path=stage_payload(ledger,'23_COMPLETE_APPEARANCE_ASSET_BAKED','RealSaS.CompleteAppearanceAssetIR.v2'); asset=complete_appearance_asset_from_dict(asset_raw)
    if not bool(dict(asset.metadata or {}).get('canonical_completion_field_bound')): raise RuntimeError('CAA_COMPLETION_FIELD_NOT_BOUND')
    if str(asset.candidate_mesh_binding_hash)!=str(candidate.candidate_lineage_hash): raise RuntimeError('CAA_CANDIDATE_BINDING_DRIFT')

    rest=np.asarray([v.P for v in candidate.vertices],dtype=np.float64); mech_faces=mechanical_faces(candidate)
    if tuple(map(str,carrier.ordered_vertex_ids))!=tuple(str(v.candidate_vertex_id) for v in candidate.vertices): raise RuntimeError('CARRIER_ORDER_DRIFT')
    axis_payload,envelope=derive_deformation_envelope_v2(skeleton=skeleton,camera_set=camera_set)
    compatibility=run_skin_topology_compatibility_carrier_v1(carrier,skeleton=skeleton,skin=skin,envelope=envelope,cameras=cameras,policy=policy)
    unsafe=set(map(int,compatibility.get('unsafe_face_indices') or ()))

    ctx={'repo_root':Path('.').resolve(),'authority_root':auth,'run_root':caa,'run_id':CAA_RUN,'run_manifest_path':caa/'run_manifest.json','run_manifest':manifest,'ledger':ledger,'stage':{'id':'RENDER'}}
    rgba_by_view,mask_by_view=_load_source_inputs(ctx,observation)
    texture_by_view={int(r.direction_index):r for r in asset.textures}
    bindings={}; visual_meshes={}; materials={}; view_info=[]
    for vi,camera in enumerate(cameras):
        mask=np.asarray(mask_by_view[vi],dtype=bool); h,w=mask.shape
        visibility=rasterize_visible_owner(candidate,camera,positions=rest,width=w,height=h,max_layers=4)
        final_labels,seed_labels,charts=partition_source_mask_by_safe_face_adjacency_v1(mask,visibility.owner_face_index,mech_faces,unsafe,minimum_seed_pixels=64)
        region=build_visual_mesh_from_region_labels_v1(mask,final_labels,target_edge_px=16)
        vm=region.mesh
        binding=bind_region_visual_vertices_to_mechanical_affine_v1(points_source_xy=vm.positions,vertex_region_id=np.asarray(region.vertex_region_id,dtype=np.int32),seed_region_labels=np.asarray(seed_labels,dtype=np.int32),owner_face_index=np.asarray(visibility.owner_face_index,dtype=np.int64),mechanical_positions_xyz=rest,mechanical_faces=mech_faces,camera=camera,candidate_seed_count=16,max_seed_distance_px=float(np.hypot(w,h)))
        rest_eval=evaluate_region_visual_binding_v1(binding,posed_mechanical_positions_xyz=rest,camera=camera)
        err=np.linalg.norm(rest_eval-np.asarray(vm.positions,float),axis=1)
        if not np.isfinite(err).all() or float(np.max(err,initial=0.0))>1e-7: raise RuntimeError(f'REST_BIND_DRIFT::V{vi}')
        tex=texture_by_view[vi]; tex_path=Path(tex.transport_png_path)
        if not tex_path.is_file() or sha256(tex_path)!=str(tex.transport_png_sha256): raise RuntimeError(f'TEXTURE_DRIFT::V{vi}')
        rgba=np.asarray(Image.open(tex_path).convert('RGBA'),dtype=np.uint8)
        prov,donor=load_visual_material(asset,view_index=vi,rgba=rgba)
        bindings[vi]=binding; visual_meshes[vi]=vm; materials[vi]=(rgba,prov,donor)
        view_info.append({'view_index':vi,'visual_vertices':int(len(vm.positions)),'visual_faces':int(len(vm.faces)),'chart_count':int(len(charts)),'max_rest_binding_error_px':float(np.max(err,initial=0.0)),'max_seed_distance_px':float(np.max(binding['nearest_safe_seed_distance_px'],initial=0.0)),'max_extrapolation_penalty':float(np.max(binding['extrapolation_penalty'],initial=0.0))})

    with np.load(a.input_root/'MIRA_AXIS41_TESSA_M_SKIN_FIELD_V55.npz',allow_pickle=False) as z:
        W=np.asarray(z['weights_axis41_canonical'],dtype=np.float64); joint_ids=tuple(map(str,np.asarray(z['canonical_joint_ids']).tolist())); basis=str(np.asarray(z['carrier_basis_sha256']).item())
    if W.shape!=(len(rest),len(joint_ids)) or set(joint_ids)!={str(j.canonical_joint_id) for j in skeleton.joints}: raise RuntimeError('W_M_BINDING_DRIFT')
    if basis!='dbac301b9c47fa7f31a024ef0612008d8315b590d7a2dd1552a75f75c7ac9ff2': raise RuntimeError('W_M_BASIS_DRIFT')
    source_report=json.loads(SOURCE_REPORT.read_text()); motions=motion_payloads(a.input_root)
    clip_order=(('demo_idle_v1','IDLE'),('demo_run_v1','RUN'),('demo_slash_v1','SLASH'))
    report={'schema':'RealSaS.KnightLatestCurrentVisualResearchRender.v1','status':'MEASURED_RESEARCH_RENDER','product_authority_claimed':False,'code_sha':'d3c4f39ddecf2dc73e7c532ec352fc97ad92579b','caa_asset_hash':asset.asset_hash,'caa_canonical_completion_field_bound':True,'caa_completion_field_consumed_by_visual_material_transport':False,'mechanical_carrier_evidence_hash':carrier.carrier_evidence_hash,'carrier_skin_lineage_hash':skin.skin_lineage_hash,'skeleton_lineage_hash':skeleton.skeleton_lineage_hash,'carrier_basis_sha256':basis,'unsafe_mechanical_face_count':int(len(unsafe)),'compatibility_passed':bool(compatibility.get('passed')),'views':view_info,'clips':{},'outputs':[]}

    for clip_id,label in clip_order:
        payload,motion_path=motions[clip_id]; tracks,mapping=_tracks_for_clip(payload,skeleton,cameras,source_report)
        times=np.linspace(0.0,float(payload['duration_seconds']),int(a.frames),endpoint=not bool(payload.get('loop')),dtype=np.float64)
        grid_frames=[]; v0_frames=[]; qa={f'V{i}':[] for i in range(8)}; render_counts={f'V{i}':{'overlap':0,'tie':0,'overflow':0} for i in range(8)}
        for frame_i,t in enumerate(times):
            mats,_,frame_hash=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,W,joint_ids,mats)
            rendered=[]
            for vi,camera in enumerate(cameras):
                vm=visual_meshes[vi]; binding=bindings[vi]; rgba,prov,donor=materials[vi]
                pos=evaluate_region_visual_binding_v1(binding,posed_mechanical_positions_xyz=posed,camera=camera)
                projected=np.asarray(project_points_xyz_v3(posed,camera),dtype=np.float64)
                depths=np.sum(projected[np.asarray(binding['mechanical_face_indices'],dtype=np.int64),2]*np.asarray(binding['mechanical_barycentric'],dtype=np.float64),axis=1)
                if not np.isfinite(depths).all() or np.any(depths<=0): raise RuntimeError(f'CANONICAL_DEPTH_INVALID::{clip_id}::V{vi}::{frame_i}')
                rr=render_visual_material(positions=pos,depths=depths,faces=np.asarray(vm.faces,dtype=np.int64),uv=np.asarray(vm.uv,dtype=np.float64),texture=rgba,provenance=prov,source_view=donor,view_index=vi,resolution=int(a.resolution))
                im=Image.fromarray(np.asarray(rr.straight_rgba_u8,dtype=np.uint8),'RGBA')
                rendered.append(im); qa[f'V{vi}'].append(visual_metrics(vm.positions,pos,vm.faces))
                render_counts[f'V{vi}']['overlap']+=int(np.count_nonzero(rr.owner_face_index>=0))
                render_counts[f'V{vi}']['tie']+=int(np.count_nonzero(rr.depth_tie)) if hasattr(rr,'depth_tie') else 0
                render_counts[f'V{vi}']['overflow']+=int(np.count_nonzero(rr.layer_overflow)) if hasattr(rr,'layer_overflow') else 0
            grid=compose_grid(rendered,[f'V{i}' for i in range(8)]); grid_frames.append(grid); v0_frames.append(rendered[0])
        duration_ms=max(40,int(round(float(payload['duration_seconds'])*1000.0/max(1,len(times)-1 if not payload.get('loop') else len(times)))))
        grid_path=out/f'KNIGHT_{label}_LATEST_CURRENT_VISUAL_ALL_VIEWS.gif'; v0_path=out/f'KNIGHT_{label}_LATEST_CURRENT_VISUAL_V0.gif'
        grid_frames[0].save(grid_path,save_all=True,append_images=grid_frames[1:],duration=duration_ms,loop=0,disposal=2)
        v0_frames[0].save(v0_path,save_all=True,append_images=v0_frames[1:],duration=duration_ms,loop=0,disposal=2)
        first_path=out/f'KNIGHT_{label}_LATEST_CURRENT_VISUAL_ALL_VIEWS_FRAME0.png'; grid_frames[0].save(first_path)
        clip_report={'motion_path':str(motion_path),'motion_sha256':sha256(motion_path),'duration_seconds':float(payload['duration_seconds']),'loop':bool(payload.get('loop')),'frame_count':len(times),'qa':{},'render_counts':render_counts,'mapping':mapping}
        for vk,rows in qa.items():
            clip_report['qa'][vk]={'maximum_flipped_triangles':max(r['flipped_triangles'] for r in rows),'maximum_p95_edge_ratio':max(r['p95_edge_ratio'] for r in rows),'maximum_p99_edge_ratio':max(r['p99_edge_ratio'] for r in rows),'maximum_edge_ratio':max(r['max_edge_ratio'] for r in rows)}
        report['clips'][clip_id]=clip_report
        for p in (grid_path,v0_path,first_path): report['outputs'].append({'path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size})
        print('RENDERED',clip_id,grid_path,sha256(grid_path),flush=True)
    report_path=out/'REPORT.json'; report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print('REPORT',json.dumps(report,sort_keys=True)[:20000],flush=True)

if __name__=='__main__': main()
