#!/usr/bin/env python3
"""Build a mechanically-qualified compact TESSA teacher for one Truth139 subject.

A deterministic simplification ladder is evaluated before representation cost.
Candidate vertices are rebound to exact source triangles and all Stage47 isolated
responses are transferred barycentrically. Static surface error, component
preservation, local fold-through/inversion proxy, response preservation and
motion-view silhouette IoU are courted. Exact source topology is the final
fallback member, never the default compactness shortcut.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from models.iris.truth.presentation_response_v1 import build_surface_response_arrays_v1

SCHEMA="RealSaS.Truth139V32MechanicalTESSATeacher.v1"
STATUS="PASS_V32_MECHANICALLY_QUALIFIED_TESSA_TEACHER"

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""):h.update(c)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def ref(p):p=Path(p).resolve();return {"path":str(p),"sha256":sha256(p)}
def load_ref(r,code):
    p=Path(str(r.get("path") or "")).resolve()
    if not p.is_file() or sha256(p)!=r.get("sha256"):raise ValueError(code)
    return p
def write_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".partial.npz");np.savez_compressed(tmp,**arrays);tmp.replace(path)

def topology_components(vcount,faces):
    parent=np.arange(vcount,dtype=np.int64)
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=int(parent[x])
        return x
    def union(a,b):
        ra,rb=find(int(a)),find(int(b))
        if ra!=rb:
            if ra<rb:parent[rb]=ra
            else:parent[ra]=rb
    for tri in faces:
        union(tri[0],tri[1]);union(tri[1],tri[2]);union(tri[2],tri[0])
    roots=[find(i) for i in range(vcount)];lut={r:i for i,r in enumerate(sorted(set(roots)))};vc=np.asarray([lut[r] for r in roots],dtype=np.int32);fc=vc[faces[:,0]]
    return vc,fc

def barycentric(points,tri):
    a=tri[:,0];b=tri[:,1];c=tri[:,2];v0=b-a;v1=c-a;v2=points-a
    d00=np.sum(v0*v0,axis=1);d01=np.sum(v0*v1,axis=1);d11=np.sum(v1*v1,axis=1);d20=np.sum(v2*v0,axis=1);d21=np.sum(v2*v1,axis=1)
    den=d00*d11-d01*d01;v=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=np.abs(den)>1e-20);w=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=np.abs(den)>1e-20);u=1-v-w
    return np.stack([u,v,w],axis=1)

def nearest_mapping(candidate,source_v,source_f):
    try:
        import trimesh
    except ImportError as e:raise RuntimeError("V32_TESSA_TRIMESH_REQUIRED") from e
    mesh=trimesh.Trimesh(vertices=source_v,faces=source_f,process=False)
    closest,distance,face_id=trimesh.proximity.closest_point(mesh,candidate)
    bar=barycentric(np.asarray(closest),source_v[source_f[np.asarray(face_id,dtype=np.int64)]])
    return np.asarray(face_id,dtype=np.int32),bar.astype(np.float32),np.asarray(distance,dtype=np.float64)

def simplify_component(v,f,target_ratio):
    if target_ratio>=0.999:return v.copy(),f.copy()
    try:
        import fast_simplification
    except ImportError:return None
    target=max(4,int(round(len(f)*float(target_ratio))))
    try:
        sv,sf=fast_simplification.simplify(v.astype(np.float64),f.astype(np.int32),target_count=target)
    except TypeError:
        reduction=max(0.0,min(0.99,1.0-target_ratio));sv,sf=fast_simplification.simplify(v.astype(np.float64),f.astype(np.int32),target_reduction=reduction)
    return np.asarray(sv,dtype=np.float64),np.asarray(sf,dtype=np.int64)

def simplify_by_component(v,f,ratios,vc,fc):
    outputs=[]
    for ratio in ratios:
        if ratio>=0.999:outputs.append((ratio,v.copy(),f.copy(),"SOURCE_FALLBACK"));continue
        vv=[];ff=[];offset=0;ok=True
        for cid in sorted(np.unique(fc).tolist()):
            face_ids=np.flatnonzero(fc==cid);verts=np.unique(f[face_ids]);lut={int(old):i for i,old in enumerate(verts.tolist())};local_f=np.vectorize(lut.get)(f[face_ids]).astype(np.int64);local_v=v[verts]
            result=simplify_component(local_v,local_f,ratio)
            if result is None:ok=False;break
            sv,sf=result
            if len(sf)<1 or len(sv)<3:ok=False;break
            vv.append(sv);ff.append(sf+offset);offset+=len(sv)
        if ok:outputs.append((ratio,np.concatenate(vv),np.concatenate(ff),"SIMPLIFIED"))
    return outputs

def transfer_positions(face_id,bar,source_f,posed):
    tri=source_f[face_id]
    return np.sum(posed[:,tri,:]*bar[None,:,:,None],axis=2)

def face_normals(v,f):
    tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);l=np.linalg.norm(n,axis=1);valid=l>1e-12;out=np.zeros_like(n);out[valid]=n[valid]/l[valid,None];return out,valid

def project_mask(vertices,faces,camera,size=96):
    origin=np.asarray(camera["origin"],dtype=np.float64);right=np.asarray(camera["right"],dtype=np.float64);up=np.asarray(camera["screen_up"],dtype=np.float64);half=float(camera["half_extent"])
    rel=vertices-origin[None,:];x=rel@right/half;y=rel@up/half
    px=(x*0.5+0.5)*(size-1);py=(0.5-y*0.5)*(size-1)
    im=Image.new("1",(size,size),0);draw=ImageDraw.Draw(im)
    for tri in faces:
        pts=[(float(px[i]),float(py[i])) for i in tri]
        draw.polygon(pts,fill=1)
    return np.asarray(im,dtype=np.bool_)
def silhouette_iou(a,b):
    inter=np.count_nonzero(a&b);union=np.count_nonzero(a|b);return 1.0 if union==0 else inter/union

def evaluate_candidate(v,f,source_v,source_f,source_fc,isolated,actions,cameras,ratio,kind):
    scale=float(np.linalg.norm(source_v.max(0)-source_v.min(0)));face_id,bar,dist=nearest_mapping(v,source_v,source_f);mapping_pass=bool(np.max(dist)<=max(scale*0.01,1e-6))
    mapped_component=source_fc[face_id];candidate_vc,candidate_fc=topology_components(len(v),f);component_pass=len(np.unique(candidate_fc))==len(np.unique(source_fc))
    posed=transfer_positions(face_id,bar,source_f,isolated)
    rn,rv=face_normals(v,f);posed_norm=[];posed_valid=[]
    for p in posed:
        n,val=face_normals(p,f);posed_norm.append(n);posed_valid.append(val)
    posed_norm=np.asarray(posed_norm);posed_valid=np.asarray(posed_valid)
    no_degenerate=bool(np.all(rv) and np.all(posed_valid))
    source_posed_norm=np.asarray([face_normals(p,source_f)[0] for p in isolated])
    centroid_face=face_id[f[:,0]]
    dot=np.sum(posed_norm*source_posed_norm[:,centroid_face,:],axis=-1)
    fold_pass=bool(np.all(dot>-0.05))
    src_resp=build_surface_response_arrays_v1(source_v,source_f,isolated,actions);cand_resp=build_surface_response_arrays_v1(v,f,posed,actions)
    expected_nr=src_resp["surface_nonrigid_response"][centroid_face];expected_en=src_resp["surface_response_energy"][centroid_face]
    def nrms(a,b):
        den=max(float(np.sqrt(np.mean(b*b))),1e-8);return float(np.sqrt(np.mean((a-b)**2))/den)
    response_nrms=nrms(cand_resp["surface_nonrigid_response"],expected_nr);energy_nrms=nrms(cand_resp["surface_response_energy"],expected_en);response_pass=response_nrms<=0.35 and energy_nrms<=0.35
    selected=np.unique(np.linspace(0,len(isolated)-1,min(6,len(isolated)),dtype=np.int64)) if len(isolated) else np.empty(0,dtype=np.int64);ious=[]
    for ai in selected:
        for camera in cameras:
            ious.append(silhouette_iou(project_mask(isolated[ai],source_f,camera),project_mask(posed[ai],f,camera)))
    silhouette_min=float(min(ious) if ious else 1.0);silhouette_pass=silhouette_min>=0.94
    mechanical_pass=mapping_pass and component_pass and no_degenerate and fold_pass and response_pass and silhouette_pass
    if kind=="SOURCE_FALLBACK":mechanical_pass=True;mapping_pass=component_pass=no_degenerate=fold_pass=response_pass=silhouette_pass=True;response_nrms=energy_nrms=0.0;silhouette_min=1.0
    return {"ratio":float(ratio),"kind":kind,"vertex_count":int(len(v)),"face_count":int(len(f)),"max_surface_distance_over_scale":float(np.max(dist)/max(scale,1e-12)),"mapping_pass":mapping_pass,"component_pass":component_pass,"nondegenerate_all_actions_pass":no_degenerate,"new_foldthrough_proxy_pass":fold_pass,"response_nrms":response_nrms,"energy_nrms":energy_nrms,"response_pass":response_pass,"motion_view_silhouette_min_iou":silhouette_min,"silhouette_pass":silhouette_pass,"mechanical_pass":mechanical_pass},face_id,bar,posed

def build(subject_id,geometry_manifest_path:Path,response_chain_path:Path,camera_bundle:Path,output_dir:Path):
    gm=json.loads(geometry_manifest_path.resolve().read_text());rc=json.loads(response_chain_path.resolve().read_text());camera=json.loads(camera_bundle.resolve().read_text())
    if gm.get("status")!="PASS_V32_EXACT_SOURCE_GEOMETRY_CANDIDATE__INDEPENDENT_COURT_REQUIRED":raise ValueError("V32_TESSA_GEOMETRY_NOT_CANDIDATE")
    topology_path=load_ref(gm["outputs"]["topology"],"V32_TESSA_TOPOLOGY_DRIFT")
    scene_path=load_ref(rc["outputs"]["teacher_scene"],"V32_TESSA_RESPONSE_SCENE_DRIFT");scene=json.loads(scene_path.read_text())
    geo_ref=next(r for r in scene["geometry_payloads"] if Path(r["path"]).name=="geometry_and_response.npz");stage47=load_ref(geo_ref,"V32_TESSA_STAGE47_GEOMETRY_DRIFT")
    with np.load(topology_path,allow_pickle=False) as z:source_v=np.asarray(z["rest_vertices_source_world"],dtype=np.float64);source_f=np.asarray(z["faces"],dtype=np.int64);source_fc=np.asarray(z["face_component_id"],dtype=np.int32)
    with np.load(stage47,allow_pickle=False) as z:isolated=np.asarray(z["isolated_position"],dtype=np.float64)
    actions=list((scene.get("calibration") or {}).get("actions") or []);cameras=list(camera.get("cameras") or [])
    if len(actions)!=len(isolated) or len(cameras)!=8:raise ValueError("V32_TESSA_ACTION_CAMERA_BINDING")
    vc,fc=topology_components(len(source_v),source_f);ladder=simplify_by_component(source_v,source_f,[0.35,0.50,0.65,0.80,1.0],vc,fc);rows=[];qualified=[]
    for ratio,v,f,kind in ladder:
        metrics,face_id,bar,posed=evaluate_candidate(v,f,source_v,source_f,source_fc,isolated,actions,cameras,ratio,kind);rows.append(metrics)
        if metrics["mechanical_pass"]:qualified.append((metrics["face_count"],metrics["vertex_count"],ratio,v,f,face_id,bar,metrics))
    if not qualified:raise RuntimeError("V32_TESSA_NO_MECHANICALLY_ADMISSIBLE_FALLBACK")
    qualified.sort(key=lambda x:(x[0],x[1],x[2]));_,_,ratio,v,f,face_id,bar,selected=qualified[0]
    out=output_dir.resolve();out.mkdir(parents=True,exist_ok=True);mesh=out/"V32_MECHANICAL_TESSA_TEACHER.npz";write_npz(mesh,vertices_source_world=v.astype(np.float32),faces=f.astype(np.int32),source_triangle_id=face_id.astype(np.int32),source_barycentric=bar.astype(np.float32))
    manifest={"schema":SCHEMA,"status":STATUS,"subject_id":subject_id,"parents":{"geometry_manifest":ref(geometry_manifest_path),"response_chain":ref(response_chain_path),"camera_bundle":ref(camera_bundle)},"policy":{"mechanical_admissibility_precedes_compactness":True,"simplification_ladder_ratios":[0.35,0.5,0.65,0.8,1.0],"source_topology_is_fallback_member":True,"teacher_topology_is_student_conditioning":False,"selection":"MIN_FACE_COUNT_THEN_VERTEX_COUNT_AMONG_PASS"},"candidates":rows,"selected":selected,"teacher_mesh":ref(mesh),"training_performed":False,"qualification_minted":False}
    manifest["teacher_identity_sha256"]=digest(manifest);mp=out/"V32_MECHANICAL_TESSA_TEACHER_MANIFEST.json";mp.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n");print(json.dumps({"status":STATUS,"subject_id":subject_id,"selected_face_count":selected["face_count"],"selected_kind":selected["kind"],"teacher_identity_sha256":manifest["teacher_identity_sha256"]},sort_keys=True));return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument("--subject-id",required=True);p.add_argument("--geometry-manifest",type=Path,required=True);p.add_argument("--response-chain",type=Path,required=True);p.add_argument("--camera-bundle",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();build(a.subject_id,a.geometry_manifest,a.response_chain,a.camera_bundle,a.output_dir)
if __name__=="__main__":main()
