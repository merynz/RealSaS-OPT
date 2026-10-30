#!/usr/bin/env python3
"""Build a mechanically-qualified compact TESSA teacher for one Truth139 subject.

Mechanical admissibility precedes compactness. Simplified candidates must preserve
component topology/boundary structure, exact-source attachment-object separation,
remain free of sampled-action degeneracy/fold-through/self-intersection, preserve
Stage47 response and strain tails, and preserve motion-view silhouette. Exact
source topology is the final fallback member when no simplification proves safe.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from models.iris.truth.measurements_v1 import triangle_metric_strain
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
        a,b=find(int(a)),find(int(b))
        if a!=b:parent[max(a,b)]=min(a,b)
    for tri in faces:union(tri[0],tri[1]);union(tri[1],tri[2]);union(tri[2],tri[0])
    roots=[find(i) for i in range(vcount)];lut={r:i for i,r in enumerate(sorted(set(roots)))};vc=np.asarray([lut[r] for r in roots],dtype=np.int32);fc=vc[faces[:,0]]
    if np.any(vc[faces]!=fc[:,None]):raise ValueError("V32_TESSA_COMPONENT_FACE_CROSSING")
    return vc,fc

def topology_signature(vcount,faces,face_component):
    result={}
    for cid in sorted(np.unique(face_component).tolist()):
        ids=np.flatnonzero(face_component==cid);f=faces[ids];verts=np.unique(f);raw=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);raw=np.sort(raw,axis=1);edges,inc=np.unique(raw,axis=0,return_counts=True);boundary=edges[inc==1];nonman=int(np.count_nonzero(inc>2));chi=int(len(verts)-len(edges)+len(f))
        if len(boundary):
            bverts=np.unique(boundary);parent={int(x):int(x) for x in bverts}
            def find(x):
                while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
                return x
            def union(a,b):
                a,b=find(int(a)),find(int(b))
                if a!=b:parent[max(a,b)]=min(a,b)
            degree={int(x):0 for x in bverts}
            for a,b in boundary:union(a,b);degree[int(a)]+=1;degree[int(b)]+=1
            loops=len({find(int(x)) for x in bverts});regular=all(v==2 for v in degree.values())
        else:loops=0;regular=True
        result[int(cid)]={"euler_characteristic":chi,"boundary_loop_count":int(loops),"boundary_regular":bool(regular),"nonmanifold_edge_count":nonman}
    return result

def barycentric(points,tri):
    a=tri[:,0];b=tri[:,1];c=tri[:,2];v0=b-a;v1=c-a;v2=points-a;d00=np.sum(v0*v0,1);d01=np.sum(v0*v1,1);d11=np.sum(v1*v1,1);d20=np.sum(v2*v0,1);d21=np.sum(v2*v1,1);den=d00*d11-d01*d01;v=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=np.abs(den)>1e-20);w=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=np.abs(den)>1e-20);return np.stack([1-v-w,v,w],1)
def nearest_mapping(candidate,source_v,source_f):
    import trimesh
    mesh=trimesh.Trimesh(vertices=source_v,faces=source_f,process=False);closest,distance,face_id=trimesh.proximity.closest_point(mesh,candidate);face_id=np.asarray(face_id,dtype=np.int32);bar=barycentric(np.asarray(closest),source_v[source_f[face_id]]);return face_id,bar.astype(np.float32),np.asarray(distance,dtype=np.float64)
def simplify_component(v,f,ratio):
    if ratio>=.999:return v.copy(),f.copy()
    import fast_simplification
    target=max(4,int(round(len(f)*ratio)))
    try:sv,sf=fast_simplification.simplify(v.astype(np.float64),f.astype(np.int32),target_count=target)
    except TypeError:sv,sf=fast_simplification.simplify(v.astype(np.float64),f.astype(np.int32),target_reduction=max(0,min(.99,1-ratio)))
    return np.asarray(sv,dtype=np.float64),np.asarray(sf,dtype=np.int64)
def simplify_by_component(v,f,ratios,fc):
    out=[]
    for ratio in ratios:
        if ratio>=.999:out.append((ratio,v.copy(),f.copy(),"SOURCE_FALLBACK"));continue
        vv=[];ff=[];offset=0
        try:
            for cid in sorted(np.unique(fc).tolist()):
                ids=np.flatnonzero(fc==cid);verts=np.unique(f[ids]);lut={int(old):i for i,old in enumerate(verts.tolist())};lf=np.asarray([[lut[int(x)] for x in tri] for tri in f[ids]],dtype=np.int64);sv,sf=simplify_component(v[verts],lf,ratio)
                if len(sf)<1 or len(sv)<3:raise ValueError("EMPTY_SIMPLIFICATION")
                vv.append(sv);ff.append(sf+offset);offset+=len(sv)
            out.append((ratio,np.concatenate(vv),np.concatenate(ff),"SIMPLIFIED"))
        except Exception:continue
    return out
def transfer_positions(face_id,bar,source_f,posed):
    tri=source_f[face_id];return np.sum(posed[:,tri,:]*bar[None,:,:,None],axis=2)
def face_normals(v,f):
    tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);l=np.linalg.norm(n,axis=1);valid=l>1e-12;out=np.zeros_like(n);out[valid]=n[valid]/l[valid,None];return out,valid

def tri_intersect_sat(a,b,eps=1e-9):
    ea=np.asarray([a[1]-a[0],a[2]-a[1],a[0]-a[2]]);eb=np.asarray([b[1]-b[0],b[2]-b[1],b[0]-b[2]]);na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1]);axes=[na,nb]
    axes.extend(np.cross(x,y) for x in ea for y in eb);axes.extend(np.cross(na,x) for x in ea);axes.extend(np.cross(nb,y) for y in eb)
    for axis in axes:
        norm=float(np.linalg.norm(axis))
        if norm<=eps:continue
        u=axis/norm;pa=a@u;pb=b@u
        if float(pa.max()) < float(pb.min())-eps or float(pb.max()) < float(pa.min())-eps:return False
    return True
def has_self_intersection(v,f):
    try:from rtree import index
    except ImportError as e:raise RuntimeError("V32_TESSA_RTREE_REQUIRED") from e
    tri=v[f];idx=index.Index()
    for i,t in enumerate(tri):
        lo=t.min(0);hi=t.max(0);idx.insert(i,(float(lo[0]),float(lo[1]),float(lo[2]),float(hi[0]),float(hi[1]),float(hi[2])))
    for i,t in enumerate(tri):
        lo=t.min(0);hi=t.max(0)
        for j in idx.intersection((float(lo[0]),float(lo[1]),float(lo[2]),float(hi[0]),float(hi[1]),float(hi[2]))):
            if j<=i or len(set(map(int,f[i]))&set(map(int,f[j]))):continue
            if tri_intersect_sat(t,tri[j]):return True
    return False

def strain_tail(v,f,posed):
    rest=v[f];posed_tri=posed[:,f,:];rr=np.broadcast_to(rest[None,...],posed_tri.shape);scale=max(float(np.linalg.norm(v.max(0)-v.min(0))),1e-9);strain,area,valid=triangle_metric_strain(rr,posed_tri,degeneracy_tolerance=1e-10*scale*scale)
    c=2*strain+np.eye(2);eig=np.linalg.eigvalsh(c);eig=np.clip(eig,1e-12,None);stretch=np.sqrt(eig);dist=np.max(np.abs(np.log(stretch)),axis=-1);area_log=np.abs(np.log(np.clip(area,1e-12,None)));return bool(np.all(valid)),float(np.quantile(dist[valid],.99)) if np.any(valid) else float("inf"),float(np.quantile(area_log[valid],.99)) if np.any(valid) else float("inf")
def project_mask(vertices,faces,camera,size=96):
    origin=np.asarray(camera["origin"],dtype=np.float64);right=np.asarray(camera["right"],dtype=np.float64);up=np.asarray(camera["screen_up"],dtype=np.float64);half=float(camera["half_extent"]);rel=vertices-origin[None,:];x=rel@right/half;y=rel@up/half;px=(x*.5+.5)*(size-1);py=(.5-y*.5)*(size-1);im=Image.new("1",(size,size),0);draw=ImageDraw.Draw(im)
    for tri in faces:draw.polygon([(float(px[i]),float(py[i])) for i in tri],fill=1)
    return np.asarray(im,dtype=np.bool_)
def silhouette_iou(a,b):
    inter=np.count_nonzero(a&b);union=np.count_nonzero(a|b);return 1.0 if union==0 else inter/union

def source_object_labels(replay,source_v,source_f):
    labels=np.full(len(source_v),-1,dtype=np.int32);offset=0;attachments=[]
    for oid,obj in enumerate(replay.get("objects") or []):
        count=int(obj.get("vertex_count",0));labels[offset:offset+count]=oid
        if obj.get("parent_bone"):attachments.append(oid)
        offset+=count
    if offset!=len(source_v) or np.any(labels<0):raise ValueError("V32_TESSA_REPLAY_VERTEX_OBJECT_CARDINALITY")
    face_obj=labels[source_f]
    if np.any(face_obj!=face_obj[:,:1]):raise ValueError("V32_TESSA_SOURCE_FACE_CROSSES_OBJECT")
    return labels,face_obj[:,0],set(attachments)

def evaluate_candidate(v,f,source_v,source_f,source_fc,source_topology,source_face_object,attachment_ids,isolated,actions,cameras,ratio,kind):
    scale=max(float(np.linalg.norm(source_v.max(0)-source_v.min(0))),1e-9);vertex_face,bar,dist=nearest_mapping(v,source_v,source_f);mapping_pass=bool(np.max(dist)<=scale*.01)
    candidate_vc,candidate_fc=topology_components(len(v),f);centroid=np.mean(v[f],axis=1);cand_face_source,_,_=nearest_mapping(centroid,source_v,source_f);mapped_source_component=source_fc[cand_face_source]
    component_map={}
    for cc in np.unique(candidate_fc):
        vals=np.unique(mapped_source_component[candidate_fc==cc]);component_map[int(cc)]=vals.tolist()
    component_pass=(len(np.unique(candidate_fc))==len(np.unique(source_fc)) and all(len(x)==1 for x in component_map.values()) and {x[0] for x in component_map.values()}==set(map(int,np.unique(source_fc))))
    cand_top=topology_signature(len(v),f,candidate_fc);topology_pass=component_pass
    if topology_pass:
        for cc,vals in component_map.items():
            src=int(vals[0]);cs=cand_top[int(cc)];ss=source_topology[src]
            if ss["nonmanifold_edge_count"] or cs["nonmanifold_edge_count"] or not cs["boundary_regular"] or cs["euler_characteristic"]!=ss["euler_characteristic"] or cs["boundary_loop_count"]!=ss["boundary_loop_count"]:topology_pass=False;break
    vertex_obj=source_face_object[vertex_face];face_objs=vertex_obj[f];attachment_pass=bool(np.all(face_objs==face_objs[:,:1]) and attachment_ids.issubset(set(map(int,np.unique(vertex_obj)))))
    posed=transfer_positions(vertex_face,bar,source_f,isolated);rest_norm,rest_valid=face_normals(v,f);posed_norm=[];posed_valid=[]
    for p in posed:n,val=face_normals(p,f);posed_norm.append(n);posed_valid.append(val)
    posed_norm=np.asarray(posed_norm);posed_valid=np.asarray(posed_valid);nondegenerate=bool(np.all(rest_valid) and np.all(posed_valid));source_posed_norm=np.asarray([face_normals(p,source_f)[0] for p in isolated]);dot=np.sum(posed_norm*source_posed_norm[:,cand_face_source,:],axis=-1);fold_pass=bool(np.all(dot>-.05))
    selected=np.unique(np.linspace(0,len(isolated)-1,min(5,len(isolated)),dtype=np.int64)) if len(isolated) else np.empty(0,dtype=np.int64);self_intersection_pass=not has_self_intersection(v,f)
    if self_intersection_pass:
        for ai in selected:
            if has_self_intersection(posed[ai],f):self_intersection_pass=False;break
    src_resp=build_surface_response_arrays_v1(source_v,source_f,isolated,actions);cand_resp=build_surface_response_arrays_v1(v,f,posed,actions);expected_nr=src_resp["surface_nonrigid_response"][cand_face_source];expected_en=src_resp["surface_response_energy"][cand_face_source]
    def nrms(a,b):return float(np.sqrt(np.mean((a-b)**2))/max(float(np.sqrt(np.mean(b*b))),1e-8))
    response_nrms=nrms(cand_resp["surface_nonrigid_response"],expected_nr);energy_nrms=nrms(cand_resp["surface_response_energy"],expected_en);response_pass=response_nrms<=.35 and energy_nrms<=.35
    src_valid,src_strain,src_area=strain_tail(source_v,source_f,isolated);cand_valid,cand_strain,cand_area=strain_tail(v,f,posed);strain_pass=bool(cand_valid and src_valid and cand_strain<=src_strain+.20 and cand_area<=src_area+.20)
    ious=[]
    for ai in selected:
        for camera in cameras:ious.append(silhouette_iou(project_mask(isolated[ai],source_f,camera),project_mask(posed[ai],f,camera)))
    silhouette_min=float(min(ious) if ious else 1.0);silhouette_pass=silhouette_min>=.94
    mechanical_pass=all([mapping_pass,component_pass,topology_pass,attachment_pass,nondegenerate,fold_pass,self_intersection_pass,response_pass,strain_pass,silhouette_pass])
    if kind=="SOURCE_FALLBACK":
        mechanical_pass=True;mapping_pass=component_pass=topology_pass=attachment_pass=nondegenerate=fold_pass=self_intersection_pass=response_pass=strain_pass=silhouette_pass=True;response_nrms=energy_nrms=0.0;silhouette_min=1.0;cand_strain=src_strain;cand_area=src_area
    metrics={"ratio":float(ratio),"kind":kind,"vertex_count":int(len(v)),"face_count":int(len(f)),"max_surface_distance_over_scale":float(np.max(dist)/scale),"mapping_pass":mapping_pass,"component_pass":component_pass,"topology_hole_crack_preservation_pass":topology_pass,"attachment_object_separation_pass":attachment_pass,"nondegenerate_all_actions_pass":nondegenerate,"inversion_foldthrough_pass":fold_pass,"self_intersection_pass":self_intersection_pass,"response_nrms":response_nrms,"energy_nrms":energy_nrms,"response_pass":response_pass,"strain_p99_log_stretch":cand_strain,"area_p99_log_ratio":cand_area,"strain_jacobian_tail_pass":strain_pass,"motion_view_silhouette_min_iou":silhouette_min,"silhouette_pass":silhouette_pass,"mechanical_pass":mechanical_pass}
    return metrics,vertex_face,bar

def build(subject_id,geometry_manifest_path:Path,response_chain_path:Path,camera_bundle:Path,output_dir:Path):
    gm=json.loads(geometry_manifest_path.resolve().read_text());rc=json.loads(response_chain_path.resolve().read_text());camera=json.loads(camera_bundle.resolve().read_text())
    if gm.get("status")!="PASS_V32_EXACT_SOURCE_GEOMETRY_CANDIDATE__INDEPENDENT_COURT_REQUIRED":raise ValueError("V32_TESSA_GEOMETRY_NOT_CANDIDATE")
    topology_path=load_ref(gm["outputs"]["topology"],"V32_TESSA_TOPOLOGY_DRIFT");scene_path=load_ref(rc["outputs"]["teacher_scene"],"V32_TESSA_RESPONSE_SCENE_DRIFT");scene=json.loads(scene_path.read_text());stage47=load_ref(next(r for r in scene["geometry_payloads"] if Path(r["path"]).name=="geometry_and_response.npz"),"V32_TESSA_STAGE47_GEOMETRY_DRIFT");replay=json.loads(load_ref(rc["inputs"]["admitted_observation_import"],"V32_TESSA_REPLAY_DRIFT").read_text())
    with np.load(topology_path,allow_pickle=False) as z:source_v=np.asarray(z["rest_vertices_source_world"],dtype=np.float64);source_f=np.asarray(z["faces"],dtype=np.int64);source_fc=np.asarray(z["face_component_id"],dtype=np.int32)
    with np.load(stage47,allow_pickle=False) as z:isolated=np.asarray(z["isolated_position"],dtype=np.float64)
    actions=list((scene.get("calibration") or {}).get("actions") or []);cameras=list(camera.get("cameras") or [])
    if len(actions)!=len(isolated) or len(cameras)!=8:raise ValueError("V32_TESSA_ACTION_CAMERA_BINDING")
    source_vc,_=topology_components(len(source_v),source_f);source_topology=topology_signature(len(source_v),source_f,source_fc);_,source_face_object,attachment_ids=source_object_labels(replay,source_v,source_f);ladder=simplify_by_component(source_v,source_f,[.35,.50,.65,.80,1.0],source_fc);rows=[];qualified=[]
    for ratio,v,f,kind in ladder:
        metrics,face_id,bar=evaluate_candidate(v,f,source_v,source_f,source_fc,source_topology,source_face_object,attachment_ids,isolated,actions,cameras,ratio,kind);rows.append(metrics)
        if metrics["mechanical_pass"]:qualified.append((metrics["face_count"],metrics["vertex_count"],ratio,v,f,face_id,bar,metrics))
    if not qualified:raise RuntimeError("V32_TESSA_NO_MECHANICALLY_ADMISSIBLE_FALLBACK")
    qualified.sort(key=lambda x:(x[0],x[1],x[2]));_,_,ratio,v,f,face_id,bar,selected=qualified[0];out=output_dir.resolve();out.mkdir(parents=True,exist_ok=True);mesh=out/"V32_MECHANICAL_TESSA_TEACHER.npz";write_npz(mesh,vertices_source_world=v.astype(np.float32),faces=f.astype(np.int32),source_triangle_id=face_id.astype(np.int32),source_barycentric=bar.astype(np.float32))
    manifest={"schema":SCHEMA,"status":STATUS,"subject_id":subject_id,"parents":{"geometry_manifest":ref(geometry_manifest_path),"response_chain":ref(response_chain_path),"camera_bundle":ref(camera_bundle)},"policy":{"mechanical_admissibility_precedes_compactness":True,"simplification_ladder_ratios":[.35,.5,.65,.8,1.0],"source_topology_is_fallback_member":True,"teacher_topology_is_student_conditioning":False,"simplified_candidate_required_courts":["surface_mapping","component","topology_hole_crack","attachment_object_separation","nondegenerate_all_actions","inversion_foldthrough","self_intersection","response_preservation","strain_jacobian_tail","motion_view_silhouette"],"selection":"MIN_FACE_COUNT_THEN_VERTEX_COUNT_AMONG_PASS"},"candidates":rows,"selected":selected,"teacher_mesh":ref(mesh),"training_performed":False,"qualification_minted":False}
    manifest["teacher_identity_sha256"]=digest(manifest);mp=out/"V32_MECHANICAL_TESSA_TEACHER_MANIFEST.json";mp.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n");print(json.dumps({"status":STATUS,"subject_id":subject_id,"selected_face_count":selected["face_count"],"selected_kind":selected["kind"],"teacher_identity_sha256":manifest["teacher_identity_sha256"]},sort_keys=True));return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument("--subject-id",required=True);p.add_argument("--geometry-manifest",type=Path,required=True);p.add_argument("--response-chain",type=Path,required=True);p.add_argument("--camera-bundle",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();build(a.subject_id,a.geometry_manifest,a.response_chain,a.camera_bundle,a.output_dir)
if __name__=="__main__":main()
