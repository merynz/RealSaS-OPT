#!/usr/bin/env python3
"""Build V3.2 exact-source geometry targets for one Truth139 subject.

Exact triangle-surface distance and source normals come from the admitted rest
mesh. Global sign/occupancy is fail-closed: topology must be closed, manifold,
nondegenerate and consistently oriented, then independent direct ray-parity
probes must agree before any occupancy label is emitted. F* is never read here.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

SCHEMA="RealSaS.Truth139V32ExactGeometry.v1"
STATUS="PASS_V32_EXACT_SOURCE_GEOMETRY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8<<20),b""):h.update(c)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def matrix4(value):
    m=np.asarray(value,dtype=np.float64)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-10,rtol=0):raise ValueError("V32_GEOMETRY_SOURCE_TO_CANONICAL_INVALID")
    if np.linalg.matrix_rank(m[:3,:3])!=3:raise ValueError("V32_GEOMETRY_SOURCE_TO_CANONICAL_SINGULAR")
    return m
def transform_points(points,m):
    p=np.asarray(points,dtype=np.float64);flat=p.reshape(-1,3);h=np.concatenate([flat,np.ones((len(flat),1))],axis=1);return (h@m.T)[:,:3].reshape(p.shape)
def transform_normals(normals,m):
    n=np.asarray(normals,dtype=np.float64).reshape(-1,3);out=n@np.linalg.inv(m[:3,:3]);length=np.linalg.norm(out,axis=1);valid=length>1e-12;z=np.zeros_like(out);z[valid]=out[valid]/length[valid,None];return z.reshape(np.asarray(normals).shape)
def load_geometry(path):
    with np.load(path,allow_pickle=False) as z:v=np.asarray(z["rest_vertices_source_world"],dtype=np.float64);f=np.asarray(z["faces"],dtype=np.int64)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():raise ValueError("V32_GEOMETRY_VERTEX_LAYOUT")
    if f.ndim!=2 or f.shape[1]!=3 or not len(f) or np.any(f<0) or np.any(f>=len(v)):raise ValueError("V32_GEOMETRY_FACE_LAYOUT")
    return v,f
def face_geometry(v,f):
    tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);dbl=np.linalg.norm(cross,axis=1);valid=np.isfinite(dbl)&(dbl>1e-12);n=np.zeros_like(cross);n[valid]=cross[valid]/dbl[valid,None];return .5*dbl,n,valid
def topology(vcount,f):
    directed=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);undirected=np.sort(directed,axis=1);edges,inverse,inc=np.unique(undirected,axis=0,return_inverse=True,return_counts=True);boundary=inc==1;nonmanifold=inc>2;orient=np.ones(len(edges),dtype=np.bool_)
    for eid in np.flatnonzero(inc==2):
        ids=np.flatnonzero(inverse==eid);a,b=directed[ids[0]],directed[ids[1]];orient[eid]=bool(a[0]==b[1] and a[1]==b[0])
    parent=np.arange(vcount,dtype=np.int64)
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=int(parent[x])
        return x
    def union(a,b):
        a,b=find(int(a)),find(int(b))
        if a!=b:parent[max(a,b)]=min(a,b)
    for a,b in edges:union(a,b)
    roots=[find(i) for i in range(vcount)];lut={r:i for i,r in enumerate(sorted(set(roots)))};vc=np.asarray([lut[r] for r in roots],dtype=np.int32);fc=vc[f[:,0]]
    if np.any(vc[f]!=fc[:,None]):raise ValueError("V32_GEOMETRY_COMPONENT_INCONSISTENCY")
    return edges.astype(np.int32),inc.astype(np.int32),boundary,nonmanifold,orient,vc,fc.astype(np.int32)
def point_triangle_distance_sq(points,tri):
    p=points[:,None,:];a,b,c=tri[None,:,0,:],tri[None,:,1,:],tri[None,:,2,:];ab=b-a;ac=c-a;ap=p-a;d1=np.sum(ab*ap,-1);d2=np.sum(ac*ap,-1);bp=p-b;d3=np.sum(ab*bp,-1);d4=np.sum(ac*bp,-1);cp=p-c;d5=np.sum(ab*cp,-1);d6=np.sum(ac*cp,-1);dist=np.full(d1.shape,np.inf)
    m=(d1<=0)&(d2<=0);dist[m]=np.sum(ap*ap,-1)[m];m=(d3>=0)&(d4<=d3);dist[m]=np.minimum(dist[m],np.sum(bp*bp,-1)[m]);vc=d1*d4-d3*d2;m=(vc<=0)&(d1>=0)&(d3<=0);den=d1-d3;t=np.divide(d1,den,out=np.zeros_like(d1),where=np.abs(den)>1e-30);q=a+t[...,None]*ab;dist[m]=np.minimum(dist[m],np.sum((p-q)**2,-1)[m]);m=(d6>=0)&(d5<=d6);dist[m]=np.minimum(dist[m],np.sum(cp*cp,-1)[m]);vb=d5*d2-d1*d6;m=(vb<=0)&(d2>=0)&(d6<=0);den=d2-d6;t=np.divide(d2,den,out=np.zeros_like(d2),where=np.abs(den)>1e-30);q=a+t[...,None]*ac;dist[m]=np.minimum(dist[m],np.sum((p-q)**2,-1)[m]);va=d3*d6-d5*d4;m=(va<=0)&((d4-d3)>=0)&((d5-d6)>=0);e=c-b;den=(d4-d3)+(d5-d6);t=np.divide(d4-d3,den,out=np.zeros_like(d4),where=np.abs(den)>1e-30);q=b+t[...,None]*e;dist[m]=np.minimum(dist[m],np.sum((p-q)**2,-1)[m]);m=~np.isfinite(dist);den=va+vb+vc;vv=np.divide(vb,den,out=np.zeros_like(vb),where=np.abs(den)>1e-30);ww=np.divide(vc,den,out=np.zeros_like(vc),where=np.abs(den)>1e-30);q=a+vv[...,None]*ab+ww[...,None]*ac;fd=np.sum((p-q)**2,-1);dist[m]=fd[m];return dist
def ray_parity(points,vertices,faces,direction,chunk=32):
    tri=vertices[faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];d=np.asarray(direction,dtype=np.float64);d/=np.linalg.norm(d);h=np.cross(np.broadcast_to(d,e2.shape),e2);aa=np.sum(e1*h,axis=1);vf=np.abs(aa)>1e-10;inv=np.zeros_like(aa);inv[vf]=1/aa[vf];out=[]
    for s0 in range(0,len(points),chunk):
        p=points[s0:s0+chunk];ss=p[:,None,:]-tri[None,:,0,:];u=np.sum(ss*h[None,:,:],-1)*inv[None,:];q=np.cross(ss,e1[None,:,:]);vv=np.sum(q*d[None,None,:],-1)*inv[None,:];tt=np.sum(q*e2[None,:,:],-1)*inv[None,:];hit=vf[None,:]&(u>=-1e-10)&(vv>=-1e-10)&((u+vv)<=1+1e-10)&(tt>1e-9);out.append(np.count_nonzero(hit,axis=1)%2==1)
    return np.concatenate(out)
def sample_surface(subject,v,f,n,valid,count):
    area,_,_=face_geometry(v,f);ids=np.flatnonzero(valid);rng=np.random.default_rng(int(hashlib.sha256((subject+"|surface").encode()).hexdigest()[:16],16));prob=area[ids]/area[ids].sum();face=rng.choice(ids,size=count,replace=True,p=prob).astype(np.int32);r1=rng.random(count);r2=rng.random(count);root=np.sqrt(r1);bar=np.stack([1-root,root*(1-r2),root*r2],axis=1);xyz=np.sum(v[f[face]]*bar[...,None],axis=1);return face,bar.astype(np.float32),xyz.astype(np.float32),n[face].astype(np.float32)
def write_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".partial.npz");np.savez_compressed(tmp,**arrays);tmp.replace(path)

def build(subject_id,geometry_npz,snapshot_json,camera_json,output_dir,surface_samples=32768,query_samples=8192):
    geometry_npz=geometry_npz.resolve();snapshot_json=snapshot_json.resolve();camera_json=camera_json.resolve();output_dir=output_dir.resolve();snapshot=json.loads(snapshot_json.read_text());vsw,faces=load_geometry(geometry_npz);m=matrix4(snapshot["source_world_to_canonical"]);v=transform_points(vsw,m);area,nsw,face_valid=face_geometry(vsw,faces);n=transform_normals(nsw,m);edges,inc,boundary,nonmanifold,orient,vc,fc=topology(len(v),faces);orientation_valid=bool(np.all(orient[inc==2]));topology_sign_valid=bool(not np.any(boundary) and not np.any(nonmanifold) and orientation_valid and np.all(face_valid));sface,sbar,sxyz,snormal=sample_surface(subject_id,v,faces,n,face_valid,int(surface_samples));rng=np.random.default_rng(int(hashlib.sha256((subject_id+"|query").encode()).hexdigest()[:16],16));near_n=query_samples//2;ids=rng.integers(0,len(sxyz),size=near_n);diag=float(np.linalg.norm(v.max(0)-v.min(0)));offset=rng.uniform(-.03,.03,size=near_n)*max(diag,1e-6);near=sxyz[ids].astype(np.float64)+snormal[ids].astype(np.float64)*offset[:,None];lo=np.minimum(v.min(0),[-1,-1,-1]);hi=np.maximum(v.max(0),[1,1,1]);query=np.concatenate([near,rng.uniform(lo,hi,size=(query_samples-near_n,3))]).astype(np.float64)
    try:
        import trimesh
    except ImportError as e:raise RuntimeError("V32_GEOMETRY_TRIMESH_RTREE_REQUIRED") from e
    mesh=trimesh.Trimesh(vertices=v,faces=faces,process=False);closest,distance,closest_face=trimesh.proximity.closest_point(mesh,query);distance=np.asarray(distance,dtype=np.float32);closest_face=np.asarray(closest_face,dtype=np.int32)
    sign_gate=False
    if topology_sign_valid:
        probe_idx=np.linspace(0,len(query)-1,min(192,len(query)),dtype=np.int64);dirs=([1,.173,.319],[.271,1,.113],[.197,.233,1]);par=np.stack([ray_parity(query[probe_idx],v,faces,np.asarray(d)) for d in dirs],axis=1);sign_gate=bool(np.all(par==par[:,:1]))
    sign_valid=np.full(len(query),bool(topology_sign_valid and sign_gate),dtype=np.bool_);occupancy=np.zeros(len(query),dtype=np.bool_)
    if np.any(sign_valid):occupancy[:]=np.asarray(mesh.contains(query),dtype=np.bool_)
    signed=distance.astype(np.float32).copy();signed[occupancy]*=-1;signed[~sign_valid]=0
    normal_valid=face_valid & orientation_valid;topology_path=output_dir/"V32_EXACT_GEOMETRY_TOPOLOGY.npz";write_npz(topology_path,rest_vertices_source_world=vsw.astype(np.float32),rest_vertices_canonical=v.astype(np.float32),faces=faces.astype(np.int32),face_normal_source_world=nsw.astype(np.float32),face_normal_canonical=n.astype(np.float32),face_normal_valid=normal_valid.astype(np.bool_),face_area_source_world=area.astype(np.float32),nondegenerate_face=face_valid.astype(np.bool_),unique_edge_index=edges,edge_face_incidence_count=inc,boundary_edge=boundary.astype(np.bool_),nonmanifold_edge=nonmanifold.astype(np.bool_),edge_orientation_consistent=orient.astype(np.bool_),vertex_component_id=vc,face_component_id=fc)
    surface_path=output_dir/"V32_EXACT_SURFACE_SAMPLES.npz";write_npz(surface_path,triangle_id=sface,barycentric=sbar,surface_xyz_canonical=sxyz,normal_canonical=snormal,normal_valid=normal_valid[sface].astype(np.bool_));query_path=output_dir/"V32_EXACT_DISTANCE_SIGN_TARGETS.npz";write_npz(query_path,query_xyz_canonical=query.astype(np.float32),unsigned_distance=distance,closest_face=closest_face,sign_valid=sign_valid,occupancy=occupancy,signed_distance=signed)
    payload={"schema":SCHEMA,"status":STATUS,"subject_id":subject_id,"parents":{"geometry_npz":{"path":str(geometry_npz),"sha256":sha256(geometry_npz)},"source_scene_snapshot":{"path":str(snapshot_json),"sha256":sha256(snapshot_json)},"camera_bundle":{"path":str(camera_json),"sha256":sha256(camera_json)},"source_sha256":snapshot.get("source_sha256")},"outputs":{"topology":{"path":str(topology_path),"sha256":sha256(topology_path)},"surface":{"path":str(surface_path),"sha256":sha256(surface_path)},"distance_sign":{"path":str(query_path),"sha256":sha256(query_path)}},"counts":{"vertex_count":int(len(v)),"face_count":int(len(faces)),"component_count":int(len(np.unique(vc))),"boundary_edge_count":int(np.count_nonzero(boundary)),"nonmanifold_edge_count":int(np.count_nonzero(nonmanifold)),"surface_sample_count":int(surface_samples),"query_sample_count":int(query_samples),"sign_valid_query_count":int(np.count_nonzero(sign_valid))},"validity":{"orientation_consistent":orientation_valid,"global_sign_topology_valid":topology_sign_valid,"independent_parity_probe_gate":sign_gate,"global_sign_policy":"FAIL_CLOSED_ALL_COMPONENTS_REQUIRED"},"authority":{"fstar_used":False,"surface_distance_authority":"EXACT_SOURCE_TRIANGLE_MESH","global_sign_fabricated_for_invalid_mesh":False,"training_performed":False,"qualification_minted":False}}
    payload["geometry_identity_sha256"]=digest(payload);manifest=output_dir/"V32_EXACT_GEOMETRY_MANIFEST.json";manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+"\n");print(json.dumps({"status":STATUS,"subject_id":subject_id,"sign_valid_query_count":payload["counts"]["sign_valid_query_count"],"geometry_identity_sha256":payload["geometry_identity_sha256"]},sort_keys=True));return payload

def main():
    p=argparse.ArgumentParser();p.add_argument("--subject-id",required=True);p.add_argument("--geometry-npz",type=Path,required=True);p.add_argument("--source-scene-snapshot",type=Path,required=True);p.add_argument("--camera-bundle",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--surface-samples",type=int,default=32768);p.add_argument("--query-samples",type=int,default=8192);a=p.parse_args();build(a.subject_id,a.geometry_npz,a.source_scene_snapshot,a.camera_bundle,a.output_dir,a.surface_samples,a.query_samples)
if __name__=="__main__":main()
