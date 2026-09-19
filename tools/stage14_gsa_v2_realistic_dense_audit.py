from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.substrate.adequacy_v1 import select_adequate_rigging_surface_v1
from compiler.realsas_compiler_core.substrate.scene_first_signed import rigging_surface_from_scene_first_zero_mesh_v1

OUT=Path(os.environ.get("REALSAS_STAGE14_REAL_OUT","stage14_real_out")); OUT.mkdir(parents=True,exist_ok=True)

BASE={"min_candidate_nodes":128,"max_candidate_nodes":8192,"candidate_growth_factor":2.0,"refinement_rounds":3}
PROFILES=[
 {"profile_id":"CURRENT_V1","max_dense_to_surface_p95_norm":.03,"max_dense_to_surface_max_norm":.08,"max_normal_p95_deg":30.0,"max_projected_p95_px":6.0,"max_projected_max_px":16.0,"component_min_dense_fraction":.001,"min_nodes_per_component":8,"max_component_alias_nodes":0},
 {"profile_id":"TIGHT_A","max_dense_to_surface_p95_norm":.020,"max_dense_to_surface_max_norm":.050,"max_normal_p95_deg":25.0,"max_projected_p95_px":4.0,"max_projected_max_px":10.0,"component_min_dense_fraction":.0005,"min_nodes_per_component":12,"max_component_alias_nodes":0},
 {"profile_id":"TIGHT_B","max_dense_to_surface_p95_norm":.015,"max_dense_to_surface_max_norm":.040,"max_normal_p95_deg":20.0,"max_projected_p95_px":3.0,"max_projected_max_px":8.0,"component_min_dense_fraction":.00025,"min_nodes_per_component":16,"max_component_alias_nodes":0},
 {"profile_id":"TIGHT_C","max_dense_to_surface_p95_norm":.010,"max_dense_to_surface_max_norm":.030,"max_normal_p95_deg":15.0,"max_projected_p95_px":2.0,"max_projected_max_px":6.0,"component_min_dense_fraction":.0001,"min_nodes_per_component":24,"max_component_alias_nodes":0},
]

def cams():
 rows=[]
 for vi in range(8):
  y=math.radians(45*vi); f=np.array([-math.sin(y),-math.cos(y),0.]); r=np.array([-math.cos(y),math.sin(y),0.])
  rows.append({"view_index":vi,"origin":tuple((-4*f).tolist()),"right":tuple(r.tolist()),"screen_up":(0,0,1),"forward":tuple(f.tolist()),"half_extent":1.0,"resolution":1024})
 return tuple(rows)

def ell(a,b,c,nlat,nlon,center=(0,0,0)):
 cx,cy,cz=center; v=[(cx,cy,cz+c)]; n=[(0,0,1)]
 for i in range(1,nlat):
  t=math.pi*i/nlat; st,ct=math.sin(t),math.cos(t)
  for j in range(nlon):
   p=2*math.pi*j/nlon; cp,sp=math.cos(p),math.sin(p)
   x,y,z=a*st*cp,b*st*sp,c*ct; v.append((cx+x,cy+y,cz+z))
   g=np.array([x/(a*a),y/(b*b),z/(c*c)]); g/=np.linalg.norm(g); n.append(tuple(g))
 bot=len(v); v.append((cx,cy,cz-c)); n.append((0,0,-1)); f=[]
 for j in range(nlon): f.append((0,1+j,1+(j+1)%nlon))
 for i in range(nlat-2):
  r0=1+i*nlon; r1=r0+nlon
  for j in range(nlon):
   a0=r0+j;a1=r0+(j+1)%nlon;b0=r1+j;b1=r1+(j+1)%nlon
   f.extend(((a0,b0,b1),(a0,b1,a1)))
 last=1+(nlat-2)*nlon
 for j in range(nlon):f.append((bot,last+(j+1)%nlon,last+j))
 return np.asarray(v,np.float32),np.asarray(f,np.int64),np.asarray(n,np.float32)

def torus(R=.48,r=.13,nu=128,nv=80):
 v=[];n=[];f=[]
 for i in range(nu):
  u=2*math.pi*i/nu;cu,su=math.cos(u),math.sin(u)
  for j in range(nv):
   w=2*math.pi*j/nv;cw,sw=math.cos(w),math.sin(w)
   v.append(((R+r*cw)*cu,(R+r*cw)*su,r*sw));n.append((cw*cu,cw*su,sw))
 for i in range(nu):
  for j in range(nv):
   a=i*nv+j;b=((i+1)%nu)*nv+j;c=((i+1)%nu)*nv+(j+1)%nv;d=i*nv+(j+1)%nv
   f.extend(((a,b,c),(a,c,d)))
 return np.asarray(v,np.float32),np.asarray(f,np.int64),np.asarray(n,np.float32)

def combine(parts):
 vs=[];fs=[];ns=[];off=0
 for v,f,n in parts:vs.append(v);fs.append(f+off);ns.append(n);off+=len(v)
 return np.concatenate(vs),np.concatenate(fs),np.concatenate(ns)

def shapes():
 return {
  "SPHERE_15K":ell(.62,.62,.62,96,160),
  "TALL_ELLIPSOID_15K":ell(.28,.34,.82,96,160),
  "TORUS_CONCAVE_10K":torus(),
  "TWO_CLOSE_COMPONENTS_12K":combine([ell(.31,.38,.52,64,96,(-.17,0,0)),ell(.20,.24,.34,64,96,(.19,0,.03))]),
  "THIN_APPENDAGE_11K":combine([ell(.42,.38,.55,72,120,(-.12,0,0)),ell(.07,.065,.55,64,48,(.50,0,.03))]),
 }

def ppass(m,p):
 return m["dense_to_surface_p95_norm"]<=p["max_dense_to_surface_p95_norm"] and m["dense_to_surface_max_norm"]<=p["max_dense_to_surface_max_norm"] and m["normal_p95_deg"]<=p["max_normal_p95_deg"] and m["projected_p95_px"]<=p["max_projected_p95_px"] and m["projected_max_px"]<=p["max_projected_max_px"] and m["minimum_nodes_per_eligible_component"]>=p["min_nodes_per_component"] and m["component_alias_node_count"]<=p["max_component_alias_nodes"]

def select(rows,p):
 ok=[m for m in rows if ppass(m,p)]
 if not ok:return None
 return int(min(ok,key=lambda m:(m["actual_node_count"],m["candidate_target_node_cap"]))["candidate_target_node_cap"])

def disp(p,k):
 p=np.asarray(p,float)
 if k=="TWIST_Z":
  a=np.deg2rad(22)*p[:,2];ca,sa=np.cos(a),np.sin(a);o=p.copy();o[:,0]=ca*p[:,0]-sa*p[:,1];o[:,1]=sa*p[:,0]+ca*p[:,1];return o-p
 if k=="BEND_Y":
  a=np.deg2rad(18)*p[:,2];ca,sa=np.cos(a),np.sin(a);o=p.copy();o[:,0]=ca*p[:,0]+sa*p[:,2];o[:,2]=-sa*p[:,0]+ca*p[:,2];return o-p
 q=p-np.array([.15,0,0]);w=np.clip((p[:,0]-.05)/.20,0,1);a=np.deg2rad(25)*w;ca,sa=np.cos(a),np.sin(a);o=q.copy();o[:,0]=ca*q[:,0]-sa*q[:,1];o[:,1]=sa*q[:,0]+ca*q[:,1];o+=np.array([.15,0,0]);return o-p

def proj(p,c):
 d=np.asarray(p)-np.asarray(c["origin"]);x=(d@np.asarray(c["right"]))/c["half_extent"];y=-(d@np.asarray(c["screen_up"]))/c["half_extent"];r=c["resolution"];return np.stack(((x+1)*.5*r-.5,(y+1)*.5*r-.5),1)

def proxy(dense,surface,cs):
 nodes=np.asarray([n.P for n in surface.surface_nodes],float);dist,idx=cKDTree(nodes).query(dense,k=min(4,len(nodes)),workers=-1)
 if idx.ndim==1:idx=idx[:,None];dist=dist[:,None]
 w=1/np.maximum(dist,1e-8);w/=w.sum(1,keepdims=True);out={}
 for k in ("TWIST_Z","BEND_Y","HINGE_Z"):
  dd=disp(dense,k);nd=disp(nodes,k);ii=np.einsum("nk,nkd->nd",w,nd[idx]);e=np.linalg.norm(ii-dd,axis=1);px=[]
  for c in cs:px.extend(np.linalg.norm(proj(dense+ii,c)-proj(dense+dd,c),axis=1))
  px=np.asarray(px)
  out[k]={"world_p95_norm":float(np.quantile(e,.95)),"world_max_norm":float(e.max()),"projected_p95_px":float(np.quantile(px,.95)),"projected_max_px":float(px.max())}
 return out

def main():
 cs=cams(); loose={**BASE,"max_dense_to_surface_p95_norm":2,"max_dense_to_surface_max_norm":2,"max_normal_p95_deg":180,"max_projected_p95_px":2048,"max_projected_max_px":2048,"component_min_dense_fraction":0.0,"min_nodes_per_component":1,"max_component_alias_nodes":10**9}
 out={"schema":"RealSaS.Stage14GSAV2RealisticDenseAudit.v1","status":"PASS","subject_inputs_used":False,"knight_result_used":False,"mage_result_used":False,"dense_vertex_floor":10000,"profiles":PROFILES,"shapes":{}}
 for name,(v,f,n) in shapes().items():
  _,report=select_adequate_rigging_surface_v1(v,f,n,cs,normalization_center=(0,0,0),normalization_half_extent=1.0,authority_label="SUBJECT_FREE_ANALYTIC",source_run_id="STAGE14_V2_REAL",source_checkpoint_sha256="0"*64,source_zero_surface_sha256="1"*64,normal_k=64,visibility_depth_tolerance_norm=.02,adequacy_policy=loose,metadata={"shape":name})
  caps={p["profile_id"]:select(report["evaluated_candidates"],p) for p in PROFILES};unique=sorted({c for c in caps.values() if c is not None});surf={}
  for cap in unique:
   s=rigging_surface_from_scene_first_zero_mesh_v1(v,f,n,cs,normalization_center=(0,0,0),normalization_half_extent=1.0,authority_label="SUBJECT_FREE_ANALYTIC",source_run_id="STAGE14_V2_REAL",source_checkpoint_sha256="0"*64,source_zero_surface_sha256="1"*64,target_nodes=cap,normal_k=64,visibility_depth_tolerance_norm=.02,metadata={"shape":name,"cap":cap})
   surf[cap]={"actual_node_count":len(s.surface_nodes),"downstream_proxy":proxy(np.asarray(v,float),s,cs)}
  out["shapes"][name]={"dense_vertex_count":len(v),"dense_face_count":len(f),"selected_caps":caps,"surface_proxy_by_cap":surf,"candidate_metrics":report["evaluated_candidates"]}
 (OUT/"STAGE14_GSA_V2_REALISTIC_DENSE_AUDIT.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":"PASS","summary":{k:v["selected_caps"] for k,v in out["shapes"].items()}},indent=2))
 return 0
if __name__=="__main__":raise SystemExit(main())
