from __future__ import annotations
import json,math,os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from compiler.realsas_compiler_core.substrate.scene_first_signed import _adaptive_voxel_compact,_self_zbuffer_support

OUT=Path(os.environ.get("REALSAS_STAGE14_CURVE_OUT","stage14_curve_out"));OUT.mkdir(parents=True,exist_ok=True)
CAPS=(1024,2048,4096,8192,12288)

def cams():
 rows=[]
 for vi in range(8):
  y=math.radians(45*vi);f=np.array([-math.sin(y),-math.cos(y),0.]);r=np.array([-math.cos(y),math.sin(y),0.])
  rows.append({"view_index":vi,"origin":tuple((-4*f).tolist()),"right":tuple(r.tolist()),"screen_up":(0,0,1),"forward":tuple(f.tolist()),"half_extent":1.0,"resolution":1024})
 return tuple(rows)

def ell(a,b,c,nlat=96,nlon=160):
 v=[(0,0,c)];n=[(0,0,1)];f=[]
 for i in range(1,nlat):
  t=math.pi*i/nlat;st,ct=math.sin(t),math.cos(t)
  for j in range(nlon):
   p=2*math.pi*j/nlon;cp,sp=math.cos(p),math.sin(p);x,y,z=a*st*cp,b*st*sp,c*ct
   v.append((x,y,z));g=np.array([x/(a*a),y/(b*b),z/(c*c)],float);g/=np.linalg.norm(g);n.append(tuple(g))
 bot=len(v);v.append((0,0,-c));n.append((0,0,-1))
 for j in range(nlon):f.append((0,1+j,1+(j+1)%nlon))
 for i in range(nlat-2):
  r0=1+i*nlon;r1=r0+nlon
  for j in range(nlon):
   a0=r0+j;a1=r0+(j+1)%nlon;b0=r1+j;b1=r1+(j+1)%nlon;f.extend(((a0,b0,b1),(a0,b1,a1)))
 last=1+(nlat-2)*nlon
 for j in range(nlon):f.append((bot,last+(j+1)%nlon,last+j))
 return np.asarray(v,float),np.asarray(f,np.int64),np.asarray(n,float)

def torus(R=.48,r=.13,nu=128,nv=80):
 v=[];n=[];f=[]
 for i in range(nu):
  u=2*math.pi*i/nu;cu,su=math.cos(u),math.sin(u)
  for j in range(nv):
   w=2*math.pi*j/nv;cw,sw=math.cos(w),math.sin(w)
   v.append(((R+r*cw)*cu,(R+r*cw)*su,r*sw));n.append((cw*cu,cw*su,sw))
 for i in range(nu):
  for j in range(nv):
   a=i*nv+j;b=((i+1)%nu)*nv+j;c=((i+1)%nu)*nv+(j+1)%nv;d=i*nv+(j+1)%nv;f.extend(((a,b,c),(a,c,d)))
 return np.asarray(v,float),np.asarray(f,np.int64),np.asarray(n,float)

def shapes():
 return {
  "SPHERE_15K":ell(.62,.62,.62),
  "TALL_ELLIPSOID_15K":ell(.28,.34,.82),
  "FLAT_ELLIPSOID_15K":ell(.78,.56,.18),
  "TORUS_CONCAVE_10K":torus(),
 }

def project(points,cam):
 p=np.asarray(points,float);d=p-np.asarray(cam["origin"]);right=np.asarray(cam["right"]);up=np.asarray(cam["screen_up"]);r=cam["resolution"]
 x=(d@right)/cam["half_extent"];y=-(d@up)/cam["half_extent"]
 return np.stack(((x+1)*.5*r-.5,(y+1)*.5*r-.5),axis=1)

def displacement(points,kind):
 p=np.asarray(points,float)
 if kind=="TWIST_Z":
  a=np.deg2rad(22)*p[:,2];ca,sa=np.cos(a),np.sin(a);o=p.copy();o[:,0]=ca*p[:,0]-sa*p[:,1];o[:,1]=sa*p[:,0]+ca*p[:,1];return o-p
 if kind=="BEND_Y":
  a=np.deg2rad(18)*p[:,2];ca,sa=np.cos(a),np.sin(a);o=p.copy();o[:,0]=ca*p[:,0]+sa*p[:,2];o[:,2]=-sa*p[:,0]+ca*p[:,2];return o-p
 if kind=="HINGE_Z":
  pivot=np.array([.15,0,0]);q=p-pivot;w=np.clip((p[:,0]-.05)/.20,0,1);a=np.deg2rad(25)*w;ca,sa=np.cos(a),np.sin(a);o=q.copy();o[:,0]=ca*q[:,0]-sa*q[:,1];o[:,1]=sa*q[:,0]+ca*q[:,1];o+=pivot;return o-p
 raise KeyError(kind)

def downstream(dense,compact,cs):
 dist,idx=cKDTree(compact).query(dense,k=min(4,len(compact)),workers=-1)
 if idx.ndim==1:idx=idx[:,None];dist=dist[:,None]
 w=1/np.maximum(dist,1e-8);w/=w.sum(1,keepdims=True);out={}
 for k in ("TWIST_Z","BEND_Y","HINGE_Z"):
  dd=displacement(dense,k);cd=displacement(compact,k);interp=np.einsum("nk,nkd->nd",w,cd[idx]);e=np.linalg.norm(interp-dd,axis=1)
  px=[]
  for c in cs:px.extend(np.linalg.norm(project(dense+interp,c)-project(dense+dd,c),axis=1))
  px=np.asarray(px,float)
  out[k]={"world_p95_norm":float(np.quantile(e,.95)),"world_max_norm":float(e.max(initial=0)),"projected_p95_px":float(np.quantile(px,.95)),"projected_max_px":float(px.max(initial=0))}
 return out

def static_metrics(dense,compact,cs,dense_support,dense_raster):
 d,_=cKDTree(compact).query(dense,k=1,workers=-1);d=np.asarray(d,float)
 support,raster,_=_self_zbuffer_support(dense,compact,cs,depth_tolerance=.02)
 vals=[];by={}
 for vi in range(8):
  dxy=dense_raster[dense_support[:,vi],vi];cxy=raster[support[:,vi],vi]
  if len(cxy):
   q,_=cKDTree(cxy).query(dxy,k=1,workers=-1);q=np.asarray(q,float);vals.extend(q.tolist());p95=float(np.quantile(q,.95));mx=float(q.max(initial=0))
  else:p95=mx=float("inf")
  by[str(vi)]={"dense_visible":int(dense_support[:,vi].sum()),"compact_support":int(support[:,vi].sum()),"p95_px":p95,"max_px":mx}
 vals=np.asarray(vals,float)
 return {"dense_to_surface_p95_norm":float(np.quantile(d,.95)),"dense_to_surface_max_norm":float(d.max(initial=0)),"projected_p95_px":float(np.quantile(vals,.95)),"projected_max_px":float(vals.max(initial=0)),"projected_by_view":by}

def main():
 cs=cams();out={"schema":"RealSaS.Stage14MechanicalCapacityCurve.v1","status":"PASS","subject_inputs_used":False,"knight_result_used":False,"mage_result_used":False,"caps":CAPS,"probe_fields":{"TWIST_Z":{"max_deg":22},"BEND_Y":{"max_deg":18},"HINGE_Z":{"max_deg":25}},"shapes":{}}
 for name,(v,f,n) in shapes().items():
  dense_support,dense_raster,_=_self_zbuffer_support(v,v,cs,depth_tolerance=.02)
  rows={}
  for cap in CAPS:
   cp,cn,e,div,inv=_adaptive_voxel_compact(v,f,n,target_nodes=cap)
   rows[str(cap)]={"actual_nodes":int(len(cp)),"divisions":int(div),"static":static_metrics(v,cp,cs,dense_support,dense_raster),"mechanical":downstream(v,cp,cs)}
  out["shapes"][name]={"dense_vertex_count":int(len(v)),"dense_face_count":int(len(f)),"curve":rows}
 (OUT/"STAGE14_MECHANICAL_CAPACITY_CURVE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":"PASS","summary":{k:{c:{"n":r["actual_nodes"],"static_p95":r["static"]["projected_p95_px"],"hinge_p95":r["mechanical"]["HINGE_Z"]["projected_p95_px"],"hinge_max":r["mechanical"]["HINGE_Z"]["projected_max_px"]} for c,r in s["curve"].items()} for k,s in out["shapes"].items()}},indent=2))
 return 0
if __name__=="__main__":raise SystemExit(main())
