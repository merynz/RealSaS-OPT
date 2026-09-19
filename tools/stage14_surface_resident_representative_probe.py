from __future__ import annotations
import json,math,os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from compiler.realsas_compiler_core.substrate.scene_first_signed import _adaptive_voxel_compact,_self_zbuffer_support

OUT=Path(os.environ.get("REALSAS_STAGE14_MEDOID_OUT","stage14_medoid_out"));OUT.mkdir(parents=True,exist_ok=True)

def cams():
 r=[]
 for vi in range(8):
  y=math.radians(45*vi);f=np.array([-math.sin(y),-math.cos(y),0.]);right=np.array([-math.cos(y),math.sin(y),0.])
  r.append({"view_index":vi,"origin":tuple((-4*f).tolist()),"right":tuple(right.tolist()),"screen_up":(0,0,1),"forward":tuple(f.tolist()),"half_extent":1.0,"resolution":1024})
 return tuple(r)

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
   a0=r0+j;a1=r0+(j+1)%nlon;b0=r1+j;b1=r1+(j+1)%nlon
   f.extend(((a0,b0,b1),(a0,b1,a1)))
 last=1+(nlat-2)*nlon
 for j in range(nlon):f.append((bot,last+(j+1)%nlon,last+j))
 return np.asarray(v,float),np.asarray(f,np.int64),np.asarray(n,float)

def medoids(p,centroids,inv):
 d2=np.sum((p-centroids[inv])**2,axis=1)
 order=np.lexsort((np.arange(len(p)),d2,inv))
 chosen=[];seen=set()
 for idx in order:
  g=int(inv[idx])
  if g not in seen:seen.add(g);chosen.append(int(idx))
 chosen=np.asarray(chosen,np.int64)
 if len(chosen)!=len(centroids):raise RuntimeError("medoid cardinality")
 # chosen order follows sorted group because lexsort inv primary; verify/reorder by inv.
 gids=inv[chosen];re=np.argsort(gids,kind="stable");chosen=chosen[re]
 if not np.array_equal(inv[chosen],np.arange(len(centroids))):raise RuntimeError("medoid group drift")
 return chosen

def projected_metrics(dense,compact,cs):
 ds,dr,_=_self_zbuffer_support(dense,dense,cs,depth_tolerance=.02)
 csup,cr,_=_self_zbuffer_support(dense,compact,cs,depth_tolerance=.02)
 by={};allv=[]
 for vi in range(8):
  dxy=dr[np.asarray(ds[:,vi],bool),vi]
  cxy=cr[np.asarray(csup[:,vi],bool),vi]
  if len(cxy)==0: p95=mx=float("inf")
  else:
   dist,_=cKDTree(cxy).query(dxy,k=1,workers=-1);dist=np.asarray(dist,float);p95=float(np.quantile(dist,.95));mx=float(dist.max(initial=0));allv.extend(dist.tolist())
  by[str(vi)]={"dense_visible":int(ds[:,vi].sum()),"compact_support":int(csup[:,vi].sum()),"p95_px":p95,"max_px":mx}
 a=np.asarray(allv,float)
 return {"p95_px":float(np.quantile(a,.95)),"max_px":float(a.max(initial=0)),"by_view":by}

def spatial(dense,compact):
 d,_=cKDTree(compact).query(dense,k=1,workers=-1);d=np.asarray(d,float)
 return {"p95_norm":float(np.quantile(d,.95)),"max_norm":float(d.max(initial=0))}


def displacement(points,kind):
 p=np.asarray(points,float)
 if kind=="TWIST_Z":
  a=np.deg2rad(22.0)*p[:,2];ca,sa=np.cos(a),np.sin(a);o=p.copy();o[:,0]=ca*p[:,0]-sa*p[:,1];o[:,1]=sa*p[:,0]+ca*p[:,1];return o-p
 if kind=="BEND_Y":
  a=np.deg2rad(18.0)*p[:,2];ca,sa=np.cos(a),np.sin(a);o=p.copy();o[:,0]=ca*p[:,0]+sa*p[:,2];o[:,2]=-sa*p[:,0]+ca*p[:,2];return o-p
 if kind=="HINGE_Z":
  pivot=np.array([.15,0,0]);q=p-pivot;w=np.clip((p[:,0]-.05)/.20,0,1);a=np.deg2rad(25.0)*w;ca,sa=np.cos(a),np.sin(a);o=q.copy();o[:,0]=ca*q[:,0]-sa*q[:,1];o[:,1]=sa*q[:,0]+ca*q[:,1];o+=pivot;return o-p
 raise KeyError(kind)

def project(points,cam):
 p=np.asarray(points,float);origin=np.asarray(cam["origin"]);right=np.asarray(cam["right"]);up=np.asarray(cam["screen_up"]);d=p-origin
 x=(d@right)/cam["half_extent"];y=-(d@up)/cam["half_extent"];r=cam["resolution"]
 return np.stack(((x+1)*.5*r-.5,(y+1)*.5*r-.5),axis=1)

def downstream_proxy(dense,compact,cs):
 k=min(4,len(compact));dist,idx=cKDTree(compact).query(dense,k=k,workers=-1)
 if k==1:dist=dist[:,None];idx=idx[:,None]
 w=1.0/np.maximum(dist,1e-8);w/=w.sum(axis=1,keepdims=True);out={}
 for kind in ("TWIST_Z","BEND_Y","HINGE_Z"):
  dd=displacement(dense,kind);cd=displacement(compact,kind);interp=np.einsum("nk,nkd->nd",w,cd[idx]);err=np.linalg.norm(interp-dd,axis=1)
  true=dense+dd;pred=dense+interp;px=[]
  for cam in cs:px.extend(np.linalg.norm(project(pred,cam)-project(true,cam),axis=1).tolist())
  px=np.asarray(px,float)
  out[kind]={"world_p95_norm":float(np.quantile(err,.95)),"world_max_norm":float(err.max(initial=0)),"projected_p95_px":float(np.quantile(px,.95)),"projected_max_px":float(px.max(initial=0))}
 return out
def main():
 cs=cams();out={"schema":"RealSaS.Stage14SurfaceResidentRepresentativeCausalProbe.v2","status":"PASS","single_changed_variable":"VOXEL_REPRESENTATIVE_CENTROID_TO_NEAREST_DENSE_ZERO_SURFACE_VERTEX","target_nodes":8192,"subject_inputs_used":False,"knight_result_used":False,"mage_result_used":False,"shapes":{}}
 for name,args in {"SPHERE_15K":(.62,.62,.62),"TALL_ELLIPSOID_15K":(.28,.34,.82)}.items():
  p,f,n=ell(*args);cent,cn,e,div,inv=_adaptive_voxel_compact(p,f,n,target_nodes=8192)
  ids=medoids(p,cent,inv);med=p[ids]
  b=projected_metrics(p,cent,cs);m=projected_metrics(p,med,cs)
  row={"dense_vertex_count":len(p),"actual_nodes":len(cent),"divisions":div,
       "baseline_centroid":{"spatial":spatial(p,cent),"projected":b,"downstream_proxy":downstream_proxy(p,cent,cs)},
       "surface_medoid":{"spatial":spatial(p,med),"projected":m,"downstream_proxy":downstream_proxy(p,med,cs)},
       "projected_p95_improvement_fraction":float((b["p95_px"]-m["p95_px"])/b["p95_px"]),
       "support_gain_by_view":{str(v):m["by_view"][str(v)]["compact_support"]-b["by_view"][str(v)]["compact_support"] for v in range(8)}}
  out["shapes"][name]=row
  if not (m["p95_px"]<b["p95_px"] and m["max_px"]<b["max_px"]):out["status"]="FAIL_NO_CAUSAL_IMPROVEMENT"
 (OUT/"STAGE14_SURFACE_RESIDENT_REPRESENTATIVE_CAUSAL_PROBE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps(out,indent=2,sort_keys=True));return 0 if out["status"]=="PASS" else 2
if __name__=="__main__":raise SystemExit(main())
