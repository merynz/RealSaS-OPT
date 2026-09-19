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

def main():
 cs=cams();out={"schema":"RealSaS.Stage14SurfaceResidentRepresentativeCausalProbe.v1","status":"PASS","single_changed_variable":"VOXEL_REPRESENTATIVE_CENTROID_TO_NEAREST_DENSE_ZERO_SURFACE_VERTEX","target_nodes":8192,"subject_inputs_used":False,"knight_result_used":False,"mage_result_used":False,"shapes":{}}
 for name,args in {"SPHERE_15K":(.62,.62,.62),"TALL_ELLIPSOID_15K":(.28,.34,.82)}.items():
  p,f,n=ell(*args);cent,cn,e,div,inv=_adaptive_voxel_compact(p,f,n,target_nodes=8192)
  ids=medoids(p,cent,inv);med=p[ids]
  b=projected_metrics(p,cent,cs);m=projected_metrics(p,med,cs)
  row={"dense_vertex_count":len(p),"actual_nodes":len(cent),"divisions":div,
       "baseline_centroid":{"spatial":spatial(p,cent),"projected":b},
       "surface_medoid":{"spatial":spatial(p,med),"projected":m},
       "projected_p95_improvement_fraction":float((b["p95_px"]-m["p95_px"])/b["p95_px"]),
       "support_gain_by_view":{str(v):m["by_view"][str(v)]["compact_support"]-b["by_view"][str(v)]["compact_support"] for v in range(8)}}
  out["shapes"][name]=row
  if not (m["p95_px"]<b["p95_px"] and m["max_px"]<b["max_px"]):out["status"]="FAIL_NO_CAUSAL_IMPROVEMENT"
 (OUT/"STAGE14_SURFACE_RESIDENT_REPRESENTATIVE_CAUSAL_PROBE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps(out,indent=2,sort_keys=True));return 0 if out["status"]=="PASS" else 2
if __name__=="__main__":raise SystemExit(main())
