from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.substrate.adequacy_v1 import select_adequate_rigging_surface_v1

OUT=Path(os.environ.get("REALSAS_STAGE14_V2_OUT","stage14_v2_out")); OUT.mkdir(parents=True,exist_ok=True)

POLICY={
 "min_candidate_nodes":128,
 "max_candidate_nodes":12288,
 "candidate_growth_factor":2.0,
 "refinement_rounds":3,
 "max_dense_to_surface_p95_norm":0.02,
 "max_dense_to_surface_max_norm":0.04,
 "max_normal_p95_deg":15.0,
 "max_projected_p95_px":16.0,
 "max_projected_max_px":24.0,
 "component_min_dense_fraction":0.001,
 "min_nodes_per_component":8,
 "max_component_alias_nodes":0,
 "visible_component_always_eligible":True,
 "component_aware_voxel_compaction":True,
 "mechanical_probe_enabled":True,
 "max_mechanical_probe_p95_norm":0.003,
 "max_mechanical_probe_max_norm":0.012
}

def cams():
 rows=[]
 for vi in range(8):
  y=math.radians(45*vi); f=np.array([-math.sin(y),-math.cos(y),0.]); r=np.array([-math.cos(y),math.sin(y),0.])
  rows.append({"view_index":vi,"origin":tuple((-4*f).tolist()),"right":tuple(r.tolist()),"screen_up":(0,0,1),"forward":tuple(f.tolist()),"half_extent":1.0,"resolution":1024})
 return tuple(rows)

def ell(a,b,c,nlat,nlon,center=(0,0,0)):
 cx,cy,cz=center; v=[(cx,cy,cz+c)]; n=[(0,0,1)]; f=[]
 for i in range(1,nlat):
  t=math.pi*i/nlat; st,ct=math.sin(t),math.cos(t)
  for j in range(nlon):
   p=2*math.pi*j/nlon; cp,sp=math.cos(p),math.sin(p); x,y,z=a*st*cp,b*st*sp,c*ct
   v.append((cx+x,cy+y,cz+z))
   g=np.array([x/(a*a),y/(b*b),z/(c*c)],float); g/=np.linalg.norm(g); n.append(tuple(g))
 bot=len(v); v.append((cx,cy,cz-c)); n.append((0,0,-1))
 for j in range(nlon): f.append((0,1+j,1+(j+1)%nlon))
 for i in range(nlat-2):
  r0=1+i*nlon; r1=r0+nlon
  for j in range(nlon):
   a0=r0+j;a1=r0+(j+1)%nlon;b0=r1+j;b1=r1+(j+1)%nlon;f.extend(((a0,b0,b1),(a0,b1,a1)))
 last=1+(nlat-2)*nlon
 for j in range(nlon): f.append((bot,last+(j+1)%nlon,last+j))
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
 for v,f,n in parts:
  vs.append(v);fs.append(f+off);ns.append(n);off+=len(v)
 return np.concatenate(vs),np.concatenate(fs),np.concatenate(ns)

def shapes():
 return {
  "SPHERE_15K":ell(.62,.62,.62,96,160),
  "TALL_ELLIPSOID_15K":ell(.28,.34,.82,96,160),
  "FLAT_ELLIPSOID_15K":ell(.78,.56,.18,96,160),
  "TORUS_CONCAVE_10K":torus(),
  "TWO_CLOSE_COMPONENTS_12K":combine([ell(.31,.38,.52,64,96,(-.17,0,0)),ell(.20,.24,.34,64,96,(.19,0,.03))]),
  "THIN_APPENDAGE_11K":combine([ell(.42,.38,.55,72,120,(-.12,0,0)),ell(.07,.065,.55,64,48,(.50,0,.03))]),
 }

def main():
 cs=cams()
 out={"schema":"RealSaS.Stage14V2SubjectFreePolicyCalibration.v1","status":"PASS","subject_inputs_used":False,"knight_result_used":False,"mage_result_used":False,"policy":POLICY,"shapes":{}}
 for name,(v,f,n) in shapes().items():
  surface,report=select_adequate_rigging_surface_v1(
   v,f,n,cs,
   normalization_center=(0,0,0),normalization_half_extent=1.0,
   authority_label="SUBJECT_FREE_STAGE14_V2",source_run_id="STAGE14_V2_CALIBRATION",
   source_checkpoint_sha256="0"*64,source_zero_surface_sha256="1"*64,
   normal_k=64,visibility_depth_tolerance_norm=.02,adequacy_policy=POLICY,
   metadata={"subject_free_shape":name},
  )
  if surface is None or report["status"]!="PASS":
   out["status"]="FAIL_SUBJECT_FREE_SHAPE"
  sel=None
  if report["selected_target_node_cap"] is not None:
   sel=next(x for x in report["evaluated_candidates"] if x["candidate_target_node_cap"]==report["selected_target_node_cap"])
   if sel["component_alias_node_count"]!=0: out["status"]="FAIL_ALIAS"
   if sel["mechanical_probe"] is None: out["status"]="FAIL_MECHANICAL_PROBE_MISSING"
   if sel["visible_dense_component_count"]>sel["eligible_dense_component_count"]: out["status"]="FAIL_VISIBLE_COMPONENT_ESCAPE"
  out["shapes"][name]={
   "dense_vertex_count":int(len(v)),"dense_face_count":int(len(f)),
   "selected_target_node_cap":report["selected_target_node_cap"],
   "selected_actual_node_count":report["selected_actual_node_count"],
   "selected_metrics":sel,
   "evaluated_candidates":report["evaluated_candidates"],
   "adequacy_report_hash":report["adequacy_report_hash"],
  }
 (OUT/"STAGE14_V2_SUBJECT_FREE_POLICY_CALIBRATION.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":out["status"],"selected":{k:{"cap":v["selected_target_node_cap"],"actual":v["selected_actual_node_count"],"mechanical":None if v["selected_metrics"] is None else v["selected_metrics"]["mechanical_probe"]} for k,v in out["shapes"].items()}},indent=2,sort_keys=True))
 return 0 if out["status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
