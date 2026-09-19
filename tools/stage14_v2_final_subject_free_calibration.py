from __future__ import annotations

import json, math, os
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.substrate.adequacy_v1 import select_adequate_rigging_surface_v1

OUT=Path(os.environ.get("REALSAS_STAGE14_V2_FINAL_OUT","stage14_v2_final_out"))
OUT.mkdir(parents=True,exist_ok=True)

POLICY={
  "min_candidate_nodes":128,
  "max_candidate_nodes":12288,
  "candidate_growth_factor":2.0,
  "refinement_rounds":3,
  "max_dense_to_surface_p95_norm":0.03,
  "max_dense_to_surface_max_norm":0.08,
  "max_normal_p95_deg":30.0,
  "max_projected_p95_px":6.0,
  "max_projected_max_px":16.0,
  "component_min_dense_fraction":0.001,
  "min_nodes_per_component":8,
  "max_component_alias_nodes":0,
  "visible_component_always_eligible":True,
  "component_aware_voxel_compaction":True,
}

def cameras():
    rows=[]
    for vi in range(8):
        yaw=math.radians(45.0*vi)
        forward=np.asarray([-math.sin(yaw),-math.cos(yaw),0.0],dtype=float)
        right=np.asarray([-math.cos(yaw),math.sin(yaw),0.0],dtype=float)
        rows.append({
            "view_index":vi,
            "origin":tuple((-4.0*forward).tolist()),
            "right":tuple(right.tolist()),
            "screen_up":(0.0,0.0,1.0),
            "forward":tuple(forward.tolist()),
            "half_extent":1.0,
            "resolution":1024,
        })
    return tuple(rows)

def ellipsoid(a,b,c,nlat,nlon,center=(0,0,0)):
    cx,cy,cz=center
    vertices=[(cx,cy,cz+c)]
    normals=[(0.0,0.0,1.0)]
    for i in range(1,nlat):
        th=math.pi*i/nlat
        st,ct=math.sin(th),math.cos(th)
        for j in range(nlon):
            ph=2.0*math.pi*j/nlon
            cp,sp=math.cos(ph),math.sin(ph)
            x,y,z=a*st*cp,b*st*sp,c*ct
            vertices.append((cx+x,cy+y,cz+z))
            g=np.asarray([x/(a*a),y/(b*b),z/(c*c)],dtype=float)
            g/=np.linalg.norm(g)
            normals.append(tuple(g))
    bottom=len(vertices)
    vertices.append((cx,cy,cz-c))
    normals.append((0.0,0.0,-1.0))
    faces=[]
    for j in range(nlon):
        faces.append((0,1+j,1+(j+1)%nlon))
    for i in range(nlat-2):
        r0=1+i*nlon
        r1=r0+nlon
        for j in range(nlon):
            a0=r0+j; a1=r0+(j+1)%nlon
            b0=r1+j; b1=r1+(j+1)%nlon
            faces.extend(((a0,b0,b1),(a0,b1,a1)))
    last=1+(nlat-2)*nlon
    for j in range(nlon):
        faces.append((bottom,last+(j+1)%nlon,last+j))
    return np.asarray(vertices,np.float32),np.asarray(faces,np.int64),np.asarray(normals,np.float32)

def torus(R=.48,r=.13,nu=128,nv=80):
    vertices=[]; normals=[]; faces=[]
    for i in range(nu):
        u=2.0*math.pi*i/nu
        cu,su=math.cos(u),math.sin(u)
        for j in range(nv):
            v=2.0*math.pi*j/nv
            cv,sv=math.cos(v),math.sin(v)
            vertices.append(((R+r*cv)*cu,(R+r*cv)*su,r*sv))
            normals.append((cv*cu,cv*su,sv))
    for i in range(nu):
        for j in range(nv):
            a=i*nv+j
            b=((i+1)%nu)*nv+j
            c=((i+1)%nu)*nv+(j+1)%nv
            d=i*nv+(j+1)%nv
            faces.extend(((a,b,c),(a,c,d)))
    return np.asarray(vertices,np.float32),np.asarray(faces,np.int64),np.asarray(normals,np.float32)

def icosahedron(center=(0.68,0.0,0.10),radius=0.035):
    phi=(1.0+5.0**0.5)/2.0
    raw=np.asarray([
        (-1, phi,0),(1,phi,0),(-1,-phi,0),(1,-phi,0),
        (0,-1,phi),(0,1,phi),(0,-1,-phi),(0,1,-phi),
        (phi,0,-1),(phi,0,1),(-phi,0,-1),(-phi,0,1),
    ],dtype=float)
    raw/=np.linalg.norm(raw,axis=1,keepdims=True)
    c=np.asarray(center,dtype=float)
    vertices=c[None,:]+radius*raw
    normals=raw.copy()
    faces=np.asarray([
      (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
      (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
      (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
      (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ],dtype=np.int64)
    return vertices.astype(np.float32),faces,normals.astype(np.float32)

def combine(parts):
    vs=[]; fs=[]; ns=[]; offset=0
    for v,f,n in parts:
        vs.append(v); fs.append(f+offset); ns.append(n); offset+=len(v)
    return np.concatenate(vs),np.concatenate(fs),np.concatenate(ns)

def shapes():
    sphere=ellipsoid(.62,.62,.62,96,160)
    return {
      "SPHERE_15K":sphere,
      "TALL_ELLIPSOID_15K":ellipsoid(.28,.34,.82,96,160),
      "FLAT_ELLIPSOID_15K":ellipsoid(.78,.56,.18,96,160),
      "TORUS_CONCAVE_10K":torus(),
      "TWO_CLOSE_COMPONENTS_12K":combine([
          ellipsoid(.31,.38,.52,64,96,(-.17,0,0)),
          ellipsoid(.20,.24,.34,64,96,(.19,0,.03)),
      ]),
      "THIN_DISCONNECTED_COMPONENT_11K":combine([
          ellipsoid(.42,.38,.55,72,120,(-.12,0,0)),
          ellipsoid(.07,.065,.55,64,48,(.50,0,.03)),
      ]),
      "VISIBLE_TINY_ACCESSORY_BELOW_0P1PCT":combine([
          sphere,
          icosahedron(),
      ]),
    }

def main():
    cams=cameras()
    out={
      "schema":"RealSaS.Stage14V2FinalSubjectFreeCalibration.v1",
      "status":"PASS",
      "subject_inputs_used":False,
      "knight_result_used":False,
      "mage_result_used":False,
      "raster_resolution":1024,
      "normal_k":64,
      "visibility_depth_tolerance_norm":0.02,
      "policy":POLICY,
      "selection_rule":"MINIMUM_ACTUAL_NODE_COUNT_AMONG_PASSING_BOUNDED_SEARCH_CANDIDATES",
      "shapes":{},
    }
    overall=True
    for name,(v,f,n) in shapes().items():
        surface,report=select_adequate_rigging_surface_v1(
            v,f,n,cams,
            normalization_center=(0.0,0.0,0.0),
            normalization_half_extent=1.0,
            authority_label="SUBJECT_FREE_ANALYTIC_STAGE14_V2",
            source_run_id="STAGE14_V2_FINAL_CALIBRATION",
            source_checkpoint_sha256="0"*64,
            source_zero_surface_sha256="1"*64,
            normal_k=64,
            visibility_depth_tolerance_norm=0.02,
            adequacy_policy=POLICY,
            metadata={"subject_free_shape":name},
        )
        passed=surface is not None and report.get("status")=="PASS"
        overall &= bool(passed)
        selected=None
        if passed:
            selected=next(
                m for m in report["evaluated_candidates"]
                if int(m["candidate_target_node_cap"])==int(report["selected_target_node_cap"])
            )
        out["shapes"][name]={
            "dense_vertex_count":int(len(v)),
            "dense_face_count":int(len(f)),
            "status":"PASS" if passed else "FAIL",
            "selected_target_node_cap":None if not passed else int(report["selected_target_node_cap"]),
            "selected_actual_node_count":None if not passed else int(report["selected_actual_node_count"]),
            "selected_metrics":selected,
            "candidate_count":int(len(report.get("evaluated_candidates",()))),
            "report_hash":report.get("adequacy_report_hash"),
        }
    out["status"]="PASS" if overall else "FAIL_SUBJECT_FREE_SHAPE"
    path=OUT/"STAGE14_V2_FINAL_SUBJECT_FREE_CALIBRATION.json"
    path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({
        "status":out["status"],
        "selected":{
            k:{
              "status":v["status"],
              "cap":v["selected_target_node_cap"],
              "actual":v["selected_actual_node_count"],
              "p95":None if v["selected_metrics"] is None else v["selected_metrics"]["projected_p95_px"],
              "max":None if v["selected_metrics"] is None else v["selected_metrics"]["projected_max_px"],
              "eligible_components":None if v["selected_metrics"] is None else v["selected_metrics"]["eligible_dense_component_count"],
              "alias_nodes":None if v["selected_metrics"] is None else v["selected_metrics"]["component_alias_node_count"],
            } for k,v in out["shapes"].items()
        }
    },indent=2,sort_keys=True))
    return 0 if overall else 2

if __name__=="__main__":
    raise SystemExit(main())
