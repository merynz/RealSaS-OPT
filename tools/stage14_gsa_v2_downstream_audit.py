from __future__ import annotations

import json, math, os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.substrate.adequacy_v1 import select_adequate_rigging_surface_v1
from compiler.realsas_compiler_core.substrate.scene_first_signed import rigging_surface_from_scene_first_zero_mesh_v1

OUT=Path(os.environ.get("REALSAS_STAGE14_AUDIT_OUT","stage14_audit_out"))
OUT.mkdir(parents=True,exist_ok=True)

CURRENT={
    "profile_id":"CURRENT_V1",
    "max_dense_to_surface_p95_norm":0.03,
    "max_dense_to_surface_max_norm":0.08,
    "max_normal_p95_deg":30.0,
    "max_projected_p95_px":6.0,
    "max_projected_max_px":16.0,
    "component_min_dense_fraction":0.001,
    "min_nodes_per_component":8,
    "max_component_alias_nodes":0,
}
CANDIDATES=[
    {
        "profile_id":"TIGHT_A",
        "max_dense_to_surface_p95_norm":0.020,
        "max_dense_to_surface_max_norm":0.050,
        "max_normal_p95_deg":25.0,
        "max_projected_p95_px":4.0,
        "max_projected_max_px":10.0,
        "component_min_dense_fraction":0.0005,
        "min_nodes_per_component":12,
        "max_component_alias_nodes":0,
    },
    {
        "profile_id":"TIGHT_B",
        "max_dense_to_surface_p95_norm":0.015,
        "max_dense_to_surface_max_norm":0.040,
        "max_normal_p95_deg":20.0,
        "max_projected_p95_px":3.0,
        "max_projected_max_px":8.0,
        "component_min_dense_fraction":0.00025,
        "min_nodes_per_component":16,
        "max_component_alias_nodes":0,
    },
    {
        "profile_id":"TIGHT_C",
        "max_dense_to_surface_p95_norm":0.010,
        "max_dense_to_surface_max_norm":0.030,
        "max_normal_p95_deg":15.0,
        "max_projected_p95_px":2.0,
        "max_projected_max_px":6.0,
        "component_min_dense_fraction":0.0001,
        "min_nodes_per_component":24,
        "max_component_alias_nodes":0,
    },
]
PROFILES=[CURRENT,*CANDIDATES]

BASE_POLICY={
    "min_candidate_nodes":128,
    "max_candidate_nodes":8192,
    "candidate_growth_factor":2.0,
    "refinement_rounds":3,
}

def cameras():
    rows=[]
    for view in range(8):
        yaw=math.radians(45.0*view)
        f=np.array([-math.sin(yaw),-math.cos(yaw),0.0])
        r=np.array([-math.cos(yaw), math.sin(yaw),0.0])
        rows.append({
            "view_index":view,
            "origin":tuple((-4.0*f).tolist()),
            "right":tuple(r.tolist()),
            "screen_up":(0.0,0.0,1.0),
            "forward":tuple(f.tolist()),
            "half_extent":1.0,
            "resolution":1024,
        })
    return tuple(rows)

def uv_ellipsoid(a,b,c,nlat=56,nlon=112,center=(0,0,0)):
    verts=[]; norms=[]
    cx,cy,cz=center
    # poles
    verts.append((cx,cy,cz+c)); norms.append((0,0,1))
    for i in range(1,nlat):
        th=math.pi*i/nlat
        st,ct=math.sin(th),math.cos(th)
        for j in range(nlon):
            ph=2*math.pi*j/nlon
            cp,sp=math.cos(ph),math.sin(ph)
            x=a*st*cp; y=b*st*sp; z=c*ct
            verts.append((cx+x,cy+y,cz+z))
            g=np.array([x/(a*a),y/(b*b),z/(c*c)],dtype=float)
            g/=np.linalg.norm(g)
            norms.append(tuple(g))
    bottom=len(verts); verts.append((cx,cy,cz-c)); norms.append((0,0,-1))
    faces=[]
    first=1
    for j in range(nlon):
        faces.append((0, first+j, first+(j+1)%nlon))
    for i in range(nlat-2):
        r0=1+i*nlon; r1=r0+nlon
        for j in range(nlon):
            a0=r0+j; a1=r0+(j+1)%nlon; b0=r1+j; b1=r1+(j+1)%nlon
            faces.append((a0,b0,b1)); faces.append((a0,b1,a1))
    last=1+(nlat-2)*nlon
    for j in range(nlon):
        faces.append((bottom,last+(j+1)%nlon,last+j))
    return np.asarray(verts,np.float32),np.asarray(faces,np.int64),np.asarray(norms,np.float32)

def torus(R=.48,r=.13,nu=96,nv=48):
    v=[]; n=[]
    for i in range(nu):
        u=2*math.pi*i/nu; cu,su=math.cos(u),math.sin(u)
        for j in range(nv):
            w=2*math.pi*j/nv; cw,sw=math.cos(w),math.sin(w)
            x=(R+r*cw)*cu; y=(R+r*cw)*su; z=r*sw
            v.append((x,y,z)); n.append((cw*cu,cw*su,sw))
    f=[]
    for i in range(nu):
        for j in range(nv):
            a=i*nv+j; b=((i+1)%nu)*nv+j; c=((i+1)%nu)*nv+(j+1)%nv; d=i*nv+(j+1)%nv
            f.append((a,b,c)); f.append((a,c,d))
    return np.asarray(v,np.float32),np.asarray(f,np.int64),np.asarray(n,np.float32)

def combine(parts):
    vv=[]; ff=[]; nn=[]; off=0
    for v,f,n in parts:
        vv.append(v); nn.append(n); ff.append(f+off); off+=len(v)
    return np.concatenate(vv),np.concatenate(ff),np.concatenate(nn)

def shapes():
    sphere=uv_ellipsoid(.62,.62,.62)
    tall=uv_ellipsoid(.28,.34,.82,nlat=64,nlon=112)
    flat=uv_ellipsoid(.78,.56,.18,nlat=56,nlon=128)
    ring=torus()
    two=combine([
        uv_ellipsoid(.48,.48,.58,nlat=56,nlon=104,center=(-.18,0,0)),
        uv_ellipsoid(.12,.10,.16,nlat=20,nlon=40,center=(.66,.05,.12)),
    ])
    thin=combine([
        uv_ellipsoid(.42,.38,.55,nlat=56,nlon=104,center=(-.12,0,0)),
        uv_ellipsoid(.09,.08,.55,nlat=48,nlon=36,center=(.50,0,.03)),
    ])
    return {
        "SPHERE":sphere,
        "TALL_THIN_ELLIPSOID":tall,
        "FLAT_WIDE_ELLIPSOID":flat,
        "TORUS_CONCAVE":ring,
        "TWO_COMPONENTS_ACCESSORY":two,
        "THIN_APPENDAGE_COMPONENT":thin,
    }

def posthoc_pass(m,p):
    return (
      m["dense_to_surface_p95_norm"]<=p["max_dense_to_surface_p95_norm"] and
      m["dense_to_surface_max_norm"]<=p["max_dense_to_surface_max_norm"] and
      m["normal_p95_deg"]<=p["max_normal_p95_deg"] and
      m["projected_p95_px"]<=p["max_projected_p95_px"] and
      m["projected_max_px"]<=p["max_projected_max_px"] and
      m["minimum_nodes_per_eligible_component"]>=p["min_nodes_per_component"] and
      m["component_alias_node_count"]<=p["max_component_alias_nodes"]
    )

def selected_cap(report,p):
    rows=[]
    for m in report["evaluated_candidates"]:
        # component eligibility itself depends on component_min_dense_fraction, so
        # report generated with floor=0 is conservative: every component counted.
        if posthoc_pass(m,p):
            rows.append(m)
    if not rows:return None
    best=min(rows,key=lambda m:(m["actual_node_count"],m["candidate_target_node_cap"]))
    return int(best["candidate_target_node_cap"])

def displacement(points,kind):
    p=np.asarray(points,float)
    if kind=="TWIST_Z":
        a=np.deg2rad(22.0)*p[:,2]
        ca,sa=np.cos(a),np.sin(a)
        out=p.copy()
        out[:,0]=ca*p[:,0]-sa*p[:,1]; out[:,1]=sa*p[:,0]+ca*p[:,1]
        return out-p
    if kind=="BEND_Y":
        a=np.deg2rad(18.0)*p[:,2]
        ca,sa=np.cos(a),np.sin(a)
        out=p.copy()
        out[:,0]=ca*p[:,0]+sa*p[:,2]; out[:,2]=-sa*p[:,0]+ca*p[:,2]
        return out-p
    if kind=="HINGE_Z":
        pivot=np.array([0.15,0.0,0.0]); q=p-pivot
        w=np.clip((p[:,0]-0.05)/0.20,0,1)
        a=np.deg2rad(25.0)*w; ca,sa=np.cos(a),np.sin(a)
        out=q.copy()
        out[:,0]=ca*q[:,0]-sa*q[:,1]; out[:,1]=sa*q[:,0]+ca*q[:,1]
        out+=pivot
        return out-p
    raise KeyError(kind)

def project(points,cam):
    p=np.asarray(points,float); origin=np.asarray(cam["origin"]); right=np.asarray(cam["right"]); up=np.asarray(cam["screen_up"])
    d=p-origin
    x=(d@right)/cam["half_extent"]; y=-(d@up)/cam["half_extent"]; r=cam["resolution"]
    return np.stack(((x+1)*.5*r-.5,(y+1)*.5*r-.5),axis=1)

def downstream_proxy(dense,surface,cams):
    nodes=np.asarray([n.P for n in surface.surface_nodes],float)
    tree=cKDTree(nodes)
    k=min(4,len(nodes))
    dist,idx=tree.query(dense,k=k,workers=-1)
    if k==1:
        dist=dist[:,None]; idx=idx[:,None]
    w=1.0/np.maximum(dist,1e-8); w/=w.sum(axis=1,keepdims=True)
    out={}
    for kind in ("TWIST_Z","BEND_Y","HINGE_Z"):
        dd=displacement(dense,kind); nd=displacement(nodes,kind)
        interp=np.einsum("nk,nkd->nd",w,nd[idx])
        err=np.linalg.norm(interp-dd,axis=1)
        true=dense+dd; pred=dense+interp
        px=[]
        for cam in cams:
            px.extend(np.linalg.norm(project(pred,cam)-project(true,cam),axis=1).tolist())
        px=np.asarray(px)
        out[kind]={
          "world_p95_norm":float(np.quantile(err,.95)),
          "world_max_norm":float(err.max(initial=0)),
          "projected_p95_px":float(np.quantile(px,.95)),
          "projected_max_px":float(px.max(initial=0)),
        }
    return out

def main():
    cams=cameras()
    # One permissive exact GSA sweep per shape. Every component is eligible.
    loose={**BASE_POLICY,
      "max_dense_to_surface_p95_norm":1.0,"max_dense_to_surface_max_norm":2.0,
      "max_normal_p95_deg":180.0,"max_projected_p95_px":1024.0,"max_projected_max_px":2048.0,
      "component_min_dense_fraction":0.0,"min_nodes_per_component":1,"max_component_alias_nodes":10**9,
    }
    result={
      "schema":"RealSaS.Stage14GSAV2DownstreamAudit.v1",
      "status":"PASS",
      "subject_inputs_used":False,"knight_result_used":False,"mage_result_used":False,
      "camera_resolution":1024,
      "normal_k":64,"visibility_depth_tolerance_norm":0.02,
      "profiles":PROFILES,
      "downstream_proxy":{
        "authority":"DIAGNOSTIC_ONLY_NOT_SHIPPING_GATE",
        "interpolation":"4NN_INVERSE_DISTANCE_FROM_GSA_NODES",
        "fields":["TWIST_Z","BEND_Y","HINGE_Z"],
      },
      "shapes":{},
    }
    for name,(v,f,n) in shapes().items():
        surface,report=select_adequate_rigging_surface_v1(
          v,f,n,cams,
          normalization_center=(0,0,0),normalization_half_extent=1.0,
          authority_label="SUBJECT_FREE_ANALYTIC",source_run_id="STAGE14_GSA_V2_AUDIT",
          source_checkpoint_sha256="0"*64,source_zero_surface_sha256="1"*64,
          normal_k=64,visibility_depth_tolerance_norm=0.02,adequacy_policy=loose,
          metadata={"subject_free_shape":name},
        )
        profile_rows={}
        for p in PROFILES:
            cap=selected_cap(report,p)
            row={"selected_target_node_cap":cap}
            if cap is not None:
                ss=rigging_surface_from_scene_first_zero_mesh_v1(
                  v,f,n,cams,normalization_center=(0,0,0),normalization_half_extent=1.0,
                  authority_label="SUBJECT_FREE_ANALYTIC",source_run_id="STAGE14_GSA_V2_AUDIT",
                  source_checkpoint_sha256="0"*64,source_zero_surface_sha256="1"*64,
                  target_nodes=cap,normal_k=64,visibility_depth_tolerance_norm=0.02,
                  metadata={"subject_free_shape":name,"profile_id":p["profile_id"]},
                )
                match=next(m for m in report["evaluated_candidates"] if int(m["candidate_target_node_cap"])==cap)
                row["static_metrics"]=match
                row["downstream_proxy"]=downstream_proxy(np.asarray(v,float),ss,cams)
                row["actual_node_count"]=len(ss.surface_nodes)
            profile_rows[p["profile_id"]]=row
        result["shapes"][name]={
          "dense_vertex_count":len(v),"dense_face_count":len(f),
          "profiles":profile_rows,
          "sweep_candidate_metrics":report["evaluated_candidates"],
        }
    (OUT/"STAGE14_GSA_V2_DOWNSTREAM_AUDIT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
      "status":result["status"],
      "summary":{s:{p:r["selected_target_node_cap"] for p,r in d["profiles"].items()} for s,d in result["shapes"].items()}
    },indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
