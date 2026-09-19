from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.substrate.adequacy_v1 import (
    _dense_component_labels,
    _metrics,
    select_adequate_rigging_surface_v1,
)
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    _self_zbuffer_support,
    rigging_surface_from_scene_first_zero_mesh_v1,
)
from models.iris.v3.zero_surface_decoder_v3 import extract_zero_surface_mesh_v3

OUT=Path(os.environ.get("REALSAS_STAGE14_V2_OUT","stage14_v2_out"))
OUT.mkdir(parents=True,exist_ok=True)
GRID=192
RES=1024
CAPS=(128,256,512,1024,2048,4096,8192)
NORMAL_K=64
VIS_TOL=0.02


def _write(name,payload):
    (OUT/name).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def _cams():
    rows=[]
    for view in range(8):
        yaw=math.radians(45.0*view)
        forward=np.asarray([-math.sin(yaw),-math.cos(yaw),0.0],dtype=np.float64)
        right=np.asarray([-math.cos(yaw),math.sin(yaw),0.0],dtype=np.float64)
        rows.append({
            "view_id":f"V{view}",
            "view_index":view,
            "origin":tuple((-4.0*forward).tolist()),
            "right":tuple(right.tolist()),
            "screen_up":(0.0,0.0,1.0),
            "forward":tuple(forward.tolist()),
            "half_extent":1.0,
            "resolution":RES,
        })
    return tuple(rows)


def _axis_grid():
    axis=np.linspace(-1.0,1.0,GRID,dtype=np.float32)
    z,y,x=np.meshgrid(axis,axis,axis,indexing="ij")
    return x,y,z


def _sphere(x,y,z,cx,cy,cz,r):
    return np.sqrt((x-cx)**2+(y-cy)**2+(z-cz)**2).astype(np.float32)-np.float32(r)


def _ellipsoid(x,y,z,cx,cy,cz,rx,ry,rz):
    # Sign-correct implicit field; exact Euclidean distance is unnecessary for zero-set extraction.
    return (np.sqrt(((x-cx)/rx)**2+((y-cy)/ry)**2+((z-cz)/rz)**2)-1.0).astype(np.float32)


def _field(kind):
    x,y,z=_axis_grid()
    if kind=="SPHERE":
        return _sphere(x,y,z,0,0,0,0.62)
    if kind=="TALL_THIN_ELLIPSOID":
        return _ellipsoid(x,y,z,0,0,0,0.28,0.28,0.78)
    if kind=="FLAT_WIDE_ELLIPSOID":
        return _ellipsoid(x,y,z,0,0,0,0.78,0.58,0.18)
    if kind=="CONNECTED_THIN_SPUR":
        body=_sphere(x,y,z,0,0,0,0.50)
        spur=_ellipsoid(x,y,z,0.48,0,0,0.42,0.018,0.018)
        return np.minimum(body,spur)
    if kind=="TINY_DETACHED_NEAR_BODY":
        body=_sphere(x,y,z,0,0,0,0.70)
        tiny=_sphere(x,y,z,0.745,0,0,0.020)
        return np.minimum(body,tiny)
    raise KeyError(kind)


CURRENT={
  "profile_id":"CURRENT_V1",
  "min_candidate_nodes":128,"max_candidate_nodes":8192,"candidate_growth_factor":2.0,"refinement_rounds":3,
  "max_dense_to_surface_p95_norm":0.03,"max_dense_to_surface_max_norm":0.08,
  "max_normal_p95_deg":30.0,"max_projected_p95_px":6.0,"max_projected_max_px":16.0,
  "component_min_dense_fraction":0.001,"min_nodes_per_component":8,"max_component_alias_nodes":0,
}
CANDIDATES=[
 {
  "profile_id":"P1_ULTRA",
  "min_candidate_nodes":128,"max_candidate_nodes":8192,"candidate_growth_factor":2.0,"refinement_rounds":3,
  "max_dense_to_surface_p95_norm":0.005,"max_dense_to_surface_max_norm":0.015,
  "max_normal_p95_deg":15.0,"max_projected_p95_px":1.5,"max_projected_max_px":4.0,
  "component_min_dense_fraction":0.0,"min_nodes_per_component":8,"max_component_alias_nodes":0,
 },
 {
  "profile_id":"P2_TIGHT",
  "min_candidate_nodes":128,"max_candidate_nodes":8192,"candidate_growth_factor":2.0,"refinement_rounds":3,
  "max_dense_to_surface_p95_norm":0.01,"max_dense_to_surface_max_norm":0.025,
  "max_normal_p95_deg":20.0,"max_projected_p95_px":2.5,"max_projected_max_px":6.0,
  "component_min_dense_fraction":0.0,"min_nodes_per_component":8,"max_component_alias_nodes":0,
 },
 {
  "profile_id":"P3_BALANCED",
  "min_candidate_nodes":128,"max_candidate_nodes":8192,"candidate_growth_factor":2.0,"refinement_rounds":3,
  "max_dense_to_surface_p95_norm":0.015,"max_dense_to_surface_max_norm":0.04,
  "max_normal_p95_deg":25.0,"max_projected_p95_px":4.0,"max_projected_max_px":8.0,
  "component_min_dense_fraction":0.0,"min_nodes_per_component":8,"max_component_alias_nodes":0,
 },
]


def _policy(p):
    return {k:v for k,v in p.items() if k!="profile_id"}


def _raw_eval(mesh,cameras):
    vn=np.asarray(mesh.vertices_normalized,dtype=np.float64)
    f=np.asarray(mesh.faces,dtype=np.int64)
    hints=np.asarray(mesh.implicit_normals,dtype=np.float64)
    dense_world=vn.copy()
    dense_normals=hints/np.linalg.norm(hints,axis=1,keepdims=True).clip(min=1e-12)
    labels=_dense_component_labels(len(vn),f)
    support,raster,_=_self_zbuffer_support(dense_world,dense_world,cameras,depth_tolerance=VIS_TOL)
    surfaces={}
    for cap in CAPS:
        surfaces[cap]=rigging_surface_from_scene_first_zero_mesh_v1(
            vn,f,hints,cameras,
            normalization_center=(0.0,0.0,0.0),normalization_half_extent=1.0,
            authority_label="SUBJECT_FREE_STAGE14_V2",
            source_run_id="SUBJECT_FREE_STAGE14_V2",
            source_checkpoint_sha256="0"*64,
            source_zero_surface_sha256="1"*64,
            target_nodes=cap,normal_k=NORMAL_K,visibility_depth_tolerance_norm=VIS_TOL,
            metadata={"subject_free":True,"candidate_target_node_cap":cap},
        )
    return vn,f,hints,dense_world,dense_normals,labels,support,raster,surfaces


def _evaluate_profile(raw,p):
    vn,f,hints,dense_world,dense_normals,labels,support,raster,surfaces=raw
    rows=[]
    pol=_policy(p)
    for cap in CAPS:
        m=_metrics(
            surface=surfaces[cap],dense_world=dense_world,dense_normals=dense_normals,
            dense_labels=labels,dense_support=support,dense_raster=raster,
            normalization_half_extent=1.0,policy=pol,
        )
        m["candidate_target_node_cap"]=cap
        rows.append(m)
    passing=[r for r in rows if r["passed"]]
    return {
        "profile_id":p["profile_id"],
        "passing_cap_exists":bool(passing),
        "minimum_passing_fixed_cap":None if not passing else min(r["candidate_target_node_cap"] for r in passing),
        "fixed_cap_metrics":rows,
    }


def main():
    import skimage
    cameras=_cams()
    shapes={}
    kinds=("SPHERE","TALL_THIN_ELLIPSOID","FLAT_WIDE_ELLIPSOID","CONNECTED_THIN_SPUR","TINY_DETACHED_NEAR_BODY")
    all_profiles=[CURRENT,*CANDIDATES]
    for kind in kinds:
        mesh=extract_zero_surface_mesh_v3(_field(kind),bounds=(-1.0,1.0),level=0.0)
        raw=_raw_eval(mesh,cameras)
        labels=raw[5]
        counts=np.bincount(labels,minlength=int(labels.max(initial=-1)+1))
        profile_results={p["profile_id"]:_evaluate_profile(raw,p) for p in all_profiles}
        shapes[kind]={
            "vertex_count":int(len(mesh.vertices_normalized)),
            "face_count":int(len(mesh.faces)),
            "dense_component_count":int(len(counts)),
            "dense_component_vertex_counts":[int(x) for x in counts],
            "dense_component_vertex_fractions":[float(x)/float(len(labels)) for x in counts],
            "profiles":profile_results,
        }

    selected=None
    for p in CANDIDATES:
        if all(shapes[k]["profiles"][p["profile_id"]]["passing_cap_exists"] for k in kinds):
            selected=p
            break

    shipping_confirmation={}
    if selected is not None:
        for kind in kinds:
            mesh=extract_zero_surface_mesh_v3(_field(kind),bounds=(-1.0,1.0),level=0.0)
            surface,report=select_adequate_rigging_surface_v1(
                mesh.vertices_normalized,mesh.faces,mesh.implicit_normals,cameras,
                normalization_center=(0.0,0.0,0.0),normalization_half_extent=1.0,
                authority_label="SUBJECT_FREE_STAGE14_V2_CONFIRM",
                source_run_id="SUBJECT_FREE_STAGE14_V2_CONFIRM",
                source_checkpoint_sha256="2"*64,source_zero_surface_sha256="3"*64,
                normal_k=NORMAL_K,visibility_depth_tolerance_norm=VIS_TOL,
                adequacy_policy=_policy(selected),metadata={"subject_free":True},
            )
            shipping_confirmation[kind]={
                "status":report["status"],
                "selected_target_node_cap":report["selected_target_node_cap"],
                "selected_actual_node_count":report["selected_actual_node_count"],
                "selected_surface_lineage_hash":report["selected_surface_lineage_hash"],
                "surface_returned":surface is not None,
            }

    tiny=shapes["TINY_DETACHED_NEAR_BODY"]
    current_tiny_eligible=[
        f for f in tiny["dense_component_vertex_fractions"]
        if f+1e-15>=CURRENT["component_min_dense_fraction"]
    ]
    result={
      "schema":"RealSaS.Stage14GSAV2SubjectFreeCalibration.v1",
      "status":"PASS" if selected is not None and all(v["status"]=="PASS" for v in shipping_confirmation.values()) else "FAIL_NO_PREREGISTERED_V2_PROFILE_CLOSES",
      "subject_inputs_used":False,"knight_result_used":False,"mage_result_used":False,
      "grid_resolution":GRID,"raster_resolution":RES,"normal_k":NORMAL_K,"visibility_depth_tolerance_norm":VIS_TOL,
      "candidate_caps":CAPS,
      "current_policy":CURRENT,
      "candidate_profiles_frozen_before_run":CANDIDATES,
      "selection_rule":"FIRST_STRICT_TO_LOOSE_CANDIDATE_WITH_A_PASSING_CAP_FOR_EVERY_SUBJECT_FREE_SHAPE_THEN_CONFIRM_WITH_SHIPPING_SELECTOR",
      "selected_profile":selected,
      "shapes":shapes,
      "shipping_selector_confirmation":shipping_confirmation,
      "tiny_detached_component_audit":{
        "component_vertex_fractions":tiny["dense_component_vertex_fractions"],
        "current_component_floor":CURRENT["component_min_dense_fraction"],
        "eligible_component_count_under_current_floor":len(current_tiny_eligible),
        "all_components_eligible_under_v2_candidates":True,
      },
      "claim_boundary":"Subject-free Stage14 mechanical-carrier adequacy calibration only; no Knight Stage14 PASS or downstream rig/skin/product claim.",
      "skimage_version":skimage.__version__,
    }
    _write("STAGE14_GSA_V2_SUBJECT_FREE_CALIBRATION.json",result)
    print(json.dumps({
      "status":result["status"],
      "selected_profile":selected,
      "tiny_detached_component_audit":result["tiny_detached_component_audit"],
      "minimum_passing_caps":{
        k:{pid:r["minimum_passing_fixed_cap"] for pid,r in v["profiles"].items()} for k,v in shapes.items()
      },
      "shipping_confirmation":shipping_confirmation,
    },indent=2,sort_keys=True))
    return 0 if result["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
