#!/usr/bin/env python3
"""Independent source-rest analytic 3x3 subpixel footprint court, V2.

No Blender imported. Source rays intersect ORIGINAL rest triangles with
Moller-Trumbore, using SHA-bound original face palette and camera facts.
New physical edited camera set: V1/V3/V5/V7, disjoint from V1 pilot.
The observed-pixel predicate is imported from independent non-renderer code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.corpus.court_corpus370_operational_palette_pixel_consequence_v1 import judge_edit_rgba

SCHEMA="RealSaS.IndependentOriginalPalettePhysicalFootprintCourt.v2"
PRODUCER="RealSaS.OriginalSourcePhysicalPaletteEditRenderCandidate.v2"
POLICY="ORIGINAL_SHADER_CONSTANT_PALETTE__CONTROLLED_UNLIT_REPLAY__CYCLES_256_16SPP__NEW_V1_V3_V5_V7"
VIEWS=(1,3,5,7)
RES=256
STRIDE=4
STENCIL=(-0.375,0.0,0.375)
MIN_SAMPLES=8

def require(ok,code):
    if not ok:raise ValueError("SOURCE_FOOTPRINT_COURT_V2_"+code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def image(path):
    with Image.open(path) as im:
        require(im.size==(RES,RES),"RASTER_RESOLUTION")
        require(im.mode=="RGBA","RASTER_RGBA_REQUIRED")
        return np.asarray(im,dtype=np.float64)/255.0

class SourceRestRayOracle:
    def __init__(self,vertices,faces,face_color_indices):
        v=np.asarray(vertices,dtype=np.float64)
        f=np.asarray(faces,dtype=np.int64)
        colors=np.asarray(face_color_indices,dtype=np.int64)
        require(v.ndim==2 and v.shape[1]==3
                and f.ndim==2 and f.shape[1]==3
                and len(f)==len(colors) and f.min()>=0 and f.max()<len(v)
                and np.isfinite(v).all() and np.all(colors>=0),
                "SOURCE_REST_ARRAY_DOMAIN")
        tri=v[f]
        self.v0=tri[:,0]
        self.e1=tri[:,1]-tri[:,0]
        self.e2=tri[:,2]-tri[:,0]
        self.colors=colors

    def view(self,camera):
        self.origin=np.asarray(camera["origin"],dtype=np.float64)
        self.right=np.asarray(camera["right"],dtype=np.float64)
        self.up=np.asarray(camera["screen_up"],dtype=np.float64)
        self.forward=np.asarray(camera["forward"],dtype=np.float64)
        self.forward/=np.linalg.norm(self.forward)
        self.half=float(camera["half_extent"])
        self.clip_start=float(camera["clip_start"])
        self.clip_end=float(camera["clip_end"])
        require(np.isfinite(self.origin).all()
            and np.isfinite(self.right).all() and np.isfinite(self.up).all()
            and np.isfinite(self.forward).all() and self.half>0
            and 0<self.clip_start<self.clip_end,
            "SOURCE_CAMERA_DOMAIN")
        self.pvec=np.cross(np.broadcast_to(self.forward,self.e2.shape),self.e2)
        self.det=np.einsum("ij,ij->i",self.e1,self.pvec)
        self.invdet=np.divide(1.0,self.det,out=np.zeros_like(self.det),
                              where=np.abs(self.det)>1e-11)

    def first_hit_palette(self,x,y):
        u=2.0*((x+0.5)/RES)-1.0
        v=1.0-2.0*((y+0.5)/RES)
        origin=self.origin+self.half*u*self.right+self.half*v*self.up
        to=origin-self.v0
        us=np.einsum("ij,ij->i",to,self.pvec)*self.invdet
        q=np.cross(to,self.e1)
        vs=np.einsum("j,ij->i",self.forward,q)*self.invdet
        t=np.einsum("ij,ij->i",self.e2,q)*self.invdet
        acceptable=(np.abs(self.det)>1e-11)&(us>=-1e-10)&(vs>=-1e-10)&(
            us+vs<=1.0+1e-10)&(t>=self.clip_start)&(t<=self.clip_end)
        if not bool(np.any(acceptable)):return None
        accepted=np.flatnonzero(acceptable)
        nearest=int(accepted[np.argmin(t[acceptable])])
        return int(self.colors[nearest])

    def conservative_stencil_samples(self,baseline):
        samples=[]
        stats={"grid_points":0,"background_or_alpha_unknown":0,
               "multi_palette_or_ray_unknown":0,"qualified_stencil_interiors":0}
        for y in range(2,RES-2,STRIDE):
            for x in range(2,RES-2,STRIDE):
                stats["grid_points"]+=1
                if baseline[y,x,3]<0.995:
                    stats["background_or_alpha_unknown"]+=1
                    continue
                keys=[self.first_hit_palette(x+dx,y+dy)
                      for dy in STENCIL for dx in STENCIL]
                if any(k is None for k in keys) or len(set(keys))!=1:
                    stats["multi_palette_or_ray_unknown"]+=1
                    continue
                samples.append({"x":x,"y":y,"color_index":keys[0]})
                stats["qualified_stencil_interiors"]+=1
        return samples,stats


def court(source_path,rest_path,camera_path,policy_path,palette_path,manifest_path,
          render_root,outdir):
    manifest=load(manifest_path)
    palette=load(palette_path)
    original=load(policy_path)
    camera=load(camera_path)
    require(manifest.get("schema")==PRODUCER and manifest.get("policy")==POLICY
        and manifest.get("rendered_view_indices")==list(VIEWS)
        and manifest.get("new_disjoint_evaluation_views")==list(VIEWS)
        and manifest.get("prior_exploratory_views")==[0,2,4,6]
        and manifest.get("render_resolution")==RES
        and manifest.get("cycles_samples")==16
        and len(manifest.get("edits",[]))==3,
        "FROZEN_NEW_EVALUATION_PLAN")
    require(sha(source_path)==manifest["original_source_sha256"]
        ==palette["original_source_sha256"]
        ==original["source_sha256"],"ORIGINAL_SOURCE_BYTES")
    require(sha(camera_path)==manifest["source_camera_bundle_sha256"]
        ==original["camera_bundle"]["sha256"]
        and sha(policy_path)==manifest["original_source_policy_sha256"]
        and sha(palette_path)==manifest["source_palette_facts_sha256"]
        and sha(rest_path)==palette["original_rest_npz_sha256"]
        and sha(manifest_path)>"" and original["material_mode"]=="unlit_preserve_alpha",
        "ORIGINAL_SOURCE_PARENT_SHA")
    require(palette["palette_operation_count"]==3
        and palette["source_face_count"]==1334
        and palette["qualified_constant_face_count"]==1334
        and palette["unsupported_face_count"]==0
        and camera["schema"]=="RealSaS.CameraProjectionBundle.v1"
        and len(camera["cameras"])==8,
        "ORIGINAL_PALETTE_OR_CAMERA_CENSUS")
    with np.load(rest_path,allow_pickle=False) as z:
        vertices=np.asarray(z["rest_vertices_source_world"],dtype=np.float64)
        faces=np.asarray(z["faces"],dtype=np.int64)
    require(hashlib.sha256(np.ascontiguousarray(faces,dtype="<i8").tobytes()).hexdigest()
            ==palette["source_face_order_sha256"],"FACE_ORDER_SHA_DRIFT")
    palette_index=palette["face_palette_color_key_index_or_null"]
    require(len(palette_index)==len(faces)==1334
        and all(type(x) is int and 0<=x<3 for x in palette_index),
        "ORIGINAL_FACE_PALETTE_INCOMPLETE")
    oracle=SourceRestRayOracle(vertices,faces,palette_index)
    totals={i:{"selected":0,"unselected":0,"selected_unchanged":0,
               "collateral":0,"max_alpha":0.0} for i in range(3)}
    receipts=[]
    failure_samples=[]
    coverage=[]
    for vi in VIEWS:
        c=camera["cameras"][vi]
        require(c["view_index"]==vi and c["view_id"]==f"V{vi}"
            and c["resolution"]==1024,"ORIGINAL_CAMERA_ORDER")
        oracle.view(c)
        base_info=manifest["baseline"][str(vi)]
        baseline_path=render_root/"BASELINE"/base_info["file"]
        require(sha(baseline_path)==base_info["sha256"],"BASELINE_RASTER_SHA")
        base=image(baseline_path)
        candidates,counts=oracle.conservative_stencil_samples(base)
        coverage.append({"view_index":vi,**counts})
        for edit in manifest["edits"]:
            idx=edit["color_key_index"]
            require(type(idx) is int and 0<=idx<3
                and edit["source_hex_rgba"]==palette["constant_palette_color_keys_hex_rgba"][idx]
                and edit["physical_palette_edit_rule"]
                  =="RGB_COMPONENT_TO_0P15_IF_GE_0P5_ELSE_0P85__ALPHA_RETAINED",
                "EDIT_POLICY_DRIFT")
            row=edit["raster"][str(vi)]
            path=render_root/f"EDIT_{idx}"/row["file"]
            require(sha(path)==row["sha256"],"EDIT_RASTER_SHA")
            observed=image(path)
            qualified=[x for x in candidates if observed[x["y"],x["x"],3]>=0.995]
            trial=judge_edit_rgba(base,observed,qualified,idx,
                                  min_selected_samples=0,min_unselected_samples=0)
            sums=totals[idx]
            for dest,key in (("selected","selected_count"),("unselected","unselected_count"),
                             ("selected_unchanged","selected_unchanged_count"),
                             ("collateral","collateral_count")):
                sums[dest]+=trial[key]
            sums["max_alpha"]=max(sums["max_alpha"],trial["alpha_max_difference"])
            if trial["collateral_count"] or trial["selected_unchanged_count"]:
                for row in qualified:
                    x,y=row["x"],row["y"]
                    diff=float(np.max(np.abs(observed[y,x,:3]-base[y,x,:3])))
                    if (row["color_index"]==idx and diff<8.0/255.0) or (
                        row["color_index"]!=idx and diff>3.0/255.0):
                        failure_samples.append({"view":vi,"edit":idx,"xy":[x,y],
                            "source_stencil_color_index":row["color_index"],
                            "max_rgb_change":diff})
                        if len(failure_samples)>=32:break
            receipts.append({"view_index":vi,"edit_color_index":idx,
                "tested_stencil_uniform_interior_count":len(qualified),
                "selected":trial["selected_count"],"unselected":trial["unselected_count"],
                "collateral":trial["collateral_count"],
                "selected_unchanged":trial["selected_unchanged_count"],
                "alpha_max_difference":trial["alpha_max_difference"]})
    failures=[]
    for idx,s in totals.items():
        if s["selected"]<MIN_SAMPLES:failures.append(f"EDIT_{idx}_SELECTED_COVERAGE_TOO_SMALL")
        if s["unselected"]<MIN_SAMPLES:failures.append(f"EDIT_{idx}_UNSELECTED_COVERAGE_TOO_SMALL")
        if s["collateral"]:failures.append(f"EDIT_{idx}_COLLATERAL_INTERIOR_CHANGED")
        if s["selected_unchanged"]:failures.append(f"EDIT_{idx}_SELECTED_INTERIOR_INERT")
    result={"schema":SCHEMA,
        "status":"PASS_ORIGINAL_PHYSICAL_EDIT_STENCIL_INTERIOR_LOCALITY__NOT_DRAWABLE_OWNER"
            if not failures else "FAIL_ORIGINAL_PHYSICAL_EDIT_STENCIL_INTERIOR_LOCALITY",
        "source_original_sha256":palette["original_source_sha256"],
        "source_rest_geometry_npz_sha256":sha(rest_path),
        "source_camera_bundle_sha256":sha(camera_path),
        "source_palette_facts_sha256":sha(palette_path),
        "physical_renderer_manifest_sha256":sha(manifest_path),
        "original_shader_edit_policy":"CONTROLLED_SOURCE_UNLIT_ONLY__NO_RAW_LIT_PARITY",
        "ray_intersector":"INDEPENDENT_NUMPY_SOURCE_REST_MOLLER_TRUMBORE_FIRST_HIT",
        "stencil_offset_pixels":list(STENCIL),"sample_stride_pixels":STRIDE,
        "new_evaluation_view_indices":list(VIEWS),
        "previously_inspected_physical_edit_views":[0,2,4,6],
        "same_pose_as_prior_run":True,
        "pose_heldout_edit_consequence":"NOT_EVALUATED",
        "source_coverage":coverage,"edit_totals":totals,
        "per_view_edit_receipts":receipts,
        "source_private_failure_samples":failure_samples,
        "failures":failures,
        "continuous_pixel_area_certified":False,
        "full_raster_physical_locality_certified":False,
        "source_artist_drawable_owner":"UNKNOWN",
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False}
    outdir.mkdir(parents=True,exist_ok=True)
    dest=outdir/"PRIVATE_INDEPENDENT_SOURCE_PHYSICAL_FOOTPRINT_V2_COURT.json"
    dest.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
                    encoding="utf-8")
    print(json.dumps({"status":result["status"],"total":totals,
        "fails":len(failures),"training_authorized":False},sort_keys=True),flush=True)
    require(not failures,"NEW_PHYSICAL_EDIT_VIEW_OR_SOURCE_PIXEL_GATE_FAIL")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("source","source-rest","camera-bundle","source-policy",
                 "palette-facts","render-manifest","render-root","output-dir"):
        p.add_argument("--"+flag,type=Path,required=True)
    a=p.parse_args()
    court(a.source,a.source_rest,a.camera_bundle,a.source_policy,
          a.palette_facts,a.render_manifest,a.render_root,a.output_dir)

if __name__=="__main__":main()
