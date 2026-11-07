#!/usr/bin/env python3
"""Source-bound, reversible normalization plan; NEVER teacher/admission.

SOURCE source-world -> IRIS [-1,1]^3 -> TESSA [-0.5,0.5]^3.
This plan does NOT infer anatomical forward/up, source meters, root joints,
bind-rest equivalence, IK, semantics or Graphics. Imported 3D source bounds
are observations, not ground-truth acceptance or a rig response test.
All matrices are 16 flat row-major scalars, column-vector application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA="RealSaS.Corpus4008ReversibleSourceNormalizationPlan.v1"
BOUNDS_SCHEMA="RealSaS.Corpus4008MeasuredImportedSourceBounds.v1"
SHA_LEN=64
EPS=1e-12


def require(x, why):
    if not x:
        raise ValueError("C4008_NORMALIZATION_"+why)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(4<<20),b""):
            h.update(part)
    return h.hexdigest()


def issha(x):
    return isinstance(x,str) and len(x)==64 and all(
        c in "0123456789abcdef" for c in x)


def matrix(scale, translation):
    require(math.isfinite(scale) and scale>EPS,"MATRIX_SCALE")
    require(len(translation)==3 and all(math.isfinite(float(x)) for x in translation),
            "MATRIX_TRANSLATION")
    x,y,z=map(float,translation)
    return [
        scale,0.0,0.0,x,
        0.0,scale,0.0,y,
        0.0,0.0,scale,z,
        0.0,0.0,0.0,1.0,
    ]


def point(mat, p):
    require(len(mat)==16 and len(p)==3,"MATRIX_POINT_SHAPE")
    require(all(math.isfinite(float(v)) for v in mat)
            and all(math.isfinite(float(v)) for v in p),
            "MATRIX_POINT_NONFINITE")
    return [sum(mat[4*i+j]*p[j] for j in range(3))+mat[4*i+3]
            for i in range(3)]


def four_by_four(mat, other):
    return [
        sum(mat[4*r+k]*other[4*k+c] for k in range(4))
        for r in range(4) for c in range(4)
    ]


def require_identity(mat,tol=1e-6):
    exact=[1.0 if i//4==i%4 else 0.0 for i in range(16)]
    require(max(abs(x-y) for x,y in zip(mat,exact))<=tol,
            "MATRIX_ROUNDTRIP")


def plan(bounds_file:Path, *, expected_sha256=None):
    bounds_file=bounds_file.resolve()
    require(bounds_file.is_file(),"BOUNDS_RECEIPT_ABSENT")
    receipt_sha=file_sha(bounds_file)
    if expected_sha256 is not None:
        require(receipt_sha==expected_sha256,"BOUNDS_RECEIPT_SHA_DRIFT")
    d=json.loads(bounds_file.read_text())
    require(d.get("schema")==BOUNDS_SCHEMA and
            d.get("status")=="IMPORTED_SOURCE_BOUNDS_ONLY" and
            d.get("training_authorized") is False and
            d.get("independent_source_truth_courted") is False,
            "UNQUALIFIED_BOUNDS_RECEIPT")
    original=d.get("source_sha256")
    require(issha(original),"ORIGINAL_SOURCE_SHA")
    aid=d.get("canonical_asset_id")
    cid=d.get("candidate_id")
    require(isinstance(aid,str) and aid and
            isinstance(cid,str) and
            (cid.endswith(original) or cid.endswith(original[:24])),
            "SOURCE_IDENTITY")
    require(d.get("source_space")=="IMPORTED_BLENDER_SOURCE_WORLD",
            "BOUNDS_SPACE")
    require(d.get("anatomical_up")=="UNKNOWN" and
            d.get("anatomical_front")=="UNKNOWN" and
            d.get("source_meter_unit")=="UNKNOWN",
            "UNAUTHORIZED_ORIENTATION_OR_UNIT")
    mn=d.get("source_world_min")
    mx=d.get("source_world_max")
    require(isinstance(mn,list) and isinstance(mx,list)
            and len(mn)==len(mx)==3,"BOUNDS_SHAPE")
    require(all(isinstance(t,(float,int)) and math.isfinite(t)
                for t in mn+mx),"BOUNDS_NONFINITE")
    require(all(b>=a for a,b in zip(mn,mx)),"BOUNDS_REVERSED")
    span=[b-a for a,b in zip(mn,mx)]
    max_span=max(span)
    require(math.isfinite(max_span) and max_span>EPS,
            "DEGENERATE_GLOBAL_SOURCE_EXTENT")
    center=[(a+b)*.5 for a,b in zip(mn,mx)]
    half=max_span*.5
    require(half>EPS and math.isfinite(half),"INVALID_HALF_EXTENT")
    scale=1.0/half
    world_to_iris=matrix(scale,[-c*scale for c in center])
    iris_to_world=matrix(half,center)
    iris_to_tessa=matrix(.5,(0,0,0))
    tessa_to_iris=matrix(2.,(0,0,0))
    world_to_tessa=four_by_four(iris_to_tessa,world_to_iris)
    tessa_to_world=four_by_four(iris_to_world,tessa_to_iris)
    for m,n in ((world_to_iris,iris_to_world),
                (world_to_tessa,tessa_to_world),
                (iris_to_tessa,tessa_to_iris)):
        require_identity(four_by_four(m,n))
        require_identity(four_by_four(n,m))
    # Source-bound corners only. Full source vertex/rig/skin/attachment
    # parity is a SEPARATE court and not claimed by an AABB roundtrip.
    corners=[
        [mn[0] if ix==0 else mx[0],
         mn[1] if iy==0 else mx[1],
         mn[2] if iz==0 else mx[2]]
        for ix in (0,1) for iy in (0,1) for iz in (0,1)
    ]
    worst=0.
    for p in corners:
        normalized=point(world_to_iris,p)
        require(max(abs(z) for z in normalized)<=1.+1e-6,
                "IRIS_DOMAIN_DRIFT")
        tessa=point(iris_to_tessa,normalized)
        require(max(abs(z) for z in tessa)<=.5+1e-6,
                "TESSA_DOMAIN_DRIFT")
        recovered=point(iris_to_world,normalized)
        worst=max(worst,max(abs(u-v) for u,v in zip(p,recovered)))
    require(worst/max_span<=1e-6,"CORNERS_ROUNDTRIP_RELATIVE_ERROR")
    result={
        "schema":SCHEMA,
        "status":"REVERSIBLE_PLAN_NUMERICALLY_CHECKED__NOT_SOURCE_PARITY",
        "canonical_asset_id":aid,"candidate_id":cid,
        "original_source_sha256":original,
        "bounds_observation_sha256":receipt_sha,
        "source_world_min":mn,"source_world_max":mx,
        "global_source_center":center,
        "global_source_half_extent":half,
        "source_world_to_iris":world_to_iris,
        "iris_to_source_world":iris_to_world,
        "iris_to_tessa":iris_to_tessa,
        "tessa_to_iris":tessa_to_iris,
        "source_world_to_tessa":world_to_tessa,
        "tessa_to_source_world":tessa_to_world,
        "coordinate_units":"SOURCE_IMPORTED_WORLD_UNITS_UNKNOWN",
        "result_unit":"DIMENSIONLESS",
        "source_anatomical_up":"UNKNOWN",
        "source_anatomical_front":"UNKNOWN",
        "source_meter_unit":"UNKNOWN",
        "common_global_scale_for_all_views_and_poses":True,
        "per_view_recenter_permitted":False,
        "per_view_renormalization_permitted":False,
        "source_bind_rest_parity_courted":False,
        "source_mesh_vertex_roundtrip_courted":False,
        "source_skin_bind_and_weight_parity_courted":False,
        "source_rig_controller_response_courted":False,
        "source_material_texture_rgba_courted":False,
        "source_attachment_transform_parity_courted":False,
        "source_camera_projection_parity_courted":False,
        "source_semantics_courted":False,
        "max_source_bbox_corner_roundtrip_error":worst,
        "max_source_bbox_corner_relative_error":worst/max_span,
        "normalization_transform_application_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bounds-observation",type=Path,required=True)
    p.add_argument("--bounds-observation-sha256",type=str)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    j=plan(a.bounds_observation,
           expected_sha256=a.bounds_observation_sha256)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(j,indent=2,sort_keys=True,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "asset_id":j["canonical_asset_id"],
        "corner_relative_error":j["max_source_bbox_corner_relative_error"],
        "source_full_parity_courted":False,
        "materialization_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
