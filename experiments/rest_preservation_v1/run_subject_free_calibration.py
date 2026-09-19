#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from compiler.realsas_compiler_core.mesh.product_coverage_v1 import mask_sha256
from compiler.realsas_compiler_core.observation_authority_v1 import (
    QualifiedObservationViewIR,build_qualified_observation_set,
)
from compiler.realsas_compiler_core.rest_preservation_v1 import (
    REST_PRESERVATION_METRIC_CONTRACT_HASH,
    measure_rest_source_preservation,
)
from compiler.realsas_compiler_core.rest_render_v1 import (
    RestRenderSetIR,RestRenderViewIR,rest_render_set_hash,rgba_sha256,
)

RESOLUTIONS=(64,128)
SHAPES=("RECTANGLE","DISK","CONCAVE_L","STEPPED_ASYMMETRIC")
SHIFT_CASES=("SHIFT_X_POS_1","SHIFT_X_NEG_1","SHIFT_Y_POS_1","SHIFT_Y_NEG_1")
G5_MESH_FLOORS={
    "min_recall":0.97,
    "min_precision":0.995,
    "max_largest_coherent_hole_fraction":0.005,
    "max_interior_uncovered_fraction":0.005,
}


def _mask(resolution:int,shape:str)->np.ndarray:
    h=w=int(resolution)
    yy,xx=np.mgrid[0:h,0:w]
    if shape=="RECTANGLE":
        return (xx>=w//4)&(xx<3*w//4)&(yy>=h//4)&(yy<3*h//4)
    if shape=="DISK":
        cx=(w-1)/2.0; cy=(h-1)/2.0; r=0.29*float(w)
        return (xx-cx)**2+(yy-cy)**2<=r*r
    if shape=="CONCAVE_L":
        vertical=(xx>=w//4)&(xx<w//4+w//7)&(yy>=h//5)&(yy<4*h//5)
        horizontal=(xx>=w//4)&(xx<3*w//4)&(yy>=3*h//5)&(yy<4*h//5)
        return vertical|horizontal
    if shape=="STEPPED_ASYMMETRIC":
        a=(xx>=w//5)&(xx<4*w//5)&(yy>=h//4)&(yy<h//2)
        b=(xx>=2*w//5)&(xx<4*w//5)&(yy>=h//2)&(yy<3*h//4)
        c=(xx>=3*w//5)&(xx<4*w//5)&(yy>=3*h//4)&(yy<4*h//5)
        return a|b|c
    raise ValueError(shape)


def _source_rgba(mask:np.ndarray,variant:int)->np.ndarray:
    h,w=mask.shape
    yy,xx=np.mgrid[0:h,0:w]
    out=np.zeros((h,w,4),dtype=np.uint8)
    out[...,0]=((3*xx+5*yy+17*variant)%251).astype(np.uint8)
    out[...,1]=((7*xx+2*yy+31+variant)%251).astype(np.uint8)
    out[...,2]=((xx+11*yy+53+3*variant)%251).astype(np.uint8)
    out[...,3]=np.where(mask,255,0).astype(np.uint8)
    out[~mask,:3]=0
    return out


def _shift(image:np.ndarray,dx:int,dy:int)->np.ndarray:
    out=np.zeros_like(image)
    h,w,_=image.shape
    src_x0=max(0,-dx); src_x1=min(w,w-dx)
    src_y0=max(0,-dy); src_y1=min(h,h-dy)
    dst_x0=max(0,dx); dst_x1=min(w,w+dx)
    dst_y0=max(0,dy); dst_y1=min(h,h+dy)
    out[dst_y0:dst_y1,dst_x0:dst_x1]=image[src_y0:src_y1,src_x0:src_x1]
    return out


def _coherent_hole(image:np.ndarray,mask:np.ndarray)->np.ndarray:
    out=image.copy()
    ys,xs=np.nonzero(mask)
    y0=int(np.percentile(ys,45)); y1=max(y0+2,int(np.percentile(ys,55))+1)
    x0=int(np.percentile(xs,45)); x1=max(x0+2,int(np.percentile(xs,55))+1)
    region=mask[y0:y1,x0:x1]
    out[y0:y1,x0:x1][region]=0
    return out


def _one_lsb(image:np.ndarray,mask:np.ndarray)->np.ndarray:
    out=image.copy()
    y,x=next(zip(*np.nonzero(mask)))
    channel=int((x+y)%3)
    value=int(out[y,x,channel])
    out[y,x,channel]=np.uint8(value+1 if value<255 else value-1)
    return out


def _observation_and_rest(
    source:np.ndarray,
    rendered:np.ndarray,
    source_mask:np.ndarray,
    *,
    cross_view:bool,
):
    h,w,_=source.shape
    fg=bytes(source_mask.astype(np.uint8).reshape(-1))
    source_id=hashlib.sha256(source.tobytes(order="C")).hexdigest()
    views=[]
    render_rows=[]
    geometry_visible=int(np.count_nonzero(rendered[...,3]>0))
    direct=0 if cross_view else geometry_visible
    cross=geometry_visible if cross_view else 0
    for view in range(8):
        views.append(QualifiedObservationViewIR(
            view,w,h,
            f"SYNTH_OBS_{source_id[:16]}_{view}",
            source_id,
            mask_sha256(fg),
            hashlib.sha256(f"CAM:{w}:{h}:{view}".encode()).hexdigest(),
            "PASS",
            (f"SYNTHETIC:{view}",),
        ))
        render_rows.append(RestRenderViewIR(
            view_index=view,
            camera_binding_hash=views[-1].camera_binding_hash,
            appearance_binding_hash=hashlib.sha256(f"APP:{view}".encode()).hexdigest(),
            composition_binding_hash=hashlib.sha256(f"COMP:{view}".encode()).hexdigest(),
            rendered_rgba_sha256=rgba_sha256(rendered),
            width=w,height=h,
            visible_pixel_count=int(np.count_nonzero(rendered[...,3])),
            geometry_visible_pixel_count=geometry_visible,
            direct_source_geometry_pixel_count=direct,
            cross_view_source_geometry_pixel_count=cross,
        ))
    obs=build_qualified_observation_set(tuple(views))
    rest=RestRenderSetIR(
        views=tuple(render_rows),
        mesh_binding_hash="SYNTH_MESH",
        presentation_binding_hash="SYNTH_PRESENTATION",
        appearance_set_binding_hash="SYNTH_APPEARANCE",
        composition_set_binding_hash="SYNTH_COMPOSITION",
        camera_set_binding_hash="SYNTH_CAMERAS",
        observation_set_binding_hash=obs.observation_set_hash,
        render_set_hash="",
        metadata={"synthetic_calibration":True},
    )
    rest=replace(rest,render_set_hash=rest_render_set_hash(rest))
    return obs,rest,fg


def _case(source:np.ndarray,source_mask:np.ndarray,case_id:str):
    cross=False
    if case_id=="IDENTITY":
        rendered=source.copy()
    elif case_id=="SHIFT_X_POS_1":
        rendered=_shift(source,1,0)
    elif case_id=="SHIFT_X_NEG_1":
        rendered=_shift(source,-1,0)
    elif case_id=="SHIFT_Y_POS_1":
        rendered=_shift(source,0,1)
    elif case_id=="SHIFT_Y_NEG_1":
        rendered=_shift(source,0,-1)
    elif case_id=="COHERENT_HOLE":
        rendered=_coherent_hole(source,source_mask)
    elif case_id=="RGB_ONE_LSB":
        rendered=_one_lsb(source,source_mask)
    elif case_id=="VISIBLE_CROSS_VIEW_DONOR":
        rendered=source.copy(); cross=True
    else:
        raise ValueError(case_id)
    obs,rest,fg=_observation_and_rest(source,rendered,source_mask,cross_view=cross)
    source_by={v:source for v in range(8)}
    rendered_by={v:rendered for v in range(8)}
    fg_by={v:fg for v in range(8)}
    measurement=measure_rest_source_preservation(
        rest_render_set=rest,
        observation_set=obs,
        source_rgba_by_view=source_by,
        rendered_rgba_by_view=rendered_by,
        source_foreground_by_view=fg_by,
    )
    return measurement


def run_calibration()->dict:
    case_rows=[]
    identity_p95=[]
    shift_p95=[]
    sensitivity={
        "identity_exact_rgba_and_direct_source":True,
        "rgb_one_lsb_detected":True,
        "visible_cross_view_detected":True,
        "coherent_hole_detected":True,
    }
    for resolution in RESOLUTIONS:
        for shape_index,shape in enumerate(SHAPES):
            mask=_mask(resolution,shape)
            source=_source_rgba(mask,shape_index+resolution)
            for case_id in ("IDENTITY",*SHIFT_CASES,"COHERENT_HOLE","RGB_ONE_LSB","VISIBLE_CROSS_VIEW_DONOR"):
                result=_case(source,mask,case_id)
                for row in result.views:
                    if case_id=="IDENTITY":
                        identity_p95.append(row.silhouette_edge_p95_px)
                        sensitivity["identity_exact_rgba_and_direct_source"] &= (
                            row.overlap_rgba_mismatch_pixel_count==0
                            and row.cross_view_source_geometry_fraction==0.0
                            and row.direct_source_geometry_fraction==1.0
                        )
                    if case_id in SHIFT_CASES:
                        shift_p95.append(row.silhouette_edge_p95_px)
                    if case_id=="RGB_ONE_LSB":
                        sensitivity["rgb_one_lsb_detected"] &= (
                            row.overlap_rgba_mismatch_pixel_count>=1
                            and row.overlap_rgba_max_abs_channel_error_u8>=1
                        )
                    if case_id=="VISIBLE_CROSS_VIEW_DONOR":
                        sensitivity["visible_cross_view_detected"] &= row.cross_view_source_geometry_fraction>0.0
                    if case_id=="COHERENT_HOLE":
                        sensitivity["coherent_hole_detected"] &= row.largest_coherent_hole_fraction>0.0
                sample=result.views[0]
                case_rows.append({
                    "resolution":resolution,
                    "shape":shape,
                    "case_id":case_id,
                    "view_metrics_identical":all(row.to_dict()==sample.to_dict() for row in result.views),
                    "silhouette_edge_p95_px":sample.silhouette_edge_p95_px,
                    "alpha_recall":sample.alpha_recall,
                    "alpha_precision":sample.alpha_precision,
                    "largest_coherent_hole_fraction":sample.largest_coherent_hole_fraction,
                    "interior_uncovered_fraction":sample.interior_uncovered_fraction,
                    "overlap_rgba_mismatch_pixel_count":sample.overlap_rgba_mismatch_pixel_count,
                    "overlap_rgba_max_abs_channel_error_u8":sample.overlap_rgba_max_abs_channel_error_u8,
                    "cross_view_source_geometry_fraction":sample.cross_view_source_geometry_fraction,
                })
    p=max(identity_p95)
    n=min(shift_p95)
    separated=bool(p<n)
    threshold=(float(p)+float(n))/2.0 if separated else None
    sensitivity_pass=all(bool(x) for x in sensitivity.values())
    status="PASS" if separated and sensitivity_pass else "FAIL"
    return {
        "schema":"RealSaS.RestSourcePreservationSubjectFreeCalibration.v1",
        "status":status,
        "authority":"SUBJECT_FREE_SYNTHETIC_ONLY__NO_KNIGHT_OR_MAGE_INPUT",
        "metric_contract_hash":REST_PRESERVATION_METRIC_CONTRACT_HASH,
        "resolutions":list(RESOLUTIONS),
        "shape_ids":list(SHAPES),
        "case_count":len(case_rows),
        "view_measurement_count":len(case_rows)*8,
        "case_rows":case_rows,
        "silhouette_selection":{
            "max_positive_identity_p95_px":float(p),
            "min_negative_one_pixel_shift_p95_px":float(n),
            "strictly_separated":separated,
            "selected_p95_ceiling_px":threshold,
            "selection_rule":"MIDPOINT(MAX_IDENTITY_P95,MIN_ONE_PIXEL_SHIFT_P95)",
        },
        "sensitivity":sensitivity,
        "inherited_g5_mesh_floors":G5_MESH_FLOORS,
        "exact_policy_candidates":{
            "max_overlap_rgba_mismatch_pixel_count":0,
            "max_overlap_rgba_max_abs_channel_error_u8":0,
            "max_cross_view_source_geometry_fraction":0.0,
        },
        "claim_boundary":"Calibration sensitivity and threshold selection only; no subject or product PASS.",
    }


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--json-out")
    args=parser.parse_args()
    report=run_calibration()
    text=json.dumps(report,indent=2,sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text+"\n",encoding="utf-8")
    return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
