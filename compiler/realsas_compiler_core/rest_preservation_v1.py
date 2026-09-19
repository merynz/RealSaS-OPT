from __future__ import annotations

"""Rest/source preservation measurements.

This module measures; it does not choose admission thresholds. Product PASS policy is a
separate preregistered authority so Knight results cannot tune their own gate.
"""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt

from .hashing import content_sha256
from .mesh.product_coverage_v1 import coverage_metrics, mask_sha256
from .observation_authority_v1 import QualifiedObservationSetIR, validate_qualified_observation_set
from .rest_render_v1 import RestRenderSetIR, rgba_sha256
from .types import QualificationError

Json=dict[str,Any]

REST_PRESERVATION_METRIC_CONTRACT={
    "schema":"RealSaS.RestPreservationMetricContract.v1",
    "source_foreground_authority":"QUALIFIED_OBSERVATION_FOREGROUND_MASK",
    "render_foreground":"RENDERED_RGBA_ALPHA_GT_ZERO",
    "coverage_metrics":"G5_COMPATIBLE_BINARY_COVERAGE",
    "silhouette_edge":"SYMMETRIC_NEAREST_BOUNDARY_DISTANCE_PIXELS",
    "color":"PREMULTIPLIED_RGB_ABSOLUTE_ERROR_ON_FOREGROUND_UNION",
    "alpha":"ABSOLUTE_ALPHA_ERROR_ON_FOREGROUND_UNION",
    "donor":"Z_VISIBLE_GEOMETRY_DONOR_ACCOUNTING_FROM_REST_RENDER",
    "thresholds":"NOT_DEFINED_HERE",
}
REST_PRESERVATION_METRIC_CONTRACT_HASH=content_sha256(REST_PRESERVATION_METRIC_CONTRACT)


@dataclass(frozen=True)
class RestPreservationViewMeasurementIR:
    view_index:int
    source_raster_sha256:str
    source_foreground_sha256:str
    rendered_rgba_sha256:str
    rendered_foreground_sha256:str
    alpha_recall:float
    alpha_precision:float
    alpha_iou:float
    largest_coherent_hole_fraction:float
    interior_uncovered_fraction:float
    silhouette_edge_mean_px:float
    silhouette_edge_p95_px:float
    silhouette_edge_max_px:float
    premultiplied_rgb_mae:float
    premultiplied_rgb_p95:float
    alpha_mae:float
    direct_source_geometry_fraction:float
    cross_view_source_geometry_fraction:float
    metric_contract_hash:str=REST_PRESERVATION_METRIC_CONTRACT_HASH
    schema_version:str="RealSaS.RestPreservationViewMeasurementIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RestPreservationMeasurementSetIR:
    views:tuple[RestPreservationViewMeasurementIR,...]
    rest_render_set_binding_hash:str
    observation_set_binding_hash:str
    measurement_set_hash:str
    schema_version:str="RealSaS.RestPreservationMeasurementSetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def rest_preservation_measurement_set_hash(value:RestPreservationMeasurementSetIR)->str:
    payload=value.to_dict()
    payload.pop("measurement_set_hash",None)
    return content_sha256(payload)


def _boundary(mask:np.ndarray)->np.ndarray:
    m=np.asarray(mask,dtype=bool)
    if m.ndim!=2:
        raise QualificationError("REST_PRESERVATION_MASK_DIMENSION_INVALID")
    h,w=m.shape
    out=np.zeros_like(m)
    ys,xs=np.nonzero(m)
    for y,x in zip(ys,xs):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0:
                    continue
                yy=y+dy; xx=x+dx
                if yy<0 or yy>=h or xx<0 or xx>=w or not m[yy,xx]:
                    out[y,x]=True
                    break
            if out[y,x]:
                break
    return out


def _silhouette_distance(source:np.ndarray,predicted:np.ndarray)->tuple[float,float,float]:
    a=_boundary(source); b=_boundary(predicted)
    if not np.any(a) or not np.any(b):
        raise QualificationError("REST_PRESERVATION_EMPTY_SILHOUETTE_BOUNDARY")
    dist_to_b=distance_transform_edt(~b)
    dist_to_a=distance_transform_edt(~a)
    values=np.concatenate((dist_to_b[a],dist_to_a[b])).astype(np.float64)
    if values.size==0 or not np.isfinite(values).all():
        raise QualificationError("REST_PRESERVATION_SILHOUETTE_DISTANCE_INVALID")
    return (
        float(np.mean(values)),
        float(np.percentile(values,95.0)),
        float(np.max(values)),
    )


def _appearance_error(source_rgba:np.ndarray,rendered_rgba:np.ndarray,union:np.ndarray):
    if source_rgba.shape!=rendered_rgba.shape or source_rgba.ndim!=3 or source_rgba.shape[2]!=4:
        raise QualificationError("REST_PRESERVATION_RGBA_SHAPE_MISMATCH")
    if source_rgba.dtype!=np.uint8 or rendered_rgba.dtype!=np.uint8:
        raise QualificationError("REST_PRESERVATION_RGBA_UINT8_REQUIRED")
    if not np.any(union):
        raise QualificationError("REST_PRESERVATION_EMPTY_FOREGROUND_UNION")
    s=source_rgba.astype(np.float64)/255.0
    r=rendered_rgba.astype(np.float64)/255.0
    sa=s[...,3:4]; ra=r[...,3:4]
    sp=s[...,:3]*sa; rp=r[...,:3]*ra
    rgb_pixel=np.mean(np.abs(sp-rp),axis=2)
    alpha=np.abs(sa[...,0]-ra[...,0])
    values=rgb_pixel[union]
    return (
        float(np.mean(values)),
        float(np.percentile(values,95.0)),
        float(np.mean(alpha[union])),
    )


def measure_rest_source_preservation(
    *,
    rest_render_set:RestRenderSetIR,
    observation_set:QualifiedObservationSetIR,
    source_rgba_by_view:Mapping[int,np.ndarray],
    rendered_rgba_by_view:Mapping[int,np.ndarray],
    source_foreground_by_view:Mapping[int,bytes],
)->RestPreservationMeasurementSetIR:
    validate_qualified_observation_set(observation_set)
    if set(source_rgba_by_view)!=set(range(8)) or set(rendered_rgba_by_view)!=set(range(8)) or set(source_foreground_by_view)!=set(range(8)):
        raise QualificationError("REST_PRESERVATION_VIEW_SET_INCOMPLETE")
    if rest_render_set.observation_set_binding_hash!=observation_set.observation_set_hash:
        raise QualificationError("REST_PRESERVATION_OBSERVATION_BINDING_DRIFT")
    obs={int(row.view_index):row for row in observation_set.views}
    render_rows={int(row.view_index):row for row in rest_render_set.views}
    rows=[]
    for view in range(8):
        authority=obs[view]; render_row=render_rows[view]
        source=np.ascontiguousarray(source_rgba_by_view[view],dtype=np.uint8)
        rendered=np.ascontiguousarray(rendered_rgba_by_view[view],dtype=np.uint8)
        h=int(authority.height); w=int(authority.width)
        if source.shape!=(h,w,4) or rendered.shape!=(h,w,4):
            raise QualificationError("REST_PRESERVATION_IMAGE_DIMENSION_DRIFT")
        if rgba_sha256(rendered)!=render_row.rendered_rgba_sha256:
            raise QualificationError("REST_PRESERVATION_RENDER_RGBA_HASH_DRIFT")
        fg=bytes(source_foreground_by_view[view])
        if len(fg)!=h*w or mask_sha256(fg)!=authority.foreground_mask_sha256:
            raise QualificationError("REST_PRESERVATION_SOURCE_FOREGROUND_AUTHORITY_DRIFT")
        source_mask=np.frombuffer(fg,dtype=np.uint8).reshape(h,w).astype(bool)
        rendered_mask=rendered[...,3]>0
        predicted=bytes(rendered_mask.astype(np.uint8).reshape(-1))
        coverage=coverage_metrics(fg,predicted,width=w,height=h)
        edge_mean,edge_p95,edge_max=_silhouette_distance(source_mask,rendered_mask)
        union=source_mask|rendered_mask
        rgb_mae,rgb_p95,alpha_mae=_appearance_error(source,rendered,union)
        geometry=max(1,int(render_row.geometry_visible_pixel_count))
        direct=float(render_row.direct_source_geometry_pixel_count)/float(geometry)
        cross=float(render_row.cross_view_source_geometry_pixel_count)/float(geometry)
        rows.append(RestPreservationViewMeasurementIR(
            view_index=view,
            source_raster_sha256=authority.source_raster_sha256,
            source_foreground_sha256=authority.foreground_mask_sha256,
            rendered_rgba_sha256=render_row.rendered_rgba_sha256,
            rendered_foreground_sha256=mask_sha256(predicted),
            alpha_recall=float(coverage["recall"]),
            alpha_precision=float(coverage["precision"]),
            alpha_iou=float(coverage["iou"]),
            largest_coherent_hole_fraction=float(coverage["largest_coherent_hole_fraction"]),
            interior_uncovered_fraction=float(coverage["interior_uncovered_fraction"]),
            silhouette_edge_mean_px=edge_mean,
            silhouette_edge_p95_px=edge_p95,
            silhouette_edge_max_px=edge_max,
            premultiplied_rgb_mae=rgb_mae,
            premultiplied_rgb_p95=rgb_p95,
            alpha_mae=alpha_mae,
            direct_source_geometry_fraction=direct,
            cross_view_source_geometry_fraction=cross,
            metadata={
                "threshold_applied":False,
                "source_raster_bytes_verified_elsewhere":True,
                "direct_source_only_rest_rule_measured":True,
            },
        ))
    value=RestPreservationMeasurementSetIR(
        views=tuple(rows),
        rest_render_set_binding_hash=rest_render_set.render_set_hash,
        observation_set_binding_hash=observation_set.observation_set_hash,
        measurement_set_hash="",
        metadata={
            "metric_contract":REST_PRESERVATION_METRIC_CONTRACT,
            "metric_contract_hash":REST_PRESERVATION_METRIC_CONTRACT_HASH,
            "admission_status":"MEASURED_NOT_ADMITTED",
            "thresholds_from_knight_results_forbidden":True,
        },
    )
    value=replace(value,measurement_set_hash=rest_preservation_measurement_set_hash(value))
    return value
