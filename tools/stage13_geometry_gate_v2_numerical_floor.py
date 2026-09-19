from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics,
    rasterize_triangles_half_integer_top_left,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    CameraProjectionV3,
    project_points_xyz_v3,
)
from compiler.realsas_compiler_core.rest_preservation_v1 import _silhouette_distance
from models.iris.v3.zero_surface_decoder_v3 import extract_zero_surface_mesh_v3

OUT = Path(os.environ.get("REALSAS_NUMERICAL_OUT", "numerical_out"))
OUT.mkdir(parents=True, exist_ok=True)
R = 256
RES = 1024
PROFILE = {
    "profile_id": "P999",
    "min_recall": 0.999,
    "min_precision": 0.999,
    "max_largest_coherent_hole_fraction": 0.00025,
    "max_interior_uncovered_fraction": 0.0005,
    "max_silhouette_edge_p95_px": 0.5,
}


def _write(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cams():
    rows=[]
    for view in range(8):
        yaw=math.radians(45.0*view)
        forward=np.asarray([-math.sin(yaw),-math.cos(yaw),0.0],dtype=np.float64)
        right=np.asarray([-math.cos(yaw),math.sin(yaw),0.0],dtype=np.float64)
        rows.append(CameraProjectionV3(
            f"V{view}", view,
            tuple((-4.0*forward).tolist()),
            tuple(right.tolist()),
            (0.0,0.0,1.0),
            tuple(forward.tolist()),
            1.0, RES,
        ))
    return rows


def _field(kind: str):
    axis=np.linspace(-1.0,1.0,R,dtype=np.float32)
    z,y,x=np.meshgrid(axis,axis,axis,indexing="ij")
    if kind=="SPHERE":
        r=0.65
        return np.sqrt(x*x+y*y+z*z).astype(np.float32)-r
    if kind=="ELLIPSOID":
        a,b,c=0.68,0.42,0.78
        return (np.sqrt((x/a)**2+(y/b)**2+(z/c)**2)-1.0).astype(np.float32)
    raise KeyError(kind)


def _source_mask(kind: str, camera: CameraProjectionV3):
    yy,xx=np.mgrid[0:RES,0:RES]
    X=xx+0.5
    Y=yy+0.5
    cx=cy=RES/2.0
    if kind=="SPHERE":
        rx=ry=0.65*(RES/2.0)
    elif kind=="ELLIPSOID":
        a,b,c=0.68,0.42,0.78
        right=np.asarray(camera.right,dtype=np.float64)
        rx=math.sqrt((a*right[0])**2+(b*right[1])**2+(c*right[2])**2)*(RES/2.0)
        ry=c*(RES/2.0)
    else:
        raise KeyError(kind)
    return (((X-cx)/rx)**2+((Y-cy)/ry)**2)<=1.0


def _render(mesh, camera):
    projected=project_points_xyz_v3(mesh.vertices_normalized,camera)
    triangles=[]
    for f in np.asarray(mesh.faces,dtype=np.int64):
        a,b,c=(projected[int(i)] for i in f)
        triangles.append(((float(a[0]),float(a[1])),(float(b[0]),float(b[1])),(float(c[0]),float(c[1]))))
    raw=rasterize_triangles_half_integer_top_left(triangles,width=RES,height=RES)
    return np.frombuffer(raw,dtype=np.uint8).reshape(RES,RES).astype(bool)


def _metrics(source,pred):
    raw_s=np.asarray(source,dtype=np.uint8).reshape(-1).tobytes()
    raw_p=np.asarray(pred,dtype=np.uint8).reshape(-1).tobytes()
    cov=coverage_metrics(raw_s,raw_p,width=RES,height=RES)
    em,ep95,emax=_silhouette_distance(source,pred)
    return {
        "recall":float(cov["recall"]),
        "precision":float(cov["precision"]),
        "largest_coherent_hole_fraction":float(cov["largest_coherent_hole_fraction"]),
        "interior_uncovered_fraction":float(cov["interior_uncovered_fraction"]),
        "silhouette_edge_mean_px":float(em),
        "silhouette_edge_p95_px":float(ep95),
        "silhouette_edge_max_px":float(emax),
    }


def _pass(m):
    return (
        m["recall"]>=PROFILE["min_recall"]
        and m["precision"]>=PROFILE["min_precision"]
        and m["largest_coherent_hole_fraction"]<=PROFILE["max_largest_coherent_hole_fraction"]
        and m["interior_uncovered_fraction"]<=PROFILE["max_interior_uncovered_fraction"]
        and m["silhouette_edge_p95_px"]<=PROFILE["max_silhouette_edge_p95_px"]
    )


def main():
    import skimage
    out={
        "schema":"RealSaS.Stage13GeometryGateV2NumericalFloor.v1",
        "status":"PASS",
        "subject_inputs_used":False,
        "knight_result_used":False,
        "mage_result_used":False,
        "decoder_resolution":R,
        "raster_resolution":RES,
        "decoder_id":"RealSaS.ZeroSurfaceDecoder.MarchingCubes.v3",
        "skimage_version":skimage.__version__,
        "tested_profile":PROFILE,
        "shapes":{},
    }
    all_pass=True
    for kind in ("SPHERE","ELLIPSOID"):
        grid=_field(kind)
        mesh=extract_zero_surface_mesh_v3(grid,bounds=(-1.0,1.0),level=0.0)
        rows=[]
        for cam in _cams():
            src=_source_mask(kind,cam)
            pred=_render(mesh,cam)
            m=_metrics(src,pred)
            m["view_index"]=cam.view_index
            m["passed"]=_pass(m)
            rows.append(m)
            all_pass &= bool(m["passed"])
        out["shapes"][kind]={
            "vertex_count":int(len(mesh.vertices_normalized)),
            "face_count":int(len(mesh.faces)),
            "field_min":float(mesh.field_min),
            "field_max":float(mesh.field_max),
            "views":rows,
            "worst":{
                "min_recall":min(r["recall"] for r in rows),
                "min_precision":min(r["precision"] for r in rows),
                "max_largest_coherent_hole_fraction":max(r["largest_coherent_hole_fraction"] for r in rows),
                "max_interior_uncovered_fraction":max(r["interior_uncovered_fraction"] for r in rows),
                "max_silhouette_edge_p95_px":max(r["silhouette_edge_p95_px"] for r in rows),
            },
        }
    out["status"]="PASS" if all_pass else "FAIL_P999_BELOW_NUMERICAL_FLOOR"
    _write("STAGE13_GEOMETRY_GATE_V2_NUMERICAL_FLOOR.json",out)
    print(json.dumps(out,indent=2,sort_keys=True))
    return 0 if all_pass else 2


if __name__=="__main__":
    raise SystemExit(main())
