from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from models.iris.v3.dense_source_sampling_v3 import camera_pixel_ray_origins_normalized
from models.iris.v4.source_constraint_v4_1 import sample_hard_positive_refresh_candidates_v41
from tools.iris_v4_3_foreground_hull_feasibility import _load_evidence


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ray_triangle_any_hit(
    origins: np.ndarray,
    directions: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    epsilon: float = 1e-9,
    triangle_chunk: int = 512,
) -> np.ndarray:
    """Continuous two-sided Moller-Trumbore surface hit test; no signed volume."""
    o=np.asarray(origins,dtype=np.float64)
    d=np.asarray(directions,dtype=np.float64)
    v=np.asarray(vertices,dtype=np.float64)
    f=np.asarray(faces,dtype=np.int64)
    if o.ndim!=2 or o.shape[1]!=3 or d.shape!=o.shape:
        raise ValueError("ray arrays must be [R,3]")
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3:
        raise ValueError("mesh arrays invalid")
    hit=np.zeros(o.shape[0],dtype=bool)
    for start in range(0,f.shape[0],int(triangle_chunk)):
        tri=v[f[start:start+int(triangle_chunk)]]
        v0=tri[:,0]; e1=tri[:,1]-v0; e2=tri[:,2]-v0
        pvec=np.cross(d[:,None,:],e2[None,:,:])
        det=np.einsum("tj,rtj->rt",e1,pvec)
        valid=np.abs(det)>float(epsilon)
        inv=np.zeros_like(det); inv[valid]=1.0/det[valid]
        tvec=o[:,None,:]-v0[None,:,:]
        u=np.einsum("rtj,rtj->rt",tvec,pvec)*inv
        valid &= (u>=-epsilon)&(u<=1.0+epsilon)
        qvec=np.cross(tvec,e1[None,:,:])
        vv=np.einsum("rj,rtj->rt",d,qvec)*inv
        valid &= (vv>=-epsilon)&((u+vv)<=1.0+epsilon)
        t=np.einsum("tj,rtj->rt",e2,qvec)*inv
        valid &= t>=-epsilon
        hit |= np.any(valid,axis=1)
        if np.all(hit):
            break
    return hit


def run_audit(
    *,
    evidence_zip: Path,
    teacher_truth_npz: Path,
    fit_seed: int=26091909,
    rays_per_view: int=256,
) -> dict:
    import tempfile, zipfile

    truth=np.load(teacher_truth_npz,allow_pickle=False)
    vertices_world=np.asarray(truth["vertices"],dtype=np.float64)
    faces=np.asarray(truth["faces"],dtype=np.int64)
    mesh_names=np.asarray(truth["mesh_names"]) if "mesh_names" in truth.files else np.asarray([])

    with tempfile.TemporaryDirectory(prefix="realsas-v43-truth-surface-") as tmp:
        root=Path(tmp)
        with zipfile.ZipFile(evidence_zip,"r") as zf:
            zf.extractall(root)
        masks,cameras,center,half=_load_evidence(root)
        vertices=(vertices_world-center[None,:])/float(half)

        rows=[]; total_hit=0; total_miss=0; boundary_miss=0; interior_miss=0; all_depth=[]
        for view,(mask,camera) in enumerate(zip(masks,cameras)):
            state_foreground=np.asarray(mask,dtype=bool)
            # Exact V4.3 fixed audit hard-positive refresh candidate semantics.
            from models.iris.v3.dense_source_sampling_v3 import admitted_pixel_mask_for_camera
            from models.iris.v4.source_constraint_v4 import build_source_constraint_view_state_v4
            admitted=admitted_pixel_mask_for_camera(
                width=1024,height=1024,
                camera_origin_normalized=camera["origin"],
                camera_right=camera["right"],
                camera_screen_up=camera["up"],
                camera_forward=camera["forward"],
                camera_half_extent_normalized=camera["half_extent"],
            )
            state=build_source_constraint_view_state_v4(state_foreground,admitted,near_boundary_max_px=32.0)
            indices=sample_hard_positive_refresh_candidates_v41(
                state,fit_seed=int(fit_seed)+404,refresh_index=0,view_index=view,count=int(rays_per_view)
            )
            origins,directions=camera_pixel_ray_origins_normalized(
                indices,width=1024,height=1024,
                camera_origin_normalized=camera["origin"],
                camera_right=camera["right"],
                camera_screen_up=camera["up"],
                camera_forward=camera["forward"],
                camera_half_extent_normalized=camera["half_extent"],
            )
            hit=_ray_triangle_any_hit(origins,directions,vertices,faces)
            interior=ndimage.binary_erosion(state_foreground,structure=np.ones((3,3),dtype=bool),border_value=0)
            boundary=state_foreground&~interior
            inside_distance=ndimage.distance_transform_edt(state_foreground)
            miss=~hit; miss_idx=indices[miss]; miss_depth=inside_distance.reshape(-1)[miss_idx]
            bmiss=int(np.count_nonzero(boundary.reshape(-1)[miss_idx]))
            imiss=int(np.count_nonzero(interior.reshape(-1)[miss_idx]))
            if bmiss+imiss!=int(miss_idx.size):
                raise RuntimeError("V43_TRUTH_SURFACE_MISS_CLASSIFICATION_DRIFT")
            row={
                "view_index":view,
                "ray_count":int(indices.size),
                "truth_surface_hit_count":int(np.count_nonzero(hit)),
                "truth_surface_miss_count":int(np.count_nonzero(miss)),
                "truth_surface_hit_fraction":float(np.mean(hit)),
                "boundary_miss_count":bmiss,
                "interior_miss_count":imiss,
                "missed_source_inside_distance_px":{
                    "min":float(np.min(miss_depth)) if miss_depth.size else None,
                    "median":float(np.median(miss_depth)) if miss_depth.size else None,
                    "max":float(np.max(miss_depth)) if miss_depth.size else None,
                },
            }
            rows.append(row)
            total_hit+=row["truth_surface_hit_count"]; total_miss+=row["truth_surface_miss_count"]
            boundary_miss+=bmiss; interior_miss+=imiss; all_depth.extend(map(float,miss_depth.tolist()))

    total=total_hit+total_miss
    out={
        "schema":"RealSaS.IRIS.V43ForegroundTruthSurfaceReferenceReproduction.v1",
        "status":"NON_NORMATIVE_TRUTH_SURFACE_REFERENCE__SOURCE_RASTER_REMAINS_AUTHORITY",
        "inputs":{
            "stage08_evidence_sha256":_sha256(evidence_zip),
            "teacher_truth_fit_only_sha256":_sha256(teacher_truth_npz),
            "truth_vertices":int(vertices_world.shape[0]),
            "truth_faces":int(faces.shape[0]),
            "truth_mesh_objects":int(mesh_names.size),
        },
        "result":{
            "total_rays":int(total),
            "truth_surface_hit_count":int(total_hit),
            "truth_surface_miss_count":int(total_miss),
            "truth_surface_hit_fraction":float(total_hit/max(total,1)),
            "truth_surface_miss_fraction":float(total_miss/max(total,1)),
            "boundary_miss_count":int(boundary_miss),
            "interior_miss_count":int(interior_miss),
            "missed_source_inside_distance_px":{
                "min":float(np.min(all_depth)) if all_depth else None,
                "median":float(np.median(all_depth)) if all_depth else None,
                "max":float(np.max(all_depth)) if all_depth else None,
            },
            "per_view":rows,
        },
        "signed_sdf_used":False,
        "inside_outside_assigned":False,
        "claim_boundary":"Knight FIT reference only; no Stage13/product/capacity/unseen/signed-volume claim.",
    }
    if out["result"]["total_rays"]!=2048 or out["result"]["truth_surface_miss_count"]!=38:
        raise RuntimeError("V43_TRUTH_SURFACE_REFERENCE_WITNESS_DRIFT")
    if out["result"]["boundary_miss_count"]!=36 or out["result"]["interior_miss_count"]!=2:
        raise RuntimeError("V43_TRUTH_SURFACE_REFERENCE_CLASSIFICATION_DRIFT")
    return out


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence-zip",type=Path,required=True)
    ap.add_argument("--teacher-truth-npz",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--fit-seed",type=int,default=26091909)
    ap.add_argument("--rays-per-view",type=int,default=256)
    a=ap.parse_args()
    result=run_audit(
        evidence_zip=a.evidence_zip,teacher_truth_npz=a.teacher_truth_npz,
        fit_seed=a.fit_seed,rays_per_view=a.rays_per_view,
    )
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
