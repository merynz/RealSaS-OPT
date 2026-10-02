from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,signed_zero_surface_from_dict,
)

MIN_ANGLE=7.5
MAX_ASPECT=16.0

def loadj(p): return json.loads(Path(p).read_text())

def faces_for(faces,inv):
    mf=inv[np.asarray(faces,dtype=np.int64)]
    valid=(mf[:,0]!=mf[:,1])&(mf[:,1]!=mf[:,2])&(mf[:,2]!=mf[:,0])
    return np.unique(np.sort(mf[valid],axis=1),axis=0)

def centroid(points,inv):
    n=int(inv.max())+1
    count=np.bincount(inv,minlength=n).astype(np.float64)
    out=np.zeros((n,3),dtype=np.float64)
    np.add.at(out,inv,points)
    out/=count[:,None]
    return out

def medoid_nearest_mean(points,inv,mean):
    d2=np.sum((points-mean[inv])**2,axis=1)
    # sort primarily by cluster id, secondarily by distance, third by vertex id
    order=np.lexsort((np.arange(len(inv),dtype=np.int64),d2,inv))
    clusters=inv[order]
    first=np.r_[True,clusters[1:]!=clusters[:-1]]
    chosen=order[first]
    if len(chosen)!=len(mean):
        raise RuntimeError(f"MEDOID_CLUSTER_CARDINALITY::{len(chosen)}::{len(mean)}")
    # chosen arrives cluster-sorted because primary key is inv
    if not np.array_equal(inv[chosen],np.arange(len(mean),dtype=np.int64)):
        raise RuntimeError("MEDOID_CLUSTER_ORDER_DRIFT")
    return np.asarray(points[chosen],dtype=np.float64),chosen

def area_weighted(points,faces,inv):
    tri=points[faces]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*0.5
    w=np.zeros(len(points),dtype=np.float64)
    np.add.at(w,faces[:,0],area/3.0)
    np.add.at(w,faces[:,1],area/3.0)
    np.add.at(w,faces[:,2],area/3.0)
    # isolated should not occur; fallback unit weight.
    w=np.where(w>1e-30,w,1.0)
    n=int(inv.max())+1
    den=np.bincount(inv,weights=w,minlength=n)
    out=np.zeros((n,3),dtype=np.float64)
    for axis in range(3):
        out[:,axis]=np.bincount(inv,weights=w*points[:,axis],minlength=n)/den
    return out

def metrics(points,faces):
    tri=points[faces]
    e0=np.linalg.norm(tri[:,1]-tri[:,0],axis=1)
    e1=np.linalg.norm(tri[:,2]-tri[:,1],axis=1)
    e2=np.linalg.norm(tri[:,0]-tri[:,2],axis=1)
    a=e1;b=e2;c=e0
    c0=np.clip((b*b+c*c-a*a)/(2*b*c+1e-30),-1,1)
    c1=np.clip((a*a+c*c-b*b)/(2*a*c+1e-30),-1,1)
    c2=np.clip((a*a+b*b-c*c)/(2*a*b+1e-30),-1,1)
    ang=np.min(np.degrees(np.arccos(np.stack((c0,c1,c2),axis=1))),axis=1)
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    longest=np.maximum.reduce((e0,e1,e2))
    alt=np.divide(area2,longest,out=np.zeros_like(area2),where=longest>1e-15)
    aspect=np.divide(longest,alt,out=np.full_like(longest,np.inf),where=alt>1e-15)
    bad=(ang<MIN_ANGLE)|(aspect>MAX_ASPECT)|(~np.isfinite(aspect))
    finite=aspect[np.isfinite(aspect)]
    return {
      "face_count":int(len(faces)),
      "bad_face_count":int(np.count_nonzero(bad)),
      "bad_fraction":float(np.mean(bad)),
      "angle_lt_7_5_count":int(np.count_nonzero(ang<MIN_ANGLE)),
      "aspect_gt_16_count":int(np.count_nonzero(aspect>MAX_ASPECT)),
      "minimum_angle_deg":float(np.min(ang)),
      "angle_p01_deg":float(np.percentile(ang,1)),
      "angle_p05_deg":float(np.percentile(ang,5)),
      "max_aspect":float(np.max(aspect)),
      "aspect_p95":float(np.percentile(finite,95)),
      "aspect_p99":float(np.percentile(finite,99)),
      "degenerate_count":int(np.count_nonzero(area2<=1e-15)),
    }

def displacement(reference,candidate,half):
    d=np.linalg.norm(candidate-reference,axis=1)
    return {
      "world_max":float(np.max(d)),
      "world_p95":float(np.percentile(d,95)),
      "normalized_max":float(np.max(d)/half),
      "normalized_p95":float(np.percentile(d,95)/half),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64)
        faces=np.asarray(z["faces"],np.int64)
    with np.load(a.inverse_npz,allow_pickle=False) as z:
        inv=np.asarray(z["final_inverse"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)
    cf=faces_for(faces,inv)

    mean=centroid(world,inv)
    med,chosen=medoid_nearest_mean(world,inv,mean)
    aw=area_weighted(world,faces,inv)

    rows={
      "CENTROID_CURRENT":metrics(mean,cf),
      "SOURCE_SURFACE_MEDOID_NEAREST_CENTROID":metrics(med,cf),
      "DENSE_FACE_AREA_WEIGHTED_CENTROID":metrics(aw,cf),
    }
    rows["SOURCE_SURFACE_MEDOID_NEAREST_CENTROID"]["vs_current_centroid_displacement"]=displacement(mean,med,float(norm.half_extent))
    rows["DENSE_FACE_AREA_WEIGHTED_CENTROID"]["vs_current_centroid_displacement"]=displacement(mean,aw,float(norm.half_extent))
    report={
      "schema":"RealSaS.KnightRepairedRepresentativeQualityCourt.v1",
      "status":"MEASURED_AUDIT_ONLY__NO_REPAIR_APPLIED",
      "topology_fixed":True,
      "nonmanifold_input_claim":0,
      "policy":{"minimum_angle_deg":MIN_ANGLE,"maximum_aspect":MAX_ASPECT},
      "rows":rows,
      "medoid":{
        "selected_dense_vertex_count":int(len(chosen)),
        "all_representatives_are_exact_dense_vertices":True,
      },
      "claim_boundary":[
        "Topology/inverse mapping and exact dense-face quotient are identical across arms.",
        "Only compact representative positions change.",
        "This court does not promote a new Stage15 operator.",
      ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_REPAIRED_REPRESENTATIVE_QUALITY="+json.dumps({
      k:{x:v for x,v in row.items() if x in {"bad_face_count","minimum_angle_deg","max_aspect","angle_p05_deg","aspect_p95"}}
      for k,row in rows.items()
    },sort_keys=True))

if __name__=="__main__":main()
