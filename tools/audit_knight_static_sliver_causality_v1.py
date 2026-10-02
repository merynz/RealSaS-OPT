from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,signed_zero_surface_from_dict,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse

MIN_ANGLE_DEG=7.5
MAX_ASPECT=16.0

def loadj(p): return json.loads(Path(p).read_text())

def compact_points(points,inv):
    inv=np.asarray(inv,dtype=np.int64)
    n=int(inv.max())+1
    counts=np.bincount(inv,minlength=n).astype(np.float64)
    out=np.zeros((n,3),dtype=np.float64)
    np.add.at(out,inv,np.asarray(points,dtype=np.float64))
    out/=counts[:,None]
    return out

def compact_faces(faces,inv):
    mapped=np.asarray(inv,dtype=np.int64)[np.asarray(faces,dtype=np.int64)]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    return np.unique(np.sort(mapped[valid],axis=1),axis=0)

def quality(points,faces):
    p=np.asarray(points,dtype=np.float64); f=np.asarray(faces,dtype=np.int64)
    tri=p[f]
    e0=np.linalg.norm(tri[:,1]-tri[:,0],axis=1)
    e1=np.linalg.norm(tri[:,2]-tri[:,1],axis=1)
    e2=np.linalg.norm(tri[:,0]-tri[:,2],axis=1)
    edges=np.stack((e0,e1,e2),axis=1)
    a=e1;b=e2;c=e0
    cos0=np.clip((b*b+c*c-a*a)/(2*b*c+1e-30),-1,1)
    cos1=np.clip((a*a+c*c-b*b)/(2*a*c+1e-30),-1,1)
    cos2=np.clip((a*a+b*b-c*c)/(2*a*b+1e-30),-1,1)
    minang=np.min(np.degrees(np.arccos(np.stack((cos0,cos1,cos2),axis=1))),axis=1)
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    longest=np.max(edges,axis=1)
    min_alt=np.divide(area2,longest,out=np.zeros_like(area2),where=longest>1e-15)
    aspect=np.divide(longest,min_alt,out=np.full_like(longest,np.inf),where=min_alt>1e-15)
    bad=(minang<MIN_ANGLE_DEG)|(aspect>MAX_ASPECT)|(~np.isfinite(aspect))
    return minang,aspect,bad,area2,edges

def summary(points,faces):
    ang,asp,bad,area2,edges=quality(points,faces)
    finite=asp[np.isfinite(asp)]
    return {
      "face_count":int(len(faces)),
      "bad_face_count":int(np.count_nonzero(bad)),
      "bad_face_fraction":float(np.mean(bad)) if len(bad) else 0.0,
      "angle_lt_7_5_count":int(np.count_nonzero(ang<MIN_ANGLE_DEG)),
      "aspect_gt_16_count":int(np.count_nonzero(asp>MAX_ASPECT)),
      "minimum_angle_deg":float(np.min(ang)) if len(ang) else None,
      "angle_p01_deg":float(np.percentile(ang,1)) if len(ang) else None,
      "angle_p05_deg":float(np.percentile(ang,5)) if len(ang) else None,
      "max_aspect":float(np.max(asp)) if len(asp) else None,
      "aspect_p95":float(np.percentile(finite,95)) if len(finite) else None,
      "aspect_p99":float(np.percentile(finite,99)) if len(finite) else None,
      "min_double_area":float(np.min(area2)) if len(area2) else None,
    }

def dense_summary_chunked(points,faces,chunk=500_000):
    counts={"face_count":0,"bad_face_count":0,"angle_lt_7_5_count":0,"aspect_gt_16_count":0}
    amin=float("inf"); amax=0.0; minarea=float("inf")
    # Keep bounded samples only for percentiles.
    angle_samples=[]; aspect_samples=[]
    stride=max(1,len(faces)//1_000_000)
    for start in range(0,len(faces),chunk):
        f=np.asarray(faces[start:start+chunk],dtype=np.int64)
        ang,asp,bad,area2,_=quality(points,f)
        counts["face_count"]+=len(f)
        counts["bad_face_count"]+=int(np.count_nonzero(bad))
        counts["angle_lt_7_5_count"]+=int(np.count_nonzero(ang<MIN_ANGLE_DEG))
        counts["aspect_gt_16_count"]+=int(np.count_nonzero(asp>MAX_ASPECT))
        amin=min(amin,float(np.min(ang)))
        finite=asp[np.isfinite(asp)]
        if len(finite): amax=max(amax,float(np.max(finite)))
        minarea=min(minarea,float(np.min(area2)))
        local=np.arange(start,start+len(f))
        take=((local%stride)==0)
        angle_samples.append(ang[take])
        aspect_samples.append(asp[take & np.isfinite(asp)])
    aa=np.concatenate(angle_samples) if angle_samples else np.zeros(0)
    ss=np.concatenate(aspect_samples) if aspect_samples else np.zeros(0)
    return {
      **counts,
      "bad_face_fraction":float(counts["bad_face_count"]/max(counts["face_count"],1)),
      "minimum_angle_deg":amin if math.isfinite(amin) else None,
      "angle_sample_p01_deg":float(np.percentile(aa,1)) if len(aa) else None,
      "angle_sample_p05_deg":float(np.percentile(aa,5)) if len(aa) else None,
      "max_aspect":amax,
      "aspect_sample_p95":float(np.percentile(ss,95)) if len(ss) else None,
      "aspect_sample_p99":float(np.percentile(ss,99)) if len(ss) else None,
      "min_double_area":minarea if math.isfinite(minarea) else None,
      "percentile_sampling_stride":int(stride),
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
        vn=np.asarray(z["vertices_normalized"],dtype=np.float64)
        faces=np.asarray(z["faces"],dtype=np.int64)
    with np.load(a.inverse_npz,allow_pickle=False) as z:
        repaired=np.asarray(z["final_inverse"],dtype=np.int64)
        sealed_base=np.asarray(z["base_inverse"],dtype=np.int64)
    world=np.asarray(norm.center_xyz,dtype=np.float64)[None,:]+vn*float(norm.half_extent)

    # Recompute exact historical smear-base mapping independently and bind it to seal.
    base=compact_inverse(world,faces,divisions=56,component_aware=True)
    if not np.array_equal(base,sealed_base):
        raise RuntimeError("SLIVER_CAUSALITY_BASE_MAPPING_DRIFT")

    bp=compact_points(world,base)
    rp=compact_points(world,repaired)
    bf=compact_faces(faces,base)
    rf=compact_faces(faces,repaired)

    dense_stats=dense_summary_chunked(world,faces)
    base_stats=summary(bp,bf)
    repaired_stats=summary(rp,rf)

    # Every final cluster must be a strict child of exactly one base cluster.
    child_parent={}
    parent_children=defaultdict(list)
    for b,r in np.unique(np.column_stack((base,repaired)),axis=0):
        old=child_parent.setdefault(int(r),int(b))
        if old!=int(b): raise RuntimeError("SLIVER_CAUSALITY_CROSS_BASE_MERGE")
        parent_children[int(b)].append(int(r))
    split_parent_count=sum(len(v)>1 for v in parent_children.values())
    split_child_count=sum(len(v) for v in parent_children.values() if len(v)>1)

    # Base quality lookup for exact compact faces.
    bang,basp,bbad,_ba,_be=quality(bp,bf)
    base_face_index={tuple(map(int,row)):i for i,row in enumerate(bf.tolist())}

    rang,rasp,rbad,_ra,_re=quality(rp,rf)
    classes=defaultdict(int)
    examples=defaultdict(list)
    bad_codes=[]
    nrep=int(repaired.max())+1
    for i,row in enumerate(rf):
        if not bool(rbad[i]): continue
        parents=tuple(sorted(child_parent[int(x)] for x in row))
        if len(set(parents))<3:
            cls="REPAIR_EXPOSED_PREVIOUSLY_COLLAPSED_DENSE_FACE"
        else:
            bi=base_face_index.get(parents)
            if bi is None:
                cls="BASE_FACE_WITNESS_MISSING_UNEXPECTED"
            elif bool(bbad[bi]):
                cls="PREEXISTING_BASE_SLIVER"
            else:
                cls="REFINEMENT_DEGRADED_PREVIOUSLY_PASSING_BASE_FACE"
        classes[cls]+=1
        if len(examples[cls])<12:
            rec={
              "repaired_face_index":int(i),
              "repaired_face":list(map(int,row)),
              "parent_base_face":list(map(int,parents)),
              "repaired_min_angle_deg":float(rang[i]),
              "repaired_aspect":float(rasp[i]),
            }
            if len(set(parents))==3 and parents in base_face_index:
                bi=base_face_index[parents]
                rec["base_min_angle_deg"]=float(bang[bi])
                rec["base_aspect"]=float(basp[bi])
            examples[cls].append(rec)
        s=sorted(map(int,row))
        bad_codes.append(((s[0]*nrep)+s[1])*nrep+s[2])

    # For repaired bad faces, inspect exact dense triangle witnesses. This is
    # diagnostic only: product 7.5deg/16 thresholds do not qualify the dense decoder
    # mesh, but they reveal whether compact slivers are inherited vs centroid-created.
    bad_codes=np.asarray(sorted(set(bad_codes)),dtype=np.int64)
    witness_count=defaultdict(int)
    witness_bad_count=defaultdict(int)
    witness_min_angle=defaultdict(lambda:float("inf"))
    witness_max_aspect=defaultdict(float)
    chunk=400_000
    for start in range(0,len(faces),chunk):
        raw=faces[start:start+chunk]
        mapped=repaired[raw]
        valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
        if not np.any(valid): continue
        rawv=raw[valid]; mv=np.sort(mapped[valid],axis=1)
        codes=((mv[:,0]*nrep)+mv[:,1])*nrep+mv[:,2]
        take=np.isin(codes,bad_codes,assume_unique=False)
        if not np.any(take): continue
        selected_raw=rawv[take]
        selected_codes=codes[take]
        ang,asp,bad,_area,_edges=quality(world,selected_raw)
        for code,aa,ss,bb in zip(selected_codes.tolist(),ang.tolist(),asp.tolist(),bad.tolist()):
            c=int(code);witness_count[c]+=1;witness_bad_count[c]+=int(bool(bb))
            witness_min_angle[c]=min(witness_min_angle[c],float(aa))
            if math.isfinite(float(ss)): witness_max_aspect[c]=max(witness_max_aspect[c],float(ss))

    dense_witness_rows=[]
    all_dense_bad=0;any_dense_good=0;zero_witness=0
    for i,row in enumerate(rf):
        if not bool(rbad[i]): continue
        s=sorted(map(int,row));code=((s[0]*nrep)+s[1])*nrep+s[2]
        wc=int(witness_count.get(code,0));wb=int(witness_bad_count.get(code,0))
        if wc==0: zero_witness+=1
        elif wb==wc: all_dense_bad+=1
        else: any_dense_good+=1
        if len(dense_witness_rows)<30:
            dense_witness_rows.append({
              "face_index":int(i),"witness_count":wc,"witness_bad_count":wb,
              "all_dense_witnesses_bad":bool(wc>0 and wb==wc),
              "dense_witness_min_angle_deg":(
                float(witness_min_angle[code]) if math.isfinite(witness_min_angle[code]) else None),
              "dense_witness_max_aspect":float(witness_max_aspect.get(code,0.0)),
            })

    report={
      "schema":"RealSaS.KnightStaticSliverCausalityCourt.v1",
      "status":"MEASURED_AUDIT_ONLY__NO_REPAIR_APPLIED",
      "policy":{"minimum_angle_deg":MIN_ANGLE_DEG,"maximum_aspect":MAX_ASPECT},
      "dense_stage12":dense_stats,
      "historical_div56_compact":base_stats,
      "repaired_div56_refinement":repaired_stats,
      "refinement_structure":{
        "base_node_count":int(base.max())+1,
        "repaired_node_count":int(repaired.max())+1,
        "split_parent_cluster_count":int(split_parent_count),
        "children_of_split_parent_count":int(split_child_count),
        "cross_base_merge_count":0,
      },
      "repaired_bad_face_causality_counts":dict(sorted(classes.items())),
      "examples":dict(examples),
      "dense_witness_diagnostic":{
        "repaired_bad_face_count":int(np.count_nonzero(rbad)),
        "all_dense_witnesses_bad_face_count":int(all_dense_bad),
        "has_at_least_one_dense_good_witness_face_count":int(any_dense_good),
        "zero_dense_witness_face_count":int(zero_witness),
        "sample_rows":dense_witness_rows,
        "claim_boundary":"Dense witness quality is diagnostic provenance, not product admission of the dense decoder tessellation.",
      },
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STATIC_SLIVER_CAUSALITY="+json.dumps({
      "dense_bad_fraction":dense_stats["bad_face_fraction"],
      "base_bad":base_stats["bad_face_count"],"base_faces":base_stats["face_count"],
      "repaired_bad":repaired_stats["bad_face_count"],"repaired_faces":repaired_stats["face_count"],
      "classes":dict(sorted(classes.items())),
      "all_dense_witnesses_bad":all_dense_bad,
      "has_dense_good_witness":any_dense_good,
      "split_parent_clusters":split_parent_count,
    },sort_keys=True))

if __name__=="__main__":main()
