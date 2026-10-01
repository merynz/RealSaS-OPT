from __future__ import annotations

import json, math, subprocess
from pathlib import Path
import numpy as np

def auc_binary(y,score):
    y=np.asarray(y,dtype=bool); s=np.asarray(score,dtype=np.float64)
    finite=np.isfinite(s)
    y=y[finite]; s=s[finite]
    pos=int(y.sum()); neg=int((~y).sum())
    if pos==0 or neg==0:return None
    order=np.argsort(s,kind="mergesort")
    ranks=np.empty(len(s),dtype=np.float64)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and s[order[j]]==s[order[i]]: j+=1
        avg=(i+1+j)/2.0
        ranks[order[i:j]]=avg
        i=j
    auc=(ranks[y].sum()-pos*(pos+1)/2)/(pos*neg)
    return float(auc)

def quant(a):
    a=np.asarray(a,dtype=np.float64)
    a=a[np.isfinite(a)]
    if not len(a): return {"count":0}
    return {"count":int(len(a)),"mean":float(a.mean()),"p05":float(np.quantile(a,.05)),"p50":float(np.quantile(a,.5)),"p95":float(np.quantile(a,.95)),"p99":float(np.quantile(a,.99)),"min":float(a.min()),"max":float(a.max())}

def best_oriented_auc(y,s):
    a=auc_binary(y,s)
    if a is None:return None
    return {"raw_auc":a,"best_oriented_auc":max(a,1.0-a),"orientation":"HIGHER_POSITIVE" if a>=.5 else "LOWER_POSITIVE"}

def main(surface_path,bank_path,source_path,out_path):
    surface=json.loads(Path(surface_path).read_text())
    nodes=surface["surface_nodes"]; rels=surface["local_relations"]
    by={str(n["surface_id"]):n for n in nodes}

    with np.load(bank_path,allow_pickle=False) as z:
        ids=tuple(map(str,z["surface_ids"].tolist()))
        W=np.asarray(z["weights"],np.float64)
        valid=np.asarray(z["teacher_valid_mask"],np.uint8).astype(bool)
        tri=np.asarray(z["source_triangle_index"],np.int64)
    with np.load(source_path,allow_pickle=False) as z:
        source_faces=np.asarray(z["faces"],np.int64)
    row={sid:i for i,sid in enumerate(ids)}
    face_sets=[set(map(int,f)) for f in source_faces.tolist()]

    def category(ia,ib):
        ta,tb=int(tri[ia]),int(tri[ib])
        if ta==tb:return "SAME_FACE"
        common=len(face_sets[ta]&face_sets[tb])
        if common==2:return "SHARE_EDGE"
        if common==1:return "SHARE_VERTEX_ONLY"
        return "NONINCIDENT"

    feats={k:[] for k in (
      "distance_3d","normal_angle_deg","normal_abs_angle_deg",
      "common_support_count","support_jaccard",
      "common_raster_min_distance","common_raster_mean_distance",
      "completed_endpoint_count",
    )}
    cats=[]; l1=[]; high=[]

    for rel in rels:
        a=str(rel["a_surface_id"]); b=str(rel["b_surface_id"])
        ia=row[a]; ib=row[b]
        if not (valid[ia] and valid[ib]): continue
        na=by[a]; nb=by[b]
        pa=np.asarray(na["P"],np.float64); pb=np.asarray(nb["P"],np.float64)
        feats["distance_3d"].append(float(np.linalg.norm(pa-pb)))

        va=np.asarray(na.get("derived_normal") or (np.nan,np.nan,np.nan),np.float64)
        vb=np.asarray(nb.get("derived_normal") or (np.nan,np.nan,np.nan),np.float64)
        if np.all(np.isfinite(va)) and np.all(np.isfinite(vb)) and np.linalg.norm(va)>1e-12 and np.linalg.norm(vb)>1e-12:
            dot=float(np.clip(np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)),-1,1))
            feats["normal_angle_deg"].append(float(np.degrees(np.arccos(dot))))
            feats["normal_abs_angle_deg"].append(float(np.degrees(np.arccos(abs(dot)))))
        else:
            feats["normal_angle_deg"].append(float("nan"))
            feats["normal_abs_angle_deg"].append(float("nan"))

        sa=set(map(int,na.get("support_views") or ())); sb=set(map(int,nb.get("support_views") or ()))
        common=sa&sb; union=sa|sb
        feats["common_support_count"].append(float(len(common)))
        feats["support_jaccard"].append(float(len(common)/len(union)) if union else 0.0)

        ra={int(v):np.asarray(xy,np.float64) for v,xy in (na.get("raster_bindings") or ())}
        rb={int(v):np.asarray(xy,np.float64) for v,xy in (nb.get("raster_bindings") or ())}
        ds=[float(np.linalg.norm(ra[v]-rb[v])) for v in sorted(set(ra)&set(rb))]
        feats["common_raster_min_distance"].append(min(ds) if ds else float("nan"))
        feats["common_raster_mean_distance"].append(float(np.mean(ds)) if ds else float("nan"))

        fa=set(map(str,na.get("validity_flags") or ())); fb=set(map(str,nb.get("validity_flags") or ()))
        feats["completed_endpoint_count"].append(float(
          int(any("COMPLETED" in x for x in fa))+int(any("COMPLETED" in x for x in fb))
        ))

        c=category(ia,ib); cats.append(c)
        d=float(np.abs(W[ia]-W[ib]).sum()); l1.append(d); high.append(d>1.0)

    cats=np.asarray(cats,object); l1=np.asarray(l1,np.float64); high=np.asarray(high,bool)
    nonincident=(cats=="NONINCIDENT")
    coherent=np.isin(cats,["SAME_FACE","SHARE_EDGE"])
    trusted=nonincident|coherent

    result_features={}
    for k,v in feats.items():
        arr=np.asarray(v,np.float64)
        result_features[k]={
          "NONINCIDENT":quant(arr[nonincident]),
          "SAME_FACE_OR_SHARE_EDGE":quant(arr[coherent]),
          "auc_nonincident_vs_coherent":best_oriented_auc(nonincident[trusted],arr[trusted]),
          "auc_high_skin_l1_gt_1":best_oriented_auc(high,arr),
        }

    # Threshold-only diagnostic ceiling per individual feature, no model promotion.
    report={
      "schema":"RealSaS.KnightObservableSeamFeatureSeparabilityAudit.v1",
      "status":"AUDIT_ONLY__TEACHER_LABELS_DIAGNOSTIC_ONLY",
      "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
      "population":{
        "both_teacher_valid_relation_count":int(len(cats)),
        "nonincident_count":int(nonincident.sum()),
        "same_face_or_share_edge_count":int(coherent.sum()),
        "share_vertex_only_count":int((cats=="SHARE_VERTEX_ONLY").sum()),
        "high_skin_l1_gt_1_count":int(high.sum()),
        "high_skin_l1_gt_1_by_category":{c:int(np.count_nonzero(high&(cats==c))) for c in ("SAME_FACE","SHARE_EDGE","SHARE_VERTEX_ONLY","NONINCIDENT")},
      },
      "features":result_features,
      "finding":{
        "id":"OBSERVABLE_GSA_FEATURE_SEPARABILITY_OF_TEACHER_TOPOLOGY_SEAMS",
        "claim_boundary":"Teacher source topology and teacher weights are labels only. This audit asks whether already-shipping GSA node features contain simple geometric/observation evidence that correlates with the seam class; it does not authorize teacher topology at inference and does not select a classifier or threshold."
      }
    }
    Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    Path(out_path).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__":
    import sys
    main(*sys.argv[1:5])
