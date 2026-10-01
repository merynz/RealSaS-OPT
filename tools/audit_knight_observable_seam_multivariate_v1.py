from __future__ import annotations

import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

def fold_of(sid,k=5):
    return int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)%k

def feature_row(a,b):
    pa=np.asarray(a["P"],np.float64); pb=np.asarray(b["P"],np.float64)
    d=float(np.linalg.norm(pa-pb))
    na=np.asarray(a.get("derived_normal") or (np.nan,np.nan,np.nan),np.float64)
    nb=np.asarray(b.get("derived_normal") or (np.nan,np.nan,np.nan),np.float64)
    if np.all(np.isfinite(na)) and np.all(np.isfinite(nb)) and np.linalg.norm(na)>1e-12 and np.linalg.norm(nb)>1e-12:
        dot=float(np.clip(np.dot(na,nb)/(np.linalg.norm(na)*np.linalg.norm(nb)),-1,1))
        ang=float(np.degrees(np.arccos(dot)))
        absang=float(np.degrees(np.arccos(abs(dot))))
    else:
        ang=np.nan; absang=np.nan
    sa=set(map(int,a.get("support_views") or ())); sb=set(map(int,b.get("support_views") or ()))
    common=sa&sb; union=sa|sb
    jac=len(common)/len(union) if union else 0.0
    ra={int(v):np.asarray(xy,np.float64) for v,xy in (a.get("raster_bindings") or ())}
    rb={int(v):np.asarray(xy,np.float64) for v,xy in (b.get("raster_bindings") or ())}
    ds=[float(np.linalg.norm(ra[v]-rb[v])) for v in sorted(set(ra)&set(rb))]
    rmin=min(ds) if ds else np.nan
    rmean=float(np.mean(ds)) if ds else np.nan
    fa=set(map(str,a.get("validity_flags") or ())); fb=set(map(str,b.get("validity_flags") or ()))
    completed=int(any("COMPLETED" in x for x in fa))+int(any("COMPLETED" in x for x in fb))
    return [d,ang,absang,float(len(common)),float(jac),rmin,rmean,float(completed)]

def main(surface_path,bank_path,source_path,out_path):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

    surface=json.loads(Path(surface_path).read_text())
    by={str(n["surface_id"]):n for n in surface["surface_nodes"]}
    rels=surface["local_relations"]
    with np.load(bank_path,allow_pickle=False) as z:
        ids=tuple(map(str,z["surface_ids"].tolist()))
        W=np.asarray(z["weights"],np.float64)
        valid=np.asarray(z["teacher_valid_mask"],np.uint8).astype(bool)
        tri=np.asarray(z["source_triangle_index"],np.int64)
    with np.load(source_path,allow_pickle=False) as z:
        faces=np.asarray(z["faces"],np.int64)
    row={sid:i for i,sid in enumerate(ids)}
    fsets=[set(map(int,f)) for f in faces.tolist()]

    def cat(ia,ib):
        ta,tb=int(tri[ia]),int(tri[ib])
        if ta==tb:return "SAME_FACE"
        c=len(fsets[ta]&fsets[tb])
        if c==2:return "SHARE_EDGE"
        if c==1:return "SHARE_VERTEX_ONLY"
        return "NONINCIDENT"

    X=[]; y_non=[]; y_seam=[]; folds=[]; cats=[]
    for rel in rels:
        a=str(rel["a_surface_id"]); b=str(rel["b_surface_id"])
        ia=row[a]; ib=row[b]
        if not(valid[ia] and valid[ib]):continue
        fa=fold_of(a); fb=fold_of(b)
        if fa!=fb: continue  # strict node-disjoint evaluation pool
        c=cat(ia,ib)
        if c=="SHARE_VERTEX_ONLY": continue
        X.append(feature_row(by[a],by[b]))
        cats.append(c)
        y_non.append(c=="NONINCIDENT")
        y_seam.append(float(np.abs(W[ia]-W[ib]).sum())>1.0)
        folds.append(fa)

    X=np.asarray(X,np.float64); folds=np.asarray(folds,np.int64)
    targets={"NONINCIDENT":np.asarray(y_non,bool),"SKIN_L1_GT_1":np.asarray(y_seam,bool)}
    feature_names=["distance_3d","normal_angle_deg","normal_abs_angle_deg","common_support_count","support_jaccard","common_raster_min_distance","common_raster_mean_distance","completed_endpoint_count"]

    models={
      "LOGISTIC": lambda: make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced",C=1.0)),
      "HIST_GBDT_SHALLOW": lambda: make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(max_depth=3,max_iter=120,learning_rate=.06,l2_regularization=1.0,random_state=0)),
    }
    results={}
    for tname,y in targets.items():
        results[tname]={}
        for mname,maker in models.items():
            fold_rows=[]
            pooled_y=[]; pooled_p=[]
            for test_fold in range(5):
                test=folds==test_fold; train=~test
                if test.sum()<10 or y[test].sum()==0 or (~y[test]).sum()==0:continue
                model=maker(); model.fit(X[train],y[train])
                p=model.predict_proba(X[test])[:,1]
                auc=float(roc_auc_score(y[test],p))
                ap=float(average_precision_score(y[test],p))
                precision,recall,thr=precision_recall_curve(y[test],p)
                # max recall achievable while precision >= .90
                rec90=float(np.max(recall[precision>=.90])) if np.any(precision>=.90) else 0.0
                # max precision achievable while recall >= .90
                p90=float(np.max(precision[recall>=.90])) if np.any(recall>=.90) else 0.0
                fold_rows.append({"fold":test_fold,"test_count":int(test.sum()),"positive_count":int(y[test].sum()),"roc_auc":auc,"average_precision":ap,"recall_at_precision_ge_0p90":rec90,"precision_at_recall_ge_0p90":p90})
                pooled_y.extend(y[test].tolist());pooled_p.extend(p.tolist())
            pooled_y=np.asarray(pooled_y,bool);pooled_p=np.asarray(pooled_p,float)
            results[tname][mname]={
              "folds":fold_rows,
              "mean_roc_auc":float(np.mean([r["roc_auc"] for r in fold_rows])),
              "mean_average_precision":float(np.mean([r["average_precision"] for r in fold_rows])),
              "pooled_roc_auc":float(roc_auc_score(pooled_y,pooled_p)),
              "pooled_average_precision":float(average_precision_score(pooled_y,pooled_p)),
              "pooled_positive_rate":float(pooled_y.mean()),
              "pooled_count":int(len(pooled_y)),
            }

    report={
      "schema":"RealSaS.KnightObservableSeamMultivariateSeparabilityAudit.v1",
      "status":"AUDIT_ONLY__TEACHER_LABELS_DIAGNOSTIC_ONLY",
      "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
      "split":"5_FOLD_SURFACE_ID_HASH__ONLY_EDGES_WITH_BOTH_ENDPOINTS_IN_SAME_FOLD__NODE_DISJOINT_TRAIN_TEST",
      "feature_names":feature_names,
      "eligible_relation_count":int(len(X)),
      "target_counts":{k:{"positive":int(v.sum()),"negative":int((~v).sum())} for k,v in targets.items()},
      "results":results,
      "finding":{
        "id":"MULTIVARIATE_OBSERVABLE_GSA_SEAM_SEPARABILITY",
        "claim_boundary":"Diagnostic only. Teacher topology/weights are labels, not inference inputs. Node-disjoint CV tests whether the already-shipping local GSA feature vector contains enough simple multivariate signal for pre-skin seam qualification; it does not prove impossibility for richer observable models."
      }
    }
    Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    Path(out_path).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__":
    main(*sys.argv[1:5])
