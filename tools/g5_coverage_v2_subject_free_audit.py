from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import coverage_metrics

OUT=Path(os.environ.get("REALSAS_G5_AUDIT_OUT","g5_audit_out")); OUT.mkdir(parents=True,exist_ok=True)
H=W=1024

def b(a): return np.asarray(a,dtype=np.uint8).reshape(-1).tobytes()
def ellipse(cx,cy,rx,ry):
    yy,xx=np.mgrid[0:H,0:W]; X=xx+.5; Y=yy+.5
    return (((X-cx)/rx)**2+((Y-cy)/ry)**2)<=1
def disk(cx,cy,r):
    yy,xx=np.mgrid[0:H,0:W]; X=xx+.5; Y=yy+.5
    return (X-cx)**2+(Y-cy)**2<=r*r
def shift(mask,px):
    out=np.zeros_like(mask)
    if px>0: out[:,px:]=mask[:,:-px]
    elif px<0: out[:,:px]=mask[:,-px:]
    else: out[:]=mask
    return out
def sparse_boundary(mask,stride=100):
    # deterministic boundary samples; same tolerance semantics as Stage13 v2 calibration
    er=binary_erosion(mask,structure=np.ones((3,3),bool))
    bd=mask&~er
    out=mask.copy()
    for k,(y,x) in enumerate(np.argwhere(bd)[::stride]):
        y=int(y);x=int(x)
        if k%2==0: out[y,x]=False
        else:
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                yy=y+dy;xx=x+dx
                if 0<=yy<H and 0<=xx<W and not mask[yy,xx]:
                    out[yy,xx]=True;break
    return out

src=ellipse(496,535,236,352)|disk(496,205,98)
hole=disk(496,535,8)
pep=np.zeros_like(src); interior=binary_erosion(src,structure=np.ones((3,3),bool),iterations=2); pts=np.argwhere(interior)[::500]; pep[pts[:,0],pts[:,1]]=True
notch=src.copy(); notch[470:500,725:745]=False
cases={
 "IDENTITY":(src,src,"ACCEPT"),
 "SPARSE_BOUNDARY_JITTER_1PCT":(src,sparse_boundary(src),"ACCEPT"),
 "SHIFT_X_1PX":(src,shift(src,1),"REJECT"),
 "ERODE_1PX":(src,binary_erosion(src,iterations=1),"REJECT"),
 "DILATE_1PX":(src,binary_dilation(src,iterations=1),"REJECT"),
 "INTERIOR_HOLE_R8":(src,src&~hole,"REJECT"),
 "INTERIOR_PEPPER_0P2PCT":(src,src&~pep,"REJECT"),
 "LOCAL_NOTCH_30X20":(src,notch,"REJECT"),
}
profiles={
 "CURRENT_MESH":{"min_recall":.97,"min_precision":.995,"max_largest_coherent_hole_fraction":.005,"max_interior_uncovered_fraction":.005},
 "CURRENT_PLANAR":{"min_recall":.99,"min_precision":.995,"max_largest_coherent_hole_fraction":.0025,"max_interior_uncovered_fraction":.0025},
 "P999":{"min_recall":.999,"min_precision":.999,"max_largest_coherent_hole_fraction":.00025,"max_interior_uncovered_fraction":.0005},
}

def passed(m,p):
    return m["recall"]>=p["min_recall"] and m["precision"]>=p["min_precision"] and m["largest_coherent_hole_fraction"]<=p["max_largest_coherent_hole_fraction"] and m["interior_uncovered_fraction"]<=p["max_interior_uncovered_fraction"]

rows={}
for cid,(a,pred,expect) in cases.items():
    m=coverage_metrics(b(a),b(pred),width=W,height=H)
    rows[cid]={"expectation":expect,"metrics":m}
evals={}
for name,p in profiles.items():
    mis=[]
    decisions={}
    for cid,row in rows.items():
        ok=passed(row["metrics"],p); decisions[cid]=ok
        if (row["expectation"]=="ACCEPT")!=ok: mis.append(cid)
    evals[name]={"profile":p,"misclassified_cases":mis,"case_pass":decisions,"semantic_separation_pass":not mis}
out={"schema":"RealSaS.G5CoveragePolicyRetrospectiveAudit.v1","status":"PASS","subject_inputs_used":False,"knight_result_used":False,"mage_result_used":False,"resolution":[W,H],"cases":rows,"profiles":evals,"conclusion":"CURRENT_G5_COVERAGE_THRESHOLDS_ARE_TOO_LOOSE_FOR_1024_SOURCE_PRESERVATION" if (evals["CURRENT_MESH"]["misclassified_cases"] or evals["CURRENT_PLANAR"]["misclassified_cases"]) else "CURRENT_G5_SEPARATES_CONTROLS"}
(OUT/"G5_COVERAGE_V2_RETROSPECTIVE_AUDIT.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({"conclusion":out["conclusion"],"profiles":evals},indent=2,sort_keys=True))
