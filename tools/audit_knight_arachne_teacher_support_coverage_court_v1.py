from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def _jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher-bank",type=Path,required=True)
    ap.add_argument("--target-npz",type=Path,required=True)
    ap.add_argument("--target-report",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    with np.load(a.teacher_bank,allow_pickle=False) as z:
        W=np.asarray(z["weights"],dtype=np.float64)
        valid=np.asarray(z["teacher_valid_mask"],dtype=np.uint8).astype(bool)
        bank_pos=np.asarray(z["target_positions_world"],dtype=np.float64) if "target_positions_world" in z.files else None
        bank_par=np.asarray(z["target_parent_indices"],dtype=np.int64) if "target_parent_indices" in z.files else None
        bank_keys=sorted(z.files)

    with np.load(a.target_npz,allow_pickle=False) as z:
        target={k:np.asarray(z[k]) for k in z.files}
        target_keys=sorted(z.files)
    report=json.loads(a.target_report.read_text())

    # Resolve Geppetto target-column alignment conservatively by exact/near-exact target positions
    # only if a compatible position matrix exists.
    position_key=None
    for k in ("positions_world","positions","target_positions_world","joint_positions","target_positions"):
        if k in target and np.asarray(target[k]).ndim==2 and np.asarray(target[k]).shape[1]==3:
            position_key=k; break
    alignment=None
    max_pos_error=None
    if bank_pos is not None and position_key is not None:
        tp=np.asarray(target[position_key],dtype=np.float64)
        if len(tp)==len(bank_pos):
            # greedy unique exact-nearest alignment is safe only if unambiguous and tiny.
            D=np.linalg.norm(bank_pos[:,None,:]-tp[None,:,:],axis=2)
            idx=np.argmin(D,axis=1)
            errs=D[np.arange(len(bank_pos)),idx]
            if len(set(map(int,idx)))==len(idx) and float(errs.max(initial=0.0))<=1e-6:
                alignment=idx.astype(int)
                max_pos_error=float(errs.max(initial=0.0))

    # Collect any role vector from NPZ or report without assuming schema.
    role_values=None
    role_source=None
    for k in ("roles","target_roles","joint_roles","role"):
        if k in target and len(np.asarray(target[k]).reshape(-1))==W.shape[1]:
            role_values=[str(x) for x in np.asarray(target[k]).reshape(-1).tolist()]
            role_source="target_npz:"+k
            break
    if role_values is None:
        candidates=[]
        def walk(x,path=""):
            if isinstance(x,dict):
                for k,v in x.items(): walk(v,path+"."+k if path else k)
            elif isinstance(x,list) and len(x)==W.shape[1] and all(isinstance(y,(str,int,float,bool)) or y is None for y in x):
                if "role" in path.lower(): candidates.append((path,x))
        walk(report)
        if candidates:
            role_source, vals=candidates[0]
            role_values=[str(x) for x in vals]
    if role_values is not None and alignment is not None:
        role_values=[role_values[int(i)] for i in alignment]

    thresholds=(0.0,0.01,0.05,0.10)
    columns=[]
    for j in range(W.shape[1]):
        row={"target_column":j}
        if role_values is not None: row["target_role"]=role_values[j]
        for t in thresholds:
            pos=W[:,j]>t
            pv=pos&valid
            pi=pos&(~valid)
            mass=np.clip(W[:,j]-t,0,None) if t>0 else np.clip(W[:,j],0,None)
            total=float(mass.sum())
            inv=float(mass[~valid].sum())
            row[str(t)]={
                "positive_row_count_all":int(pos.sum()),
                "positive_row_count_valid":int(pv.sum()),
                "positive_row_count_invalid":int(pi.sum()),
                "positive_weight_mass_all":total,
                "positive_weight_mass_valid":float(mass[valid].sum()),
                "positive_weight_mass_invalid":inv,
                "invalid_mass_fraction":float(inv/total) if total>0 else None,
                "zero_support_all":bool(pos.sum()==0),
                "invalid_only_support":bool(pos.sum()>0 and pv.sum()==0),
            }
        columns.append(row)

    t0=[r["0.0"] for r in columns]
    result={
        "schema":"RealSaS.KnightArachneTeacherSupportCoverageCourt.v1",
        "status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
        "teacher_bank_keys":bank_keys,
        "target_npz_keys":target_keys,
        "target_report_schema":report.get("schema"),
        "surface_row_count":int(W.shape[0]),
        "target_column_count":int(W.shape[1]),
        "teacher_valid_count":int(valid.sum()),
        "teacher_invalid_count":int((~valid).sum()),
        "teacher_coverage":float(valid.mean()),
        "geppetto_alignment":{
            "position_key":position_key,
            "exact_nearest_alignment_available":alignment is not None,
            "max_position_error":max_pos_error,
            "role_source":role_source,
        },
        "summary_at_zero":{
            "zero_support_all_column_count":int(sum(x["zero_support_all"] for x in t0)),
            "invalid_only_support_column_count":int(sum(x["invalid_only_support"] for x in t0)),
            "columns_with_any_invalid_positive_support":int(sum(x["positive_row_count_invalid"]>0 for x in t0)),
            "columns_with_any_valid_positive_support":int(sum(x["positive_row_count_valid"]>0 for x in t0)),
        },
        "columns":columns,
        "claim_boundary":"Read-only teacher support classification. No model or product mutation."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_ARACHNE_TEACHER_SUPPORT_COVERAGE_COURT_PASS",json.dumps({
        "summary_at_zero":result["summary_at_zero"],
        "zero_support_columns":[r["target_column"] for r in columns if r["0.0"]["zero_support_all"]],
        "invalid_only_columns":[r["target_column"] for r in columns if r["0.0"]["invalid_only_support"]],
        "alignment":result["geppetto_alignment"],
    },sort_keys=True))

if __name__=="__main__":
    main()
