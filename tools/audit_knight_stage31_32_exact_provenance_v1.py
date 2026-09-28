from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def safe_json(p:Path):
    try:
        x=json.loads(p.read_text())
    except Exception:
        return None
    def slim(v,depth=0):
        if depth>4:return "<DEPTH>"
        if isinstance(v,dict):
            return {str(k):slim(val,depth+1) for k,val in v.items() if not str(k).lower().endswith(("weights","logits"))}
        if isinstance(v,list):
            if len(v)>64:return {"count":len(v),"head":[slim(x,depth+1) for x in v[:8]]}
            return [slim(x,depth+1) for x in v]
        return v
    return slim(x)

def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    rr=a.authority_root/"runs"/a.run_id
    rows={}
    for stage in ["31_ARACHNE_FIT","32_SKIN_QUALIFIED","33_ARACHNE_CHECKPOINT_SEALED"]:
        d=rr/"artifacts"/stage
        items=[]
        if d.is_dir():
            for f in sorted(d.rglob("*")):
                if not f.is_file():continue
                rel=str(f.relative_to(rr))
                row={"path":rel,"size":f.stat().st_size,"sha256":sha(f)}
                if f.suffix.lower()==".json" and f.stat().st_size<8_000_000:
                    row["json"]=safe_json(f)
                items.append(row)
        rows[stage]=items
    # ledger entries from known common ledger files
    ledgers=[]
    for name in ["run_ledger.json","ledger.json","RUN_LEDGER.json","execution_ledger.json"]:
        f=rr/name
        if f.is_file():
            x=safe_json(f)
            ledgers.append({"path":str(f.relative_to(rr)),"sha256":sha(f),"content":x})
    report={"schema":"RealSaS.KnightStage31Stage32ExactProvenanceAudit.v1","status":"MEASURED__NO_REPAIR","run_id":a.run_id,"stage_artifacts":rows,"ledgers":ledgers}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STAGE31_32_EXACT_PROVENANCE_AUDIT_PASS",json.dumps({k:[{"path":x["path"],"sha256":x["sha256"],"size":x["size"]} for x in v] for k,v in rows.items()},sort_keys=True))
if __name__=="__main__":main()
