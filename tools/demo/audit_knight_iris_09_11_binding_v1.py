from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path

def sha256(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    a=ap.parse_args()
    rr=(a.authority_root/"runs"/a.run_id).resolve()
    ledger=json.loads((rr/"ACTIVE_RUN_V2.json").read_text())
    def row(s): return next(x for x in ledger["stages"] if x["id"]==s)
    out={}
    for sid in ("09_IRIS_FIT_PREREGISTERED","10_IRIS_FIT","11_IRIS_CHECKPOINT_SEALED"):
        r=row(sid)
        rows=[]
        for o in r.get("outputs") or ():
            p=Path(o["path"]).resolve()
            item={"schema":o.get("schema"),"path":str(p),"ledger_sha256":o.get("sha256"),
                  "exists":p.is_file()}
            if p.is_file():
                item["actual_sha256"]=sha256(p)
                if p.suffix.lower()==".json":
                    try:
                        payload=json.loads(p.read_text())
                        item["payload_keys"]=sorted(payload.keys())
                        for k in ("upstream_bindings","preregistration_binding_hash","execution_binding_hash",
                                  "checkpoint_seal_hash","execution_hash","preregistration_hash",
                                  "qualified_output_binding_hash","checkpoint_sha256","result_sha256",
                                  "observation_set_binding_hash","normalization_binding_hash"):
                            if k in payload: item[k]=payload[k]
                    except Exception as exc:
                        item["parse_error"]=str(exc)
            rows.append(item)
        out[sid]={"status":r.get("status"),"outputs":rows}
    print("IRIS_09_11_BINDING="+json.dumps(out,sort_keys=True))
if __name__=="__main__": main()
