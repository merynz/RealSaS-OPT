from __future__ import annotations
import argparse,json,glob,hashlib
from pathlib import Path

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    a=ap.parse_args()
    rr=a.authority_root/"runs"/a.run_id
    targets={
      "12_ZERO_SURFACE_DECODED":sha256(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"),
      "13_GEOMETRY_SUBSTRATE_QUALIFIED":sha256(rr/"artifacts/13_GEOMETRY_SUBSTRATE_QUALIFIED/geometry_substrate_qualification.json"),
      "14_GSA_BUILD":sha256(rr/"artifacts/14_GSA_BUILD/rigging_surface_candidate.json"),
      "15_RIGGING_SURFACE_QUALIFIED":sha256(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"),
    }
    snaps=sorted(rr.glob("ACTIVE_RUN_V2.before_frozen_c.*.json"))
    rows=[]
    for lp in snaps:
        try:l=json.loads(lp.read_text())
        except Exception:continue
        sm={str(x.get("id")):x for x in l.get("stages") or ()}
        item={"path":str(lp.resolve()),"status":l.get("status"),"stages":{}}
        exact=0
        for sid in ("05_CAMERA_CONTRACT_SOLVED","07_OBSERVATION_CONTRACT_QUALIFIED",
                    "08_NORMALIZATION_DOMAIN_QUALIFIED","09_IRIS_FIT_PREREGISTERED",
                    "10_IRIS_FIT","11_IRIS_CHECKPOINT_SEALED","12_ZERO_SURFACE_DECODED",
                    "13_GEOMETRY_SUBSTRATE_QUALIFIED","14_GSA_BUILD","15_RIGGING_SURFACE_QUALIFIED"):
            r=sm.get(sid)
            if r is None:continue
            outs=[]
            for o in r.get("outputs") or ():
                outs.append({"schema":o.get("schema"),"sha256":o.get("sha256"),"path":o.get("path")})
                if sid in targets and o.get("sha256")==targets[sid]: exact+=1
            item["stages"][sid]={"status":r.get("status"),"outputs":outs}
        item["target_exact_output_match_count"]=exact
        rows.append(item)
    print("HISTORICAL_LEDGER_AUTHORITY="+json.dumps({"targets":targets,"snapshots":rows},sort_keys=True))
if __name__=="__main__":main()
