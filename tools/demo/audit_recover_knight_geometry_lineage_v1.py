from __future__ import annotations
import argparse, json, os
from pathlib import Path

NEEDLES={
 "camera_set_hash":"2485d184c7fef6c453b6ff817d3eac525e438fe023350b294436589d62d75cca",
 "observation_set_hash":"87a3ba4ba20ea680669d9957593489bafd8fef285f7a5678c8eb419160792efa",
 "normalization_hash":"81b1d0d8d2e7cbaf8a5dc14d174bdb65f6310d2868dd8af6e2f3915471526e23",
 "zero_surface_hash":"266346ba5ada150d2ba3bc5b02f6527b671c0609c1a63ad1c04abea1fbe95797",
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    a=ap.parse_args()
    roots=[a.authority_root/"runs",a.authority_root/"imports"]
    hits={k:[] for k in NEEDLES}
    scanned=0
    for root in roots:
        if not root.exists(): continue
        for dirpath,dirs,files in os.walk(root):
            # avoid giant binary/cache trees where possible
            dirs[:] = [d for d in dirs if d not in {"__pycache__",".git","cache","tmp"}]
            for name in files:
                if not name.lower().endswith(".json"): continue
                p=Path(dirpath)/name
                scanned+=1
                try:
                    text=p.read_text(errors="ignore")
                except Exception:
                    continue
                matched=[k for k,v in NEEDLES.items() if v in text]
                if not matched: continue
                try:
                    payload=json.loads(text)
                    schema=str(payload.get("schema") or payload.get("schema_version") or "")
                    status=payload.get("status")
                except Exception:
                    schema=""; status=None
                for k in matched:
                    hits[k].append({
                        "path":str(p.resolve()),"schema":schema,"status":status,
                        "size":p.stat().st_size,
                    })
    # Inspect ledgers for runs that contain exact matching stage hashes.
    run_candidates=[]
    runs_root=a.authority_root/"runs"
    if runs_root.exists():
        for rr in sorted(x for x in runs_root.iterdir() if x.is_dir()):
            lp=rr/"ACTIVE_RUN_V2.json"
            if not lp.is_file(): continue
            try: ledger=json.loads(lp.read_text())
            except Exception: continue
            stages={str(x.get("id")):x for x in ledger.get("stages") or ()}
            summary={"run_id":rr.name}
            useful=False
            for sid in ("05_CAMERA_CONTRACT_SOLVED","07_OBSERVATION_CONTRACT_QUALIFIED",
                        "08_NORMALIZATION_DOMAIN_QUALIFIED","12_ZERO_SURFACE_DECODED",
                        "13_GEOMETRY_SUBSTRATE_QUALIFIED","15_RIGGING_SURFACE_QUALIFIED"):
                r=stages.get(sid)
                if r is not None:
                    summary[sid]={"status":r.get("status"),"outputs":[
                        {"schema":o.get("schema"),"path":o.get("path"),"sha256":o.get("sha256")}
                        for o in r.get("outputs") or ()
                    ]}
            # mark candidate if run path appears in any hit
            if any(any(f"/runs/{rr.name}/" in h["path"] for h in rows) for rows in hits.values()):
                useful=True
            if useful: run_candidates.append(summary)
    print("RECOVER_GEOMETRY_LINEAGE="+json.dumps({
      "needles":NEEDLES,"scanned_json_count":scanned,"hits":hits,
      "run_candidates":run_candidates,
    },sort_keys=True))
if __name__=="__main__": main()
