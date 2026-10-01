from __future__ import annotations
import argparse,json,hashlib,os
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
    stage28=rr/"artifacts"/"28_SKELETON_QUALIFIED"/"qualified_skeleton.json"
    stage34=rr/"artifacts"/"34_DEFORMATION_CAPABILITY_ENVELOPE"/"deformation_envelope.json"
    targets={}
    if stage28.is_file():targets["stage28_sha256"]=sha256(stage28)
    if stage34.is_file():targets["stage34_sha256"]=sha256(stage34)
    if stage28.is_file():
        p=json.loads(stage28.read_text())
        targets["skeleton_lineage_hash"]=p.get("skeleton_lineage_hash")
    hits=[]
    for root in (rr,a.authority_root/"imports"):
        if not root.exists():continue
        for dp,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if d not in {"__pycache__",".git","cache","tmp"}]
            for name in files:
                if not name.endswith(".json"):continue
                p=Path(dp)/name
                try:text=p.read_text(errors="ignore")
                except Exception:continue
                matched=[k for k,v in targets.items() if isinstance(v,str) and v and v in text]
                if not matched:continue
                try:
                    obj=json.loads(text)
                    schema=str(obj.get("schema") or obj.get("schema_version") or "")
                    status=obj.get("status")
                except Exception:
                    schema="";status=None
                hits.append({"path":str(p.resolve()),"schema":schema,"status":status,"matched":matched})
    ledger_hits=[]
    for lp in sorted(rr.glob("ACTIVE_RUN_V2*.json")):
        try:l=json.loads(lp.read_text())
        except Exception:continue
        sm={str(x.get("id")):x for x in l.get("stages") or ()}
        row={"path":str(lp.resolve()),"ledger_status":l.get("status"),"stages":{}}
        found=False
        for sid in ("27_GEPPETTO_FIT","28_SKELETON_QUALIFIED","32_SKIN_QUALIFIED","34_DEFORMATION_CAPABILITY_ENVELOPE"):
            s=sm.get(sid)
            if not s:continue
            outs=[{"schema":o.get("schema"),"sha256":o.get("sha256"),"path":o.get("path")} for o in s.get("outputs") or ()]
            row["stages"][sid]={"status":s.get("status"),"outputs":outs}
            if any(o.get("sha256") in {targets.get("stage28_sha256"),targets.get("stage34_sha256")} for o in s.get("outputs") or ()):
                found=True
        if found:ledger_hits.append(row)
    print("RECOVER_STAGE28_AUTHORITY="+json.dumps({"targets":targets,"hits":hits,"ledger_hits":ledger_hits},sort_keys=True))
if __name__=="__main__":main()
