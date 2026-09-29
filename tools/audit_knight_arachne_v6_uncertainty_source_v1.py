from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from compiler.realsas_compiler_core.preproduct_authority_v1 import model_fit_preregistration_from_dict
from tools.demo.render_knight_motion_preview_v1 import _ctx

PREREG=Path("canonical/KNIGHT_ARACHNE_V6_UNCERTAINTY_SOURCE_AUDIT_PREREG_V1_20260929.json")
TOKENS=("log_sigma","sigma","uncertainty","confidence","entropy","variance","reliability","teacher_valid","coverage","SkinProposalIR","metadata")

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_ARACHNE_V6_SOURCE_AUDIT_RESULT":raise RuntimeError("V6_SOURCE_AUDIT_PREREG_DRIFT")
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    pp=json.loads((rr/"artifacts/30_ARACHNE_FIT_PREREGISTERED/model_fit_preregistration.json").read_text())
    pr=model_fit_preregistration_from_dict(pp)
    path=Path(str(pr.model_source_path)).expanduser().resolve()
    if not path.is_file():raise RuntimeError("ARACHNE_MODEL_SOURCE_MISSING:"+str(path))
    actual=sha(path)
    if actual!=pr.model_source_sha256:raise RuntimeError("ARACHNE_MODEL_SOURCE_SHA_DRIFT")
    raw=path.read_bytes()
    try:text=raw.decode("utf-8")
    except UnicodeDecodeError: text=""
    lines=text.splitlines()
    hits=[]
    for i,line in enumerate(lines,1):
        low=line.lower()
        matched=[t for t in TOKENS if t.lower() in low]
        if matched:
            hits.append({"line":i,"tokens":matched,"text":line[:500]})
    report={
      "schema":"RealSaS.KnightArachneV6UncertaintySourceAudit.v1",
      "status":"READ_ONLY_HASH_SEALED_SOURCE_AUDIT",
      "preregistration":str(PREREG),
      "teacher_loaded":False,
      "architecture_id":pr.architecture_id,
      "model_source_path":str(path),
      "model_source_sha256":actual,
      "source_utf8":bool(text),
      "source_line_count":len(lines),
      "token_hit_counts":{t:sum(t in h["tokens"] for h in hits) for t in TOKENS},
      "hits":hits[:500],
      "finding":{
        "has_log_sigma":any("log_sigma" in h["tokens"] for h in hits),
        "has_uncertainty_literal":any("uncertainty" in h["tokens"] for h in hits),
        "has_confidence_literal":any("confidence" in h["tokens"] for h in hits),
        "has_reliability_literal":any("reliability" in h["tokens"] for h in hits),
        "has_skin_proposal_literal":any("SkinProposalIR" in h["tokens"] for h in hits)
      },
      "claim_boundary":"Only the exact Stage30 hash-sealed model source is inspected. No execution, teacher data, or model mutation occurs."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_ARACHNE_V6_UNCERTAINTY_SOURCE_AUDIT_PASS",json.dumps({k:v for k,v in report.items() if k!="hits"},sort_keys=True))
    for h in hits[:80]:print("SOURCE_HIT",json.dumps(h,sort_keys=True))
if __name__=="__main__":main()
