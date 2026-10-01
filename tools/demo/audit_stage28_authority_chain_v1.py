from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def pick(d,*keys):
    return {k:d.get(k) for k in keys if k in d}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    a=ap.parse_args()
    rr=a.authority_root/"runs"/a.run_id
    paths={
      "s28":rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json",
      "s29":rr/"artifacts/29_GEPPETTO_CHECKPOINT_SEALED/model_checkpoint_seal.json",
      "a_exec":rr/"imports/arachne_stage31_v6/ARACHNE_V6_MAINLINE_EXECUTION_RECEIPT.json",
      "a_rebind":rr/"imports/arachne_stage31_v6/ARACHNE_KNIGHT_V6_SEMANTIC_ID_REBIND_RECEIPT.json",
      "a_verify":rr/"imports/arachne_stage31_v6/ARACHNE_V6_SEMANTIC_SKIN_MIGRATION_VERIFICATION.json",
      "s30":rr/"artifacts/30_ARACHNE_FIT_PREREGISTERED/model_fit_preregistration.json",
      "s31":rr/"artifacts/31_ARACHNE_FIT/model_fit_execution.json",
    }
    p={k:json.loads(v.read_text()) for k,v in paths.items()}
    sk=p["s28"]
    lineage=sk["skeleton_lineage_hash"]
    report={
      "schema":"RealSaS.Stage28AuthorityAudit.v1",
      "skeleton":{"path":str(paths["s28"].resolve()),"sha256":sha256(paths["s28"]),"lineage":lineage},
      "stage29":{"path":str(paths["s29"].resolve()),"sha256":sha256(paths["s29"]),
        **pick(p["s29"],"execution_binding_hash","checkpoint_sha256","result_sha256","qualified_output_binding_hash","checkpoint_seal_hash")},
      "arachne_execution":{"path":str(paths["a_exec"].resolve()),"sha256":sha256(paths["a_exec"]),
        "status":p["a_exec"].get("status"),"matching_fields":{k:v for k,v in p["a_exec"].items() if isinstance(v,str) and v==lineage}},
      "arachne_rebind":{"path":str(paths["a_rebind"].resolve()),"sha256":sha256(paths["a_rebind"]),
        "status":p["a_rebind"].get("status"),"matching_fields":{k:v for k,v in p["a_rebind"].items() if isinstance(v,str) and v==lineage}},
      "arachne_verify":{"path":str(paths["a_verify"].resolve()),"sha256":sha256(paths["a_verify"]),
        "status":p["a_verify"].get("status"),"matching_fields":{k:v for k,v in p["a_verify"].items() if isinstance(v,str) and v==lineage}},
      "stage30":{"path":str(paths["s30"].resolve()),"sha256":sha256(paths["s30"]),
        "upstream_bindings":p["s30"].get("upstream_bindings"),"preregistration_hash":p["s30"].get("preregistration_hash")},
      "stage31":{"path":str(paths["s31"].resolve()),"sha256":sha256(paths["s31"]),
        "upstream_bindings":p["s31"].get("upstream_bindings"),"preregistration_binding_hash":p["s31"].get("preregistration_binding_hash"),
        "execution_hash":p["s31"].get("execution_hash")},
    }
    report["checks"]={
      "stage29_qualified_output_matches_skeleton":p["s29"].get("qualified_output_binding_hash")==lineage,
      "stage30_mentions_skeleton":any(str(v)==lineage for _,v in (p["s30"].get("upstream_bindings") or [])),
      "stage31_mentions_skeleton":any(str(v)==lineage for _,v in (p["s31"].get("upstream_bindings") or [])),
      "arachne_execution_mentions_skeleton":any(isinstance(v,str) and v==lineage for v in p["a_exec"].values()),
      "arachne_rebind_mentions_skeleton":any(isinstance(v,str) and v==lineage for v in p["a_rebind"].values()),
    }
    print("STAGE28_AUTHORITY_AUDIT="+json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
