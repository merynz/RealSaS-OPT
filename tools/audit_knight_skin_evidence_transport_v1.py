from __future__ import annotations

import argparse
import json
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import qualified_skin_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import model_fit_execution_from_dict
from tools.demo.render_knight_motion_preview_v1 import _ctx

PREREG=Path("canonical/KNIGHT_SKIN_EVIDENCE_TRANSPORT_AUDIT_PREREG_V1_20260929.json")

PER_ROW_TOKENS=("confidence","uncertainty","sigma","variance","valid","coverage","support")


def _find_candidate_fields(payload):
    hits=[]
    def walk(value,path):
        if isinstance(value,dict):
            for k,v in value.items():
                p=path+[str(k)]
                if any(t in str(k).lower() for t in PER_ROW_TOKENS):
                    hits.append({"path":".".join(p),"type":type(v).__name__,"length":len(v) if isinstance(v,(list,dict,tuple)) else None})
                walk(v,p)
        elif isinstance(value,list):
            # Do not recurse into all influences; sample first few structural rows only.
            for i,v in enumerate(value[:3]):
                walk(v,path+[str(i)])
    walk(payload,[])
    return hits


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()

    if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_SKIN_EVIDENCE_TRANSPORT_RESULT":
        raise RuntimeError("SKIN_EVIDENCE_TRANSPORT_PREREG_DRIFT")

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    execution_payload=json.loads(
        (rr/"artifacts/31_ARACHNE_FIT/model_fit_execution.json").read_text()
    )
    execution=model_fit_execution_from_dict(execution_payload)
    proposal_path=Path(str(execution.proposal_path)).expanduser().resolve()
    if not proposal_path.is_file():
        raise RuntimeError("SKIN_PROPOSAL_PATH_MISSING")
    proposal=json.loads(proposal_path.read_text())

    skin_payload=json.loads(
        (rr/"artifacts/32_SKIN_QUALIFIED/qualified_skin.json").read_text()
    )
    skin=qualified_skin_from_dict(skin_payload)
    q=dict(skin.qualification_report or {})
    meta=dict(proposal.get("metadata") or {})

    influence_rows=list(proposal.get("influences") or [])
    influence_keys=sorted(set().union(*(set(x) for x in influence_rows[:100] if isinstance(x,dict)))) if influence_rows else []

    row_payloads=list(skin_payload.get("rows") or [])
    qualified_row_keys=sorted(set().union(*(set(x) for x in row_payloads[:100] if isinstance(x,dict)))) if row_payloads else []

    proposal_candidate_fields=_find_candidate_fields({
        k:v for k,v in proposal.items() if k!="influences"
    })

    per_row_top_level=[]
    for k,v in proposal.items():
        if k=="influences":
            continue
        if isinstance(v,list) and len(v) in {len(skin.rows), len(set(x.get("surface_id") for x in influence_rows if isinstance(x,dict) and x.get("surface_id")))}:
            per_row_top_level.append({"key":str(k),"length":len(v)})

    report={
      "schema":"RealSaS.KnightSkinEvidenceTransportAudit.v1",
      "status":"READ_ONLY_EVIDENCE_TRANSPORT_AUDIT",
      "preregistration":str(PREREG),
      "teacher_loaded":False,
      "stage31":{
        "proposal_schema":str(proposal.get("schema") or proposal.get("schema_version") or ""),
        "proposal_top_level_keys":sorted(map(str,proposal.keys())),
        "proposal_metadata_keys":sorted(map(str,meta.keys())),
        "proposal_metadata":meta,
        "influence_count":len(influence_rows),
        "influence_row_keys":influence_keys,
        "candidate_confidence_uncertainty_fields":proposal_candidate_fields,
        "candidate_per_surface_top_level_arrays":per_row_top_level,
      },
      "stage32":{
        "qualified_row_count":len(skin.rows),
        "qualified_row_serialized_keys":qualified_row_keys,
        "qualification_report":q,
        "row_schema_can_represent_confidence_or_uncertainty":bool(
            any(any(t in k.lower() for t in PER_ROW_TOKENS) for k in qualified_row_keys)
        ),
      },
      "finding":{
        "stage31_has_per_influence_confidence":bool(any(any(t in k.lower() for t in PER_ROW_TOKENS) for k in influence_keys)),
        "stage31_has_candidate_per_surface_array":bool(per_row_top_level),
        "stage32_row_has_per_row_evidence_field":bool(any(any(t in k.lower() for t in PER_ROW_TOKENS) for k in qualified_row_keys)),
        "stage32_product_skin_evidence_complete_claim":q.get("product_skin_evidence_complete"),
        "stage32_row_confidence_available_claim":q.get("row_confidence_available"),
      },
      "claim_boundary":"Exact sealed Stage31 proposal and Stage32 QualifiedSkin are inspected read-only. No teacher data or inferred correctness labels are used."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+"\n")
    print("KNIGHT_SKIN_EVIDENCE_TRANSPORT_AUDIT_PASS",json.dumps(report,sort_keys=True,default=str))


if __name__=="__main__":
    main()
