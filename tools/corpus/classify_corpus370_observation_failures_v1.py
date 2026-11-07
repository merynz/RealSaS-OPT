"""Exact-source 370 observation-failure strata. Accounting, never admission.

Read a sealed family-input bootstrap and the original observation policy. Never
infer one family repair from another or turn the 76 historical FAILs into MASKED.
The resulting typed per-subject worklist is suitable for a scoped repair notebook
only AFTER its producer/independent courts are versioned and qualified.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "RealSaS.Corpus370ObservationFailureStrata.v1"
MASK = re.compile(r"^ADMISSION_MASK_DRIFT:([^:]+):V([0-7]):([^:]+)$")
BIND = re.compile(r"^ADMISSION_BIND_VERTEX_DRIFT:([^:]+):([^:]+)$")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def classify(manifest_path, policy_path):
    mp, pp = Path(manifest_path), Path(policy_path)
    m = json.loads(mp.read_text()); policy = json.loads(pp.read_text())
    if (m.get("schema") != "RealSaS.Corpus370PlatformFamilyInputBundle.v2"
            or m.get("subject_count") != 370
            or m.get("candidate_subject_count") != 370
            or m.get("training_authorized") is not False
            or policy.get("schema") != "RealSaS.IrisObservationPreparationPolicy.v1"
            or policy.get("view_count") != 8
            or policy.get("training_authorized") is not False):
        raise ValueError("CORPUS370_OBSERVATION_STRATA_SOURCE_SCOPE")
    rows=m.get("subjects") or []
    if len(rows)!=370 or len(set(row["subject_id"] for row in rows))!=370:
        raise ValueError("CORPUS370_OBSERVATION_STRATA_IDENTITY")
    strata=[]
    success=0
    for row in rows:
        sid=row["subject_id"]
        original=row["observation_preparation"]
        if original.get("status")=="PASS":
            success+=1
            continue
        if original.get("status")!="FAIL":
            raise ValueError("CORPUS370_OBSERVATION_STRATA_UNKNOWN_OUTCOME")
        reason=original.get("reason") or ""
        mask=MASK.fullmatch(reason)
        bind=BIND.fullmatch(reason)
        if bool(mask)==bool(bind):
            raise ValueError("CORPUS370_OBSERVATION_STRATA_UNCLASSIFIED:"+sid)
        if (mask and mask.group(1)!=sid) or (bind and bind.group(1)!=sid):
            raise ValueError("CORPUS370_OBSERVATION_STRATA_SUBJECT_DRIFT:"+sid)
        measured=float(mask.group(3) if mask else bind.group(2))
        if (not math.isfinite(measured) or measured<=(
                float(policy["maximum_mask_gap_px"]) if mask
                else float(policy["maximum_relative_vertex_error"]))):
            raise ValueError("CORPUS370_OBSERVATION_STRATA_NO_MEASURED_VIOLATION:"+sid)
        kind="MASK" if mask else "BIND_VERTEX"
        strata.append({
            "subject_id":sid,
            "source_sha256":row["raw_source"]["sha256"],
            "display_name":row["display_name"],
            "pack_id":row["pack_id"],
            "source_format":original["source_format"],
            "repair_lane":kind,
            "observation_view_id":("V"+mask.group(2)) if mask else None,
            "measured_violation":measured,
            "unchanged_policy_limit":float(policy["maximum_mask_gap_px"] if mask else policy["maximum_relative_vertex_error"]),
            "failure_receipt":original["failure_receipt"],
            "parent_failure_reason":reason,
            "historical_admission_status":"FAIL",
            "training_authorized":False,
        })
    if success!=m.get("observation_qualified_count") or len(strata)!=m.get("observation_failed_count"):
        raise ValueError("CORPUS370_OBSERVATION_STRATA_PARTITION")
    mask_rows=[r for r in strata if r["repair_lane"]=="MASK"]
    bind_rows=[r for r in strata if r["repair_lane"]=="BIND_VERTEX"]
    return {
        "schema":SCHEMA,
        "status":"HISTORICAL_FAILURE_STRATA_ACCOUNTED__NO_REPAIR_OR_ADMISSION",
        "parents":{"bootstrap":{"path":str(mp.resolve()),"sha256":sha(mp)},
                   "policy":{"path":str(pp.resolve()),"sha256":sha(pp)}},
        "candidate_count":370,
        "old_observation_pass_count":success,
        "old_observation_failure_count":len(strata),
        "mask_failures":len(mask_rows),
        "bind_vertex_failures":len(bind_rows),
        "by_lane_and_source_format":{
            lane:dict(sorted(Counter(r["source_format"] for r in group).items()))
            for lane,group in (("MASK",mask_rows),("BIND_VERTEX",bind_rows))
        },
        "by_lane_and_pack":{
            lane:dict(sorted(Counter(r["pack_id"] for r in group).items()))
            for lane,group in (("MASK",mask_rows),("BIND_VERTEX",bind_rows))
        },
        "subjects":sorted(strata,key=lambda r:r["subject_id"]),
        "authority":{"existing_failures_unchanged":True,"repair_qualified":False,
                     "training_authorized":False,
                     "all_370_admitted":False},
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap",type=Path,required=True)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=classify(a.bootstrap,a.policy)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ("old_observation_pass_count",
      "old_observation_failure_count","mask_failures","bind_vertex_failures")},sort_keys=True))


if __name__=="__main__":main()
