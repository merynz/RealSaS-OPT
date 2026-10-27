#!/usr/bin/env python3
from __future__ import annotations

"""Read-only inventory of the admitted FIT8 source-observation authority.

This court does not choose an opacity policy.  It verifies the exact admission
bytes sealed by ``NATIVE_TEACHER_REPLAY_REQUEST_V1`` and inventories the nearby
metadata/receipt artifacts so a later court can identify the already-admitted
foreground/alpha witness instead of guessing from Blender material sockets.
"""

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "RealSaS.SourceObservationOpacityAuthorityInventory.v1"
TOKENS = (
    "alpha", "foreground", "mask", "rgba", "raster", "render", "view",
    "observation", "candidate", "source", "path", "sha256", "status", "admission",
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _interesting(value, prefix=""):
    rows=[]
    if isinstance(value, dict):
        for key, child in value.items():
            path=f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child,(dict,list)):
                rows.extend(_interesting(child,path))
            elif any(token in path.lower() for token in TOKENS):
                rows.append({"path":path,"value":child})
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path=f"{prefix}[{index}]"
            if isinstance(child,(dict,list)):
                rows.extend(_interesting(child,path))
            elif any(token in path.lower() for token in TOKENS):
                rows.append({"path":path,"value":child})
    return rows


def audit(request_path: Path, authority_root: Path) -> dict:
    request=json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema")!="RealSaS.NativeTeacherReplayRequest.v1":
        raise RuntimeError("SOURCE_OBSERVATION_AUTHORITY_REQUEST_SCHEMA")
    subjects=tuple(request.get("subjects") or ())
    if len(subjects)!=8:
        raise RuntimeError("SOURCE_OBSERVATION_AUTHORITY_EXPECTED_FIT8")

    rows=[]
    for subject in subjects:
        sid=str(subject["subject_id"])
        ref=subject["observation_admission"]
        admission=(authority_root/str(ref["relative"])).resolve()
        if not admission.is_file():
            raise RuntimeError("SOURCE_OBSERVATION_ADMISSION_MISSING:"+sid)
        digest=_sha(admission)
        if digest!=str(ref["sha256"]):
            raise RuntimeError("SOURCE_OBSERVATION_ADMISSION_HASH_DRIFT:"+sid)
        value=json.loads(admission.read_text(encoding="utf-8"))
        directory=admission.parent
        nearby=[]
        # Metadata only; never copy binary source/raster payload bytes into the receipt.
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            rel=path.relative_to(directory).as_posix()
            suffix=path.suffix.lower()
            row={"relative":rel,"size_bytes":path.stat().st_size,"suffix":suffix}
            if suffix in {".json",".txt",".md"} and path.stat().st_size<=2_000_000:
                row["sha256"]=_sha(path)
                if suffix==".json":
                    try:
                        parsed=json.loads(path.read_text(encoding="utf-8"))
                        row["schema"]=parsed.get("schema") if isinstance(parsed,dict) else None
                        row["interesting_fields"]=_interesting(parsed)
                    except Exception as exc:
                        row["json_parse_error"]=type(exc).__name__
            nearby.append(row)
        rows.append({
            "subject_id":sid,
            "admission_relative":str(ref["relative"]),
            "admission_sha256":digest,
            "admission_schema":value.get("schema"),
            "admission_top_level_keys":sorted(value) if isinstance(value,dict) else [],
            "admission_interesting_fields":_interesting(value),
            "nearby_metadata_files":nearby,
        })

    return {
        "schema":SCHEMA,
        "request_sha256":_sha(request_path),
        "subject_count":len(rows),
        "qualification_minted":False,
        "opacity_policy_selected":False,
        "sampler_qualified":False,
        "student_target_allowed":False,
        "training_allowed":False,
        "court_status":"PASS_READ_ONLY_SOURCE_OBSERVATION_AUTHORITY_INVENTORY",
        "subjects":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--request",required=True,type=Path)
    p.add_argument("--authority-root",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    result=audit(a.request,a.authority_root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="subjects"},sort_keys=True))

if __name__=="__main__":
    main()
