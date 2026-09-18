from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def _write(path:Path,payload:dict[str,Any])->dict:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return {"path":str(path),"sha256":_sha256(path),"authority_class":"SEALED_SOURCE_EVIDENCE","schema":payload["schema"]}

def seal_source_bytes(ctx:dict)->dict:
    manifest=dict(ctx["run_manifest"]); rows=list(manifest.get("source_files") or ())
    if not rows: return {"status":"BLOCKED","blockers":["SOURCE_FILES_MISSING"],"diagnostics":{}}
    sealed=[]
    for row in rows:
        role=str(row.get("role","")).strip(); path=Path(str(row.get("path",""))).expanduser().resolve()
        if not role or not path.is_file(): return {"status":"FAIL","blockers":[f"SOURCE_FILE_INVALID:{role}:{path}"],"diagnostics":{}}
        digest=_sha256(path); size=path.stat().st_size; expected_sha=str(row.get("expected_sha256","") or ""); expected_size=row.get("expected_size_bytes")
        if expected_sha and expected_sha!=digest: return {"status":"FAIL","blockers":[f"SOURCE_SHA_MISMATCH:{role}"],"diagnostics":{"actual_sha256":digest}}
        if expected_size is not None and int(expected_size)!=size: return {"status":"FAIL","blockers":[f"SOURCE_SIZE_MISMATCH:{role}"],"diagnostics":{"actual_size_bytes":size}}
        sealed.append({"role":role,"path":str(path),"sha256":digest,"size_bytes":size})
    payload={"schema":"RealSaS.SourceByteSeal.v1","run_id":ctx["run_id"],"subject_id":manifest.get("subject_id",""),"files":sorted(sealed,key=lambda x:x["role"]),"filename_is_not_authority":True}
    out=ctx["run_root"]/"artifacts"/"01_SOURCE_BYTES_SEALED"/"source_byte_seal.json"
    return {"status":"PASS","outputs":[_write(out,payload)],"diagnostics":{"file_count":len(sealed)}}

def seal_source_license_provenance(ctx:dict)->dict:
    manifest=dict(ctx["run_manifest"]); info=dict(manifest.get("source_license") or {}); required=("source_pack","license_name","license_ref")
    missing=[k for k in required if not str(info.get(k,"")).strip()]
    if missing: return {"status":"BLOCKED","blockers":["SOURCE_LICENSE_INCOMPLETE:"+",".join(missing)],"diagnostics":{}}
    payload={"schema":"RealSaS.SourceLicenseSeal.v1","run_id":ctx["run_id"],"subject_id":manifest.get("subject_id",""),"source_pack":info["source_pack"],"license_name":info["license_name"],"license_ref":info["license_ref"],"source_url_or_locator":info.get("source_url_or_locator",""),"notes":info.get("notes","")}
    out=ctx["run_root"]/"artifacts"/"02_SOURCE_LICENSE_PROVENANCE"/"source_license_seal.json"
    return {"status":"PASS","outputs":[_write(out,payload)],"diagnostics":{"provenance_complete":True}}
