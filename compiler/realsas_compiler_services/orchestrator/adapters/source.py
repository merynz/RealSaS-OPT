from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

_MOVING_ALIAS_TOKENS = {"latest", "current", "newest"}
_EXACT_PROVENANCE_PROFILE = "EXACT_SOURCE_PROVENANCE_V2"
_SHARED_IRIS_PROFILE = "SHARED_IRIS_IMAGE_DERIVED_FSTAR_V1"


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
    seen_roles=set()
    for row in rows:
        role=str(row.get("role","")).strip()
        raw_path=Path(str(row.get("path",""))).expanduser()
        if any(str(part).lower() in _MOVING_ALIAS_TOKENS for part in raw_path.parts):
            return {"status":"FAIL","blockers":[f"SOURCE_MOVING_ALIAS_FORBIDDEN:{role}"],"diagnostics":{"path":str(raw_path)}}
        path=raw_path.resolve()
        if role in seen_roles:
            return {"status":"FAIL","blockers":[f"SOURCE_ROLE_DUPLICATE:{role}"],"diagnostics":{}}
        seen_roles.add(role)
        if not role or not path.is_file(): return {"status":"FAIL","blockers":[f"SOURCE_FILE_INVALID:{role}:{path}"],"diagnostics":{}}
        digest=_sha256(path); size=path.stat().st_size; expected_sha=str(row.get("expected_sha256","") or ""); expected_size=row.get("expected_size_bytes")
        if expected_sha and expected_sha!=digest: return {"status":"FAIL","blockers":[f"SOURCE_SHA_MISMATCH:{role}"],"diagnostics":{"actual_sha256":digest}}
        if expected_size is not None and int(expected_size)!=size: return {"status":"FAIL","blockers":[f"SOURCE_SIZE_MISMATCH:{role}"],"diagnostics":{"actual_size_bytes":size}}
        sealed.append({"role":role,"path":str(path),"sha256":digest,"size_bytes":size})
    payload={"schema":"RealSaS.SourceByteSeal.v1","run_id":ctx["run_id"],"subject_id":manifest.get("subject_id",""),"files":sorted(sealed,key=lambda x:x["role"]),"filename_is_not_authority":True}
    out=ctx["run_root"]/"artifacts"/"01_SOURCE_BYTES_SEALED"/"source_byte_seal.json"
    return {"status":"PASS","outputs":[_write(out,payload)],"diagnostics":{"file_count":len(sealed)}}


def _exact_provenance(ctx:dict,info:dict)->dict|None:
    profile=str(info.get("profile") or "")
    shared_profile=str((ctx["run_manifest"].get("iris_fit") or {}).get("training_profile") or "")
    if shared_profile==_SHARED_IRIS_PROFILE and profile!=_EXACT_PROVENANCE_PROFILE:
        raise ValueError("SHARED_IRIS_EXACT_SOURCE_PROVENANCE_REQUIRED")
    if not profile:
        return None
    if profile!=_EXACT_PROVENANCE_PROFILE:
        raise ValueError("SOURCE_PROVENANCE_PROFILE_UNKNOWN")
    from compiler.realsas_compiler_core.types import QualificationError
    from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import load_file_ref, stage_output_payload
    source=stage_output_payload(ctx,"01_SOURCE_BYTES_SEALED","RealSaS.SourceByteSeal.v1")
    hashes={str(row.get("sha256") or "") for row in source.get("files") or ()}
    refs={
        "license_bytes":dict(info.get("license_bytes") or {}),
        "source_page_snapshot":dict(info.get("source_page_snapshot") or {}),
        "corpus_seal":dict(info.get("corpus_seal") or {}),
    }
    bound={}
    for key,ref in refs.items():
        digest=str(ref.get("sha256") or "")
        if len(digest)!=64 or digest not in hashes:
            raise QualificationError(f"SOURCE_PROVENANCE_{key.upper()}_NOT_STAGE1_SEALED")
        path=load_file_ref(ref,json_required=False)
        bound[key]={"sha256":digest,"size_bytes":path.stat().st_size}
    if str(info.get("license_name") or "")!="CC0-1.0":
        raise QualificationError("SOURCE_PROVENANCE_LICENSE_POLICY_DRIFT")
    expected_page=str(info.get("expected_source_page_sha256") or "")
    if len(expected_page)!=64 or expected_page!=bound["source_page_snapshot"]["sha256"]:
        raise QualificationError("SOURCE_PROVENANCE_SOURCE_PAGE_HASH_DRIFT")
    expected_source=str(info.get("primary_source_sha256") or "")
    if len(expected_source)!=64 or expected_source not in hashes:
        raise QualificationError("SOURCE_PROVENANCE_PRIMARY_SOURCE_NOT_STAGE1_SEALED")
    return {
        "profile":_EXACT_PROVENANCE_PROFILE,
        "primary_source_sha256":expected_source,
        **bound,
        "authority_scope":"EXACT_SOURCE_LICENSE_AND_SOURCE_PAGE_PROVENANCE_ONLY",
    }


def seal_source_license_provenance(ctx:dict)->dict:
    manifest=dict(ctx["run_manifest"]); info=dict(manifest.get("source_license") or {}); required=("source_pack","license_name","license_ref")
    missing=[k for k in required if not str(info.get(k,"")).strip()]
    if missing: return {"status":"BLOCKED","blockers":["SOURCE_LICENSE_INCOMPLETE:"+",".join(missing)],"diagnostics":{}}
    try:
        exact=_exact_provenance(ctx,info)
    except ValueError as exc:
        return {"status":"BLOCKED","blockers":[str(exc)],"diagnostics":{}}
    payload={"schema":"RealSaS.SourceLicenseSeal.v1","run_id":ctx["run_id"],"subject_id":manifest.get("subject_id",""),"source_pack":info["source_pack"],"license_name":info["license_name"],"license_ref":info["license_ref"],"source_url_or_locator":info.get("source_url_or_locator",""),"notes":info.get("notes","")}
    if exact is not None:
        payload["exact_provenance"]=exact
    out=ctx["run_root"]/"artifacts"/"02_SOURCE_LICENSE_PROVENANCE"/"source_license_seal.json"
    return {"status":"PASS","outputs":[_write(out,payload)],"diagnostics":{"provenance_complete":True,"exact_provenance_bound":exact is not None}}
