from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload

_MOVING_ALIAS_TOKENS = {"latest", "current", "newest"}
SOURCE_CLOSURE_PROFILE = "EXACT_SOURCE_CLOSURE_V2"
LICENSE_PROVENANCE_PROFILE = "EXACT_SOURCE_PROVENANCE_V2"
_SOURCE_KINDS = {"PRIMARY", "DEPENDENCY", "EVIDENCE", "LICENSE", "PROVENANCE"}


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def _write(path:Path,payload:dict[str,Any])->dict:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return {"path":str(path),"sha256":_sha256(path),"authority_class":"SEALED_SOURCE_EVIDENCE","schema":payload["schema"]}


def _local_gltf_dependencies(primary:Path)->set[Path]:
    try:
        payload=json.loads(primary.read_text(encoding="utf-8"))
    except Exception as exc:
        raise QualificationError("SOURCE_GLTF_PARSE_FAILED") from exc
    refs=[]
    for row in list(payload.get("buffers") or ()) + list(payload.get("images") or ()):
        uri=str(row.get("uri") or "").strip()
        if not uri or uri.startswith("data:"):
            continue
        parsed=urlparse(uri)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise QualificationError("SOURCE_GLTF_EXTERNAL_URI_MUST_BE_LOCAL_FILE")
        decoded=unquote(parsed.path)
        if not decoded:
            raise QualificationError("SOURCE_GLTF_EXTERNAL_URI_INVALID")
        refs.append((primary.parent/decoded).resolve())
    return set(refs)


def seal_source_bytes(ctx:dict)->dict:
    manifest=dict(ctx["run_manifest"]); rows=list(manifest.get("source_files") or ())
    if not rows: return {"status":"BLOCKED","blockers":["SOURCE_FILES_MISSING"],"diagnostics":{}}
    sealed=[]
    seen_roles=set()
    primary_rows=[]
    declared_dependency_paths=set()
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
        kind=str(row.get("source_kind") or "").strip().upper()
        if kind and kind not in _SOURCE_KINDS:
            return {"status":"FAIL","blockers":[f"SOURCE_KIND_INVALID:{role}:{kind}"],"diagnostics":{}}
        if kind=="PRIMARY": primary_rows.append((role,path))
        if kind=="DEPENDENCY": declared_dependency_paths.add(path)
        sealed.append({"role":role,"path":str(path),"sha256":digest,"size_bytes":size,**({"source_kind":kind} if kind else {})})

    closure_profile=""
    primary_role=""
    dependency_count=0
    if primary_rows:
        if len(primary_rows)!=1:
            return {"status":"FAIL","blockers":["SOURCE_PRIMARY_CARDINALITY_MUST_BE_ONE"],"diagnostics":{"primary_count":len(primary_rows)}}
        closure_profile=SOURCE_CLOSURE_PROFILE
        primary_role,primary_path=primary_rows[0]
        suffix=primary_path.suffix.lower()
        if suffix==".gltf":
            try:
                required=_local_gltf_dependencies(primary_path)
            except QualificationError as exc:
                return {"status":"FAIL","blockers":[str(exc)],"diagnostics":{"primary_role":primary_role}}
            missing=sorted(str(x) for x in required-declared_dependency_paths)
            if missing:
                return {"status":"FAIL","blockers":["SOURCE_GLTF_DEPENDENCY_CLOSURE_INCOMPLETE"],"diagnostics":{"missing":missing,"primary_role":primary_role}}
            dependency_count=len(required)
        else:
            dependency_count=len(declared_dependency_paths)

    payload={"schema":"RealSaS.SourceByteSeal.v1","run_id":ctx["run_id"],"subject_id":manifest.get("subject_id",""),"files":sorted(sealed,key=lambda x:x["role"]),"filename_is_not_authority":True,
             "source_closure_profile":closure_profile,"primary_source_role":primary_role,"resolved_dependency_count":dependency_count}
    out=ctx["run_root"]/"artifacts"/"01_SOURCE_BYTES_SEALED"/"source_byte_seal.json"
    return {"status":"PASS","outputs":[_write(out,payload)],"diagnostics":{"file_count":len(sealed),"source_closure_profile":closure_profile or "LEGACY","resolved_dependency_count":dependency_count}}


def seal_source_license_provenance(ctx:dict)->dict:
    manifest=dict(ctx["run_manifest"]); info=dict(manifest.get("source_license") or {}); required=("source_pack","license_name","license_ref")
    missing=[k for k in required if not str(info.get(k,"")).strip()]
    if missing: return {"status":"BLOCKED","blockers":["SOURCE_LICENSE_INCOMPLETE:"+",".join(missing)],"diagnostics":{}}

    profile=str(info.get("profile") or "").strip()
    exact={}
    if profile:
        if profile!=LICENSE_PROVENANCE_PROFILE:
            return {"status":"FAIL","blockers":["SOURCE_LICENSE_PROFILE_UNKNOWN"],"diagnostics":{"profile":profile}}
        license_role=str(info.get("license_file_role") or "").strip()
        page_role=str(info.get("source_page_role") or "").strip()
        legalcode_role=str(info.get("license_legalcode_role") or "").strip()
        if not license_role or not page_role:
            return {"status":"BLOCKED","blockers":["SOURCE_LICENSE_EXACT_ROLES_REQUIRED"],"diagnostics":{}}
        source=stage_output_payload(ctx,"01_SOURCE_BYTES_SEALED","RealSaS.SourceByteSeal.v1")
        roles={str(row.get("role")):row for row in source.get("files") or ()}
        for label,role in (("license_file",license_role),("source_page",page_role)):
            if role not in roles:
                return {"status":"FAIL","blockers":[f"SOURCE_LICENSE_STAGE1_ROLE_MISSING:{role}"],"diagnostics":{}}
            exact[label]={"role":role,"sha256":roles[role]["sha256"],"size_bytes":roles[role]["size_bytes"]}
        if legalcode_role:
            if legalcode_role not in roles:
                return {"status":"FAIL","blockers":[f"SOURCE_LICENSE_STAGE1_ROLE_MISSING:{legalcode_role}"],"diagnostics":{}}
            exact["license_legalcode"]={"role":legalcode_role,"sha256":roles[legalcode_role]["sha256"],"size_bytes":roles[legalcode_role]["size_bytes"]}

    payload={"schema":"RealSaS.SourceLicenseSeal.v1","run_id":ctx["run_id"],"subject_id":manifest.get("subject_id",""),"source_pack":info["source_pack"],"license_name":info["license_name"],"license_ref":info["license_ref"],"source_url_or_locator":info.get("source_url_or_locator",""),"notes":info.get("notes",""),
             "provenance_profile":profile,"exact_byte_bindings":exact}
    out=ctx["run_root"]/"artifacts"/"02_SOURCE_LICENSE_PROVENANCE"/"source_license_seal.json"
    return {"status":"PASS","outputs":[_write(out,payload)],"diagnostics":{"provenance_complete":True,"provenance_profile":profile or "LEGACY","exact_byte_binding_count":len(exact)}}
