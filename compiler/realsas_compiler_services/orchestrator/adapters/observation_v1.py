from __future__ import annotations

"""Observation/camera authority adapters."""

import hashlib, json, os
from pathlib import Path

from compiler.realsas_compiler_core.camera_authority_v1 import build_qualified_camera_set
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import _write_ir


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def _path(raw:str)->Path:
    p=Path(os.path.expandvars(str(raw))).expanduser().resolve()
    if any(token in p.parts for token in ("latest","current","newest")):
        raise QualificationError("OBSERVATION_ADAPTER_MOVING_ALIAS_FORBIDDEN")
    return p


def qualify_camera_contract_stage(ctx:dict)->dict:
    cfg=dict(ctx["run_manifest"].get("observation") or {})
    ref=dict(cfg.get("camera_bundle") or {})
    path=_path(str(ref.get("path") or ""))
    expected=str(ref.get("sha256") or "")
    if not path.is_file() or len(expected)!=64 or _sha256(path)!=expected:
        return {"status":"BLOCKED","blockers":["CAMERA_BUNDLE_FILE_REF_INVALID"],"diagnostics":{}}
    payload=json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema") or payload.get("schema_version") or "")!="RealSaS.CameraProjectionBundle.v1":
        return {"status":"FAIL","blockers":["CAMERA_BUNDLE_SCHEMA_INVALID"],"diagnostics":{}}
    camera_set=build_qualified_camera_set(
        tuple(payload.get("cameras") or ()),
        source_bundle_sha256=expected,
        metadata={"source_path_role":"RUN_MANIFEST_EXACT_HASH_MATERIALIZATION"},
    )
    root=ctx["run_root"]/"artifacts"/"05_CAMERA_CONTRACT_SOLVED"
    return {
        "status":"PASS",
        "outputs":[_write_ir(root/"qualified_camera_set.json",camera_set,authority_class="QUALIFIED_CAMERA_SET")],
        "diagnostics":{
            "camera_set_hash":camera_set.camera_set_hash,
            "camera_count":len(camera_set.cameras),
            "resolution":camera_set.cameras[0].resolution,
            "half_extent":camera_set.cameras[0].half_extent,
        },
    }
