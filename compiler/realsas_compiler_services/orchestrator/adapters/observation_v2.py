from __future__ import annotations

"""V2 observation/camera authority adapter."""

import json

from compiler.realsas_compiler_core.camera_authority_v1 import (
    build_qualified_camera_set,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    resolved_path,
    sha256_file,
    write_ir,
)


def qualify_camera_contract_stage(ctx: dict) -> dict:
    cfg = dict(ctx["run_manifest"].get("observation") or {})
    ref = dict(cfg.get("camera_bundle") or {})
    path = resolved_path(str(ref.get("path") or ""))
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
        return {
            "status": "BLOCKED",
            "blockers": ["CAMERA_BUNDLE_FILE_REF_INVALID"],
            "diagnostics": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        str(payload.get("schema") or payload.get("schema_version") or "")
        != "RealSaS.CameraProjectionBundle.v1"
    ):
        return {
            "status": "FAIL",
            "blockers": ["CAMERA_BUNDLE_SCHEMA_INVALID"],
            "diagnostics": {},
        }
    camera_set = build_qualified_camera_set(
        tuple(payload.get("cameras") or ()),
        source_bundle_sha256=expected,
        metadata={
            "source_path_role": "RUN_MANIFEST_EXACT_HASH_MATERIALIZATION",
            "v2_authority": True,
            "camera_refit_downstream_forbidden": True,
        },
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "qualified_camera_set.json",
                camera_set,
                authority_class="QUALIFIED_CAMERA_SET",
            )
        ],
        "diagnostics": {
            "camera_set_hash": camera_set.camera_set_hash,
            "camera_count": len(camera_set.cameras),
            "resolution": camera_set.cameras[0].resolution,
            "half_extent": camera_set.cameras[0].half_extent,
        },
    }
