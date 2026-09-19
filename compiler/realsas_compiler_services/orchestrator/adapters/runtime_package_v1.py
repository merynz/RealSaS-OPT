from __future__ import annotations

"""Stage-37 compact .rss materialization from exact Stage-36 projection."""

from dataclasses import replace
from pathlib import Path

from compiler.realsas_compiler_core.runtime_package_v1 import build_runtime_v4_package_seal
from compiler.realsas_compiler_core.runtime_projection_v1 import (
    runtime_v4_projection_from_dict,to_runtime_v4_objects,
)
from compiler.realsas_compiler_core.playback_runtime_v3 import RuntimeV3TexturePayload
from compiler.realsas_compiler_services.export.runtime_v4 import materialize_runtime_v4_archive
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _sha256,_stage_output_payload,_write_ir,
)
from compiler.realsas_compiler_core.types import QualificationError


def materialize_runtime_v4_stage(ctx:dict)->dict:
    projection=runtime_v4_projection_from_dict(
        _stage_output_payload(ctx,"36_RUNTIME_V4_PROJECT","RealSaS.RuntimeV4ProjectionIR.v1")
    )
    contract,textures,clips=to_runtime_v4_objects(projection)
    parents={Path(t.texture_path).resolve().parent for t in textures}
    if len(parents)!=1:
        raise QualificationError("RUNTIME_STAGE37_TEXTURE_ROOT_NOT_UNIQUE")
    texture_root=next(iter(parents))
    relative=tuple(RuntimeV3TexturePayload(
        t.view_id,Path(t.texture_path).name,t.texture_sha256,t.texture_crc32,t.width,t.height
    ) for t in textures)
    root=ctx["run_root"]/"artifacts"/"37_RSS_MATERIALIZE_COMPACT"
    root.mkdir(parents=True,exist_ok=True)
    archive=root/"product_runtime_v4.rss"
    result=materialize_runtime_v4_archive(
        out_path=archive,texture_root=texture_root,contract=contract,textures=relative,clips=clips,
        source_product_state_hash=projection.product_state_binding_hash,
        source_proof_bundle_hash=projection.dynamic_motion_binding_hash,
        source_authority_kind="RUNTIME_V4_ADMISSION_CERTIFICATE",
    )
    if _sha256(archive)!=result["archive_sha256"]:
        raise QualificationError("RUNTIME_STAGE37_ARCHIVE_HASH_REPLAY_DRIFT")
    seal=build_runtime_v4_package_seal(projection=projection,result=result)
    return {
        "status":"PASS",
        "outputs":[
            _write_ir(root/"runtime_v4_package_seal.json",seal,authority_class="QUALIFIED_RUNTIME_V4_PACKAGE_SEAL"),
            {"path":str(archive),"sha256":result["archive_sha256"],"authority_class":"RUNTIME_V4_RSS_PACKAGE","schema":"application/x-realsas-rss"},
        ],
        "diagnostics":{
            "package_seal_hash":seal.package_seal_hash,
            "archive_sha256":seal.archive_sha256,
            "runtime_binary_sha256":seal.runtime_binary_sha256,
            "package_total_bytes":seal.package_total_bytes,
            "solver_replay_performed":False,
        },
    }
