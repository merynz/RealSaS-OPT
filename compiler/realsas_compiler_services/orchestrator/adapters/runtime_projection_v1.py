from __future__ import annotations

"""Stage-36 compact Runtime-v4 projection adapter."""

from dataclasses import replace
from pathlib import Path
import zlib

from PIL import Image

from compiler.realsas_compiler_core.motion_dynamic_proof_v1 import qualified_dynamic_motion_from_dict
from compiler.realsas_compiler_core.motion_presentation_v1 import build_qualified_motion_presentation_v1
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    qualified_appearance_set_from_dict, qualified_camera_set_from_dict,
    qualified_composition_set_from_dict, qualified_mesh_from_dict,
    qualified_observation_set_from_dict, qualified_presentation_graph_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.runtime_projection_v1 import (
    RuntimeTextureBindingIR, build_runtime_v4_projection, runtime_v4_projection_hash,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _resolved_path, _sha256, _stage_output_payload, _write_ir,
)


def _transport_textures(ctx,observation_set,root:Path):
    cfg=dict(ctx["run_manifest"].get("observation") or {})
    rows=tuple(cfg.get("source_rasters") or ())
    if len(rows)!=8 or {int(r["view_index"]) for r in rows}!=set(range(8)):
        raise QualificationError("RUNTIME_STAGE36_SOURCE_RASTER_MATRIX_INCOMPLETE")
    authority={int(v.view_index):v for v in observation_set.views}
    texture_root=root/"textures"; texture_root.mkdir(parents=True,exist_ok=True)
    out=[]
    for row in sorted(rows,key=lambda r:int(r["view_index"])):
        vi=int(row["view_index"]); ref=dict(row.get("image") or {})
        source=_resolved_path(str(ref.get("path") or "")); expected=str(ref.get("sha256") or "")
        if not source.is_file() or len(expected)!=64 or _sha256(source)!=expected:
            raise QualificationError("RUNTIME_STAGE36_SOURCE_RASTER_REF_INVALID")
        if expected!=authority[vi].source_raster_sha256:
            raise QualificationError("RUNTIME_STAGE36_SOURCE_RASTER_AUTHORITY_DRIFT")
        with Image.open(source) as image:
            rgba=image.convert("RGBA")
            if rgba.size!=(int(authority[vi].width),int(authority[vi].height)):
                raise QualificationError("RUNTIME_STAGE36_SOURCE_RASTER_DIMENSION_DRIFT")
            target=texture_root/f"V{vi}.png"
            rgba.save(target,format="PNG",compress_level=1,optimize=False)
        raw=target.read_bytes()
        out.append(RuntimeTextureBindingIR(
            vi,expected,str(target),_sha256(target),zlib.crc32(raw)&0xffffffff,
            int(authority[vi].width),int(authority[vi].height),
            metadata={"png_is_transport_only":True,"source_raster_is_appearance_authority":True},
        ))
    return tuple(out)


def project_runtime_v4_stage(ctx:dict)->dict:
    camera_set=qualified_camera_set_from_dict(
        _stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    observation_set=qualified_observation_set_from_dict(
        _stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
    )
    surface=rigging_surface_from_dict(
        _stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1")
    )
    mesh=qualified_mesh_from_dict(
        _stage_output_payload(ctx,"27_QUALIFIED_MESH_GATE","RealSaS.QualifiedMeshIR.v1")
    )
    appearance=qualified_appearance_set_from_dict(
        _stage_output_payload(ctx,"30_QUALIFIED_PRESENTATION_GRAPH","RealSaS.QualifiedAppearanceSetIR.v1")
    )
    composition=qualified_composition_set_from_dict(
        _stage_output_payload(ctx,"30_QUALIFIED_PRESENTATION_GRAPH","RealSaS.QualifiedCompositionSetIR.v1")
    )
    presentation=qualified_presentation_graph_from_dict(
        _stage_output_payload(ctx,"30_QUALIFIED_PRESENTATION_GRAPH","RealSaS.QualifiedPresentationGraphIR.v1")
    )
    dynamic=qualified_dynamic_motion_from_dict(
        _stage_output_payload(ctx,"35_MOTION_DYNAMIC_PROOF","RealSaS.QualifiedDynamicMotionIR.v1")
    )
    root=ctx["run_root"]/"artifacts"/"36_RUNTIME_V4_PROJECT"
    textures=_transport_textures(ctx,observation_set,root)
    motion_presentation=build_qualified_motion_presentation_v1(
        dynamic=dynamic,mesh=mesh,surface=surface,appearance=appearance,
        composition=composition,camera_set=camera_set,
    )
    projection=build_runtime_v4_projection(
        mesh=mesh,presentation=presentation,appearance=appearance,composition=composition,
        camera_set=camera_set,dynamic=dynamic,observation_set_hash=observation_set.observation_set_hash,
        textures=textures,
    )
    projection=replace(
        projection,
        metadata={
            **dict(projection.metadata or {}),
            "motion_presentation_binding_hash":motion_presentation.motion_presentation_hash,
            "presentation_optimization_domain":"PRESENTATION_ONLY__NO_POSE_MUTATION",
        },
        projection_hash="",
    )
    projection=replace(projection,projection_hash=runtime_v4_projection_hash(projection))
    outputs=[
        _write_ir(root/"qualified_motion_presentation.json",motion_presentation,authority_class="QUALIFIED_2D_MOTION_PRESENTATION"),
        _write_ir(root/"runtime_v4_projection.json",projection,authority_class="QUALIFIED_RUNTIME_V4_PROJECTION"),
    ]
    outputs.extend({
        "path":t.transport_png_path,"sha256":t.transport_png_sha256,
        "authority_class":"RUNTIME_SOURCE_TEXTURE_TRANSPORT","schema":"image/png",
    } for t in textures)
    return {
        "status":"PASS","outputs":outputs,
        "diagnostics":{
            "projection_hash":projection.projection_hash,"asset_count":len(projection.assets),
            "clip_count":len(projection.clips),"view_count":len(projection.views),
            "canonical_geometry_mutated":False,
            "canonical_motion_mutated_by_presentation":False,
            "motion_presentation_hash":motion_presentation.motion_presentation_hash,
        },
    }
