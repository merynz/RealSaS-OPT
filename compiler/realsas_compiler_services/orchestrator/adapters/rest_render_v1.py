from __future__ import annotations

"""Stage-31 deterministic rest-render adapter."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    qualified_appearance_set_from_dict,
    qualified_camera_set_from_dict,
    qualified_composition_set_from_dict,
    qualified_mesh_from_dict,
    qualified_observation_set_from_dict,
    qualified_presentation_graph_from_dict,
)
from compiler.realsas_compiler_core.rest_render_v1 import (
    render_rest_views,
    rest_render_set_hash,
    rgba_sha256,
    validate_rest_render_set,
)
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _resolved_path,
    _sha256,
    _stage_output_payload,
    _write_ir,
)
from compiler.realsas_compiler_core.types import QualificationError


def _load_source_rasters(ctx,observation_set):
    cfg=dict(ctx["run_manifest"].get("observation") or {})
    rows=tuple(cfg.get("source_rasters") or ())
    if len(rows)!=8 or {int(row["view_index"]) for row in rows}!=set(range(8)):
        raise QualificationError("REST_RENDER_SOURCE_RASTER_MATRIX_INCOMPLETE")
    authority={int(row.view_index):row for row in observation_set.views}
    out={}
    for row in rows:
        view=int(row["view_index"])
        ref=dict(row.get("image") or {})
        path=_resolved_path(str(ref.get("path") or ""))
        expected=str(ref.get("sha256") or "")
        if not path.is_file() or len(expected)!=64 or _sha256(path)!=expected:
            raise QualificationError("REST_RENDER_SOURCE_RASTER_REF_INVALID")
        if expected!=authority[view].source_raster_sha256:
            raise QualificationError("REST_RENDER_SOURCE_RASTER_AUTHORITY_DRIFT")
        with Image.open(path) as image:
            rgba=np.asarray(image.convert("RGBA"),dtype=np.uint8)
        if rgba.shape!=(int(authority[view].height),int(authority[view].width),4):
            raise QualificationError("REST_RENDER_SOURCE_RASTER_DIMENSION_DRIFT")
        out[view]=np.ascontiguousarray(rgba)
    return out


def qualify_rest_render_stage(ctx:dict)->dict:
    camera_set=qualified_camera_set_from_dict(
        _stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    observation_set=qualified_observation_set_from_dict(
        _stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
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
    graph=qualified_presentation_graph_from_dict(
        _stage_output_payload(ctx,"30_QUALIFIED_PRESENTATION_GRAPH","RealSaS.QualifiedPresentationGraphIR.v1")
    )
    source_rgba=_load_source_rasters(ctx,observation_set)
    render_set,images=render_rest_views(
        mesh=mesh,presentation_graph=graph,appearance_set=appearance,
        composition_set=composition,camera_set=camera_set,
        observation_set=observation_set,source_rgba_by_view=source_rgba,
    )
    root=ctx["run_root"]/"artifacts"/"31_REST_RENDER_8VIEW"
    root.mkdir(parents=True,exist_ok=True)
    png_outputs=[]
    rows=[]
    for view in range(8):
        path=root/f"rest_V{view}.png"
        Image.fromarray(images[view],"RGBA").save(
            path,format="PNG",compress_level=1,optimize=False
        )
        with Image.open(path) as image:
            replay=np.asarray(image.convert("RGBA"),dtype=np.uint8)
        if rgba_sha256(replay)!=render_set.views[view].rendered_rgba_sha256:
            raise QualificationError("REST_RENDER_PNG_REPLAY_RGBA_DRIFT")
        png_sha=_sha256(path)
        rows.append(replace(
            render_set.views[view],
            metadata={
                **dict(render_set.views[view].metadata or {}),
                "png_sha256":png_sha,
                "png_filename":path.name,
                "png_is_evidence_transport_only":True,
            },
        ))
        png_outputs.append({
            "path":str(path),
            "sha256":png_sha,
            "authority_class":"REST_RENDER_EVIDENCE_PNG",
            "schema":"image/png",
        })
    render_set=replace(
        render_set,
        views=tuple(rows),
        render_set_hash="",
        metadata={
            **dict(render_set.metadata or {}),
            "png_transport":"PIL_RGBA_PNG_COMPRESS_LEVEL_1",
            "png_is_authority":False,
        },
    )
    render_set=replace(render_set,render_set_hash=rest_render_set_hash(render_set))
    validate_rest_render_set(
        render_set,mesh=mesh,presentation_graph=graph,appearance_set=appearance,
        composition_set=composition,camera_set=camera_set,observation_set=observation_set,
    )
    outputs=[
        _write_ir(root/"rest_render_set.json",render_set,authority_class="QUALIFIED_REST_RENDER_SET"),
        *png_outputs,
    ]
    return {
        "status":"PASS",
        "outputs":outputs,
        "diagnostics":{
            "render_set_hash":render_set.render_set_hash,
            "view_count":len(render_set.views),
            "visible_pixel_counts":[row.visible_pixel_count for row in render_set.views],
        },
    }
