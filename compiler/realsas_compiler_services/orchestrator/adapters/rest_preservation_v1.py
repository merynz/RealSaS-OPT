from __future__ import annotations

"""Stage-32 rest/source preservation adapter.

The stage replays exact source bytes and stage-31 evidence bytes, measures with the
frozen metric contract, re-derives the product policy from sealed subject-free
calibration, and admits only if every one of eight views passes every hard rule.
"""

from pathlib import Path
import json

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    mesh_policy_from_dict,
    qualified_observation_set_from_dict,
    rest_render_set_from_dict,
)
from compiler.realsas_compiler_core.rest_preservation_v1 import (
    measure_rest_source_preservation,
    rgba_sha256,
)
from compiler.realsas_compiler_core.rest_preservation_policy_v1 import (
    build_rest_source_preservation_policy,
    qualify_rest_source_preservation,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _load_file_ref,
    _resolved_path,
    _sha256,
    _stage_output_payload,
    _write_ir,
)


def _manifest_binary_rows(ctx, observation_set, *, key:str, ref_key:str, expected_attr:str):
    cfg=dict(ctx["run_manifest"].get("observation") or {})
    rows=tuple(cfg.get(key) or ())
    if len(rows)!=8 or {int(row["view_index"]) for row in rows}!=set(range(8)):
        raise QualificationError(f"REST_PRESERVATION_{key.upper()}_MATRIX_INCOMPLETE")
    authority={int(row.view_index):row for row in observation_set.views}
    result={}
    for row in rows:
        view=int(row["view_index"])
        ref=dict(row.get(ref_key) or {})
        path=_load_file_ref(ref,json_required=False)
        expected=getattr(authority[view],expected_attr)
        if _sha256(path)!=expected:
            raise QualificationError(f"REST_PRESERVATION_{key.upper()}_AUTHORITY_DRIFT")
        result[view]=path
    return result


def _load_source_rgba(ctx,observation_set):
    paths=_manifest_binary_rows(
        ctx,observation_set,key="source_rasters",ref_key="image",
        expected_attr="source_raster_sha256",
    )
    out={}
    authority={int(row.view_index):row for row in observation_set.views}
    for view,path in paths.items():
        with Image.open(path) as image:
            rgba=np.asarray(image.convert("RGBA"),dtype=np.uint8)
        expected=(int(authority[view].height),int(authority[view].width),4)
        if rgba.shape!=expected:
            raise QualificationError("REST_PRESERVATION_SOURCE_RASTER_DIMENSION_DRIFT")
        out[view]=np.ascontiguousarray(rgba)
    return out


def _load_source_foreground(ctx,observation_set):
    paths=_manifest_binary_rows(
        ctx,observation_set,key="source_foreground_masks",ref_key="mask",
        expected_attr="foreground_mask_sha256",
    )
    return {view:path.read_bytes() for view,path in paths.items()}


def _load_rendered_rgba(ctx,rest_render_set):
    stage=next(
        (row for row in ctx["ledger"]["stages"] if row["id"]=="31_REST_RENDER_8VIEW"),
        None,
    )
    if stage is None or stage.get("status") not in {"PASS","CACHE_HIT"}:
        raise QualificationError("REST_PRESERVATION_STAGE31_NOT_PASS")
    image_outputs=tuple(out for out in stage.get("outputs",()) if out.get("schema")=="image/png")
    if len(image_outputs)!=8:
        raise QualificationError("REST_PRESERVATION_STAGE31_PNG_CARDINALITY")
    by_name={Path(str(out["path"])).name:out for out in image_outputs}
    result={}
    for row in rest_render_set.views:
        name=str((row.metadata or {}).get("png_filename") or "")
        out=by_name.get(name)
        if out is None:
            raise QualificationError("REST_PRESERVATION_STAGE31_PNG_MISSING")
        path=_resolved_path(str(out["path"]))
        if not path.is_file() or _sha256(path)!=str(out.get("sha256") or ""):
            raise QualificationError("REST_PRESERVATION_STAGE31_PNG_SHA_DRIFT")
        if str((row.metadata or {}).get("png_sha256") or "")!=str(out["sha256"]):
            raise QualificationError("REST_PRESERVATION_RENDER_METADATA_PNG_DRIFT")
        with Image.open(path) as image:
            rgba=np.asarray(image.convert("RGBA"),dtype=np.uint8)
        if rgba.shape!=(int(row.height),int(row.width),4):
            raise QualificationError("REST_PRESERVATION_RENDER_PNG_DIMENSION_DRIFT")
        if rgba_sha256(rgba)!=row.rendered_rgba_sha256:
            raise QualificationError("REST_PRESERVATION_RENDER_RGBA_AUTHORITY_DRIFT")
        result[int(row.view_index)]=np.ascontiguousarray(rgba)
    if set(result)!=set(range(8)):
        raise QualificationError("REST_PRESERVATION_RENDERED_VIEW_SET_INCOMPLETE")
    return result


def qualify_rest_source_preservation_stage(ctx:dict)->dict:
    appearance_cfg=dict(ctx["run_manifest"].get("appearance") or {})
    policy_ref=dict(appearance_cfg.get("rest_preservation_policy") or {})
    calibration_ref=dict(appearance_cfg.get("rest_preservation_calibration") or {})
    policy_document=_load_file_ref(
        policy_ref,expected_schema="RealSaS.RestSourcePreservationProductPolicy.v2"
    )
    calibration_result=_load_file_ref(
        calibration_ref,expected_schema="RealSaS.RestSourcePreservationCalibrationResult.v1"
    )

    observation_set=qualified_observation_set_from_dict(
        _stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
    )
    mesh_policy=mesh_policy_from_dict(
        _stage_output_payload(ctx,"26_MESH_CANDIDATE_BUILD","RealSaS.MeshQualificationPolicyIR.v1")
    )
    rest_render_set=rest_render_set_from_dict(
        _stage_output_payload(ctx,"31_REST_RENDER_8VIEW","RealSaS.RestRenderSetIR.v1")
    )
    source_rgba=_load_source_rgba(ctx,observation_set)
    source_foreground=_load_source_foreground(ctx,observation_set)
    rendered_rgba=_load_rendered_rgba(ctx,rest_render_set)

    measurements=measure_rest_source_preservation(
        rest_render_set=rest_render_set,
        observation_set=observation_set,
        source_rgba_by_view=source_rgba,
        rendered_rgba_by_view=rendered_rgba,
        source_foreground_by_view=source_foreground,
    )
    policy=build_rest_source_preservation_policy(
        policy_document=policy_document,
        calibration_result=calibration_result,
        mesh_policy=mesh_policy,
    )
    qualified=qualify_rest_source_preservation(
        measurements,
        policy=policy,
        rest_render_set=rest_render_set,
        observation_set=observation_set,
    )

    root=ctx["run_root"]/"artifacts"/"32_REST_SOURCE_PRESERVATION_GATE"
    outputs=[
        _write_ir(root/"rest_preservation_measurements.json",measurements,authority_class="REST_PRESERVATION_MEASUREMENTS"),
        _write_ir(root/"rest_preservation_policy.json",policy,authority_class="REST_PRESERVATION_POLICY"),
        _write_ir(root/"qualified_rest_source_preservation.json",qualified,authority_class="QUALIFIED_REST_SOURCE_PRESERVATION"),
    ]
    return {
        "status":"PASS",
        "outputs":outputs,
        "diagnostics":{
            "measurement_set_hash":measurements.measurement_set_hash,
            "policy_lineage_hash":policy.policy_lineage_hash,
            "preservation_lineage_hash":qualified.preservation_lineage_hash,
            "view_count":len(qualified.view_decisions),
            "all_views_passed":all(row.status=="PASS" for row in qualified.view_decisions),
            "motion_authorization_precondition_satisfied":bool(
                qualified.qualification_report.get("motion_authorization_precondition_satisfied",False)
            ),
        },
    }
