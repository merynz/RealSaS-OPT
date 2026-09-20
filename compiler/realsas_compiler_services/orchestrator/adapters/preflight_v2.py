from __future__ import annotations

"""V2 stages 03-08: source admission, exact observation authority and normalization."""

from dataclasses import replace
import math

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.camera_authority_v1 import camera_projection_binding_hash
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.observation_authority_v1 import (
    QualifiedObservationViewIR, build_qualified_observation_set,
)
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    FullSubjectAdmissionIR, NormalizationDomainIR, ObservationRenderSetIR,
    ObservationRenderViewIR, SourceMechanicalAuditIR, SourceMechanicalAuditViewIR,
    full_subject_admission_from_dict, full_subject_admission_hash, normalization_domain_hash,
    observation_render_set_from_dict, observation_render_set_hash,
    source_mechanical_audit_from_dict, source_mechanical_audit_hash,
)
from compiler.realsas_compiler_core.product_artifact_codec_v1 import qualified_camera_set_from_dict
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    load_file_ref, stage_output_payload, write_ir,
)


def _observation_rows(ctx:dict,key:str,ref_key:str):
    cfg=dict(ctx["run_manifest"].get("observation") or {})
    rows=tuple(cfg.get(key) or ())
    if len(rows)!=8 or {int(row.get("view_index",-1)) for row in rows}!=set(range(8)):
        raise QualificationError(f"PREFLIGHT_{key.upper()}_REQUIRES_EXACT_8_VIEWS")
    out={}
    for row in rows:
        vi=int(row["view_index"])
        path=load_file_ref(dict(row.get(ref_key) or {}),json_required=False)
        out[vi]=(dict(row),path)
    return out


def _foreground_mask(path,width:int,height:int)->bytes:
    raw=path.read_bytes()
    if len(raw)!=int(width)*int(height) or any(x not in (0,1) for x in raw):
        raise QualificationError("PREFLIGHT_FOREGROUND_MASK_MUST_BE_RAW_BINARY_WxH")
    if not any(raw):
        raise QualificationError("PREFLIGHT_FOREGROUND_MASK_EMPTY")
    return raw


def _margin(mask:bytes,width:int,height:int)->int:
    idx=np.frombuffer(mask,dtype=np.uint8).reshape(int(height),int(width))
    ys,xs=np.nonzero(idx)
    if not len(xs):
        raise QualificationError("PREFLIGHT_FOREGROUND_MASK_EMPTY")
    return int(min(xs.min(),ys.min(),int(width)-1-xs.max(),int(height)-1-ys.max()))


def audit_source_mechanics_stage(ctx:dict)->dict:
    source=stage_output_payload(ctx,"01_SOURCE_BYTES_SEALED","RealSaS.SourceByteSeal.v1")
    source_hashes={str(row["sha256"]) for row in source.get("files") or ()}
    cfg=dict(ctx["run_manifest"].get("source_audit") or {})
    required_margin=int(cfg.get("minimum_border_margin_px",0))
    if required_margin<1:
        return {"status":"BLOCKED","blockers":["SOURCE_AUDIT_MINIMUM_BORDER_MARGIN_MUST_BE_AT_LEAST_1"],"diagnostics":{}}

    rasters=_observation_rows(ctx,"source_rasters","image")
    masks=_observation_rows(ctx,"source_foreground_masks","mask")
    views=[]
    for vi in range(8):
        _rrow,raster_path=rasters[vi]
        _mrow,mask_path=masks[vi]
        raster_sha=str(_rrow["image"]["sha256"])
        mask_sha=str(_mrow["mask"]["sha256"])
        if raster_sha not in source_hashes:
            raise QualificationError("SOURCE_AUDIT_RASTER_NOT_BOUND_BY_STAGE1_SOURCE_BYTES")
        with Image.open(raster_path) as image:
            width,height=map(int,image.size)
            image.verify()
        mask=_foreground_mask(mask_path,width,height)
        margin=_margin(mask,width,height)
        views.append(SourceMechanicalAuditViewIR(
            vi,raster_sha,mask_sha,width,height,int(sum(mask)),margin,margin<=0,
            metadata={"foreground_authority":"EXACT_STAGE07_RAW_BINARY_MASK","source_file_role_bound":True},
        ))
    passed=all(v.border_margin_px>=required_margin and not v.touches_border for v in views)
    audit=SourceMechanicalAuditIR(
        content_sha256(source),tuple(views),required_margin,
        {
            "status":"PASS" if passed else "FAIL",
            "exact_view_count":len(views),
            "all_foreground_nonempty":all(v.foreground_pixel_count>0 for v in views),
            "all_views_meet_nonzero_safety_margin":passed,
            "teacher_truth_used":False,
        },"",
        metadata={"controlled_full_subject_audit":True,"unknown_out_of_frame_is_not_negative":True},
    )
    audit=replace(audit,audit_hash=source_mechanical_audit_hash(audit))
    if not passed:
        return {"status":"FAIL","blockers":["SOURCE_SUBJECT_TOUCHES_OR_VIOLATES_BORDER_MARGIN"],"diagnostics":{
            "border_margins_px":[v.border_margin_px for v in views],
            "required_margin_px":required_margin,
        }}
    root=ctx["run_root"]/"artifacts"/ctx["stage"]["id"]
    return {"status":"PASS","outputs":[write_ir(root/"source_mechanical_audit.json",audit,authority_class="SOURCE_MECHANICAL_AUDIT")],
        "diagnostics":{"audit_hash":audit.audit_hash,"border_margins_px":[v.border_margin_px for v in views]}}


def admit_full_subject_stage(ctx:dict)->dict:
    audit=source_mechanical_audit_from_dict(
        stage_output_payload(ctx,"03_SOURCE_MECHANICAL_AUDIT","RealSaS.SourceMechanicalAuditIR.v1")
    )
    cfg=dict(ctx["run_manifest"].get("admission") or {})
    mode=str(cfg.get("mode") or "")
    if mode!="CONTROLLED_FULL_SUBJECT_8VIEW_V1":
        return {"status":"BLOCKED","blockers":["FULL_SUBJECT_ADMISSION_MODE_REQUIRED"],"diagnostics":{"mode":mode}}
    if audit.audit_report.get("status")!="PASS" or len(audit.views)!=8:
        raise QualificationError("FULL_SUBJECT_ADMISSION_REQUIRES_STAGE3_PASS")
    admitted=tuple(v.view_index for v in audit.views if v.border_margin_px>=audit.minimum_required_border_margin_px)
    if admitted!=tuple(range(8)):
        raise QualificationError("FULL_SUBJECT_ADMISSION_VIEW_SET_INCOMPLETE")
    value=FullSubjectAdmissionIR(
        audit.audit_hash,admitted,mode,
        {
            "status":"PASS",
            "full_subject_present_in_all_views":True,
            "nonzero_safety_margin_all_views":True,
            "out_of_frame_policy":"UNKNOWN_NEVER_NEGATIVE",
            "view_count":8,
        },"",
        metadata={"subject_specific_logic":False},
    )
    value=replace(value,admission_hash=full_subject_admission_hash(value))
    root=ctx["run_root"]/"artifacts"/ctx["stage"]["id"]
    return {"status":"PASS","outputs":[write_ir(root/"full_subject_admission.json",value,authority_class="FULL_SUBJECT_ADMISSION")],
        "diagnostics":{"admission_hash":value.admission_hash,"view_count":8}}


def materialize_observation_render_stage(ctx:dict)->dict:
    admission=full_subject_admission_from_dict(
        stage_output_payload(ctx,"04_FULL_SUBJECT_ADMISSION","RealSaS.FullSubjectAdmissionIR.v1")
    )
    cameras=qualified_camera_set_from_dict(
        stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    rasters=_observation_rows(ctx,"source_rasters","image")
    views=[]
    for vi in range(8):
        row,path=rasters[vi]
        with Image.open(path) as image:
            width,height=map(int,image.size)
        camera=next(c for c in cameras.cameras if int(c.view_index)==vi)
        if width!=int(camera.resolution) or height!=int(camera.resolution):
            raise QualificationError("OBSERVATION_RENDER_SOURCE_CAMERA_RESOLUTION_DRIFT")
        views.append(ObservationRenderViewIR(
            vi,str(path),str(row["image"]["sha256"]),width,height,
            metadata={"materialization":"EXACT_SOURCE_OBSERVATION_BYTES","synthetic_rendering":False},
        ))
    value=ObservationRenderSetIR(admission.admission_hash,cameras.camera_set_hash,tuple(views),"",
        metadata={"exact_8_view_observation_materialization":True})
    value=replace(value,render_set_hash=observation_render_set_hash(value))
    root=ctx["run_root"]/"artifacts"/ctx["stage"]["id"]
    return {"status":"PASS","outputs":[write_ir(root/"observation_render_set.json",value,authority_class="EXACT_OBSERVATION_RENDER_SET")],
        "diagnostics":{"render_set_hash":value.render_set_hash,"view_count":8}}


def qualify_observation_contract_stage(ctx:dict)->dict:
    audit=source_mechanical_audit_from_dict(
        stage_output_payload(ctx,"03_SOURCE_MECHANICAL_AUDIT","RealSaS.SourceMechanicalAuditIR.v1")
    )
    admission=full_subject_admission_from_dict(
        stage_output_payload(ctx,"04_FULL_SUBJECT_ADMISSION","RealSaS.FullSubjectAdmissionIR.v1")
    )
    cameras=qualified_camera_set_from_dict(
        stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    renders=observation_render_set_from_dict(
        stage_output_payload(ctx,"06_OBSERVATION_RENDER_8VIEW","RealSaS.ObservationRenderSetIR.v1")
    )
    masks=_observation_rows(ctx,"source_foreground_masks","mask")
    cam_by={int(c.view_index):c for c in cameras.cameras}
    render_by={int(r.view_index):r for r in renders.views}
    audit_by={int(r.view_index):r for r in audit.views}
    rows=[]
    for vi in range(8):
        render=render_by[vi]; av=audit_by[vi]; camera=cam_by[vi]
        _mrow,mask_path=masks[vi]
        mask_sha=str(_mrow["mask"]["sha256"])
        if mask_sha!=av.foreground_mask_sha256:
            raise QualificationError("OBSERVATION_CONTRACT_FOREGROUND_AUTHORITY_DRIFT")
        raw=_foreground_mask(mask_path,render.width,render.height)
        if len(raw)!=render.width*render.height:
            raise QualificationError("OBSERVATION_CONTRACT_MASK_DIMENSION_DRIFT")
        cam_hash=camera_projection_binding_hash(camera)
        source_observation_hash=content_sha256({
            "schema":"RealSaS.SourceObservationAuthority.v1",
            "view_index":vi,
            "source_raster_sha256":render.source_raster_sha256,
            "foreground_mask_sha256":mask_sha,
            "camera_binding_hash":cam_hash,
            "admission_hash":admission.admission_hash,
        })
        rows.append(QualifiedObservationViewIR(
            vi,render.width,render.height,source_observation_hash,render.source_raster_sha256,
            mask_sha,cam_hash,"PASS",(audit.audit_hash,admission.admission_hash,renders.render_set_hash),
            metadata={"evidence_semantics":"POSITIVE_FOREGROUND__NEGATIVE_INFRAME_BACKGROUND__UNKNOWN_OUT_OF_FRAME"},
        ))
    value=build_qualified_observation_set(tuple(rows),metadata={
        "full_subject_admission_hash":admission.admission_hash,
        "camera_set_hash":cameras.camera_set_hash,
        "unknown_cannot_veto_positive":True,
    })
    root=ctx["run_root"]/"artifacts"/ctx["stage"]["id"]
    return {"status":"PASS","outputs":[write_ir(root/"qualified_observation_set.json",value,authority_class="QUALIFIED_OBSERVATION_SET")],
        "diagnostics":{"observation_set_hash":value.observation_set_hash,"view_count":8}}


def qualify_normalization_domain_stage(ctx:dict)->dict:
    observation_payload=stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
    cameras=qualified_camera_set_from_dict(
        stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    cfg=dict(ctx["run_manifest"].get("normalization") or {})
    center=tuple(map(float,cfg.get("center_xyz") or ()))
    half=float(cfg.get("half_extent",0.0))
    frame=str(cfg.get("coordinate_frame") or "")
    if len(center)!=3 or not all(math.isfinite(x) for x in center) or not math.isfinite(half) or half<=0:
        return {"status":"BLOCKED","blockers":["NORMALIZATION_EXACT_CENTER_HALF_EXTENT_REQUIRED"],"diagnostics":{}}
    if frame!="REALSAS_OBJECT_FRAME":
        return {"status":"BLOCKED","blockers":["NORMALIZATION_COORDINATE_FRAME_INVALID"],"diagnostics":{"coordinate_frame":frame}}
    value=NormalizationDomainIR(
        str(observation_payload["observation_set_hash"]),cameras.camera_set_hash,center,half,
        (-1.0,-1.0,-1.0),(1.0,1.0,1.0),frame,"",
        metadata={"normalization_policy":"EXACT_MANIFEST_OBJECT_CUBE_V1","camera_refit_performed":False},
    )
    value=replace(value,normalization_hash=normalization_domain_hash(value))
    root=ctx["run_root"]/"artifacts"/ctx["stage"]["id"]
    return {"status":"PASS","outputs":[write_ir(root/"normalization_domain.json",value,authority_class="QUALIFIED_NORMALIZATION_DOMAIN")],
        "diagnostics":{"normalization_hash":value.normalization_hash,"half_extent":half,"center_xyz":center}}
