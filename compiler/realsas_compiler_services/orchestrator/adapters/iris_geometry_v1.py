from __future__ import annotations

"""Mainline stages 09-15: IRIS fit authority, signed zero-surface and GSA qualification."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import tensorize_rigging_surface_v1

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics, rasterize_triangles_half_integer_top_left,
    source_connected_component_recall_metrics,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import project_points_xyz_v3
from compiler.realsas_compiler_core.rest_preservation_v1 import silhouette_distance_metrics
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    RestReprojectionGeometryGateIR, RestReprojectionGeometryViewIR,
    RiggingSurfaceQualificationIR, SignedZeroSurfaceSealIR,
    model_checkpoint_seal_from_dict, model_fit_execution_from_dict,
    model_fit_preregistration_from_dict, normalization_domain_from_dict,
    rest_reprojection_geometry_gate_hash, rigging_surface_qualification_hash,
    signed_zero_surface_from_dict, signed_zero_surface_hash,
)
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
    qualified_camera_set_from_dict, qualified_observation_set_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.substrate.adequacy_v1 import (
    select_adequate_rigging_surface_v1, substrate_adequacy_report_hash_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.model_execution_common_v1 import (
    build_fit_execution, build_fit_preregistration, seal_model_checkpoint,
)
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _load_file_ref, _resolved_path, _sha256, _stage_output_payload, _write_ir,
)


def _array_hash(array:np.ndarray)->str:
    a=np.ascontiguousarray(array)
    return content_sha256({
        "dtype":str(a.dtype),
        "shape":tuple(map(int,a.shape)),
        "bytes_sha256":sha256(a.tobytes(order="C")).hexdigest(),
    })


def _load_zero_arrays(seal:SignedZeroSurfaceSealIR):
    path=_resolved_path(seal.npz_path)
    if not path.is_file() or _sha256(path)!=seal.npz_sha256:
        raise QualificationError("ZERO_SURFACE_NPZ_BYTES_DRIFT")
    with np.load(path,allow_pickle=False) as data:
        required={"vertices_normalized","faces","implicit_normals"}
        if not required.issubset(set(data.files)):
            raise QualificationError("ZERO_SURFACE_NPZ_ARRAYS_MISSING")
        vertices=np.asarray(data["vertices_normalized"]).copy()
        faces=np.asarray(data["faces"]).copy()
        normals=np.asarray(data["implicit_normals"]).copy()
    if _array_hash(vertices)!=seal.vertices_sha256 or _array_hash(faces)!=seal.faces_sha256 or _array_hash(normals)!=seal.implicit_normals_sha256:
        raise QualificationError("ZERO_SURFACE_ARRAY_HASH_DRIFT")
    return vertices,faces,normals


def preregister_iris_fit_stage(ctx:dict)->dict:
    obs=qualified_observation_set_from_dict(
        _stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
    )
    norm=normalization_domain_from_dict(
        _stage_output_payload(ctx,"08_NORMALIZATION_DOMAIN_QUALIFIED","RealSaS.NormalizationDomainIR.v1")
    )
    prereg,out=build_fit_preregistration(
        ctx,lane="IRIS",section_key="iris_fit",
        upstream_bindings={"observation_set":obs.observation_set_hash,"normalization":norm.normalization_hash},
        stage_id="09_IRIS_FIT_PREREGISTERED",
    )
    return {"status":"PASS","outputs":[out],"diagnostics":{
        "preregistration_hash":prereg.preregistration_hash,"architecture_id":prereg.architecture_id,
        "executor_kind":prereg.executor_kind,
    }}


def execute_iris_fit_stage(ctx:dict)->dict:
    prereg=model_fit_preregistration_from_dict(
        _stage_output_payload(ctx,"09_IRIS_FIT_PREREGISTERED","RealSaS.ModelFitPreregistrationIR.v1")
    )
    execution,out=build_fit_execution(
        ctx,lane="IRIS",section_key="iris_fit",prereg=prereg,
        stage_id="10_IRIS_FIT",proposal_required=False,
    )
    return {"status":"PASS","outputs":[out],"diagnostics":{
        "execution_hash":execution.execution_hash,"checkpoint_sha256":execution.checkpoint_sha256,
        "result_sha256":execution.result_sha256,"product_authority_minted":False,
    }}


def seal_iris_checkpoint_stage(ctx:dict)->dict:
    execution=model_fit_execution_from_dict(
        _stage_output_payload(ctx,"10_IRIS_FIT","RealSaS.ModelFitExecutionIR.v1")
    )
    value,out=seal_model_checkpoint(
        ctx,execution=execution,qualified_output_binding_hash=None,stage_id="11_IRIS_CHECKPOINT_SEALED"
    )
    return {"status":"PASS","outputs":[out],"diagnostics":{
        "checkpoint_seal_hash":value.checkpoint_seal_hash,"checkpoint_sha256":value.checkpoint_sha256,
    }}


def decode_zero_surface_stage(ctx:dict)->dict:
    checkpoint=model_checkpoint_seal_from_dict(
        _stage_output_payload(ctx,"11_IRIS_CHECKPOINT_SEALED","RealSaS.ModelCheckpointSealIR.v1")
    )
    observation=qualified_observation_set_from_dict(
        _stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
    )
    normalization=normalization_domain_from_dict(
        _stage_output_payload(ctx,"08_NORMALIZATION_DOMAIN_QUALIFIED","RealSaS.NormalizationDomainIR.v1")
    )
    cfg=dict(ctx["run_manifest"].get("geometry_decode") or {})
    npz_ref=dict(cfg.get("zero_surface_npz") or {})
    metadata_ref=dict(cfg.get("zero_surface_metadata") or {})
    npz_path=_load_file_ref(npz_ref,json_required=False)
    metadata=_load_file_ref(metadata_ref,expected_schema="RealSaS.SignedZeroSurfaceArtifact.v1")
    metadata_path=_resolved_path(str(metadata_ref.get("path") or ""))
    npz_sha=_sha256(npz_path); metadata_sha=_sha256(metadata_path)
    expected={
        "checkpoint_seal_binding_hash":checkpoint.checkpoint_seal_hash,
        "checkpoint_sha256":checkpoint.checkpoint_sha256,
        "observation_set_binding_hash":observation.observation_set_hash,
        "normalization_binding_hash":normalization.normalization_hash,
        "npz_sha256":npz_sha,
    }
    for key,wanted in expected.items():
        if str(metadata.get(key) or "")!=wanted:
            raise QualificationError(f"ZERO_SURFACE_METADATA_BINDING_DRIFT:{key}")
    if bool(metadata.get("teacher_truth_used",False)):
        raise QualificationError("ZERO_SURFACE_TEACHER_TRUTH_FORBIDDEN")
    with np.load(npz_path,allow_pickle=False) as data:
        required={"vertices_normalized","faces","implicit_normals"}
        if not required.issubset(set(data.files)):
            raise QualificationError("ZERO_SURFACE_NPZ_ARRAYS_MISSING")
        vertices=np.asarray(data["vertices_normalized"]).copy()
        faces=np.asarray(data["faces"]).copy()
        normals=np.asarray(data["implicit_normals"]).copy()
    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<4 or not np.isfinite(vertices).all():
        raise QualificationError("ZERO_SURFACE_VERTICES_INVALID")
    if faces.ndim!=2 or faces.shape[1]!=3 or len(faces)==0 or not np.issubdtype(faces.dtype,np.integer):
        raise QualificationError("ZERO_SURFACE_FACES_INVALID")
    if np.any(faces<0) or np.any(faces>=len(vertices)):
        raise QualificationError("ZERO_SURFACE_FACE_INDEX_INVALID")
    if normals.shape!=vertices.shape or not np.isfinite(normals).all() or np.any(np.linalg.norm(normals,axis=1)<=1e-12):
        raise QualificationError("ZERO_SURFACE_NORMALS_INVALID")
    value=SignedZeroSurfaceSealIR(
        checkpoint.checkpoint_seal_hash,observation.observation_set_hash,normalization.normalization_hash,
        str(npz_path),npz_sha,str(metadata_path),metadata_sha,int(len(vertices)),int(len(faces)),
        _array_hash(vertices),_array_hash(faces),_array_hash(normals),"",
        metadata={
            "decoder_id":str(metadata.get("decoder_id") or "IRIS_SIGNED_ZERO_SURFACE_EXTERNAL_DECODE_V1"),
            "teacher_truth_used":False,
            "normalized_coordinate_domain":"[-1,1]^3",
        },
    )
    value=replace(value,zero_surface_hash=signed_zero_surface_hash(value))
    root=ctx["run_root"]/"artifacts"/"12_ZERO_SURFACE_DECODED"
    return {"status":"PASS","outputs":[_write_ir(root/"signed_zero_surface_seal.json",value,authority_class="IRIS_SIGNED_ZERO_SURFACE_SEAL")],
        "diagnostics":{"zero_surface_hash":value.zero_surface_hash,"vertex_count":value.vertex_count,"face_count":value.face_count}}


def _source_foreground(ctx,observation):
    cfg=dict(ctx["run_manifest"].get("observation") or {})
    rows=tuple(cfg.get("source_foreground_masks") or ())
    if len(rows)!=8 or {int(r["view_index"]) for r in rows}!=set(range(8)):
        raise QualificationError("GEOMETRY_GATE_FOREGROUND_MATRIX_INCOMPLETE")
    authority={int(v.view_index):v for v in observation.views}
    out={}
    for row in rows:
        vi=int(row["view_index"]); ref=dict(row.get("mask") or {})
        path=_load_file_ref(ref,json_required=False)
        if _sha256(path)!=authority[vi].foreground_mask_sha256:
            raise QualificationError("GEOMETRY_GATE_FOREGROUND_AUTHORITY_DRIFT")
        raw=path.read_bytes()
        if len(raw)!=int(authority[vi].width)*int(authority[vi].height) or any(x not in (0,1) for x in raw):
            raise QualificationError("GEOMETRY_GATE_FOREGROUND_MASK_INVALID")
        out[vi]=raw
    return out


def _geometry_gate_policy_v2(cfg:dict)->dict|None:
    keys=(
        "min_recall","min_precision",
        "max_largest_coherent_hole_fraction","max_interior_uncovered_fraction",
        "min_component_recall","component_min_foreground_fraction",
        "max_silhouette_edge_p95_px",
    )
    if any(k not in cfg for k in keys):
        return None
    policy={k:float(cfg[k]) for k in keys}
    if not (
        0<=policy["min_recall"]<=1 and 0<=policy["min_precision"]<=1 and
        0<=policy["max_largest_coherent_hole_fraction"]<=1 and
        0<=policy["max_interior_uncovered_fraction"]<=1 and
        0<=policy["min_component_recall"]<=1 and
        0<=policy["component_min_foreground_fraction"]<=1 and
        math.isfinite(policy["max_silhouette_edge_p95_px"]) and
        policy["max_silhouette_edge_p95_px"]>=0
    ):
        raise QualificationError("GEOMETRY_GATE_THRESHOLD_RANGE_INVALID")
    return policy


def qualify_rest_reprojection_geometry_stage(ctx:dict)->dict:
    zero=signed_zero_surface_from_dict(
        _stage_output_payload(ctx,"12_ZERO_SURFACE_DECODED","RealSaS.SignedZeroSurfaceSealIR.v1")
    )
    normalization=normalization_domain_from_dict(
        _stage_output_payload(ctx,"08_NORMALIZATION_DOMAIN_QUALIFIED","RealSaS.NormalizationDomainIR.v1")
    )
    cameras=qualified_camera_set_from_dict(
        _stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    observation=qualified_observation_set_from_dict(
        _stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
    )
    vertices,faces,_normals=_load_zero_arrays(zero)
    world=np.asarray(normalization.center_xyz,dtype=np.float64)[None,:]+np.asarray(vertices,dtype=np.float64)*float(normalization.half_extent)
    source=_source_foreground(ctx,observation)
    cfg=dict(ctx["run_manifest"].get("geometry_gate") or {})
    policy=_geometry_gate_policy_v2(cfg)
    if policy is None:
        required=(
            "min_recall","min_precision",
            "max_largest_coherent_hole_fraction","max_interior_uncovered_fraction",
            "min_component_recall","component_min_foreground_fraction",
            "max_silhouette_edge_p95_px",
        )
        return {"status":"BLOCKED","blockers":["GEOMETRY_GATE_EXPLICIT_THRESHOLDS_REQUIRED"],"diagnostics":{"required":list(required)}}

    obs={int(v.view_index):v for v in observation.views}; rows=[]
    for camera in sorted(cameras.cameras,key=lambda c:c.view_index):
        vi=int(camera.view_index); authority=obs[vi]
        projected=project_points_xyz_v3(world,camera)
        if not np.isfinite(projected).all() or np.any(projected[:,2]<=0.0):
            raise QualificationError("GEOMETRY_GATE_ZERO_SURFACE_OUTSIDE_CAMERA_FORWARD_DOMAIN")
        triangles=[]
        for face in np.asarray(faces,dtype=np.int64):
            a,b,c=(projected[int(i)] for i in face)
            triangles.append(((float(a[0]),float(a[1])),(float(b[0]),float(b[1])),(float(c[0]),float(c[1]))))
        predicted=rasterize_triangles_half_integer_top_left(
            triangles,width=int(authority.width),height=int(authority.height)
        )
        metrics=coverage_metrics(source[vi],predicted,width=int(authority.width),height=int(authority.height))
        source_mask=np.frombuffer(source[vi],dtype=np.uint8).reshape(int(authority.height),int(authority.width)).astype(bool)
        predicted_mask=np.frombuffer(predicted,dtype=np.uint8).reshape(int(authority.height),int(authority.width)).astype(bool)
        silhouette_edge_mean_px,silhouette_edge_p95_px,silhouette_edge_max_px=silhouette_distance_metrics(
            source_mask,predicted_mask
        )
        component_metrics=source_connected_component_recall_metrics(
            source[vi],predicted,
            width=int(authority.width),height=int(authority.height),
            minimum_foreground_fraction=policy["component_min_foreground_fraction"],
        )
        if metrics["foreground_pixel_count"]>0 and component_metrics["eligible_component_count"]<=0:
            raise QualificationError("GEOMETRY_GATE_COMPONENT_POLICY_SELECTS_NO_FOREGROUND")
        passed=(
            metrics["recall"]>=policy["min_recall"] and metrics["precision"]>=policy["min_precision"] and
            metrics["largest_coherent_hole_fraction"]<=policy["max_largest_coherent_hole_fraction"] and
            metrics["interior_uncovered_fraction"]<=policy["max_interior_uncovered_fraction"] and
            component_metrics["minimum_eligible_component_recall"]>=policy["min_component_recall"] and
            silhouette_edge_p95_px<=policy["max_silhouette_edge_p95_px"]
        )
        rows.append(RestReprojectionGeometryViewIR(
            vi,float(metrics["recall"]),float(metrics["precision"]),
            float(metrics["largest_coherent_hole_fraction"]),float(metrics["interior_uncovered_fraction"]),
            int(metrics["foreground_pixel_count"]),int(metrics["predicted_pixel_count"]),bool(passed),
            metadata={
                "source_observation_hash":authority.source_observation_hash,
                "source_component_recall":component_metrics,
                "component_recall_passed":bool(
                    component_metrics["minimum_eligible_component_recall"]>=policy["min_component_recall"]
                ),
                "silhouette_edge_mean_px":float(silhouette_edge_mean_px),
                "silhouette_edge_p95_px":float(silhouette_edge_p95_px),
                "silhouette_edge_max_px":float(silhouette_edge_max_px),
                "silhouette_edge_passed":bool(
                    silhouette_edge_p95_px<=policy["max_silhouette_edge_p95_px"]
                ),
            },
        ))
    all_pass=all(row.passed for row in rows)
    value=RestReprojectionGeometryGateIR(
        zero.zero_surface_hash,observation.observation_set_hash,cameras.camera_set_hash,policy,tuple(rows),
        {"status":"PASS" if all_pass else "FAIL","every_view_passed":all_pass,"view_count":8,
         "appearance_authority_used":False,"teacher_truth_used":False},"",
        metadata={
            "raster_fill":"HALF_INTEGER_TOP_LEFT",
            "metric_contract":"STAGE13_GEOMETRY_V2__G5_COVERAGE_PLUS_SYMMETRIC_SILHOUETTE_DISTANCE",
            "silhouette_metric":"SYMMETRIC_NEAREST_BOUNDARY_DISTANCE_PIXELS",
        },
    )
    value=replace(value,geometry_gate_hash=rest_reprojection_geometry_gate_hash(value))
    if not all_pass:
        return {"status":"FAIL","blockers":["REST_REPROJECTION_GEOMETRY_GATE_FAILED"],"diagnostics":{
            "per_view":[r.to_dict() for r in rows],"geometry_gate_hash":value.geometry_gate_hash}}
    root=ctx["run_root"]/"artifacts"/"13_REST_REPROJECTION_GEOMETRY_GATE"
    return {"status":"PASS","outputs":[_write_ir(root/"rest_reprojection_geometry_gate.json",value,authority_class="QUALIFIED_REST_GEOMETRY_GATE")],
        "diagnostics":{"geometry_gate_hash":value.geometry_gate_hash,"all_views_passed":True}}


def build_gsa_stage(ctx:dict)->dict:
    zero=signed_zero_surface_from_dict(
        _stage_output_payload(ctx,"12_ZERO_SURFACE_DECODED","RealSaS.SignedZeroSurfaceSealIR.v1")
    )
    gate_payload=_stage_output_payload(ctx,"13_REST_REPROJECTION_GEOMETRY_GATE","RealSaS.RestReprojectionGeometryGateIR.v1")
    normalization=normalization_domain_from_dict(
        _stage_output_payload(ctx,"08_NORMALIZATION_DOMAIN_QUALIFIED","RealSaS.NormalizationDomainIR.v1")
    )
    cameras=qualified_camera_set_from_dict(
        _stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    checkpoint=model_checkpoint_seal_from_dict(
        _stage_output_payload(ctx,"11_IRIS_CHECKPOINT_SEALED","RealSaS.ModelCheckpointSealIR.v1")
    )
    vertices,faces,normals=_load_zero_arrays(zero)
    cfg=dict(ctx["run_manifest"].get("gsa") or {})
    required=("policy_document","normal_k","visibility_depth_tolerance_norm","adequacy_policy")
    if any(k not in cfg for k in required):
        return {"status":"BLOCKED","blockers":["GSA_EXPLICIT_POLICY_REQUIRED"],"diagnostics":{"required":list(required)}}

    policy_ref=dict(cfg.get("policy_document") or {})
    policy_document=_load_file_ref(
        policy_ref,expected_schema="RealSaS.Stage14SubstrateAdequacyPolicy.v2"
    )
    if not str(policy_document.get("status") or "").startswith("FROZEN_"):
        raise QualificationError("GSA_POLICY_DOCUMENT_NOT_FROZEN")
    document_gsa=dict(policy_document.get("gsa") or {})
    document_adequacy=dict(document_gsa.get("adequacy_policy") or {})

    normal_k=int(cfg["normal_k"]); tolerance=float(cfg["visibility_depth_tolerance_norm"])
    if normal_k<3 or not math.isfinite(tolerance) or tolerance<=0:
        raise QualificationError("GSA_POLICY_INVALID")
    if normal_k!=int(document_gsa.get("normal_k",-1)):
        raise QualificationError("GSA_POLICY_DOCUMENT_DRIFT:normal_k")
    if abs(tolerance-float(document_gsa.get("visibility_depth_tolerance_norm",-1.0)))>1e-15:
        raise QualificationError("GSA_POLICY_DOCUMENT_DRIFT:visibility_depth_tolerance_norm")
    if content_sha256(dict(cfg.get("adequacy_policy") or {}))!=content_sha256(document_adequacy):
        raise QualificationError("GSA_POLICY_DOCUMENT_DRIFT:adequacy_policy")
    camera_dicts=tuple(asdict(c) for c in sorted(cameras.cameras,key=lambda c:c.view_index))
    surface,adequacy=select_adequate_rigging_surface_v1(
        vertices,faces,normals,camera_dicts,
        normalization_center=normalization.center_xyz,
        normalization_half_extent=normalization.half_extent,
        authority_label="IRIS_SCENE_FIRST_SIGNED_ZERO_SURFACE",
        source_run_id=str(ctx["ledger"].get("run_id") or ""),
        source_checkpoint_sha256=checkpoint.checkpoint_sha256,
        source_zero_surface_sha256=zero.npz_sha256,
        normal_k=normal_k,
        visibility_depth_tolerance_norm=tolerance,
        adequacy_policy=document_adequacy,
        metadata={
            "rest_reprojection_geometry_gate_hash":str(gate_payload["geometry_gate_hash"]),
            "observation_set_hash":zero.observation_set_binding_hash,
            "stage14_policy_document_path":str(policy_ref.get("path") or ""),
            "stage14_policy_document_sha256":str(policy_ref.get("sha256") or ""),
        },
    )
    adequacy["policy_document_path"]=str(policy_ref.get("path") or "")
    adequacy["policy_document_sha256"]=str(policy_ref.get("sha256") or "")
    adequacy["adequacy_report_hash"]=""
    adequacy["adequacy_report_hash"]=substrate_adequacy_report_hash_v1(adequacy)
    if surface is None:
        return {"status":"FAIL","blockers":["SUBSTRATE_ADEQUACY_NO_PASSING_CANDIDATE"],"diagnostics":adequacy}
    root=ctx["run_root"]/"artifacts"/"14_GSA_BUILD"
    return {"status":"PASS","outputs":[
        _write_ir(root/"rigging_surface_candidate.json",surface,authority_class="GSA_RIGGING_SURFACE_CANDIDATE"),
        _write_ir(root/"substrate_adequacy_report.json",adequacy,authority_class="SUBSTRATE_ADEQUACY_REPORT"),
    ],
        "diagnostics":{"surface_lineage_hash":surface.geometry_lineage_hash,"node_count":len(surface.surface_nodes),
                       "relation_count":len(surface.local_relations),"adequacy_report_hash":adequacy["adequacy_report_hash"],
                       "selected_target_node_cap":adequacy["selected_target_node_cap"]}}


def qualify_rigging_surface_stage(ctx:dict)->dict:
    surface=rigging_surface_from_dict(
        _stage_output_payload(ctx,"14_GSA_BUILD","RealSaS.RiggingSurfaceIR.v1")
    )
    adequacy=_stage_output_payload(ctx,"14_GSA_BUILD","RealSaS.SubstrateAdequacyReport.v1")
    if adequacy.get("adequacy_report_hash")!=substrate_adequacy_report_hash_v1(adequacy):
        raise QualificationError("RIGGING_SURFACE_ADEQUACY_HASH_DRIFT")
    if str(adequacy.get("status"))!="PASS" or str(adequacy.get("selected_surface_lineage_hash"))!=surface.geometry_lineage_hash:
        raise QualificationError("RIGGING_SURFACE_ADEQUACY_BINDING_DRIFT")
    if int(adequacy.get("selected_actual_node_count",-1))!=len(surface.surface_nodes):
        raise QualificationError("RIGGING_SURFACE_ADEQUACY_NODE_COUNT_DRIFT")
    tensor=tensorize_rigging_surface_v1(surface,require_scene_first=True)
    if tensor.source_surface_hash!=surface.geometry_lineage_hash:
        raise QualificationError("RIGGING_SURFACE_TENSORIZATION_BINDING_DRIFT")
    observed=int(np.count_nonzero(tensor.observed)); completed=int(np.count_nonzero(tensor.completed))
    if observed<=0 or observed+completed!=tensor.node_count:
        raise QualificationError("RIGGING_SURFACE_OBSERVED_COMPLETED_ACCOUNTING_INVALID")
    value=RiggingSurfaceQualificationIR(
        surface.geometry_lineage_hash,tensor.tensorization_hash,tensor.certificate_hash,
        tensor.node_count,tensor.edge_count,observed,completed,
        {
            "status":"PASS",
            "scene_first_signed_geometry":True,
            "teacher_truth_contamination_rejected":True,
            "lossless_fieldwise_tensorization_passed":True,
            "local_relation_graph_present":tensor.edge_count>0,
            "substrate_adequacy_passed":True,
            "substrate_adequacy_report_hash":adequacy["adequacy_report_hash"],
        },"",
        metadata={"tensorization_schema":tensor.schema_version,"substrate_adequacy_report_hash":adequacy["adequacy_report_hash"]},
    )
    value=replace(value,qualification_hash=rigging_surface_qualification_hash(value))
    root=ctx["run_root"]/"artifacts"/"15_RIGGING_SURFACE_QUALIFIED"
    return {"status":"PASS","outputs":[
        _write_ir(root/"qualified_rigging_surface.json",surface,authority_class="QUALIFIED_RIGGING_SURFACE"),
        _write_ir(root/"rigging_surface_qualification.json",value,authority_class="RIGGING_SURFACE_QUALIFICATION"),
    ],"diagnostics":{
        "surface_lineage_hash":surface.geometry_lineage_hash,"qualification_hash":value.qualification_hash,
        "tensorization_hash":value.tensorization_hash,"node_count":value.node_count,"edge_count":value.edge_count,
    }}
