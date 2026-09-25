from __future__ import annotations

"""Current V2 mesh/mechanics adapters.

This module is exact-stage V2 authority. Historical V1 compatibility belongs outside
the current 46-stage DAG.
"""

from dataclasses import asdict, replace
import hashlib, json, os
from pathlib import Path
from typing import Any, Callable

from compiler.realsas_compiler_core.canonical_cdt_adapter_v1 import build_canonical_cdt_candidate
from compiler.realsas_compiler_core.canonical_puppet_state_v1 import build_canonical_puppet_state
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import derive_deformation_envelope_v1
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.conditioning_v1 import triangle_rest_metric
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    build_g5_coverage_matrix,
    derive_component_observation_rasters_v1,
    camera_projection_binding_hash,
    component_surface_set_hash,
    g5_coverage_evidence_hash,
    mask_sha256,
)
from compiler.realsas_compiler_core.mesh.product_policy_v1 import load_mesh_policy_document
from compiler.realsas_compiler_core.motion_3d_adapter_v1 import parse_axis_contract_v1
from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_mesh_from_dict,
    qualified_mesh_skin_from_dict,
    qualified_observation_set_from_dict,
    qualified_camera_set_from_dict,
    component_carrier_policy_from_dict,
    deformation_envelope_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    read_json,
    rigging_surface_from_dict,
    write_ir_json,
)
from compiler.realsas_compiler_core.product_mesh_skin_v1 import bind_product_mesh_skin
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentBoundaryConstraintIR,
    ComponentCarrierDecisionIR,
    DeformationCapabilityEnvelopeIR,
    JointCapabilityRangeIR,
    build_component_carrier_policy,
    deformation_envelope_lineage_hash,
    qualify_canonical_mesh_candidate,
    validate_component_carrier_policy,
    validate_deformation_capability_envelope,
    validate_mechanical_partition,
    validate_mesh_qualification_policy,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)

Json=dict[str,Any]


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def _resolved_path(raw:str)->Path:
    value=os.path.expandvars(str(raw))
    if any(token in Path(value).parts for token in ("latest","current","newest")):
        raise QualificationError("PRODUCT_ADAPTER_MOVING_ALIAS_FORBIDDEN")
    return Path(value).expanduser().resolve()


def _load_file_ref(ref:dict, *, expected_schema:str|None=None, json_required:bool=True):
    path=_resolved_path(str(ref.get("path","")))
    expected=str(ref.get("sha256",""))
    if not path.is_file() or len(expected)!=64:
        raise QualificationError("PRODUCT_ADAPTER_FILE_REF_INCOMPLETE")
    actual=_sha256(path)
    if actual!=expected:
        raise QualificationError("PRODUCT_ADAPTER_FILE_SHA_MISMATCH")
    if not json_required:
        return path
    payload=json.loads(path.read_text(encoding="utf-8"))
    if expected_schema is not None:
        actual_schema=str(payload.get("schema") or payload.get("schema_version") or "")
        if actual_schema!=expected_schema:
            raise QualificationError(f"PRODUCT_ADAPTER_FILE_SCHEMA_MISMATCH:{actual_schema}!={expected_schema}")
    return payload


def _stage_output_payload(ctx:dict, stage_id:str, schema:str)->dict:
    return stage_output_payload(ctx, stage_id, schema)


def _write_json(path:Path,payload:dict,*,authority_class:str,schema:str)->dict:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
    return {"path":str(path),"sha256":_sha256(path),"authority_class":authority_class,"schema":schema}


def _write_ir(path:Path,value,*,authority_class:str)->dict:
    write_ir_json(path,value)
    return {
        "path":str(path),
        "sha256":_sha256(path),
        "authority_class":authority_class,
        "schema":str(value.schema_version),
    }


def _load_observation_set(ctx):
    return qualified_observation_set_from_dict(
        _stage_output_payload(ctx,"07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1")
    )


def _load_surface(ctx):
    return rigging_surface_from_dict(_stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))


def _artifact_root(ctx,fallback_stage_id:str):
    stage_id=str((ctx.get("stage") or {}).get("id") or fallback_stage_id)
    return ctx["run_root"]/"artifacts"/stage_id


def _load_skeleton(ctx):
    stage_id="28_SKELETON_QUALIFIED"
    return qualified_skeleton_from_dict(_stage_output_payload(ctx,stage_id,"RealSaS.QualifiedSkeletonIR.v1"))


def _load_skin(ctx):
    stage_id="32_SKIN_QUALIFIED"
    return qualified_skin_from_dict(_stage_output_payload(ctx,stage_id,"RealSaS.QualifiedSkinIR.v1"))


def _load_partition_and_carrier(ctx):
    stage_id="17_MECHANICAL_PARTITION_QUALIFIED"
    partition=mechanical_partition_from_dict(
        _stage_output_payload(ctx,stage_id,"RealSaS.MechanicalPartitionIR.v1")
    )
    carrier=component_carrier_policy_from_dict(
        _stage_output_payload(ctx,stage_id,"RealSaS.ComponentCarrierPolicyIR.v1")
    )
    return partition,carrier


def _load_envelope(ctx):
    stage_id="34_DEFORMATION_CAPABILITY_ENVELOPE"
    return deformation_envelope_from_dict(
        _stage_output_payload(ctx,stage_id,"RealSaS.DeformationCapabilityEnvelopeIR.v1")
    )


def _load_candidate_and_policy(ctx):
    stage_id="18_CANONICAL_MESH_ADDRESSING_BUILD"
    candidate=canonical_mesh_candidate_from_dict(
        _stage_output_payload(ctx,stage_id,"RealSaS.CanonicalMeshCandidateIR.v1")
    )
    policy=mesh_policy_from_dict(
        _stage_output_payload(ctx,stage_id,"RealSaS.MeshQualificationPolicyIR.v1")
    )
    return candidate,policy


def _load_camera_set(ctx):
    return qualified_camera_set_from_dict(
        _stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )


def _axis_contract(ctx):
    stage_id="34_DEFORMATION_CAPABILITY_ENVELOPE"
    axis_payload=_stage_output_payload(ctx,stage_id,"RealSaS.DerivedAxisContract.v1")
    _,axis_hash=parse_axis_contract_v1(axis_payload)
    return axis_payload,axis_hash


def qualify_mechanical_partition_and_carriers(ctx:dict)->dict:
    surface=_load_surface(ctx)
    component_cfg=dict(ctx["run_manifest"].get("components") or {})
    carrier_cfg=dict(ctx["run_manifest"].get("carrier_policy") or {})
    if component_cfg:
        return {"status":"BLOCKED","blockers":["MANUAL_COMPONENT_BOUNDARY_AUTHORING_FORBIDDEN"],
                "diagnostics":{"unsupported_keys":sorted(component_cfg)}}
    if carrier_cfg:
        return {"status":"BLOCKED","blockers":["MANUAL_CARRIER_AUTHORING_FORBIDDEN"],
                "diagnostics":{"unsupported_keys":sorted(carrier_cfg)}}

    partition=build_structural_partition(surface,boundary_overrides=())
    decisions=tuple(
        ComponentCarrierDecisionIR(
            component.component_id,
            "MESH",
            ("AUTOMATIC_CONSERVATIVE_MESH_CARRIER_V1",),
            metadata={
                "automatic":True,
                "semantic_recognition_used":False,
                "planar_optimization_deferred":True,
            },
        )
        for component in partition.components
    )
    carrier=build_component_carrier_policy(
        partition=partition,
        decisions=decisions,
        metadata={
            "default_carrier":"MESH",
            "automatic":True,
            "manual_carrier_authoring_used":False,
            "clip_is_presentation_only":True,
        },
    )
    validate_mechanical_partition(partition,surface)
    validate_component_carrier_policy(carrier,partition)
    root=_artifact_root(ctx,"24_MECHANICAL_PARTITION_QUALIFIED")
    return {
        "status":"PASS",
        "outputs":[
            _write_ir(root/"mechanical_partition.json",partition,authority_class="QUALIFIED_MECHANICAL_PARTITION"),
            _write_ir(root/"component_carrier_policy.json",carrier,authority_class="QUALIFIED_COMPONENT_CARRIER_POLICY"),
        ],
        "diagnostics":{
            "component_count":len(partition.components),
            "unknown_boundary_count":sum(x.decision=="UNKNOWN" for x in partition.boundary_constraints),
            "manual_boundary_authoring":False,
            "manual_carrier_authoring":False,
            "all_components_mesh_carrier":True,
        },
    }

def seal_deformation_capability_envelope(ctx:dict)->dict:
    skeleton=_load_skeleton(ctx)
    cfg=dict(ctx["run_manifest"].get("deformation_envelope") or {})
    if cfg:
        return {"status":"BLOCKED","blockers":["PER_CHARACTER_DEFORMATION_AUTHORING_FORBIDDEN"],
                "diagnostics":{"unsupported_keys":sorted(cfg)}}
    camera_set=_load_camera_set(ctx)
    axis_payload,envelope=derive_deformation_envelope_v1(skeleton=skeleton,camera_set=camera_set)
    root=_artifact_root(ctx,"25_DEFORMATION_CAPABILITY_ENVELOPE")
    return {
        "status":"PASS",
        "outputs":[
            _write_json(root/"derived_axis_contract.json",axis_payload,authority_class="QUALIFIED_DERIVED_AXIS_CONTRACT",schema="RealSaS.DerivedAxisContract.v1"),
            _write_ir(root/"deformation_envelope.json",envelope,authority_class="QUALIFIED_DEFORMATION_CAPABILITY_ENVELOPE"),
        ],
        "diagnostics":{
            "joint_range_count":len(envelope.joint_ranges),
            "camera_count":len(camera_set.cameras),
            "probe_plan_hash":envelope.probe_plan_hash,
            "axis_contract_hash":envelope.axis_contract_hash,
            "per_character_joint_range_authoring":False,
            "per_character_axis_authoring":False,
        },
    }

def _relation_parent_quality_report(candidate, policy)->dict:
    vertices={str(v.candidate_vertex_id):tuple(map(float,v.P)) for v in candidate.vertices}
    rows=[]
    for face in candidate.faces:
        points=tuple(vertices[str(vid)] for vid in face)
        metric=triangle_rest_metric(points)
        rows.append(metric)
    if not rows:
        raise QualificationError("RELATION_PARENT_QUALITY_REQUIRES_FACES")
    bad_angle=sum(
        bool(row["degenerate"]) or float(row["min_angle_deg"]) + 1e-9 < float(policy.g3_min_angle_deg)
        for row in rows
    )
    bad_aspect=sum(
        bool(row["degenerate"]) or float(row["aspect_longest_over_min_altitude"]) - 1e-9 > float(policy.g3_max_aspect_longest_over_min_altitude)
        for row in rows
    )
    bad_union=sum(
        bool(row["degenerate"])
        or float(row["min_angle_deg"]) + 1e-9 < float(policy.g3_min_angle_deg)
        or float(row["aspect_longest_over_min_altitude"]) - 1e-9 > float(policy.g3_max_aspect_longest_over_min_altitude)
        for row in rows
    )
    finite_aspects=[
        float(row["aspect_longest_over_min_altitude"])
        for row in rows
        if not bool(row["degenerate"]) and float(row["aspect_longest_over_min_altitude"]) < float("inf")
    ]
    return {
        "schema":"RealSaS.RelationParentQualityReport.v1",
        "face_count":len(rows),
        "degenerate_face_count":sum(bool(row["degenerate"]) for row in rows),
        "min_angle_deg":min(float(row["min_angle_deg"]) for row in rows),
        "max_aspect_longest_over_min_altitude":max(finite_aspects,default=float("inf")),
        "policy_min_angle_deg":float(policy.g3_min_angle_deg),
        "policy_max_aspect_longest_over_min_altitude":float(policy.g3_max_aspect_longest_over_min_altitude),
        "below_min_angle_face_count":int(bad_angle),
        "above_max_aspect_face_count":int(bad_aspect),
        "policy_violating_face_count":int(bad_union),
        "below_min_angle_face_fraction":float(bad_angle/len(rows)),
        "above_max_aspect_face_fraction":float(bad_aspect/len(rows)),
        "policy_violating_face_fraction":float(bad_union/len(rows)),
        "producer_semantics":"THREE_CLIQUES_OF_RIGGING_SURFACE_LOCAL_RELATION_GRAPH",
        "not_raw_marching_cubes_parent_faces":True,
        "cdt_v1_boundary_split_policy":"FORBIDDEN",
        "cdt_v1_min_angle_repairability":(
            "UNREPAIRABLE_IF_ANY_PARENT_CORNER_IS_BELOW_TARGET"
            if bad_angle else "NO_PARENT_MIN_ANGLE_OBSTRUCTION_DETECTED"
        ),
        "proof_note":"Without boundary splits, any triangulation covering a parent triangle preserves each parent corner as a sum of incident child angles; a parent corner below the target cannot be raised above the target by interior Steiner insertion alone.",
    }


def build_canonical_mesh_candidate_stage(ctx:dict)->dict:
    surface=_load_surface(ctx)
    partition,carrier=_load_partition_and_carrier(ctx)
    mesh_cfg=dict(ctx["run_manifest"].get("mesh") or {})
    policy_cfg=dict(ctx["run_manifest"].get("mesh_policy") or {})
    policy_doc_ref=dict(policy_cfg.get("document") or {})
    policy_path=_resolved_path(str(policy_doc_ref.get("path","")))
    expected=str(policy_doc_ref.get("sha256",""))
    if not policy_path.is_file() or len(expected)!=64 or _sha256(policy_path)!=expected:
        return {"status":"BLOCKED","blockers":["MESH_POLICY_DOCUMENT_REF_INVALID"],"diagnostics":{}}
    policy=load_mesh_policy_document(policy_path)
    validate_mesh_qualification_policy(policy)

    backend=str(mesh_cfg.get("backend") or "")
    baseline_policy_hash=content_sha256({
        "schema":"RealSaS.CanonicalRelationBaselinePolicy.v1",
        "mesh_config":mesh_cfg,
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
    })
    baseline=build_canonical_relation_candidate(
        surface,partition,carrier,producer_policy_hash=baseline_policy_hash
    )
    parent_quality=_relation_parent_quality_report(baseline,policy)
    root=_artifact_root(ctx,"26_MESH_CANDIDATE_BUILD")
    parent_quality_artifact=_write_json(
        root/"relation_parent_quality_report.json",
        parent_quality,
        authority_class="DIAGNOSTIC_RELATION_PARENT_QUALITY",
        schema=parent_quality["schema"],
    )

    if backend=="CANONICAL_RELATION_BASELINE_V1":
        candidate=baseline
    elif backend=="CANONICAL_CDT_LOCAL_CHART_V1":
        if int(parent_quality["below_min_angle_face_count"])>0:
            return {
                "status":"FAIL",
                "blockers":["CDT_PARENT_MIN_ANGLE_UNREPAIRABLE_WITHOUT_BOUNDARY_SPLIT"],
                "diagnostics":{
                    "backend":backend,
                    "relation_parent_quality_sha256":parent_quality_artifact["sha256"],
                    **parent_quality,
                },
            }
        candidate=build_canonical_cdt_candidate(
            surface,partition,carrier,policy,
            relation_baseline_policy_hash=baseline_policy_hash,
            max_constraint_recovery_iterations=int(mesh_cfg.get("max_constraint_recovery_iterations",96)),
            max_quality_iterations=int(mesh_cfg.get("max_quality_iterations",96)),
        )
    else:
        return {"status":"BLOCKED","blockers":["MESH_BACKEND_NOT_EXPLICIT_OR_UNSUPPORTED"],"diagnostics":{"backend":backend}}
    return {
        "status":"PASS",
        "outputs":[
            _write_ir(root/"canonical_mesh_candidate.json",candidate,authority_class="DERIVED_MESH_CANDIDATE"),
            _write_ir(root/"mesh_qualification_policy.json",policy,authority_class="FROZEN_MESH_QUALIFICATION_POLICY"),
            parent_quality_artifact,
        ],
        "diagnostics":{
            "backend":backend,
            "vertex_count":len(candidate.vertices),
            "face_count":len(candidate.faces),
            "candidate_lineage_hash":candidate.candidate_lineage_hash,
            "mesh_policy_hash":policy.qualification_policy_lineage_hash,
            "relation_parent_quality_sha256":parent_quality_artifact["sha256"],
            "relation_parent_below_min_angle_face_count":parent_quality["below_min_angle_face_count"],
            "relation_parent_above_max_aspect_face_count":parent_quality["above_max_aspect_face_count"],
            "relation_parent_policy_violating_face_count":parent_quality["policy_violating_face_count"],
        },
    }


def _component_observations(ctx, *, surface, partition, carrier, cameras):
    cfg=dict(ctx["run_manifest"].get("observation") or {})
    if cfg.get("component_masks"):
        raise QualificationError("PRODUCT_ADAPTER_EXTERNAL_COMPONENT_MASKS_FORBIDDEN")
    observation_set=_load_observation_set(ctx)
    authority_by_view={int(row.view_index):row for row in observation_set.views}
    foreground_rows=tuple(cfg.get("source_foreground_masks") or ())
    if len(foreground_rows)!=8 or {int(row["view_index"]) for row in foreground_rows}!=set(range(8)):
        raise QualificationError("PRODUCT_ADAPTER_SOURCE_FOREGROUND_MATRIX_INCOMPLETE")
    source_foreground_masks={}
    for row in foreground_rows:
        vi=int(row["view_index"])
        path=_load_file_ref(dict(row.get("mask") or {}),json_required=False)
        raw=path.read_bytes()
        authority=authority_by_view[vi]
        if _sha256(path)!=authority.foreground_mask_sha256:
            raise QualificationError("PRODUCT_ADAPTER_SOURCE_FOREGROUND_AUTHORITY_DRIFT")
        if len(raw)!=int(authority.width)*int(authority.height) or any(x not in (0,1) for x in raw):
            raise QualificationError("PRODUCT_ADAPTER_SOURCE_FOREGROUND_INVALID")
        source_foreground_masks[vi]=raw
    observations=derive_component_observation_rasters_v1(
        surface=surface,partition=partition,carrier_policy=carrier,
        observation_set=observation_set,source_foreground_masks=source_foreground_masks,
        cameras=cameras,
    )
    return observations,source_foreground_masks,observation_set

def qualify_canonical_mesh_stage(ctx:dict)->dict:
    surface=_load_surface(ctx)
    skeleton=_load_skeleton(ctx)
    skin=_load_skin(ctx)
    partition,carrier=_load_partition_and_carrier(ctx)
    envelope=_load_envelope(ctx)
    candidate,policy=_load_candidate_and_policy(ctx)
    axis_payload,axis_hash=_axis_contract(ctx)
    camera_set=_load_camera_set(ctx)
    cameras=camera_set.cameras
    if tuple(camera_set.camera_binding_hashes)!=tuple(envelope.camera_binding_hashes):
        return {"status":"FAIL","blockers":["STAGE27_CAMERA_SET_DRIFT"],"diagnostics":{}}
    if axis_hash!=envelope.axis_contract_hash:
        return {"status":"FAIL","blockers":["STAGE27_AXIS_CONTRACT_DRIFT"],"diagnostics":{}}

    g3=run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    observations,source_foreground_masks,observation_set=_component_observations(
        ctx,surface=surface,partition=partition,carrier=carrier,cameras=cameras
    )
    g5_rows=build_g5_coverage_matrix(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        mesh_policy=policy,
        observations=observations,
        cameras=cameras,
        observation_set=observation_set,
        source_foreground_masks=source_foreground_masks,
    )
    unknown_rows=tuple(
        {
            "constraint_id":row.constraint_id,
            "a_surface_id":row.a_surface_id,
            "b_surface_id":row.b_surface_id,
            "decision":row.decision,
            "classification":"CONSEQUENTIAL_UNTIL_EXPLICITLY_RESOLVED",
        }
        for row in partition.boundary_constraints
        if row.decision=="UNKNOWN"
    )
    unknown_report={
        "schema":"RealSaS.UnknownBoundaryAnalysis.v1",
        "partition_lineage_hash":partition.partition_lineage_hash,
        "candidate_lineage_hash":candidate.candidate_lineage_hash,
        "conservative_rule":"ALL_ADMITTED_UNKNOWN_BOUNDARIES_ARE_CONSEQUENTIAL_V1",
        "rows":unknown_rows,
        "consequential_unknown_boundary_count":len(unknown_rows),
    }
    unknown_hash=content_sha256(unknown_report)

    blockers=[]
    if not g3.passed:
        blockers.append("G3_DEFORMATION_STRESS_FAIL")
    failed_g5=[row for row in g5_rows if row["status"]!="PASS"]
    if failed_g5:
        blockers.append("G5_MULTIVIEW_COMPONENT_COVERAGE_FAIL")
    if unknown_rows:
        blockers.append("G4_CONSEQUENTIAL_UNKNOWN_BOUNDARY")

    root=_artifact_root(ctx,"27_QUALIFIED_MESH_GATE")
    # Diagnostic evidence is written even when admission fails, but only PASS outputs
    # are sealed by the mainline ledger.
    _write_ir(root/"g3_deformation_stress.json",g3,authority_class="DIAGNOSTIC_G3_EVIDENCE")
    g5_payload={
        "schema":"RealSaS.G5CoverageEvidence.v1",
        "evidence_hash":g5_coverage_evidence_hash(g5_rows),
        "rows":g5_rows,
    }
    _write_json(root/"g5_coverage_evidence.json",g5_payload,authority_class="DIAGNOSTIC_G5_EVIDENCE",schema=g5_payload["schema"])
    _write_json(root/"unknown_boundary_analysis.json",unknown_report,authority_class="DIAGNOSTIC_G4_EVIDENCE",schema=unknown_report["schema"])

    if blockers:
        return {
            "status":"FAIL",
            "blockers":blockers,
            "diagnostics":{
                "g3_report_hash":g3.report_hash,
                "g3_failures":g3.failure_invariants,
                "g5_failed_cell_count":len(failed_g5),
                "g5_evidence_hash":g5_payload["evidence_hash"],
                "consequential_unknown_boundary_count":len(unknown_rows),
                "unknown_boundary_analysis_hash":unknown_hash,
            },
        }

    qualification_report={
        "gates":{
            "G1_SUPPORT_LINEAGE":"PASS",
            "G2_TOPOLOGY":"PASS",
            "G3_DEFORMATION":"PASS",
            "G4_COMPONENT_BOUNDARY":"PASS",
            "G5_MULTIVIEW_COVERAGE":"PASS",
        },
        "single_aggregate_score_authority":False,
        "view_component_coverage_matrix_complete":True,
        "consequential_unknown_boundary_count":0,
        "unknown_boundary_analysis_hash":unknown_hash,
        "g3_envelope_binding_hash":envelope.envelope_lineage_hash,
        "g3_motion_capability_claimed":False,
        "g3_role":"LOCAL_3D_NUMERICAL_CONDITIONING_ONLY",
        "g3_stress_probe_hash":g3.report_hash,
        "g3_stress_probe_status":"PASS",
        "carrier_policy_hash":carrier.carrier_policy_lineage_hash,
        "g5_evidence_hash":g5_payload["evidence_hash"],
        "view_component_coverage":g5_rows,
    }
    mesh=qualify_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        envelope=envelope,
        policy=policy,
        qualification_report=qualification_report,
    )
    outputs=[
        _write_ir(root/"qualified_mesh.json",mesh,authority_class="QUALIFIED_PRODUCT_GEOMETRY"),
        _write_ir(root/"g3_deformation_stress.json",g3,authority_class="QUALIFIED_G3_EVIDENCE"),
        _write_json(root/"g5_coverage_evidence.json",g5_payload,authority_class="QUALIFIED_G5_EVIDENCE",schema=g5_payload["schema"]),
        _write_json(root/"unknown_boundary_analysis.json",unknown_report,authority_class="QUALIFIED_G4_EVIDENCE",schema=unknown_report["schema"]),
    ]
    return {
        "status":"PASS",
        "outputs":outputs,
        "diagnostics":{
            "mesh_lineage_hash":mesh.mesh_lineage_hash,
            "g3_report_hash":g3.report_hash,
            "g5_evidence_hash":g5_payload["evidence_hash"],
            "coverage_cell_count":len(g5_rows),
        },
    }


def bind_qualified_mesh_skin_stage(ctx:dict)->dict:
    surface=_load_surface(ctx)
    skeleton=_load_skeleton(ctx)
    skin=_load_skin(ctx)
    partition,carrier=_load_partition_and_carrier(ctx)
    envelope=_load_envelope(ctx)
    _,policy=_load_candidate_and_policy(ctx)
    mesh=qualified_mesh_from_dict(
        _stage_output_payload(ctx,"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED","RealSaS.QualifiedMeshIR.v1")
    )
    bound=bind_product_mesh_skin(
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        mesh=mesh,
        partition=partition,
        carrier_policy=carrier,
        envelope=envelope,
        policy=policy,
    )
    root=_artifact_root(ctx,"28_QUALIFIED_MESH_SKIN_TRANSFER")
    return {
        "status":"PASS",
        "outputs":[_write_ir(root/"qualified_mesh_skin.json",bound,authority_class="QUALIFIED_PRODUCT_MESH_SKIN")],
        "diagnostics":{
            "mesh_skin_lineage_hash":bound.mesh_skin_lineage_hash,
            "mesh_binding_hash":bound.mesh_binding_hash,
            "skin_binding_hash":bound.skin_binding_hash,
            "row_count":len(bound.rows),
        },
    }


def seal_canonical_puppet_state_stage(ctx:dict)->dict:
    surface=_load_surface(ctx)
    skeleton=_load_skeleton(ctx)
    skin=_load_skin(ctx)
    partition,carrier=_load_partition_and_carrier(ctx)
    envelope=_load_envelope(ctx)
    _,policy=_load_candidate_and_policy(ctx)
    mesh=qualified_mesh_from_dict(
        _stage_output_payload(ctx,"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED","RealSaS.QualifiedMeshIR.v1")
    )
    mesh_skin=qualified_mesh_skin_from_dict(
        _stage_output_payload(ctx,"36_QUALIFIED_MESH_SKIN_TRANSFER","RealSaS.QualifiedMeshSkinIR.v1")
    )
    state=build_canonical_puppet_state(
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        carrier_policy=carrier,
        envelope=envelope,
        policy=policy,
        mesh=mesh,
        mesh_skin=mesh_skin,
        metadata={
            "run_id":ctx["run_id"],
            "stage_id":"29_CANONICAL_PUPPET_STATE_SEALED",
        },
    )
    root=_artifact_root(ctx,"29_CANONICAL_PUPPET_STATE_SEALED")
    return {
        "status":"PASS",
        "outputs":[_write_ir(root/"canonical_puppet_state.json",state,authority_class="CANONICAL_MECHANICAL_PRODUCT_STATE")],
        "diagnostics":{
            "product_state_hash":state.product_state_hash,
            "mesh_lineage_hash":state.mesh_lineage_hash,
            "mesh_skin_lineage_hash":state.mesh_skin_lineage_hash,
            "qualification_ledger_entries":len(state.qualification_ledger),
        },
    }
