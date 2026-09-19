from __future__ import annotations

"""Production adapters for mainline stages 24-27.

External files are never trusted by filename. Every manifest file reference carries an
exact SHA-256, and every upstream stage artifact is selected by exact schema from the
sealed active-run ledger.
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
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    expected_g3_probe_plan_hash,
    run_g3_deformation_stress,
)
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    ComponentObservationRasterIR,
    build_g5_coverage_matrix,
    camera_projection_binding_hash,
    component_surface_set_hash,
    g5_coverage_evidence_hash,
    mask_sha256,
)
from compiler.realsas_compiler_core.mesh.product_policy_v1 import load_mesh_policy_document
from compiler.realsas_compiler_core.motion_3d_adapter_v1 import parse_axis_contract_v1
from compiler.realsas_compiler_core.playback_full_surface_v3 import qualify_camera_v3
from compiler.realsas_compiler_core.playback_runtime_v3 import ReferenceRasterContractV1
from compiler.realsas_compiler_core.product_artifact_codec_v1 import (
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
    row=next((x for x in ctx["ledger"]["stages"] if x["id"]==stage_id),None)
    if row is None or row.get("status") not in {"PASS","CACHE_HIT"}:
        raise QualificationError(f"PRODUCT_ADAPTER_UPSTREAM_NOT_PASS:{stage_id}")
    matches=[out for out in row.get("outputs",()) if out.get("schema")==schema]
    if len(matches)!=1:
        raise QualificationError(f"PRODUCT_ADAPTER_UPSTREAM_SCHEMA_CARDINALITY:{stage_id}:{schema}:{len(matches)}")
    out=matches[0]
    path=_resolved_path(out["path"])
    if not path.is_file() or _sha256(path)!=out.get("sha256"):
        raise QualificationError(f"PRODUCT_ADAPTER_UPSTREAM_OUTPUT_DRIFT:{stage_id}:{schema}")
    payload=json.loads(path.read_text(encoding="utf-8"))
    actual_schema=str(payload.get("schema") or payload.get("schema_version") or "")
    if actual_schema!=schema:
        raise QualificationError(f"PRODUCT_ADAPTER_UPSTREAM_EMBEDDED_SCHEMA_DRIFT:{stage_id}:{schema}")
    return payload


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


def _load_skeleton(ctx):
    return qualified_skeleton_from_dict(_stage_output_payload(ctx,"18_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))


def _load_skin(ctx):
    return qualified_skin_from_dict(_stage_output_payload(ctx,"22_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))


def _load_partition_and_carrier(ctx):
    partition=mechanical_partition_from_dict(
        _stage_output_payload(ctx,"24_MECHANICAL_PARTITION_QUALIFIED","RealSaS.MechanicalPartitionIR.v1")
    )
    carrier=component_carrier_policy_from_dict(
        _stage_output_payload(ctx,"24_MECHANICAL_PARTITION_QUALIFIED","RealSaS.ComponentCarrierPolicyIR.v1")
    )
    return partition,carrier


def _load_envelope(ctx):
    return deformation_envelope_from_dict(
        _stage_output_payload(ctx,"25_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1")
    )


def _load_candidate_and_policy(ctx):
    candidate=canonical_mesh_candidate_from_dict(
        _stage_output_payload(ctx,"26_MESH_CANDIDATE_BUILD","RealSaS.CanonicalMeshCandidateIR.v1")
    )
    policy=mesh_policy_from_dict(
        _stage_output_payload(ctx,"26_MESH_CANDIDATE_BUILD","RealSaS.MeshQualificationPolicyIR.v1")
    )
    return candidate,policy


def _load_camera_set(ctx):
    return qualified_camera_set_from_dict(
        _stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )


def _axis_contract(ctx):
    axis_payload=_stage_output_payload(ctx,"25_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DerivedAxisContract.v1")
    _,axis_hash=parse_axis_contract_v1(axis_payload)
    return axis_payload,axis_hash


def qualify_mechanical_partition_and_carriers(ctx:dict)->dict:
    surface=_load_surface(ctx)
    cfg=dict(ctx["run_manifest"].get("components") or {})
    overrides=[]
    for row in tuple(cfg.get("boundary_overrides") or ()):
        overrides.append(ComponentBoundaryConstraintIR(
            constraint_id=str(row["constraint_id"]),
            a_surface_id=str(row["a_surface_id"]),
            b_surface_id=str(row["b_surface_id"]),
            decision=str(row["decision"]),
            evidence_refs=tuple(map(str,row.get("evidence_refs") or ())),
            confidence=float(row.get("confidence",1.0)),
            metadata=dict(row.get("metadata") or {}),
        ))
    partition=build_structural_partition(surface,boundary_overrides=tuple(overrides))

    carrier_cfg=dict(ctx["run_manifest"].get("carrier_policy") or {})
    explicit={str(row["component_id"]):row for row in tuple(carrier_cfg.get("decisions") or ())}
    unknown=set(explicit)-{c.component_id for c in partition.components}
    if unknown:
        return {"status":"FAIL","blockers":["CARRIER_POLICY_UNKNOWN_COMPONENT"],"diagnostics":{"unknown_component_ids":sorted(unknown)}}
    decisions=[]
    for component in partition.components:
        row=explicit.get(component.component_id)
        if row is None:
            decisions.append(ComponentCarrierDecisionIR(
                component.component_id,
                "MESH",
                ("CONSERVATIVE_DEFAULT_MESH_CARRIER_V1",),
                metadata={"conservative_default":True,"planar_promotion_requires_explicit_evidence":True},
            ))
        else:
            decisions.append(ComponentCarrierDecisionIR(
                component.component_id,
                str(row["carrier_class"]),
                tuple(map(str,row.get("evidence_refs") or ())),
                metadata=dict(row.get("metadata") or {}),
            ))
    carrier=build_component_carrier_policy(
        partition=partition,
        decisions=tuple(decisions),
        metadata={"default_carrier":"MESH","clip_is_presentation_only":True},
    )
    validate_mechanical_partition(partition,surface)
    validate_component_carrier_policy(carrier,partition)
    root=ctx["run_root"]/"artifacts"/"24_MECHANICAL_PARTITION_QUALIFIED"
    return {
        "status":"PASS",
        "outputs":[
            _write_ir(root/"mechanical_partition.json",partition,authority_class="QUALIFIED_MECHANICAL_PARTITION"),
            _write_ir(root/"component_carrier_policy.json",carrier,authority_class="QUALIFIED_COMPONENT_CARRIER_POLICY"),
        ],
        "diagnostics":{
            "component_count":len(partition.components),
            "unknown_boundary_count":sum(x.decision=="UNKNOWN" for x in partition.boundary_constraints),
            "explicit_carrier_decision_count":len(explicit),
            "conservative_mesh_default_count":len(partition.components)-len(explicit),
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
    root=ctx["run_root"]/"artifacts"/"25_DEFORMATION_CAPABILITY_ENVELOPE"
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
    if backend=="CANONICAL_RELATION_BASELINE_V1":
        candidate=build_canonical_relation_candidate(
            surface,partition,carrier,producer_policy_hash=baseline_policy_hash
        )
    elif backend=="CANONICAL_CDT_LOCAL_CHART_V1":
        candidate=build_canonical_cdt_candidate(
            surface,partition,carrier,policy,
            relation_baseline_policy_hash=baseline_policy_hash,
            max_constraint_recovery_iterations=int(mesh_cfg.get("max_constraint_recovery_iterations",96)),
            max_quality_iterations=int(mesh_cfg.get("max_quality_iterations",96)),
        )
    else:
        return {"status":"BLOCKED","blockers":["MESH_BACKEND_NOT_EXPLICIT_OR_UNSUPPORTED"],"diagnostics":{"backend":backend}}
    root=ctx["run_root"]/"artifacts"/"26_MESH_CANDIDATE_BUILD"
    return {
        "status":"PASS",
        "outputs":[
            _write_ir(root/"canonical_mesh_candidate.json",candidate,authority_class="DERIVED_MESH_CANDIDATE"),
            _write_ir(root/"mesh_qualification_policy.json",policy,authority_class="FROZEN_MESH_QUALIFICATION_POLICY"),
        ],
        "diagnostics":{
            "backend":backend,
            "vertex_count":len(candidate.vertices),
            "face_count":len(candidate.faces),
            "candidate_lineage_hash":candidate.candidate_lineage_hash,
            "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        },
    }


def _component_observations(ctx, *, partition, carrier, cameras):
    cfg=dict(ctx["run_manifest"].get("observation") or {})
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
        source_foreground_masks[vi]=raw

    rows=tuple(cfg.get("component_masks") or ())
    expected={(vi,c.component_id) for vi in range(8) for c in partition.components}
    supplied={(int(row["view_index"]),str(row["component_id"])) for row in rows}
    if supplied!=expected or len(rows)!=len(expected):
        raise QualificationError("PRODUCT_ADAPTER_COMPONENT_MASK_MATRIX_INCOMPLETE")
    camera_by_view={camera.view_index:camera for camera in cameras}
    carrier_by_component={row.component_id:row.carrier_class for row in carrier.decisions}
    component_by_id={row.component_id:row for row in partition.components}
    raster_hash=ReferenceRasterContractV1().contract_hash
    out=[]
    for row in rows:
        vi=int(row["view_index"]); component_id=str(row["component_id"])
        path=_load_file_ref(dict(row.get("mask") or {}),json_required=False)
        raw=path.read_bytes()
        camera=camera_by_view[vi]
        authority=authority_by_view[vi]
        expected_count=int(camera.resolution)*int(camera.resolution)
        if len(raw)!=expected_count:
            raise QualificationError("PRODUCT_ADAPTER_COMPONENT_MASK_SIZE_INVALID")
        if any(value not in (0,1) for value in raw):
            raise QualificationError("PRODUCT_ADAPTER_COMPONENT_MASK_BINARY_REQUIRED")
        if "source_observation_hash" in row and str(row["source_observation_hash"])!=authority.source_observation_hash:
            raise QualificationError("PRODUCT_ADAPTER_COMPONENT_SOURCE_OBSERVATION_DRIFT")
        out.append(ComponentObservationRasterIR(
            view_index=vi,
            component_id=component_id,
            carrier_class=carrier_by_component[component_id],
            partition_binding_hash=partition.partition_lineage_hash,
            carrier_policy_binding_hash=carrier.carrier_policy_lineage_hash,
            component_surface_set_hash=component_surface_set_hash(component_by_id[component_id]),
            width=int(camera.resolution),
            height=int(camera.resolution),
            mask_bytes=raw,
            mask_sha256=mask_sha256(raw),
            source_observation_hash=authority.source_observation_hash,
            camera_binding_hash=camera_projection_binding_hash(camera),
            raster_contract_hash=raster_hash,
            metadata={"mask_file_sha256":_sha256(path),"observation_set_hash":observation_set.observation_set_hash},
        ))
    return tuple(out),source_foreground_masks,observation_set


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

    g3=run_g3_deformation_stress(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        axis_contract=axis_payload,
        policy=policy,
    )
    observations,source_foreground_masks,observation_set=_component_observations(
        ctx,partition=partition,carrier=carrier,cameras=cameras
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

    root=ctx["run_root"]/"artifacts"/"27_QUALIFIED_MESH_GATE"
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
        _stage_output_payload(ctx,"27_QUALIFIED_MESH_GATE","RealSaS.QualifiedMeshIR.v1")
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
    root=ctx["run_root"]/"artifacts"/"28_QUALIFIED_MESH_SKIN_TRANSFER"
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
        _stage_output_payload(ctx,"27_QUALIFIED_MESH_GATE","RealSaS.QualifiedMeshIR.v1")
    )
    mesh_skin=qualified_mesh_skin_from_dict(
        _stage_output_payload(ctx,"28_QUALIFIED_MESH_SKIN_TRANSFER","RealSaS.QualifiedMeshSkinIR.v1")
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
    root=ctx["run_root"]/"artifacts"/"29_CANONICAL_PUPPET_STATE_SEALED"
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
