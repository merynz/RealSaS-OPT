from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_holeless_partitioned_dense_candidate,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_repartition_v2 import (
    build_repartitioned_partition_v2,
    repartition_authorization_hash_v1,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from tools.audit_knight_teacher_free_weight_completion_court_v1 import motion_metrics, stress_arbitrary_weights
from tools.demo.render_knight_motion_preview_v1 import _ctx


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load(path: Path, codec):
    if not path.is_file():
        raise RuntimeError(f"MISSING::{path}")
    return codec(load_json(path))


def stage_output_path(rr: Path, stage_id: str, schema: str) -> Path:
    ledger=load_json(rr/"ACTIVE_RUN_V2.json")
    row=next(x for x in ledger["stages"] if x["id"]==stage_id)
    hits=[x for x in row.get("outputs") or () if x.get("schema")==schema]
    if len(hits)!=1:
        raise RuntimeError(f"OUTPUT_LOOKUP::{stage_id}::{schema}::{len(hits)}")
    return Path(hits[0]["path"]).resolve()


def face_indices(candidate):
    ix={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    return np.asarray([[ix[str(x)] for x in face] for face in candidate.faces],dtype=np.int64)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--arm",required=True)
    ap.add_argument("--skin-json",type=Path,required=True)
    ap.add_argument("--directive-json",type=Path,required=True)
    ap.add_argument("--a100-audit-json",type=Path,required=True)
    ap.add_argument("--rebound-report-json",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()

    rr=_ctx(args.authority_root,args.run_id)["run_root"]
    surface=load(stage_output_path(rr,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"),rigging_surface_from_dict)
    face_prov=load_json(stage_output_path(rr,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.CompactedDenseFaceProvenance.v1"))
    parent=load(stage_output_path(rr,"17_MECHANICAL_PARTITION_QUALIFIED","RealSaS.MechanicalPartitionIR.v1"),mechanical_partition_from_dict)
    parent_candidate=load(stage_output_path(rr,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"),canonical_mesh_candidate_from_dict)
    policy=load(stage_output_path(rr,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1"),mesh_policy_from_dict)
    skeleton=load(stage_output_path(rr,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"),qualified_skeleton_from_dict)
    cameras=tuple(sorted(load(stage_output_path(rr,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"),qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    envelope=load(stage_output_path(rr,"34_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1"),deformation_envelope_from_dict)
    skin=load(args.skin_json,qualified_skin_from_dict)
    directive=load_json(args.directive_json)
    a100=load_json(args.a100_audit_json)
    rebound=load_json(args.rebound_report_json)

    if directive["source_skin_lineage_hash"]!=skin.skin_lineage_hash:
        raise RuntimeError("DIRECTIVE_SKIN_DRIFT")
    if directive["source_partition_lineage_hash"]!=parent.partition_lineage_hash:
        raise RuntimeError("DIRECTIVE_PARTITION_DRIFT")
    if directive["source_candidate_lineage_hash"]!=parent_candidate.candidate_lineage_hash:
        raise RuntimeError("DIRECTIVE_CANDIDATE_DRIFT")
    if directive["unresolved_unsafe_face_count"]!=0:
        raise RuntimeError("DIRECTIVE_UNRESOLVED_UNSAFE")
    if directive["face_deletion_count"]!=0 or directive["weight_mutation"] is not False:
        raise RuntimeError("DIRECTIVE_MUTATION_SCOPE")

    arm_a100=a100["arms"][args.arm]
    evidence_payload={
        "schema":"RealSaS.KnightArachneWeightReliabilityEvidence.v1",
        "arm":args.arm,
        "a100_result_sha256":arm_a100["result_sha256"],
        "a100_weights_sha256":arm_a100["weights_sha256"],
        "a100_final_valid_gate":arm_a100["final_valid_gate"],
        "a100_valid":arm_a100["valid"],
        "a100_invalid_diagnostic":arm_a100["invalid"],
        "compiler_requalification":rebound["compiler_requalification"],
        "teacher_inputs_used_by_predictor":False,
        "projected_rows_are_product_teacher_authority":False,
        "scope":"DEMO_REPAIR_CHILD_COURT_ONLY",
    }
    evidence_hash=content_sha256(evidence_payload)
    auth={
        "schema":"RealSaS.TrustworthySkinRepartitionAuthorization.v1",
        "status":"PASS_TRUSTWORTHY_SKIN_REPARTITION_AUTHORIZATION",
        "directive_hash":directive["directive_hash"],
        "source_surface_lineage_hash":surface.geometry_lineage_hash,
        "source_partition_lineage_hash":parent.partition_lineage_hash,
        "source_skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "source_skin_lineage_hash":skin.skin_lineage_hash,
        "teacher_inputs_used_by_predictor":False,
        "weight_reliability_closure_passed":True,
        "weight_reliability_evidence_hash":evidence_hash,
        "scope":"DEMO_REPAIR_CHILD_COURT_ONLY",
        "product_authority_claimed":False,
        "projected_invalid_rows_product_teacher_authority":False,
        "authorization_hash":"",
    }
    auth["authorization_hash"]=repartition_authorization_hash_v1(auth)

    child_partition=build_repartitioned_partition_v2(
        surface=surface,
        parent_partition=parent,
        directive=directive,
        authorization=auth,
    )
    decisions=tuple(
        ComponentCarrierDecisionIR(c.component_id,"MESH",("AUTOMATIC_CONSERVATIVE_MESH_CARRIER_V1",),
            metadata={"automatic":True,"semantic_recognition_used":False,"planar_optimization_deferred":True})
        for c in child_partition.components
    )
    carrier=build_component_carrier_policy(
        partition=child_partition,
        decisions=decisions,
        metadata={"default_carrier":"MESH","automatic":True,"manual_carrier_authoring_used":False,"clip_is_presentation_only":True},
    )

    explicit_faces=tuple(tuple(map(str,row)) for row in face_prov["compact_faces"])
    producer_policy_hash=content_sha256({
        "schema":"RealSaS.CanonicalRelationBaselinePolicy.v1",
        "mesh_config":{"backend":"CANONICAL_CDT_LOCAL_CHART_V1"},
        "mesh_policy_hash":policy.qualification_policy_lineage_hash,
        "repair_child_attempt":True,
        "parent_candidate_lineage_hash":parent_candidate.candidate_lineage_hash,
        "repartition_authorization_hash":auth["authorization_hash"],
    })
    child_candidate=build_holeless_partitioned_dense_candidate(
        surface,child_partition,carrier,
        producer_policy_hash=producer_policy_hash,
        explicit_face_provenance=explicit_faces,
    )
    validate_canonical_mesh_candidate(child_candidate,surface=surface,partition=child_partition,carrier_policy=carrier)

    comp=run_skin_topology_compatibility_v1(
        child_candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,cameras=cameras,policy=policy
    )

    rest,weights,faces=_candidate_skin_matrix(child_candidate,surface=surface,skeleton=skeleton,skin=skin)
    jids=tuple(j.canonical_joint_id for j in skeleton.joints)
    stress=stress_arbitrary_weights(np.asarray(rest,dtype=np.float64),np.asarray(weights,dtype=np.float64),np.asarray(faces,dtype=np.int64),jids,skeleton,cameras,envelope,policy)
    source_report=load_json(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))
    motion=motion_metrics(np.asarray(rest,dtype=np.float64),np.asarray(weights,dtype=np.float64),np.asarray(faces,dtype=np.int64),jids,skeleton,cameras,rr,source_report)

    report={
        "schema":"RealSaS.KnightArachneV6Stage17Stage18RepairChildCourt.v1",
        "status":"COMPLETE__NO_PARENT_MUTATION",
        "arm":args.arm,
        "authorization":auth,
        "weight_reliability_evidence":evidence_payload,
        "parent":{
            "partition_lineage_hash":parent.partition_lineage_hash,
            "component_count":len(parent.components),
            "candidate_lineage_hash":parent_candidate.candidate_lineage_hash,
            "vertex_count":len(parent_candidate.vertices),
            "face_count":len(parent_candidate.faces),
        },
        "child":{
            "partition_lineage_hash":child_partition.partition_lineage_hash,
            "component_count":len(child_partition.components),
            "separate_boundary_count":sum(x.decision=="SEPARATE" for x in child_partition.boundary_constraints),
            "candidate_lineage_hash":child_candidate.candidate_lineage_hash,
            "vertex_count":len(child_candidate.vertices),
            "face_count":len(child_candidate.faces),
            "candidate_metadata":dict(child_candidate.metadata or {}),
        },
        "post_repair_g3b":{
            "passed":bool(comp["passed"]),
            "risky_face_count":int(comp["risky_face_count"]),
            "unsafe_face_count":int(comp["unsafe_face_count"]),
            "report_hash":comp["report_hash"],
            "weight_mutation":bool(comp["weight_mutation"]),
        },
        "post_repair_synthetic_stress":{k:v for k,v in stress.items() if k!="unsafe_face_indices"},
        "post_repair_actual_motion":{k:v for k,v in motion.items() if k!="frames"},
        "invariants":{
            "face_deletion_count":int(child_candidate.metadata.get("face_deletion_count",-1)),
            "rest_area_relative_error":float(child_candidate.metadata.get("rest_area_relative_error",-1)),
            "dual_geometry_skin_support_required":bool(child_candidate.metadata.get("dual_geometry_skin_support_required")),
            "parent_state_mutated":False,
            "skin_weight_mutation":False,
        },
        "claim_boundary":[
            "This is an in-memory repair child court; it does not mutate the parent product run.",
            "Authorization is demo-child scoped and does not promote projected invalid rows to product teacher authority.",
            "The child partition is built only through mechanical_repartition_v2 and the child candidate only through holeless dense subdivision.",
            "A product/mainline mutation requires a separate promotion decision after this child court.",
        ],
    }

    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/"authorization.json").write_text(json.dumps(auth,indent=2,sort_keys=True)+"\n")
    (args.out_dir/"child_partition.json").write_text(json.dumps(child_partition.to_dict(),indent=2,sort_keys=True)+"\n")
    (args.out_dir/"child_candidate.json").write_text(json.dumps(child_candidate.to_dict(),indent=2,sort_keys=True)+"\n")
    (args.out_dir/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_ARACHNE_STAGE17_STAGE18_CHILD="+json.dumps({
        "arm":args.arm,
        "components":len(child_partition.components),
        "separate":report["child"]["separate_boundary_count"],
        "vertices":len(child_candidate.vertices),
        "faces":len(child_candidate.faces),
        "mixed_source_faces":child_candidate.metadata.get("mixed_source_face_count"),
        "g3b_unsafe":report["post_repair_g3b"]["unsafe_face_count"],
        "stress_unsafe":report["post_repair_synthetic_stress"]["unsafe_face_count"],
        "motion_gt10":report["post_repair_actual_motion"]["max_edge_gt_10"],
        "motion_gt4":report["post_repair_actual_motion"]["max_edge_gt_4"],
        "motion_worst":report["post_repair_actual_motion"]["worst_edge_max"],
        "face_deletion":report["invariants"]["face_deletion_count"],
        "area_error":report["invariants"]["rest_area_relative_error"],
    },sort_keys=True))


if __name__=="__main__":
    main()
