from __future__ import annotations

import argparse, json
from pathlib import Path
from time import perf_counter
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_canonical_relation_candidate,
    build_holeless_partitioned_dense_candidate,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.mechanical_repartition_v2 import (
    build_repartitioned_partition_v2,
    repartition_authorization_hash_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    propose_mechanical_repartition_directive_v2,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    validate_canonical_mesh_candidate,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import (
    load_json,
    replay_compacted_dense_face_provenance,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import motion_metrics
from tools.demo.render_knight_motion_preview_v1 import _ctx


def build_candidate(rr, surface, skeleton, cameras, envelope, skin, explicit):
    parent=build_structural_partition(surface)
    parent_carrier=build_component_carrier_policy(
        partition=parent,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id,"MESH",("PRODUCTION_G3_PARENT",),
            metadata={"automatic":True}) for c in parent.components),
        metadata={"audit_only":True},
    )
    source_candidate=build_canonical_relation_candidate(
        surface,parent,parent_carrier,
        producer_policy_hash=content_sha256({"audit":"PRODUCTION_G3_SOURCE"}),
        explicit_face_provenance=explicit,
    )
    compatibility={
        "schema":"RealSaS.SkinTopologyCompatibilityReport.v1",
        "report_hash":content_sha256({
            "audit":"PRODUCTION_G3_TRIGGER",
            "candidate":source_candidate.candidate_lineage_hash,
            "skin":skin.skin_lineage_hash,
        }),
        "unsafe_face_count":1,
        "unsafe_face_indices":(0,),
    }
    directive=propose_mechanical_repartition_directive_v2(
        source_candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=parent,
        compatibility_report=compatibility,
        envelope=envelope,
        cameras=cameras,
        seed_strategy="SOURCE_EDGE_PROBE_RATIO_V1",
    )
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
        "weight_reliability_evidence_hash":content_sha256({
            "audit":"CORRECTED_SKIN_RELIABILITY_BINDING",
            "skin":skin.skin_lineage_hash,
        }),
        "authorization_hash":"",
    }
    auth["authorization_hash"]=repartition_authorization_hash_v1(auth)
    child=build_repartitioned_partition_v2(
        surface=surface,parent_partition=parent,directive=directive,authorization=auth)
    carrier=build_component_carrier_policy(
        partition=child,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id,"MESH",("PRODUCTION_G3_CHILD",),
            metadata={"automatic":True}) for c in child.components),
        metadata={"audit_only":True},
    )
    candidate=build_holeless_partitioned_dense_candidate(
        surface,child,carrier,
        producer_policy_hash=content_sha256({
            "audit":"PRODUCTION_G3_STAGE18",
            "directive":directive["directive_hash"],
            "transfer":"COMPONENT_HARMONIC_DIRICHLET_V1",
        }),
        explicit_face_provenance=explicit,
        mechanical_skin_transfer="COMPONENT_HARMONIC_DIRICHLET_V1",
    )
    validate_canonical_mesh_candidate(
        candidate,surface=surface,partition=child,carrier_policy=carrier)
    return parent,directive,child,candidate


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--skin-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    surface=rigging_surface_from_dict(load_json(
        rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    skeleton=qualified_skeleton_from_dict(load_json(
        rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"))
    cameras=tuple(sorted(qualified_camera_set_from_dict(load_json(
        rr/"artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json"
    )).cameras,key=lambda x:int(x.view_index)))
    envelope=deformation_envelope_from_dict(load_json(
        rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json"))
    policy=mesh_policy_from_dict(load_json(
        rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    skin=qualified_skin_from_dict(load_json(a.skin_json))
    explicit,prov=replay_compacted_dense_face_provenance(rr,surface)

    build_start=perf_counter()
    parent,directive,child,candidate=build_candidate(
        rr,surface,skeleton,cameras,envelope,skin,explicit)
    build_seconds=perf_counter()-build_start

    g3_start=perf_counter()
    g3=run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    g3_seconds=perf_counter()-g3_start

    rest,W,faces=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=skin)
    source_report=load_json(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))
    motion_start=perf_counter()
    motion=motion_metrics(
        np.asarray(rest,dtype=np.float64),
        np.asarray(W,dtype=np.float64),
        np.asarray(faces,dtype=np.int64),
        tuple(j.canonical_joint_id for j in skeleton.joints),
        skeleton,cameras,rr,source_report)
    motion_seconds=perf_counter()-motion_start

    closure=dict(child.metadata.get("partition_cut_closure") or {})
    report={
        "schema":"RealSaS.KnightProductionG3MotionCourt.v1",
        "status":(
            "PASS__PRODUCTION_MECHANICAL_QUALIFICATION"
            if g3.passed and motion["max_edge_gt_4"]==0 and motion["max_edge_gt_10"]==0
            else "FAIL__PRODUCTION_MECHANICAL_QUALIFICATION"
        ),
        "stage35_source_edge":{
            "unsafe_source_edge_count":directive.get("unsafe_source_edge_count"),
            "candidate_separate_pair_count":directive.get("candidate_separate_pair_count"),
            "source_edge_probe_hash":directive.get("source_edge_probe_hash"),
        },
        "stage17":{
            "component_count":len(child.components),
            "closure":closure,
        },
        "stage18":{
            "vertex_count":len(candidate.vertices),
            "face_count":len(candidate.faces),
            "mechanical_skin_transfer":candidate.metadata.get("mechanical_skin_transfer"),
            "candidate_lineage_hash":candidate.candidate_lineage_hash,
        },
        "g3":{
            "passed":bool(g3.passed),
            "failure_invariants":list(g3.failure_invariants),
            "probe_count":int(g3.probe_count),
            "face_count":int(g3.face_count),
            "minimum_area_ratio":float(g3.minimum_area_ratio),
            "maximum_area_ratio":float(g3.maximum_area_ratio),
            "maximum_condition_number":float(g3.maximum_condition_number),
            "minimum_edge_ratio":float(g3.minimum_edge_ratio),
            "maximum_edge_ratio":float(g3.maximum_edge_ratio),
            "report_hash":g3.report_hash,
        },
        "actual_motion":motion,
        "performance":{
            "build_stage35_17_18_seconds":float(build_seconds),
            "g3_seconds":float(g3_seconds),
            "actual_motion_seconds":float(motion_seconds),
            "total_measured_seconds":float(build_seconds+g3_seconds+motion_seconds),
        },
        "face_provenance_replay":prov,
        "claim_boundary":[
            "All topology, partition and seam-transfer operators are production core implementations.",
            "The trustworthy-skin authorization is audit-only; no product authority is minted.",
            "G3 uses the production vectorized local-frame operator.",
            "Actual motion uses the exact Knight idle/run/slash clip sampling court."
        ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_PRODUCTION_G3_MOTION="+json.dumps({
        "status":report["status"],
        "components":len(child.components),
        "direct_seeds":closure.get("direct_seed_count"),
        "g3_pass":g3.passed,
        "g3_failures":list(g3.failure_invariants),
        "g3_seconds":g3_seconds,
        "motion_gt10":motion["max_edge_gt_10"],
        "motion_gt4":motion["max_edge_gt_4"],
        "motion_worst":motion["worst_edge_max"],
        "motion_p99":motion["max_edge_p99"],
        "build_seconds":build_seconds,
        "motion_seconds":motion_seconds,
    },sort_keys=True))
    if report["status"]!="PASS__PRODUCTION_MECHANICAL_QUALIFICATION":
        raise SystemExit(2)


if __name__=="__main__":
    main()
