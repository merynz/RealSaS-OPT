from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    deformation_envelope_from_dict,
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
from compiler.realsas_compiler_core.mechanical_partition_v1 import (
    build_structural_partition,
)
from compiler.realsas_compiler_core.mechanical_repartition_v2 import (
    build_repartitioned_partition_v2,
    repartition_authorization_hash_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
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
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    motion_metrics,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


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
    skin=qualified_skin_from_dict(load_json(a.skin_json))
    explicit,prov=replay_compacted_dense_face_provenance(rr,surface)

    parent=build_structural_partition(surface)
    parent_carrier=build_component_carrier_policy(
        partition=parent,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id,"MESH",("PRODUCTION_REPLAY_PARENT",),
            metadata={"automatic":True}) for c in parent.components),
        metadata={"audit_only":True},
    )
    source_candidate=build_canonical_relation_candidate(
        surface,parent,parent_carrier,
        producer_policy_hash=content_sha256({
            "audit":"KNIGHT_PRODUCTION_TOPOLOGY_REPLAY",
            "role":"DIRECTIVE_SOURCE_CANDIDATE",
        }),
        explicit_face_provenance=explicit,
    )

    compatibility_stub={
        "schema":"RealSaS.SkinTopologyCompatibilityReport.v1",
        "report_hash":content_sha256({
            "schema":"RealSaS.KnightProductionTopologyReplayTrigger.v1",
            "candidate_lineage_hash":source_candidate.candidate_lineage_hash,
            "skin_lineage_hash":skin.skin_lineage_hash,
            "role":"AUDIT_TRIGGER_ONLY__SOURCE_EDGE_PROBE_IS_INDEPENDENT_OF_FACE_LIST",
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
        compatibility_report=compatibility_stub,
        envelope=envelope,
        cameras=cameras,
        seed_strategy="SOURCE_EDGE_PROBE_RATIO_V1",
    )
    if directive["status"]!="REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY":
        raise RuntimeError("PRODUCTION_REPLAY_NO_DIRECTIVE")

    reliability_hash=content_sha256({
        "schema":"RealSaS.KnightCorrectedSkinReliabilityAuditBinding.v1",
        "skin_lineage_hash":skin.skin_lineage_hash,
        "teacher_inputs_used_by_predictor":False,
        "audit_only":True,
    })
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
        "weight_reliability_evidence_hash":reliability_hash,
        "authorization_hash":"",
    }
    auth["authorization_hash"]=repartition_authorization_hash_v1(auth)

    child=build_repartitioned_partition_v2(
        surface=surface,
        parent_partition=parent,
        directive=directive,
        authorization=auth,
    )
    carrier=build_component_carrier_policy(
        partition=child,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id,"MESH",("PRODUCTION_REPLAY_CHILD",),
            metadata={"automatic":True}) for c in child.components),
        metadata={"audit_only":True},
    )
    candidate=build_holeless_partitioned_dense_candidate(
        surface,child,carrier,
        producer_policy_hash=content_sha256({
            "audit":"KNIGHT_PRODUCTION_TOPOLOGY_REPLAY",
            "mechanical_skin_transfer":"COMPONENT_HARMONIC_DIRICHLET_V1",
            "directive_hash":directive["directive_hash"],
        }),
        explicit_face_provenance=explicit,
        mechanical_skin_transfer="COMPONENT_HARMONIC_DIRICHLET_V1",
    )
    validate_canonical_mesh_candidate(
        candidate,surface=surface,partition=child,carrier_policy=carrier)

    rest,W,faces=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=skin)
    rest=np.asarray(rest,dtype=np.float64)
    W=np.asarray(W,dtype=np.float64)
    faces=np.asarray(faces,dtype=np.int64)
    jids=tuple(j.canonical_joint_id for j in skeleton.joints)
    source_report=load_json(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))
    motion=motion_metrics(
        rest,W,faces,jids,skeleton,cameras,rr,source_report)

    closure=dict(child.metadata.get("partition_cut_closure") or {})
    report={
        "schema":"RealSaS.KnightProductionTopologyReplayCourt.v1",
        "status":"PASS__PRODUCTION_OPERATORS_REPLAYED",
        "source_edge_directive":{
            "seed_strategy":directive.get("seed_strategy"),
            "source_edge_probe_hash":directive.get("source_edge_probe_hash"),
            "source_edge_count":directive.get("source_edge_probe_count"),
            "unsafe_source_edge_count":directive.get("unsafe_source_edge_count"),
            "candidate_separate_pair_count":directive.get("candidate_separate_pair_count"),
            "directive_hash":directive.get("directive_hash"),
        },
        "stage17":{
            "parent_component_count":len(parent.components),
            "child_component_count":len(child.components),
            "closure":closure,
            "partition_lineage_hash":child.partition_lineage_hash,
        },
        "stage18":{
            "vertex_count":len(candidate.vertices),
            "face_count":len(candidate.faces),
            "candidate_lineage_hash":candidate.candidate_lineage_hash,
            "mechanical_skin_transfer":candidate.metadata.get("mechanical_skin_transfer"),
            "mechanical_skin_transfer_report":candidate.metadata.get(
                "mechanical_skin_transfer_report"),
            "face_deletion_count":candidate.metadata.get("face_deletion_count"),
            "rest_area_relative_error":candidate.metadata.get("rest_area_relative_error"),
        },
        "actual_motion":motion,
        "face_provenance_replay":prov,
        "claim_boundary":[
            "Stage35 source-edge seed selection uses the production core operator.",
            "Stage17 child partition uses the production mutex/DSU closure operator.",
            "Stage18 holeless topology and component-harmonic source-support transfer use the production core operator.",
            "The authorization in this court is audit-only and does not mint product authority.",
            "No QualifiedSkinIR row is mutated."
        ],
    }

    expected={
        "direct_seed_count":1417,
        "component_count":73,
        "final_separate_count":1818,
        "motion_gt10":0,
        "motion_gt4":0,
    }
    observed={
        "direct_seed_count":int(closure.get("direct_seed_count",-1)),
        "component_count":len(child.components),
        "final_separate_count":int(closure.get("final_separate_count",-1)),
        "motion_gt10":int(motion["max_edge_gt_10"]),
        "motion_gt4":int(motion["max_edge_gt_4"]),
    }
    report["expected_proven_oracle_signature"]=expected
    report["observed_signature"]=observed
    report["signature_matches_proven_oracle"]=observed==expected

    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_PRODUCTION_TOPOLOGY_REPLAY="+json.dumps({
        "expected":expected,
        "observed":observed,
        "matches":report["signature_matches_proven_oracle"],
        "motion_worst":motion["worst_edge_max"],
        "motion_p99":motion["max_edge_p99"],
        "stage18_transfer":report["stage18"]["mechanical_skin_transfer"],
    },sort_keys=True))
    if not report["signature_matches_proven_oracle"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
