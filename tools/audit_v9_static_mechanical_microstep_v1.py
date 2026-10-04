"""One-operator checkpointed static/mechanical microstep court.

This is the short-running replacement for monolithic fixed-point static rounds.
It reconstructs or loads one exact candidate, runs exactly one static operator
behind the proposal-level mechanical guard, immediately checkpoints the trial,
then performs global production + all-face G3B non-regression before admitting it.

Audit only. No skin/partition mutation. No product authority minted.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_holeless_partitioned_dense_candidate,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_batched_collapse_v2 import (
    repair_candidate_endpoint_collapses_batched_v2,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_cavity_remesh_v1 import (
    repair_candidate_cavity_retriangulation_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_edge_cavity_remesh_v1 import (
    repair_candidate_edge_cavity_retriangulation_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    mechanical_quality_protected_surface_ids_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import (
    repair_candidate_fixed_vertex_flips_topology_safe_v2,
)
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import (
    derive_deformation_envelope_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_repartition_v2 import build_repartitioned_partition_v2
from compiler.realsas_compiler_core.mesh.mechanical_proposal_admissibility_v1 import (
    MechanicalProposalAdmissibilityGuardV1,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    validate_canonical_mesh_candidate,
    validate_mechanical_partition,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.audit_knight_repaired_quality_collapse_v1 import (
    build_repaired_surface,
    report_quality,
)
from tools.audit_v9_teacher_oracle_iterative_repartition_v1 import (
    authorization_for,
    carrier_for,
    propose_with_generic_fallback,
    read,
    sha256,
)
from tools.audit_v9_teacher_projection_oracle_v1 import teacher_to_skin
from tools.demo.render_knight_motion_preview_v1 import _ctx


OPS = ("flip", "collapse", "vertex_cavity", "edge_cavity")


def _compat(candidate, *, surface, skeleton, skin, envelope, cameras, policy, all_faces):
    return run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
        risk_l1_min=0.0 if all_faces else 0.5,
        stress_all_faces=bool(all_faces),
    )


def _severity(report, policy):
    rows=tuple(report.get("top_unsafe_faces") or ())
    if not rows:
        return 1.0
    out=1.0
    for row in rows:
        out=max(
            out,
            float(row["max_edge_ratio"])/4.0,
            float(row["max_condition_number"])/float(policy.g3_max_dynamic_condition_number),
            float(row["max_area_ratio"])/float(policy.g3_max_dynamic_area_ratio),
            float(policy.g3_min_dynamic_area_ratio)/max(float(row["min_area_ratio"]),1e-15),
        )
    return float(out)


def _nonregression(before, after, policy):
    ub=int(before["unsafe_face_count"])
    ua=int(after["unsafe_face_count"])
    if ua<ub:
        return True
    if ua>ub:
        return False
    return _severity(after,policy) <= _severity(before,policy)*(1.0+1e-9)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--surface-json",type=Path,required=True)
    ap.add_argument("--partition-json",type=Path,required=True)
    ap.add_argument("--candidate-json",type=Path)
    ap.add_argument("--fresh-skeleton-json",type=Path,required=True)
    ap.add_argument("--teacher-bank",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--operator",choices=OPS,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--cycle-tag",default="MICROSTEP")
    ap.add_argument("--repartition-first",action="store_true")
    ap.add_argument("--repartition-iteration",type=int,default=1)
    a=ap.parse_args()

    t0=time.perf_counter()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    ctx=_ctx(a.authority_root,a.run_id)
    rr=ctx["run_root"]
    surface=rigging_surface_from_dict(read(a.surface_json))
    partition=mechanical_partition_from_dict(read(a.partition_json))
    validate_mechanical_partition(partition,surface)
    skeleton=qualified_skeleton_from_dict(read(a.fresh_skeleton_json))
    camera_set=qualified_camera_set_from_dict(
        stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    cameras=tuple(sorted(camera_set.cameras,key=lambda x:int(x.view_index)))
    _,envelope=derive_deformation_envelope_v1(skeleton=skeleton,camera_set=camera_set)
    policy=mesh_policy_from_dict(
        stage_output_payload(
            ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1"
        )
    )
    teacher_skin,_,valid,_,_,_=teacher_to_skin(surface,skeleton,a.teacher_bank)
    rebuilt_surface,explicit_faces=build_repaired_surface(rr,a.inverse_npz)
    if rebuilt_surface.geometry_lineage_hash!=surface.geometry_lineage_hash:
        raise RuntimeError("MICROSTEP_REBUILT_SURFACE_LINEAGE_DRIFT")

    if a.candidate_json is not None:
        candidate=canonical_mesh_candidate_from_dict(read(a.candidate_json))
        candidate_source="EXPLICIT_CANDIDATE"
        carrier=None
    else:
        carrier=carrier_for(partition,99)
        candidate=build_holeless_partitioned_dense_candidate(
            surface,partition,carrier,
            producer_policy_hash=content_sha256({
                "schema":"RealSaS.V9StaticMechanicalMicrostepRebuildPolicy.v1",
                "partition":partition.partition_lineage_hash,
                "cycle_tag":str(a.cycle_tag),
            }),
            explicit_face_provenance=explicit_faces,
            mechanical_skin_transfer="COMPONENT_HARMONIC_DIRICHLET_V1",
        )
        validate_canonical_mesh_candidate(
            candidate,surface=surface,partition=partition,carrier_policy=carrier
        )
        candidate_source="REBUILT_FROM_PARTITION"

    repartition_report=None
    if a.repartition_first:
        if a.candidate_json is None:
            raise RuntimeError("MICROSTEP_REPARTITION_REQUIRES_EXPLICIT_PARENT_CANDIDATE")
        parent_compat=_compat(
            candidate,surface=surface,skeleton=skeleton,skin=teacher_skin,
            envelope=envelope,cameras=cameras,policy=policy,all_faces=False,
        )
        directive,strategy=propose_with_generic_fallback(
            candidate=candidate,
            surface=surface,
            skeleton=skeleton,
            skin=teacher_skin,
            partition=partition,
            compatibility=parent_compat,
            envelope=envelope,
            cameras=cameras,
        )
        if (
            directive.get("status")!="REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY"
            or int(directive.get("candidate_separate_pair_count") or 0)<=0
        ):
            raise RuntimeError("MICROSTEP_REPARTITION_NO_QUALIFIED_DIRECTIVE")
        evidence={
            "schema":"RealSaS.V9TeacherOracleMechanicalStaticFixedPointEvidence.v1",
            "teacher_bank_sha256":sha256(a.teacher_bank),
            "teacher_clean_row_count":int(np.count_nonzero(valid)),
            "teacher_row_count":int(len(valid)),
            "teacher_coverage":float(np.mean(valid)),
            "surface_lineage_hash":surface.geometry_lineage_hash,
            "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
            "product_authority_claimed":False,
        }
        evidence_hash=content_sha256(evidence)
        auth=authorization_for(
            directive=directive,
            surface=surface,
            partition=partition,
            skeleton=skeleton,
            skin=teacher_skin,
            evidence_hash=evidence_hash,
            iteration=int(a.repartition_iteration),
        )
        child_partition=build_repartitioned_partition_v2(
            surface=surface,
            parent_partition=partition,
            directive=directive,
            authorization=auth,
        )
        validate_mechanical_partition(child_partition,surface)
        child_carrier=carrier_for(child_partition,int(a.repartition_iteration))
        child=build_holeless_partitioned_dense_candidate(
            surface,
            child_partition,
            child_carrier,
            producer_policy_hash=content_sha256({
                "schema":"RealSaS.V9MechanicalStaticFixedPointChildPolicy.v1",
                "cycle":int(a.repartition_iteration),
                "parent_candidate":candidate.candidate_lineage_hash,
                "parent_partition":partition.partition_lineage_hash,
                "directive":directive["directive_hash"],
                "authorization":auth["authorization_hash"],
                "teacher_evidence":evidence_hash,
            }),
            explicit_face_provenance=explicit_faces,
            mechanical_skin_transfer="COMPONENT_HARMONIC_DIRICHLET_V1",
        )
        validate_canonical_mesh_candidate(
            child,surface=surface,partition=child_partition,carrier_policy=child_carrier
        )
        child_compat=_compat(
            child,surface=surface,skeleton=skeleton,skin=teacher_skin,
            envelope=envelope,cameras=cameras,policy=policy,all_faces=False,
        )
        if int(child_compat["unsafe_face_count"])>=int(parent_compat["unsafe_face_count"]):
            raise RuntimeError(
                "MICROSTEP_REPARTITION_DID_NOT_IMPROVE_G3B:"
                f"{parent_compat['unsafe_face_count']}->{child_compat['unsafe_face_count']}"
            )
        repartition_report={
            "strategy":strategy,
            "parent_g3b_unsafe":int(parent_compat["unsafe_face_count"]),
            "child_g3b_unsafe":int(child_compat["unsafe_face_count"]),
            "candidate_separate_pair_count":int(directive["candidate_separate_pair_count"]),
            "parent_partition_hash":partition.partition_lineage_hash,
            "child_partition_hash":child_partition.partition_lineage_hash,
            "parent_candidate_hash":candidate.candidate_lineage_hash,
            "child_candidate_hash":child.candidate_lineage_hash,
            "iteration":int(a.repartition_iteration),
        }
        (a.out_dir/"REPARTITION_REPORT.json").write_text(
            json.dumps(repartition_report,indent=2,sort_keys=True)+"\n"
        )
        partition=child_partition
        candidate=child
        candidate_source="REPARTITIONED_FROM_EXPLICIT_PARENT"
        print("V9_MICROSTEP_REPARTITION="+json.dumps(repartition_report,sort_keys=True),flush=True)

    (a.out_dir/"INPUT_PARTITION.json").write_text(
        json.dumps(partition.to_dict(),indent=2,sort_keys=True)+"\n"
    )
    (a.out_dir/"INPUT_CANDIDATE.json").write_text(
        json.dumps(candidate.to_dict(),indent=2,sort_keys=True)+"\n"
    )

    q0=report_quality(candidate,policy)
    p0=_compat(
        candidate,surface=surface,skeleton=skeleton,skin=teacher_skin,
        envelope=envelope,cameras=cameras,policy=policy,all_faces=False,
    )
    a0=_compat(
        candidate,surface=surface,skeleton=skeleton,skin=teacher_skin,
        envelope=envelope,cameras=cameras,policy=policy,all_faces=True,
    )
    baseline={
        "static_violations":int(q0["policy_violating_face_count"]),
        "g3b_unsafe":int(p0["unsafe_face_count"]),
        "all_face_g3b_unsafe":int(a0["unsafe_face_count"]),
        "g3b_severity":_severity(p0,policy),
        "all_face_g3b_severity":_severity(a0,policy),
        "vertices":len(candidate.vertices),
        "faces":len(candidate.faces),
    }
    (a.out_dir/"BASELINE.json").write_text(json.dumps(baseline,indent=2,sort_keys=True)+"\n")
    print("V9_MICROSTEP_BASELINE="+json.dumps(baseline,sort_keys=True),flush=True)

    protected=mechanical_quality_protected_surface_ids_v1(partition)
    tg=time.perf_counter()
    guard=MechanicalProposalAdmissibilityGuardV1(
        candidate,surface=surface,skeleton=skeleton,skin=teacher_skin,
        envelope=envelope,cameras=cameras,policy=policy,
    )
    guard_seconds=time.perf_counter()-tg

    to=time.perf_counter()
    if a.operator=="flip":
        trial,op_report=repair_candidate_fixed_vertex_flips_topology_safe_v2(
            candidate,policy,max_passes=2,proposal_admissibility=guard
        )
    elif a.operator=="collapse":
        trial,op_report=repair_candidate_endpoint_collapses_batched_v2(
            candidate,policy,max_batches=6,max_collapses=192,
            proposal_admissibility=guard,
        )
    elif a.operator=="vertex_cavity":
        trial,op_report=repair_candidate_cavity_retriangulation_v1(
            candidate,policy,protected_surface_ids=protected,
            max_batches=6,max_removed_vertices=192,
            proposal_admissibility=guard,
        )
    else:
        trial,op_report=repair_candidate_edge_cavity_retriangulation_v1(
            candidate,policy,protected_surface_ids=protected,
            max_batches=6,max_removed_edges=192,
            proposal_admissibility=guard,
        )
    operator_seconds=time.perf_counter()-to

    (a.out_dir/"TRIAL_CANDIDATE.json").write_text(
        json.dumps(trial.to_dict(),indent=2,sort_keys=True)+"\n"
    )
    (a.out_dir/"OPERATOR_REPORT.json").write_text(
        json.dumps(op_report,indent=2,sort_keys=True,default=str)+"\n"
    )

    tv=time.perf_counter()
    q1=report_quality(trial,policy)
    p1=_compat(
        trial,surface=surface,skeleton=skeleton,skin=teacher_skin,
        envelope=envelope,cameras=cameras,policy=policy,all_faces=False,
    )
    a1=_compat(
        trial,surface=surface,skeleton=skeleton,skin=teacher_skin,
        envelope=envelope,cameras=cameras,policy=policy,all_faces=True,
    )
    validation_seconds=time.perf_counter()-tv

    static_improved=int(q1["policy_violating_face_count"]) < int(q0["policy_violating_face_count"])
    prod_ok=_nonregression(p0,p1,policy)
    all_ok=_nonregression(a0,a1,policy)
    accepted=bool(static_improved and prod_ok and all_ok)
    final=trial if accepted else candidate
    final_metrics={
        "static_violations":int((q1 if accepted else q0)["policy_violating_face_count"]),
        "g3b_unsafe":int((p1 if accepted else p0)["unsafe_face_count"]),
        "all_face_g3b_unsafe":int((a1 if accepted else a0)["unsafe_face_count"]),
    }

    (a.out_dir/"FINAL_CANDIDATE.json").write_text(
        json.dumps(final.to_dict(),indent=2,sort_keys=True)+"\n"
    )
    report={
        "schema":"RealSaS.V9StaticMechanicalMicrostep.v1",
        "status":"COMPLETE__AUDIT_ONLY",
        "product_authority_minted":False,
        "skin_weight_mutation":False,
        "partition_mutation":False,
        "candidate_source":candidate_source,
        "repartition_first":bool(a.repartition_first),
        "repartition_report":repartition_report,
        "operator":a.operator,
        "cycle_tag":str(a.cycle_tag),
        "baseline":baseline,
        "trial":{
            "static_violations":int(q1["policy_violating_face_count"]),
            "g3b_unsafe":int(p1["unsafe_face_count"]),
            "all_face_g3b_unsafe":int(a1["unsafe_face_count"]),
            "g3b_severity":_severity(p1,policy),
            "all_face_g3b_severity":_severity(a1,policy),
            "vertices":len(trial.vertices),
            "faces":len(trial.faces),
        },
        "decision":{
            "static_improved":bool(static_improved),
            "production_g3b_nonregression":bool(prod_ok),
            "all_face_g3b_nonregression":bool(all_ok),
            "accepted":accepted,
        },
        "final":final_metrics,
        "timing_seconds":{
            "guard_build":float(guard_seconds),
            "operator":float(operator_seconds),
            "global_validation":float(validation_seconds),
            "total":float(time.perf_counter()-t0),
        },
        "operator_report":op_report,
    }
    (a.out_dir/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+"\n")
    print("V9_MICROSTEP_RESULT="+json.dumps({
        "operator":a.operator,
        "accepted":accepted,
        "baseline":baseline,
        "trial":report["trial"],
        "final":final_metrics,
        "timing_seconds":report["timing_seconds"],
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
