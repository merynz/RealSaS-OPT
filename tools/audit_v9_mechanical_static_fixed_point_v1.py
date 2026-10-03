"""Closed-loop V9 mechanical repartition x constrained static-quality court.

Audit only.  This composes the two previously separate repair domains:
  G3B -> Stage17 repartition -> Stage18 holeless rebuild
      -> proposal-level mechanically guarded static repair
      -> global G3B non-regression -> repeat.

No skin weights are mutated and no product authority is minted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
from compiler.realsas_compiler_core.mechanical_repartition_v2 import (
    build_repartitioned_partition_v2,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
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
from tools.audit_direct_lbs_mesh_weight_projection_v1 import exact_motion_court
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


MAX_MECHANICAL_CYCLES = 8
MAX_STATIC_ROUNDS_PER_CYCLE = 2


def _compat(candidate, *, surface, skeleton, skin, envelope, cameras, policy, all_faces=False):
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


def _guarded_static_rounds(
    candidate,
    *,
    partition,
    surface,
    skeleton,
    skin,
    envelope,
    cameras,
    policy,
):
    start_quality=report_quality(candidate,policy)
    start_prod=_compat(
        candidate,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy,all_faces=False,
    )
    start_all=_compat(
        candidate,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy,all_faces=True,
    )
    protected=mechanical_quality_protected_surface_ids_v1(partition)
    rows=[]
    current=candidate

    for round_index in range(MAX_STATIC_ROUNDS_PER_CYCLE):
        before=report_quality(current,policy)
        guard=MechanicalProposalAdmissibilityGuardV1(
            current,
            surface=surface,
            skeleton=skeleton,
            skin=skin,
            envelope=envelope,
            cameras=cameras,
            policy=policy,
        )
        op_rows=[]

        nxt,rep=repair_candidate_fixed_vertex_flips_topology_safe_v2(
            current,policy,max_passes=6,proposal_admissibility=guard,
        )
        op_rows.append({
            "operator":"flip",
            "accepted":int(rep["accepted_flip_count"]),
            "rejected_mechanical":int(rep["rejected_mechanical_admissibility_count"]),
            "violations_after":int(report_quality(nxt,policy)["policy_violating_face_count"]),
        })
        current=nxt

        # Rebuild the guard after topology changes. Vertex positions/supports are
        # unchanged, but rebuilding binds the predicate to the exact current lineage.
        guard=MechanicalProposalAdmissibilityGuardV1(
            current,surface=surface,skeleton=skeleton,skin=skin,
            envelope=envelope,cameras=cameras,policy=policy,
        )
        nxt,rep=repair_candidate_endpoint_collapses_batched_v2(
            current,policy,max_batches=24,max_collapses=768,
            proposal_admissibility=guard,
        )
        op_rows.append({
            "operator":"collapse",
            "accepted":int(rep["accepted_collapse_count"]),
            "rejected_mechanical":int(rep["rejected_mechanical_admissibility_count"]),
            "violations_after":int(report_quality(nxt,policy)["policy_violating_face_count"]),
        })
        current=nxt

        guard=MechanicalProposalAdmissibilityGuardV1(
            current,surface=surface,skeleton=skeleton,skin=skin,
            envelope=envelope,cameras=cameras,policy=policy,
        )
        nxt,rep=repair_candidate_cavity_retriangulation_v1(
            current,policy,protected_surface_ids=protected,
            max_batches=24,max_removed_vertices=768,
            proposal_admissibility=guard,
        )
        op_rows.append({
            "operator":"vertex_cavity",
            "accepted":int(rep["accepted_cavity_count"]),
            "rejected_mechanical":int(rep["rejected_counts"].get("mechanical_admissibility",0)),
            "violations_after":int(report_quality(nxt,policy)["policy_violating_face_count"]),
        })
        current=nxt

        guard=MechanicalProposalAdmissibilityGuardV1(
            current,surface=surface,skeleton=skeleton,skin=skin,
            envelope=envelope,cameras=cameras,policy=policy,
        )
        nxt,rep=repair_candidate_edge_cavity_retriangulation_v1(
            current,policy,protected_surface_ids=protected,
            max_batches=24,max_removed_edges=768,
            proposal_admissibility=guard,
        )
        op_rows.append({
            "operator":"edge_cavity",
            "accepted":int(rep["accepted_edge_cavity_count"]),
            "rejected_mechanical":int(rep["rejected_counts"].get("mechanical_admissibility",0)),
            "violations_after":int(report_quality(nxt,policy)["policy_violating_face_count"]),
        })
        current=nxt

        after=report_quality(current,policy)
        row={
            "round":round_index,
            "before_violations":int(before["policy_violating_face_count"]),
            "after_violations":int(after["policy_violating_face_count"]),
            "operators":op_rows,
        }
        rows.append(row)
        print("V9_FIXED_POINT_STATIC_ROUND="+json.dumps(row,sort_keys=True),flush=True)
        if int(after["policy_violating_face_count"])>=int(before["policy_violating_face_count"]):
            break
        if int(after["policy_violating_face_count"])==0:
            break

    final_quality=report_quality(current,policy)
    final_prod=_compat(
        current,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy,all_faces=False,
    )
    final_all=_compat(
        current,surface=surface,skeleton=skeleton,skin=skin,
        envelope=envelope,cameras=cameras,policy=policy,all_faces=True,
    )
    if int(final_quality["policy_violating_face_count"])>int(start_quality["policy_violating_face_count"]):
        raise RuntimeError("V9_FIXED_POINT_STATIC_QUALITY_REGRESSION")
    if int(final_prod["unsafe_face_count"])>int(start_prod["unsafe_face_count"]):
        raise RuntimeError("V9_FIXED_POINT_PRODUCTION_G3B_REGRESSION")
    if int(final_all["unsafe_face_count"])>int(start_all["unsafe_face_count"]):
        raise RuntimeError("V9_FIXED_POINT_ALL_FACE_G3B_REGRESSION")
    return current,{
        "start_static":int(start_quality["policy_violating_face_count"]),
        "final_static":int(final_quality["policy_violating_face_count"]),
        "start_g3b_unsafe":int(start_prod["unsafe_face_count"]),
        "final_g3b_unsafe":int(final_prod["unsafe_face_count"]),
        "start_all_face_g3b_unsafe":int(start_all["unsafe_face_count"]),
        "final_all_face_g3b_unsafe":int(final_all["unsafe_face_count"]),
        "rounds":rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--surface-json",type=Path,required=True)
    ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--partition-json",type=Path,required=True)
    ap.add_argument("--fresh-skeleton-json",type=Path,required=True)
    ap.add_argument("--teacher-bank",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()

    ctx=_ctx(a.authority_root,a.run_id)
    rr=ctx["run_root"]
    surface=rigging_surface_from_dict(read(a.surface_json))
    partition=mechanical_partition_from_dict(read(a.partition_json))
    candidate=canonical_mesh_candidate_from_dict(read(a.candidate_json))
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
    rebuilt_surface,explicit_faces=build_repaired_surface(rr,a.inverse_npz)
    if rebuilt_surface.geometry_lineage_hash!=surface.geometry_lineage_hash:
        raise RuntimeError("V9_FIXED_POINT_REBUILT_SURFACE_LINEAGE_DRIFT")
    validate_mechanical_partition(partition,surface)

    teacher_skin,_,valid,_,_,_=teacher_to_skin(surface,skeleton,a.teacher_bank)
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

    a.out_dir.mkdir(parents=True,exist_ok=True)
    history=[]

    for cycle in range(MAX_MECHANICAL_CYCLES+1):
        quality=report_quality(candidate,policy)
        compat=_compat(
            candidate,surface=surface,skeleton=skeleton,skin=teacher_skin,
            envelope=envelope,cameras=cameras,policy=policy,all_faces=False,
        )
        all_compat=_compat(
            candidate,surface=surface,skeleton=skeleton,skin=teacher_skin,
            envelope=envelope,cameras=cameras,policy=policy,all_faces=True,
        )
        row={
            "cycle":cycle,
            "partition_hash":partition.partition_lineage_hash,
            "candidate_hash":candidate.candidate_lineage_hash,
            "components":len(partition.components),
            "separate":sum(x.decision=="SEPARATE" for x in partition.boundary_constraints),
            "vertices":len(candidate.vertices),
            "faces":len(candidate.faces),
            "static_violations":int(quality["policy_violating_face_count"]),
            "g3b_unsafe":int(compat["unsafe_face_count"]),
            "all_face_g3b_unsafe":int(all_compat["unsafe_face_count"]),
        }
        print("V9_FIXED_POINT_EVAL="+json.dumps(row,sort_keys=True),flush=True)

        if bool(compat["passed"]) and int(quality["policy_violating_face_count"])==0:
            row["action"]="STATIC_AND_G3B_CLOSED"
            history.append(row)
            break
        if cycle>=MAX_MECHANICAL_CYCLES:
            row["action"]="CYCLE_BUDGET_EXHAUSTED"
            history.append(row)
            break

        # If mechanical incompatibility remains, improve the partition first.
        if not bool(compat["passed"]):
            directive,strategy=propose_with_generic_fallback(
                candidate=candidate,surface=surface,skeleton=skeleton,skin=teacher_skin,
                partition=partition,compatibility=compat,envelope=envelope,cameras=cameras,
            )
            if (
                directive.get("status")!="REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY"
                or int(directive.get("candidate_separate_pair_count") or 0)<=0
            ):
                row["action"]="ABSTAIN_NO_NEW_REPARTITION"
                row["strategy"]=strategy
                history.append(row)
                break
            auth=authorization_for(
                directive=directive,surface=surface,partition=partition,skeleton=skeleton,
                skin=teacher_skin,evidence_hash=evidence_hash,iteration=cycle+1,
            )
            child_partition=build_repartitioned_partition_v2(
                surface=surface,parent_partition=partition,directive=directive,authorization=auth,
            )
            validate_mechanical_partition(child_partition,surface)
            carrier=carrier_for(child_partition,cycle+1)
            child=build_holeless_partitioned_dense_candidate(
                surface,child_partition,carrier,
                producer_policy_hash=content_sha256({
                    "schema":"RealSaS.V9MechanicalStaticFixedPointChildPolicy.v1",
                    "cycle":cycle+1,
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
                child,surface=surface,partition=child_partition,carrier_policy=carrier,
            )
            child_compat=_compat(
                child,surface=surface,skeleton=skeleton,skin=teacher_skin,
                envelope=envelope,cameras=cameras,policy=policy,all_faces=False,
            )
            if int(child_compat["unsafe_face_count"])>=int(compat["unsafe_face_count"]):
                raise RuntimeError(
                    "V9_FIXED_POINT_REPARTITION_DID_NOT_IMPROVE_G3B:"
                    f"{compat['unsafe_face_count']}->{child_compat['unsafe_face_count']}"
                )
            row.update({
                "action":"REPARTITION_THEN_GUARDED_STATIC",
                "strategy":strategy,
                "candidate_separate_pair_count":int(directive["candidate_separate_pair_count"]),
                "child_components":len(child_partition.components),
                "child_g3b_unsafe_before_static":int(child_compat["unsafe_face_count"]),
            })
            partition=child_partition
            candidate=child

        # Static optimization is allowed only inside the mechanical feasible region.
        candidate,static_report=_guarded_static_rounds(
            candidate,partition=partition,surface=surface,skeleton=skeleton,
            skin=teacher_skin,envelope=envelope,cameras=cameras,policy=policy,
        )
        row["static_composition"]=static_report
        history.append(row)

        idir=a.out_dir/f"cycle_{cycle+1:02d}"
        idir.mkdir(parents=True,exist_ok=True)
        (idir/"PARTITION.json").write_text(
            json.dumps(partition.to_dict(),indent=2,sort_keys=True)+"\n"
        )
        (idir/"CANDIDATE.json").write_text(
            json.dumps(candidate.to_dict(),indent=2,sort_keys=True)+"\n"
        )
        (idir/"STEP.json").write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")

    final_quality=report_quality(candidate,policy)
    final_compat=_compat(
        candidate,surface=surface,skeleton=skeleton,skin=teacher_skin,
        envelope=envelope,cameras=cameras,policy=policy,all_faces=False,
    )
    final_all=_compat(
        candidate,surface=surface,skeleton=skeleton,skin=teacher_skin,
        envelope=envelope,cameras=cameras,policy=policy,all_faces=True,
    )
    g3=run_g3_local_frame_micro_stress_v2(
        candidate,surface=surface,skeleton=skeleton,skin=teacher_skin,
        envelope=envelope,cameras=cameras,policy=policy,
    )
    rest,W,faces=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=teacher_skin
    )
    motion=exact_motion_court(
        ctx,np.asarray(rest,np.float64),np.asarray(W,np.float64),np.asarray(faces,np.int64),
        tuple(str(j.canonical_joint_id) for j in skeleton.joints),
        skeleton,cameras,policy,
    )
    static_close=bool(int(final_quality["policy_violating_face_count"])==0)
    mechanical_close=bool(
        final_compat["passed"] and final_all["passed"] and g3.passed
        and int(motion["failed_motion_frame_count"])==0
    )
    report={
        "schema":"RealSaS.V9MechanicalStaticFixedPointCourt.v1",
        "status":"COMPLETE__NO_PRODUCT_MUTATION",
        "product_authority_minted":False,
        "skin_weight_mutation":False,
        "teacher_oracle_only":True,
        "history":history,
        "final":{
            "components":len(partition.components),
            "separate":sum(x.decision=="SEPARATE" for x in partition.boundary_constraints),
            "vertices":len(candidate.vertices),
            "faces":len(candidate.faces),
            "static_violations":int(final_quality["policy_violating_face_count"]),
            "g3b_unsafe":int(final_compat["unsafe_face_count"]),
            "all_face_g3b_unsafe":int(final_all["unsafe_face_count"]),
            "g3_pass":bool(g3.passed),
            "g3_max_condition":float(g3.maximum_condition_number),
            "g3_min_area":float(g3.minimum_area_ratio),
            "g3_max_area":float(g3.maximum_area_ratio),
            "failed_motion_frames":int(motion["failed_motion_frame_count"]),
            "max_motion_edge":float(motion["maximum_motion_edge_ratio"]),
            "max_motion_condition":float(motion["maximum_motion_condition_number"]),
        },
        "decision":{
            "static_close":static_close,
            "mechanical_close":mechanical_close,
            "both_close":bool(static_close and mechanical_close),
            "a100_target_eligible":bool(static_close and mechanical_close),
        },
        "claim_boundary":[
            "Proposal-level static operators use a precomputed G3B-equivalent local mechanical non-regression guard.",
            "Every static batch is followed by global production and all-face G3B non-regression checks.",
            "Repartition must strictly reduce production G3B unsafe-face count.",
            "No skin weight is mutated and no product authority is minted.",
        ],
    }
    (a.out_dir/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (a.out_dir/"FINAL_PARTITION.json").write_text(json.dumps(partition.to_dict(),indent=2,sort_keys=True)+"\n")
    (a.out_dir/"FINAL_CANDIDATE.json").write_text(json.dumps(candidate.to_dict(),indent=2,sort_keys=True)+"\n")
    print("V9_FIXED_POINT_RESULT="+json.dumps({
        **report["final"],**report["decision"]
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
