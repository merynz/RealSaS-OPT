"""V9 teacher-oracle production repartition feedback court.

Diagnostic only. Tests whether the existing generic Stage35 -> Stage17 -> Stage18
feedback loop closes the topology/skin incompatibility exposed by the exact V9
teacher oracle, without mutating any QualifiedSkinIR weight.

Authority:
- current V9 refined RiggingSurface + parent partition + static candidate;
- exact V9 teacher bank reprojected from artist/source truth;
- production SOURCE_EDGE_PROBE_RATIO_V1 directive;
- production mechanical_repartition_v2 cut closure;
- production holeless Stage18 candidate with COMPONENT_HARMONIC_DIRICHLET_V1;
- production G3 plus independent 51-frame exact-motion court.
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
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import (
    derive_deformation_envelope_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_repartition_v2 import (
    build_repartitioned_partition_v2,
    repartition_authorization_hash_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    propose_mechanical_repartition_directive_v2,
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
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
from tools.audit_v9_teacher_projection_oracle_v1 import teacher_to_skin
from tools.demo.render_knight_motion_preview_v1 import _ctx


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--surface-json", type=Path, required=True)
    ap.add_argument("--candidate-json", type=Path, required=True)
    ap.add_argument("--partition-json", type=Path, required=True)
    ap.add_argument("--fresh-skeleton-json", type=Path, required=True)
    ap.add_argument("--teacher-bank", type=Path, required=True)
    ap.add_argument("--inverse-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    ctx = _ctx(a.authority_root, a.run_id)
    rr = ctx["run_root"]
    surface = rigging_surface_from_dict(read(a.surface_json))
    parent_candidate = canonical_mesh_candidate_from_dict(read(a.candidate_json))
    parent_partition = mechanical_partition_from_dict(read(a.partition_json))
    skeleton = qualified_skeleton_from_dict(read(a.fresh_skeleton_json))
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda x: int(x.view_index)))
    _, envelope = derive_deformation_envelope_v1(
        skeleton=skeleton, camera_set=camera_set
    )
    policy = mesh_policy_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.MeshQualificationPolicyIR.v1",
        )
    )

    rebuilt_surface, explicit_faces = build_repaired_surface(rr, a.inverse_npz)
    if rebuilt_surface.geometry_lineage_hash != surface.geometry_lineage_hash:
        raise RuntimeError(
            "V9_REPARTITION_REBUILT_SURFACE_DRIFT:"
            f"{rebuilt_surface.geometry_lineage_hash}:"
            f"{surface.geometry_lineage_hash}"
        )
    validate_mechanical_partition(parent_partition, surface)

    teacher_skin, _, valid, _, _, _ = teacher_to_skin(
        surface, skeleton, a.teacher_bank
    )

    # Parent measurement: this is the current V9 carrier that the teacher oracle
    # already showed to be mechanically incompatible.
    parent_g3b = run_skin_topology_compatibility_v1(
        parent_candidate,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    parent_g3 = run_g3_local_frame_micro_stress_v2(
        parent_candidate,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )

    directive = propose_mechanical_repartition_directive_v2(
        parent_candidate,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        partition=parent_partition,
        compatibility_report=parent_g3b,
        envelope=envelope,
        cameras=cameras,
        seed_strategy="SOURCE_EDGE_PROBE_RATIO_V1",
    )
    if (
        directive.get("status")
        != "REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY"
    ):
        raise RuntimeError(
            "V9_REPARTITION_TEACHER_ORACLE_NO_DIRECTIVE:"
            + str(directive.get("status"))
        )
    if directive.get("weight_mutation") is not False:
        raise RuntimeError("V9_REPARTITION_DIRECTIVE_MUTATES_WEIGHT")

    evidence = {
        "schema": "RealSaS.V9ExactTeacherOracleRepartitionEvidence.v1",
        "status": "ORACLE_ONLY",
        "teacher_bank_sha256": content_sha256(a.teacher_bank.read_bytes().hex()),
        "teacher_clean_row_count": int(np.count_nonzero(valid)),
        "teacher_row_count": int(len(valid)),
        "teacher_coverage": float(np.mean(valid)),
        "parent_g3b_report_hash": str(parent_g3b["report_hash"]),
        "parent_candidate_lineage_hash": parent_candidate.candidate_lineage_hash,
        "teacher_inputs_used_by_predictor": False,
        "product_authority_claimed": False,
    }
    evidence_hash = content_sha256(evidence)
    auth = {
        "schema": "RealSaS.TrustworthySkinRepartitionAuthorization.v1",
        "status": "PASS_TRUSTWORTHY_SKIN_REPARTITION_AUTHORIZATION",
        "directive_hash": directive["directive_hash"],
        "source_surface_lineage_hash": surface.geometry_lineage_hash,
        "source_partition_lineage_hash": parent_partition.partition_lineage_hash,
        "source_skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "source_skin_lineage_hash": teacher_skin.skin_lineage_hash,
        "teacher_inputs_used_by_predictor": False,
        "weight_reliability_closure_passed": True,
        "weight_reliability_evidence_hash": evidence_hash,
        "scope": "V9_EXACT_TEACHER_ORACLE_CHILD_ONLY",
        "product_authority_claimed": False,
        "authorization_hash": "",
    }
    auth["authorization_hash"] = repartition_authorization_hash_v1(auth)

    child_partition = build_repartitioned_partition_v2(
        surface=surface,
        parent_partition=parent_partition,
        directive=directive,
        authorization=auth,
    )
    carrier = build_component_carrier_policy(
        partition=child_partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(
                c.component_id,
                "MESH",
                ("V9_TEACHER_ORACLE_PRODUCTION_REPARTITION_CHILD",),
                metadata={
                    "automatic": True,
                    "semantic_recognition_used": False,
                    "audit_only": True,
                },
            )
            for c in child_partition.components
        ),
        metadata={
            "default_carrier": "MESH",
            "automatic": True,
            "audit_only": True,
        },
    )
    child_candidate = build_holeless_partitioned_dense_candidate(
        surface,
        child_partition,
        carrier,
        producer_policy_hash=content_sha256(
            {
                "audit": "V9_TEACHER_ORACLE_PRODUCTION_REPARTITION_CHILD_V1",
                "parent_candidate": parent_candidate.candidate_lineage_hash,
                "directive": directive["directive_hash"],
                "policy": policy.qualification_policy_lineage_hash,
            }
        ),
        explicit_face_provenance=explicit_faces,
        mechanical_skin_transfer="COMPONENT_HARMONIC_DIRICHLET_V1",
    )
    validate_canonical_mesh_candidate(
        child_candidate,
        surface=surface,
        partition=child_partition,
        carrier_policy=carrier,
    )

    child_quality = report_quality(child_candidate, policy)
    child_g3b = run_skin_topology_compatibility_v1(
        child_candidate,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    child_g3 = run_g3_local_frame_micro_stress_v2(
        child_candidate,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    rest, W, faces = _candidate_skin_matrix(
        child_candidate,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
    )
    joint_ids = tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    motion = exact_motion_court(
        ctx,
        np.asarray(rest, dtype=np.float64),
        np.asarray(W, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        joint_ids,
        skeleton,
        cameras,
        policy,
    )

    closure = dict(child_partition.metadata.get("partition_cut_closure") or {})
    report = {
        "schema": "RealSaS.V9TeacherOracleProductionRepartitionFeedbackCourt.v1",
        "status": "DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
        "product_authority_minted": False,
        "teacher_oracle_authorization_only": True,
        "weights_mutated": False,
        "parent": {
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "partition_lineage_hash": parent_partition.partition_lineage_hash,
            "partition_component_count": len(parent_partition.components),
            "candidate_lineage_hash": parent_candidate.candidate_lineage_hash,
            "candidate_vertex_count": len(parent_candidate.vertices),
            "candidate_face_count": len(parent_candidate.faces),
            "g3b_passed": bool(parent_g3b["passed"]),
            "g3b_risky_face_count": int(parent_g3b["risky_face_count"]),
            "g3b_unsafe_face_count": int(parent_g3b["unsafe_face_count"]),
            "g3_passed": bool(parent_g3.passed),
            "g3_max_condition": float(parent_g3.maximum_condition_number),
            "g3_max_edge_ratio": float(parent_g3.maximum_edge_ratio),
        },
        "directive": {
            "hash": directive["directive_hash"],
            "candidate_separate_pair_count": int(
                directive["candidate_separate_pair_count"]
            ),
            "unsafe_source_edge_count": int(
                directive.get("unsafe_source_edge_count") or 0
            ),
            "source_edge_count": int(directive.get("source_edge_probe_count") or 0),
            "seed_strategy": directive.get("seed_strategy"),
            "weight_mutation": bool(directive["weight_mutation"]),
            "face_deletion_count": int(directive["face_deletion_count"]),
        },
        "child": {
            "partition_lineage_hash": child_partition.partition_lineage_hash,
            "partition_component_count": len(child_partition.components),
            "partition_cut_closure": closure,
            "candidate_lineage_hash": child_candidate.candidate_lineage_hash,
            "candidate_vertex_count": len(child_candidate.vertices),
            "candidate_face_count": len(child_candidate.faces),
            "candidate_metadata": dict(child_candidate.metadata or {}),
            "static_quality": child_quality,
            "g3b_passed": bool(child_g3b["passed"]),
            "g3b_risky_face_count": int(child_g3b["risky_face_count"]),
            "g3b_unsafe_face_count": int(child_g3b["unsafe_face_count"]),
            "g3_passed": bool(child_g3.passed),
            "g3_failure_invariants": list(child_g3.failure_invariants),
            "g3_min_area_ratio": float(child_g3.minimum_area_ratio),
            "g3_max_area_ratio": float(child_g3.maximum_area_ratio),
            "g3_max_condition": float(child_g3.maximum_condition_number),
            "g3_min_edge_ratio": float(child_g3.minimum_edge_ratio),
            "g3_max_edge_ratio": float(child_g3.maximum_edge_ratio),
            "failed_motion_frames": int(motion["failed_motion_frame_count"]),
            "max_motion_edge_ratio": float(motion["maximum_motion_edge_ratio"]),
            "max_motion_condition_number": float(
                motion["maximum_motion_condition_number"]
            ),
        },
        "decision": {
            "production_repartition_feedback_mechanically_closes_teacher_oracle": bool(
                child_g3.passed and motion["failed_motion_frame_count"] == 0
            ),
            "static_quality_also_closed": bool(
                child_quality["policy_violating_face_count"] == 0
            ),
            "a100_still_blocked_until_mechanically_admissible_target_exists": bool(
                not (child_g3.passed and motion["failed_motion_frame_count"] == 0)
            ),
        },
        "claim_boundary": [
            "Teacher weights authorize this child only as an oracle ceiling.",
            "No product/main authority is mutated.",
            "No QualifiedSkinIR weight is changed by the repartition path.",
            "SOURCE_EDGE_PROBE_RATIO_V1, mechanical_repartition_v2, holeless Stage18, and harmonic support transfer are production compiler operators.",
        ],
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "DIRECTIVE.json").write_text(
        json.dumps(directive, indent=2, sort_keys=True) + "\n"
    )
    (a.out_dir / "CHILD_PARTITION.json").write_text(
        json.dumps(child_partition.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (a.out_dir / "CHILD_CANDIDATE.json").write_text(
        json.dumps(child_candidate.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (a.out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "V9_TEACHER_REPARTITION_RESULT="
        + json.dumps(
            {
                "parent_components": len(parent_partition.components),
                "parent_g3b_unsafe": int(parent_g3b["unsafe_face_count"]),
                "direct_seeds": int(
                    directive["candidate_separate_pair_count"]
                ),
                "child_components": len(child_partition.components),
                "final_separate": closure.get("final_separate_count"),
                "child_vertices": len(child_candidate.vertices),
                "child_faces": len(child_candidate.faces),
                "child_static_violations": child_quality[
                    "policy_violating_face_count"
                ],
                "child_g3b_unsafe": int(child_g3b["unsafe_face_count"]),
                "child_g3_passed": bool(child_g3.passed),
                "child_g3_max_condition": float(
                    child_g3.maximum_condition_number
                ),
                "child_failed_motion_frames": int(
                    motion["failed_motion_frame_count"]
                ),
                "child_max_motion_edge": float(
                    motion["maximum_motion_edge_ratio"]
                ),
                "mechanical_close": report["decision"][
                    "production_repartition_feedback_mechanically_closes_teacher_oracle"
                ],
                "static_close": report["decision"]["static_quality_also_closed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
