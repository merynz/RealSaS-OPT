"""Bounded iterative V9 teacher-oracle production repartition court.

Diagnostic only. Exercises the existing generic Stage35 -> Stage17 -> Stage18
feedback machinery on the exact V9 teacher field. No QualifiedSkinIR weight is
mutated and no product authority is minted.

Per repair iteration:
  1. measure G3B on the current candidate;
  2. propose SOURCE_EDGE_PROBE_RATIO_V1 cannot-links;
  3. if that source-edge path has no new admissible pair, fall back to the
     existing UNSAFE_FACE_LOCAL_L1_V1 residual proposal;
  4. authorization-gated mechanical_repartition_v2 cut closure;
  5. rebuild holeless Stage18 with COMPONENT_HARMONIC_DIRICHLET_V1.

The loop is bounded by DEFAULT_MAX_REPAIR_ITERATIONS from compiler core.
Final candidate is evaluated by static quality, G3B, production G3 and the
independent 51-frame exact-motion court.
"""
from __future__ import annotations

import argparse
import hashlib
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
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    DEFAULT_MAX_REPAIR_ITERATIONS,
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
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def carrier_for(partition, iteration):
    return build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(
                c.component_id,
                "MESH",
                ("V9_ITERATIVE_PRODUCTION_REPARTITION",),
                metadata={
                    "automatic": True,
                    "semantic_recognition_used": False,
                    "iteration": int(iteration),
                    "audit_only": True,
                },
            )
            for c in partition.components
        ),
        metadata={
            "default_carrier": "MESH",
            "automatic": True,
            "audit_only": True,
            "iteration": int(iteration),
        },
    )


def authorization_for(*, directive, surface, partition, skeleton, skin, evidence_hash, iteration):
    payload = {
        "schema": "RealSaS.TrustworthySkinRepartitionAuthorization.v1",
        "status": "PASS_TRUSTWORTHY_SKIN_REPARTITION_AUTHORIZATION",
        "directive_hash": directive["directive_hash"],
        "source_surface_lineage_hash": surface.geometry_lineage_hash,
        "source_partition_lineage_hash": partition.partition_lineage_hash,
        "source_skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "source_skin_lineage_hash": skin.skin_lineage_hash,
        "teacher_inputs_used_by_predictor": False,
        "weight_reliability_closure_passed": True,
        "weight_reliability_evidence_hash": evidence_hash,
        "scope": "V9_EXACT_TEACHER_ORACLE_ITERATIVE_CHILD_ONLY",
        "repair_iteration": int(iteration),
        "product_authority_claimed": False,
        "authorization_hash": "",
    }
    payload["authorization_hash"] = repartition_authorization_hash_v1(payload)
    return payload


def propose_with_generic_fallback(
    *,
    candidate,
    surface,
    skeleton,
    skin,
    partition,
    compatibility,
    envelope,
    cameras,
):
    source_edge = propose_mechanical_repartition_directive_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        compatibility_report=compatibility,
        envelope=envelope,
        cameras=cameras,
        seed_strategy="SOURCE_EDGE_PROBE_RATIO_V1",
    )
    if (
        source_edge.get("status")
        == "REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY"
        and int(source_edge.get("candidate_separate_pair_count") or 0) > 0
    ):
        return source_edge, "SOURCE_EDGE_PROBE_RATIO_V1"

    residual = propose_mechanical_repartition_directive_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        compatibility_report=compatibility,
        seed_strategy="UNSAFE_FACE_LOCAL_L1_V1",
    )
    return residual, "UNSAFE_FACE_LOCAL_L1_V1"


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
    partition = mechanical_partition_from_dict(read(a.partition_json))
    candidate = canonical_mesh_candidate_from_dict(read(a.candidate_json))
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
        raise RuntimeError("V9_ITER_REBUILT_SURFACE_LINEAGE_DRIFT")
    validate_mechanical_partition(partition, surface)

    teacher_skin, _, valid, _, _, _ = teacher_to_skin(
        surface, skeleton, a.teacher_bank
    )
    bank_sha = sha256(a.teacher_bank)
    evidence = {
        "schema": "RealSaS.V9ExactTeacherOracleIterativeRepartitionEvidence.v1",
        "status": "ORACLE_ONLY",
        "teacher_bank_sha256": bank_sha,
        "teacher_clean_row_count": int(np.count_nonzero(valid)),
        "teacher_row_count": int(len(valid)),
        "teacher_coverage": float(np.mean(valid)),
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "teacher_inputs_used_by_predictor": False,
        "product_authority_claimed": False,
    }
    evidence_hash = content_sha256(evidence)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    iterations = []
    closed_g3b = False
    repairs_used = 0

    for evaluation_index in range(int(DEFAULT_MAX_REPAIR_ITERATIONS) + 1):
        compatibility = run_skin_topology_compatibility_v1(
            candidate,
            surface=surface,
            skeleton=skeleton,
            skin=teacher_skin,
            envelope=envelope,
            cameras=cameras,
            policy=policy,
        )
        quality = report_quality(candidate, policy)
        step = {
            "evaluation_index": int(evaluation_index),
            "partition_lineage_hash": partition.partition_lineage_hash,
            "candidate_lineage_hash": candidate.candidate_lineage_hash,
            "component_count": len(partition.components),
            "separate_boundary_count": sum(
                x.decision == "SEPARATE" for x in partition.boundary_constraints
            ),
            "vertex_count": len(candidate.vertices),
            "face_count": len(candidate.faces),
            "static_policy_violating_face_count": int(
                quality["policy_violating_face_count"]
            ),
            "g3b": {
                "passed": bool(compatibility["passed"]),
                "report_hash": compatibility["report_hash"],
                "risky_face_count": int(compatibility["risky_face_count"]),
                "unsafe_face_count": int(compatibility["unsafe_face_count"]),
            },
        }

        print(
            "V9_ITER_EVAL="
            + json.dumps(
                {
                    "evaluation": evaluation_index,
                    "components": step["component_count"],
                    "separate": step["separate_boundary_count"],
                    "vertices": step["vertex_count"],
                    "faces": step["face_count"],
                    "static_violations": step[
                        "static_policy_violating_face_count"
                    ],
                    "g3b_pass": step["g3b"]["passed"],
                    "g3b_unsafe": step["g3b"]["unsafe_face_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

        if compatibility["passed"]:
            step["action"] = "G3B_CLOSED"
            iterations.append(step)
            closed_g3b = True
            break

        if repairs_used >= int(DEFAULT_MAX_REPAIR_ITERATIONS):
            step["action"] = "REPAIR_BUDGET_EXHAUSTED"
            iterations.append(step)
            break

        directive, strategy = propose_with_generic_fallback(
            candidate=candidate,
            surface=surface,
            skeleton=skeleton,
            skin=teacher_skin,
            partition=partition,
            compatibility=compatibility,
            envelope=envelope,
            cameras=cameras,
        )
        if (
            directive.get("status")
            != "REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY"
            or int(directive.get("candidate_separate_pair_count") or 0) <= 0
        ):
            step["action"] = "ABSTAIN_NO_NEW_REPARTITION"
            step["directive_status"] = directive.get("status")
            step["strategy"] = strategy
            iterations.append(step)
            break
        if directive.get("weight_mutation") is not False:
            raise RuntimeError("V9_ITER_WEIGHT_MUTATION_FORBIDDEN")
        if int(directive.get("face_deletion_count") or 0) != 0:
            raise RuntimeError("V9_ITER_FACE_DELETION_FORBIDDEN")

        auth = authorization_for(
            directive=directive,
            surface=surface,
            partition=partition,
            skeleton=skeleton,
            skin=teacher_skin,
            evidence_hash=evidence_hash,
            iteration=repairs_used + 1,
        )
        child_partition = build_repartitioned_partition_v2(
            surface=surface,
            parent_partition=partition,
            directive=directive,
            authorization=auth,
        )
        validate_mechanical_partition(child_partition, surface)
        carrier = carrier_for(child_partition, repairs_used + 1)
        child_candidate = build_holeless_partitioned_dense_candidate(
            surface,
            child_partition,
            carrier,
            producer_policy_hash=content_sha256(
                {
                    "schema": "RealSaS.V9TeacherOracleIterativeRepartitionChildPolicy.v1",
                    "repair_iteration": repairs_used + 1,
                    "parent_candidate_lineage_hash": candidate.candidate_lineage_hash,
                    "parent_partition_lineage_hash": partition.partition_lineage_hash,
                    "directive_hash": directive["directive_hash"],
                    "authorization_hash": auth["authorization_hash"],
                    "teacher_evidence_hash": evidence_hash,
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
        md = dict(child_candidate.metadata or {})
        if int(md.get("face_deletion_count", -1)) != 0:
            raise RuntimeError("V9_ITER_CHILD_FACE_DELETION_FORBIDDEN")
        if float(md.get("rest_area_relative_error", 1.0)) > 1e-10:
            raise RuntimeError("V9_ITER_CHILD_REST_AREA_DRIFT")

        step.update(
            {
                "action": "REPARTITION_AND_HOLELESS_REBUILD",
                "strategy": strategy,
                "directive_hash": directive["directive_hash"],
                "authorization_hash": auth["authorization_hash"],
                "candidate_separate_pair_count": int(
                    directive["candidate_separate_pair_count"]
                ),
                "unresolved_unsafe_face_count": int(
                    directive.get("unresolved_unsafe_face_count") or 0
                ),
                "child_partition_lineage_hash": child_partition.partition_lineage_hash,
                "child_candidate_lineage_hash": child_candidate.candidate_lineage_hash,
                "child_component_count": len(child_partition.components),
                "child_separate_boundary_count": sum(
                    x.decision == "SEPARATE"
                    for x in child_partition.boundary_constraints
                ),
                "child_vertex_count": len(child_candidate.vertices),
                "child_face_count": len(child_candidate.faces),
                "child_rest_area_relative_error": float(
                    md.get("rest_area_relative_error", 0.0)
                ),
                "child_generated_seam_vertex_count": int(
                    md.get("generated_seam_vertex_count", 0)
                ),
                "child_generated_centroid_vertex_count": int(
                    md.get("generated_centroid_vertex_count", 0)
                ),
            }
        )
        iterations.append(step)

        idir = a.out_dir / f"iteration_{repairs_used + 1:02d}"
        idir.mkdir(parents=True, exist_ok=True)
        (idir / "DIRECTIVE.json").write_text(
            json.dumps(directive, indent=2, sort_keys=True) + "\n"
        )
        (idir / "AUTHORIZATION.json").write_text(
            json.dumps(auth, indent=2, sort_keys=True) + "\n"
        )
        (idir / "PARTITION.json").write_text(
            json.dumps(child_partition.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        (idir / "CANDIDATE.json").write_text(
            json.dumps(child_candidate.to_dict(), indent=2, sort_keys=True) + "\n"
        )

        partition = child_partition
        candidate = child_candidate
        repairs_used += 1

    final_compatibility = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    final_quality = report_quality(candidate, policy)
    g3 = run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    rest, W, faces = _candidate_skin_matrix(
        candidate, surface=surface, skeleton=skeleton, skin=teacher_skin
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

    mechanical_close = bool(
        final_compatibility["passed"]
        and g3.passed
        and int(motion["failed_motion_frame_count"]) == 0
    )
    static_close = bool(final_quality["policy_violating_face_count"] == 0)

    report = {
        "schema": "RealSaS.V9TeacherOracleIterativeProductionRepartitionCourt.v1",
        "status": "COMPLETE__NO_PRODUCT_MUTATION",
        "product_authority_minted": False,
        "weights_mutated": False,
        "teacher_oracle_authorization_only": True,
        "max_repair_iterations": int(DEFAULT_MAX_REPAIR_ITERATIONS),
        "teacher_evidence": evidence,
        "teacher_evidence_hash": evidence_hash,
        "iterations": iterations,
        "repairs_used": int(repairs_used),
        "g3b_closed_within_budget": bool(
            closed_g3b and final_compatibility["passed"]
        ),
        "final": {
            "partition_lineage_hash": partition.partition_lineage_hash,
            "candidate_lineage_hash": candidate.candidate_lineage_hash,
            "component_count": len(partition.components),
            "separate_boundary_count": sum(
                x.decision == "SEPARATE" for x in partition.boundary_constraints
            ),
            "vertex_count": len(candidate.vertices),
            "face_count": len(candidate.faces),
            "static_quality": final_quality,
            "g3b": {
                "passed": bool(final_compatibility["passed"]),
                "unsafe_face_count": int(final_compatibility["unsafe_face_count"]),
                "risky_face_count": int(final_compatibility["risky_face_count"]),
                "report_hash": final_compatibility["report_hash"],
            },
            "g3": {
                "passed": bool(g3.passed),
                "failure_invariants": list(g3.failure_invariants),
                "minimum_area_ratio": float(g3.minimum_area_ratio),
                "maximum_area_ratio": float(g3.maximum_area_ratio),
                "maximum_condition_number": float(g3.maximum_condition_number),
                "minimum_edge_ratio": float(g3.minimum_edge_ratio),
                "maximum_edge_ratio": float(g3.maximum_edge_ratio),
                "report_hash": g3.report_hash,
            },
            "motion": {
                "frame_count": int(motion["frame_count"]),
                "failed_motion_frame_count": int(
                    motion["failed_motion_frame_count"]
                ),
                "maximum_motion_edge_ratio": float(
                    motion["maximum_motion_edge_ratio"]
                ),
                "maximum_motion_condition_number": float(
                    motion["maximum_motion_condition_number"]
                ),
            },
        },
        "decision": {
            "mechanical_close": mechanical_close,
            "static_close": static_close,
            "needs_static_quality_composition": bool(
                mechanical_close and not static_close
            ),
            "a100_target_topology_mechanically_admissible": mechanical_close,
        },
        "claim_boundary": [
            "All topology-repair operators are existing production compiler operators.",
            "Teacher field is used only as an oracle reliability authorization and never as product inference input.",
            "No skin weight is mutated.",
            "No product/main authority is minted.",
            "If mechanics closes but static quality does not, static quality repair must be composed and all dynamic courts rerun before any A100 escalation.",
        ],
    }
    (a.out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (a.out_dir / "FINAL_PARTITION.json").write_text(
        json.dumps(partition.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (a.out_dir / "FINAL_CANDIDATE.json").write_text(
        json.dumps(candidate.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    print(
        "V9_ITERATIVE_REPARTITION_RESULT="
        + json.dumps(
            {
                "repairs_used": repairs_used,
                "g3b_closed": report["g3b_closed_within_budget"],
                "components": report["final"]["component_count"],
                "separate": report["final"]["separate_boundary_count"],
                "vertices": report["final"]["vertex_count"],
                "faces": report["final"]["face_count"],
                "static_violations": final_quality[
                    "policy_violating_face_count"
                ],
                "g3_pass": g3.passed,
                "g3_max_condition": g3.maximum_condition_number,
                "g3b_unsafe": final_compatibility["unsafe_face_count"],
                "failed_motion_frames": motion["failed_motion_frame_count"],
                "max_motion_edge": motion["maximum_motion_edge_ratio"],
                "mechanical_close": mechanical_close,
                "static_close": static_close,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
