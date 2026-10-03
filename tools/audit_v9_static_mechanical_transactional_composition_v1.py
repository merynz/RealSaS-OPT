"""Transactional static-quality x mechanical-admissibility composition pilot.

Diagnostic only. Every static operator produces a trial candidate. The trial is
accepted only if the frozen teacher-skin G3B mechanical signature does not
regress. Static quality may move temporarily inside the transaction, but the
transaction is admitted only if the final static violation count improves.

No skin weight, partition, threshold or subject-specific region is changed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_batched_collapse_v2 import (
    repair_candidate_endpoint_collapses_batched_v2,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_batched_relaxation_v2 import (
    repair_candidate_projected_relaxation_batched_v2,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_cavity_remesh_v1 import (
    repair_candidate_cavity_retriangulation_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_edge_cavity_remesh_v1 import (
    repair_candidate_edge_cavity_retriangulation_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_patch_purge_v1 import (
    repair_candidate_patch_interior_purge_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    mechanical_quality_protected_surface_ids_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import (
    _manifold_report,
    repair_candidate_fixed_vertex_flips_topology_safe_v2,
)
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import (
    derive_deformation_envelope_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.audit_knight_repaired_quality_collapse_v1 import report_quality
from tools.audit_knight_static_quality_split_cycle_v8 import synchronized_long_edge_split
from tools.audit_v9_teacher_projection_oracle_v1 import teacher_to_skin
from tools.demo.render_knight_motion_preview_v1 import _ctx

_TOL = 1e-9


def read(path):
    return json.loads(Path(path).read_text())


def _g3b_severity(report, policy):
    rows = tuple(report.get("top_unsafe_faces") or ())
    if not rows:
        return 1.0
    out = 1.0
    for row in rows:
        out = max(
            out,
            float(row["max_edge_ratio"]) / 4.0,
            float(row["max_condition_number"]) / float(policy.g3_max_dynamic_condition_number),
            float(row["max_area_ratio"]) / float(policy.g3_max_dynamic_area_ratio),
            float(policy.g3_min_dynamic_area_ratio) / max(float(row["min_area_ratio"]), 1e-15),
        )
    return float(out)


def _mechanical_signature(candidate, *, surface, skeleton, skin, envelope, cameras, policy):
    report = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
        risk_l1_min=0.0,
        stress_all_faces=True,
    )
    return {
        "passed": bool(report["passed"]),
        "unsafe": int(report["unsafe_face_count"]),
        "severity": _g3b_severity(report, policy),
        "report_hash": str(report["report_hash"]),
    }


def _mechanical_nonregression(before, after):
    if bool(before["passed"]):
        return bool(after["passed"])
    if int(after["unsafe"]) < int(before["unsafe"]):
        return True
    if int(after["unsafe"]) > int(before["unsafe"]):
        return False
    return float(after["severity"]) <= float(before["severity"]) * (1.0 + _TOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--surface-json", type=Path, required=True)
    ap.add_argument("--child-candidate-json", type=Path, required=True)
    ap.add_argument("--child-partition-json", type=Path, required=True)
    ap.add_argument("--fresh-skeleton-json", type=Path, required=True)
    ap.add_argument("--teacher-bank", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    ctx = _ctx(a.authority_root, a.run_id)
    surface = rigging_surface_from_dict(read(a.surface_json))
    partition = mechanical_partition_from_dict(read(a.child_partition_json))
    current = canonical_mesh_candidate_from_dict(read(a.child_candidate_json))
    reference = current
    skeleton = qualified_skeleton_from_dict(read(a.fresh_skeleton_json))
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1")
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda x: int(x.view_index)))
    _, envelope = derive_deformation_envelope_v1(skeleton=skeleton, camera_set=camera_set)
    policy = mesh_policy_from_dict(
        stage_output_payload(
            ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.MeshQualificationPolicyIR.v1"
        )
    )
    teacher_skin, _, _, _, _, _ = teacher_to_skin(surface, skeleton, a.teacher_bank)
    protected = mechanical_quality_protected_surface_ids_v1(partition)

    if not _manifold_report(current.faces)["passed"]:
        raise RuntimeError("V9_TRANSACTION_INPUT_NONMANIFOLD")

    start_quality = report_quality(current, policy)
    start_mech = _mechanical_signature(
        current, surface=surface, skeleton=skeleton, skin=teacher_skin,
        envelope=envelope, cameras=cameras, policy=policy,
    )
    rows = []
    t0 = time.perf_counter()

    def attempt(name, producer):
        nonlocal current
        q0 = report_quality(current, policy)
        m0 = _mechanical_signature(
            current, surface=surface, skeleton=skeleton, skin=teacher_skin,
            envelope=envelope, cameras=cameras, policy=policy,
        )
        trial, operator_report = producer(current)
        topo = _manifold_report(trial.faces)
        q1 = report_quality(trial, policy)
        if topo["passed"]:
            m1 = _mechanical_signature(
                trial, surface=surface, skeleton=skeleton, skin=teacher_skin,
                envelope=envelope, cameras=cameras, policy=policy,
            )
            mechanical_ok = _mechanical_nonregression(m0, m1)
        else:
            m1 = {"passed": False, "unsafe": 10**9, "severity": float("inf"), "report_hash": ""}
            mechanical_ok = False
        accepted = bool(topo["passed"] and mechanical_ok)
        row = {
            "operator": name,
            "accepted": accepted,
            "topology_pass": bool(topo["passed"]),
            "static_before": int(q0["policy_violating_face_count"]),
            "static_after_trial": int(q1["policy_violating_face_count"]),
            "g3b_before": m0,
            "g3b_after_trial": m1,
            "operator_report": operator_report,
            "trial_vertices": len(trial.vertices),
            "trial_faces": len(trial.faces),
        }
        rows.append(row)
        print("V9_TRANSACTION_STEP=" + json.dumps({
            k: v for k, v in row.items() if k != "operator_report"
        }, sort_keys=True), flush=True)
        if accepted:
            current = trial

    attempt("split", lambda x: synchronized_long_edge_split(x, policy, max_splits=16))
    attempt(
        "flip",
        lambda x: repair_candidate_fixed_vertex_flips_topology_safe_v2(
            x, policy, max_passes=4
        ),
    )
    attempt(
        "collapse",
        lambda x: repair_candidate_endpoint_collapses_batched_v2(
            x, policy, max_batches=16, max_collapses=256
        ),
    )
    attempt(
        "relax",
        lambda x: repair_candidate_projected_relaxation_batched_v2(
            x, reference, policy, protected_surface_ids=protected,
            max_batches=16, max_moves=256
        ),
    )
    attempt(
        "vertex_cavity",
        lambda x: repair_candidate_cavity_retriangulation_v1(
            x, policy, protected_surface_ids=protected,
            max_batches=16, max_removed_vertices=256
        ),
    )
    attempt(
        "edge_cavity",
        lambda x: repair_candidate_edge_cavity_retriangulation_v1(
            x, policy, protected_surface_ids=protected,
            max_batches=16, max_removed_edges=256
        ),
    )
    attempt(
        "patch_purge",
        lambda x: repair_candidate_patch_interior_purge_v1(
            x, policy, protected_surface_ids=protected,
            max_hops=5, max_patch_faces=128, max_batches=16, max_patches=256
        ),
    )

    final_quality = report_quality(current, policy)
    final_mech = _mechanical_signature(
        current, surface=surface, skeleton=skeleton, skin=teacher_skin,
        envelope=envelope, cameras=cameras, policy=policy,
    )
    final_g3 = run_g3_local_frame_micro_stress_v2(
        current,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    transaction_static_improved = (
        int(final_quality["policy_violating_face_count"])
        < int(start_quality["policy_violating_face_count"])
    )
    transaction_mechanical_nonregression = _mechanical_nonregression(start_mech, final_mech)

    report = {
        "schema": "RealSaS.V9StaticMechanicalTransactionalCompositionPilot.v1",
        "status": "COMPLETE__DIAGNOSTIC_ONLY",
        "product_authority_minted": False,
        "skin_weight_mutation": False,
        "partition_mutation": False,
        "start_quality": start_quality,
        "start_g3b": start_mech,
        "steps": rows,
        "final_quality": final_quality,
        "final_g3b": final_mech,
        "final_g3": {
            "passed": bool(final_g3.passed),
            "failure_invariants": list(final_g3.failure_invariants),
            "maximum_condition_number": float(final_g3.maximum_condition_number),
            "maximum_edge_ratio": float(final_g3.maximum_edge_ratio),
            "minimum_area_ratio": float(final_g3.minimum_area_ratio),
            "maximum_area_ratio": float(final_g3.maximum_area_ratio),
        },
        "decision": {
            "static_improved": bool(transaction_static_improved),
            "mechanical_nonregression": bool(transaction_mechanical_nonregression),
            "transaction_admissible": bool(
                transaction_static_improved and transaction_mechanical_nonregression
            ),
            "next": (
                "MOVE_GUARD_INSIDE_ACCEPTED_STATIC_OPERATORS"
                if transaction_static_improved and transaction_mechanical_nonregression
                else "ATTRIBUTE_REJECTED_OPERATORS_AND_REFINE_GUARD"
            ),
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (a.out_dir / "FINAL_CANDIDATE.json").write_text(
        json.dumps(current.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    print("V9_TRANSACTION_RESULT=" + json.dumps({
        "start_static": start_quality["policy_violating_face_count"],
        "final_static": final_quality["policy_violating_face_count"],
        "start_g3b_unsafe": start_mech["unsafe"],
        "final_g3b_unsafe": final_mech["unsafe"],
        "start_g3b_severity": start_mech["severity"],
        "final_g3b_severity": final_mech["severity"],
        "g3_pass": final_g3.passed,
        "g3_condition": final_g3.maximum_condition_number,
        "transaction_admissible": report["decision"]["transaction_admissible"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
