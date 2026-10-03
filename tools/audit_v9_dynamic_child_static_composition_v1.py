"""Compose V9 static-quality closure after one production dynamic repartition.

Diagnostic only. Starts from the one-pass teacher-oracle Stage17/18 child and
replays the already-developed generic static quality operator family against an
immutable copy of that child as the projection/reference domain. QualifiedSkinIR
is frozen. Final candidate is re-evaluated with static policy + G3B + G3 +
independent 51-frame exact motion.

No Knight-specific thresholds or regions are introduced.
"""
from __future__ import annotations

import argparse
import json
import time
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
from compiler.realsas_compiler_core.canonical_mesh_quality_source_surface_optimize_v1 import (
    repair_candidate_source_surface_quality_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import (
    _manifold_report,
    repair_candidate_fixed_vertex_flips_topology_safe_v2,
)
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import (
    derive_deformation_envelope_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
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
from tools.audit_direct_lbs_mesh_weight_projection_v1 import exact_motion_court
from tools.audit_knight_repaired_quality_collapse_v1 import report_quality
from tools.audit_knight_static_quality_alternating_v3 import classify
from tools.audit_knight_static_quality_split_cycle_v8 import (
    synchronized_long_edge_split,
)
from tools.audit_v9_teacher_projection_oracle_v1 import teacher_to_skin
from tools.demo.render_knight_motion_preview_v1 import _ctx

MAX_STATIC_CYCLES = 4


def read(path):
    return json.loads(Path(path).read_text())


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
    teacher_skin, _, _, _, _, _ = teacher_to_skin(
        surface, skeleton, a.teacher_bank
    )
    protected = mechanical_quality_protected_surface_ids_v1(partition)

    topo0 = _manifold_report(current.faces)
    if not topo0["passed"]:
        raise RuntimeError("V9_STATIC_COMPOSE_INPUT_NONMANIFOLD")
    initial = report_quality(current, policy)
    previous = int(initial["policy_violating_face_count"])
    cycles = []
    t0 = time.perf_counter()

    for ci in range(MAX_STATIC_CYCLES):
        row = {"cycle": ci, "before_violations": previous}
        cycle_input = current

        split, sr = synchronized_long_edge_split(
            cycle_input, policy, max_splits=64
        )
        qs = report_quality(split, policy)
        row.update(
            {
                "split_count": int(sr["accepted_split_count"]),
                "after_split": int(qs["policy_violating_face_count"]),
            }
        )

        flipped, fr = repair_candidate_fixed_vertex_flips_topology_safe_v2(
            split, policy, max_passes=12
        )
        qf = report_quality(flipped, policy)
        row.update(
            {
                "flip_count": int(fr["accepted_flip_count"]),
                "after_flip": int(qf["policy_violating_face_count"]),
            }
        )

        collapsed, cr = repair_candidate_endpoint_collapses_batched_v2(
            flipped, policy, max_batches=64, max_collapses=2048
        )
        qc = report_quality(collapsed, policy)
        row.update(
            {
                "collapse_count": int(cr["accepted_collapse_count"]),
                "after_collapse": int(qc["policy_violating_face_count"]),
            }
        )

        relaxed, rrpt = repair_candidate_projected_relaxation_batched_v2(
            collapsed,
            reference,
            policy,
            protected_surface_ids=protected,
            max_batches=64,
            max_moves=2048,
        )
        qr = report_quality(relaxed, policy)
        row.update(
            {
                "relax_count": int(rrpt["accepted_move_count"]),
                "after_relax": int(qr["policy_violating_face_count"]),
            }
        )

        vc, vr = repair_candidate_cavity_retriangulation_v1(
            relaxed,
            policy,
            protected_surface_ids=protected,
            max_batches=64,
            max_removed_vertices=2048,
        )
        qv = report_quality(vc, policy)
        row.update(
            {
                "vertex_cavity_count": int(vr["accepted_cavity_count"]),
                "after_vertex_cavity": int(qv["policy_violating_face_count"]),
            }
        )

        ec, er = repair_candidate_edge_cavity_retriangulation_v1(
            vc,
            policy,
            protected_surface_ids=protected,
            max_batches=64,
            max_removed_edges=2048,
        )
        qe = report_quality(ec, policy)
        row.update(
            {
                "edge_cavity_count": int(er["accepted_edge_cavity_count"]),
                "after_edge_cavity": int(qe["policy_violating_face_count"]),
            }
        )

        pp, pr = repair_candidate_patch_interior_purge_v1(
            ec,
            policy,
            protected_surface_ids=protected,
            max_hops=5,
            max_patch_faces=128,
            max_batches=64,
            max_patches=2048,
        )
        qp = report_quality(pp, policy)
        topo = _manifold_report(pp.faces)
        now = int(qp["policy_violating_face_count"])
        row.update(
            {
                "patch_purge_count": int(pr["accepted_patch_count"]),
                "after_patch_purge": now,
                "min_angle_deg": float(qp["min_angle_deg"]),
                "max_aspect": float(
                    qp["max_aspect_longest_over_min_altitude"]
                ),
                "vertices": len(pp.vertices),
                "faces": len(pp.faces),
                "topology_pass": bool(topo["passed"]),
            }
        )

        admitted = bool(topo["passed"] and now < previous)
        row["cycle_admitted"] = admitted
        cycles.append(row)
        print(
            "V9_STATIC_COMPOSE_CYCLE=" + json.dumps(row, sort_keys=True),
            flush=True,
        )
        if not admitted:
            break
        current = pp
        previous = now
        if now == 0:
            break

    # Same bounded final source-surface search used by V9. It may move only
    # vertices that still exist in the immutable one-pass child reference.
    before_source = report_quality(current, policy)
    optimized, source_report = repair_candidate_source_surface_quality_v1(
        current,
        reference,
        policy,
        protected_surface_ids=protected,
        grid_schedule=(8, 16, 32, 64),
        max_batches=8,
        max_moves=64,
    )
    after_source = report_quality(optimized, policy)
    if (
        int(after_source["policy_violating_face_count"])
        <= int(before_source["policy_violating_face_count"])
        and _manifold_report(optimized.faces)["passed"]
    ):
        current = optimized

    final_quality = report_quality(current, policy)
    final_topology = _manifold_report(current.faces)
    final_class = classify(current, policy, partition)

    compatibility = run_skin_topology_compatibility_v1(
        current,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    g3 = run_g3_local_frame_micro_stress_v2(
        current,
        surface=surface,
        skeleton=skeleton,
        skin=teacher_skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    rest, W, faces = _candidate_skin_matrix(
        current, surface=surface, skeleton=skeleton, skin=teacher_skin
    )
    motion = exact_motion_court(
        ctx,
        np.asarray(rest, dtype=np.float64),
        np.asarray(W, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        tuple(str(j.canonical_joint_id) for j in skeleton.joints),
        skeleton,
        cameras,
        policy,
    )

    static_close = bool(
        final_topology["passed"]
        and int(final_quality["policy_violating_face_count"]) == 0
    )
    mechanical_close = bool(
        compatibility["passed"]
        and g3.passed
        and int(motion["failed_motion_frame_count"]) == 0
    )
    report = {
        "schema": "RealSaS.V9DynamicChildStaticCompositionCourt.v1",
        "status": "COMPLETE__NO_PRODUCT_MUTATION",
        "product_authority_minted": False,
        "skin_weight_mutation": False,
        "initial_quality": initial,
        "cycles": cycles,
        "source_surface_final_pass": source_report,
        "final_quality": final_quality,
        "final_topology": final_topology,
        "final_classification": final_class,
        "final_g3b": {
            "passed": bool(compatibility["passed"]),
            "unsafe_face_count": int(compatibility["unsafe_face_count"]),
            "risky_face_count": int(compatibility["risky_face_count"]),
        },
        "final_g3": {
            "passed": bool(g3.passed),
            "failure_invariants": list(g3.failure_invariants),
            "maximum_condition_number": float(g3.maximum_condition_number),
            "maximum_edge_ratio": float(g3.maximum_edge_ratio),
            "minimum_edge_ratio": float(g3.minimum_edge_ratio),
            "minimum_area_ratio": float(g3.minimum_area_ratio),
            "maximum_area_ratio": float(g3.maximum_area_ratio),
        },
        "final_motion": {
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
        "decision": {
            "static_close": static_close,
            "mechanical_close": mechanical_close,
            "both_close": bool(static_close and mechanical_close),
        },
        "elapsed_seconds": time.perf_counter() - t0,
        "claim_boundary": [
            "Static repair uses existing generic compiler/audit operators and frozen policy constants.",
            "Immutable reference is the exact admitted one-pass dynamic child.",
            "QualifiedSkinIR is not mutated.",
            "No product/main authority is minted.",
        ],
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (a.out_dir / "FINAL_CANDIDATE.json").write_text(
        json.dumps(current.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (a.out_dir / "PARTITION.json").write_text(
        json.dumps(partition.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    print(
        "V9_STATIC_COMPOSITION_RESULT="
        + json.dumps(
            {
                "initial_violations": initial["policy_violating_face_count"],
                "final_violations": final_quality[
                    "policy_violating_face_count"
                ],
                "interior_unprotected": final_class[
                    "interior_unprotected_face_count"
                ],
                "boundary_or_protected": final_class[
                    "boundary_or_protected_face_count"
                ],
                "vertices": len(current.vertices),
                "faces": len(current.faces),
                "g3b_unsafe": compatibility["unsafe_face_count"],
                "g3_pass": g3.passed,
                "g3_max_condition": g3.maximum_condition_number,
                "failed_motion_frames": motion[
                    "failed_motion_frame_count"
                ],
                "max_motion_edge": motion["maximum_motion_edge_ratio"],
                "static_close": static_close,
                "mechanical_close": mechanical_close,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
