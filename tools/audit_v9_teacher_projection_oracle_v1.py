"""V9 exact teacher reprojection oracle audit.

Diagnostic only. This court does not mutate product authority and does not use
teacher data in product inference. It asks whether the exact same-witness artist
source skin, reprojected onto the current V9 RiggingSurface using the historical
frozen teacher projector, defines a mechanically coherent target for an Arachne
readout refit.

The generated teacher bank is supplied by the workflow; this script only checks
its binding to the current V9 surface / fresh Geppetto skeleton and evaluates:
  1) fresh Arachne prediction error against the projected bank, stratified by the
     frozen teacher-valid mask;
  2) an ALL_PROJECTED teacher oracle under the exact frozen G3 probe bank;
  3) the independent 51-frame authored-motion court.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
)
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import (
    SkinInfluenceProposal,
    SkinProposalIR,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.audit_direct_lbs_mesh_weight_projection_v1 import (
    exact_motion_court,
    evaluate_g3_and_collect_constraints,
    probe_bank,
    unique_edges,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


def read(path):
    return json.loads(Path(path).read_text())


def q(x, p):
    a = np.asarray(x, dtype=np.float64)
    return float(np.quantile(a, p)) if a.size else None


def matrix_from_skin(surface, skeleton, skin):
    sids = tuple(str(n.surface_id) for n in surface.surface_nodes)
    jids = tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    si = {sid: i for i, sid in enumerate(sids)}
    ji = {jid: i for i, jid in enumerate(jids)}
    if len(si) != len(sids) or len(ji) != len(jids):
        raise RuntimeError("V9_TEACHER_ORACLE_DUPLICATE_AXIS")
    rows = {str(r.surface_id): r for r in skin.rows}
    if set(rows) != set(sids):
        raise RuntimeError("V9_TEACHER_ORACLE_SKIN_SURFACE_ACCOUNTING")
    W = np.zeros((len(sids), len(jids)), dtype=np.float64)
    for r, sid in enumerate(sids):
        for jid, w in rows[sid].influences:
            if str(jid) not in ji:
                raise RuntimeError("V9_TEACHER_ORACLE_UNKNOWN_SKIN_JOINT")
            W[r, ji[str(jid)]] = float(w)
    if np.any(W < -1e-12) or not np.allclose(W.sum(1), 1.0, atol=1e-8, rtol=0.0):
        raise RuntimeError("V9_TEACHER_ORACLE_SKIN_SIMPLEX")
    return sids, jids, W


def teacher_to_skin(surface, skeleton, bank_path):
    with np.load(bank_path, allow_pickle=False) as z:
        required = {
            "weights",
            "surface_ids",
            "teacher_valid_mask",
            "target_source_indices_provenance_only",
            "source_surface_distance",
            "source_triangle_index",
            "source_triangle_barycentric",
        }
        if not required.issubset(z.files):
            raise RuntimeError(
                "V9_TEACHER_ORACLE_BANK_SCHEMA_MISSING:"
                + repr(sorted(required - set(z.files)))
            )
        weights = np.asarray(z["weights"], dtype=np.float64)
        bank_sids = tuple(map(str, z["surface_ids"].tolist()))
        valid = np.asarray(z["teacher_valid_mask"], dtype=np.uint8).astype(bool)
        target_source = np.asarray(
            z["target_source_indices_provenance_only"], dtype=np.int64
        )
        dist = np.asarray(z["source_surface_distance"], dtype=np.float64)
        tri = np.asarray(z["source_triangle_index"], dtype=np.int64)
        bary = np.asarray(z["source_triangle_barycentric"], dtype=np.float64)

    surface_sids = tuple(str(n.surface_id) for n in surface.surface_nodes)
    if bank_sids != surface_sids:
        raise RuntimeError("V9_TEACHER_ORACLE_SURFACE_ID_BINDING_DRIFT")
    if weights.shape[0] != len(surface_sids) or valid.shape != (len(surface_sids),):
        raise RuntimeError("V9_TEACHER_ORACLE_BANK_ROW_SHAPE_DRIFT")
    if target_source.shape != (weights.shape[1],):
        raise RuntimeError("V9_TEACHER_ORACLE_TARGET_AXIS_SHAPE_DRIFT")
    if tri.shape != (len(surface_sids),) or bary.shape != (len(surface_sids), 3):
        raise RuntimeError("V9_TEACHER_ORACLE_PROVENANCE_SHAPE_DRIFT")
    if np.any(weights < -1e-12) or not np.allclose(
        weights.sum(1), 1.0, atol=1e-6, rtol=0.0
    ):
        raise RuntimeError("V9_TEACHER_ORACLE_BANK_SIMPLEX")

    joints = tuple(skeleton.joints)
    source_to_joint = {}
    source_indices = []
    for j in joints:
        raw = str(j.source_proposal_id)
        if not raw.startswith("P:GRS:"):
            raise RuntimeError("V9_TEACHER_ORACLE_SOURCE_PROPOSAL_DRIFT:" + raw)
        idx = int(raw.rsplit(":", 1)[1])
        source_indices.append(idx)
        if idx in source_to_joint:
            raise RuntimeError("V9_TEACHER_ORACLE_SOURCE_PROPOSAL_DUPLICATE")
        source_to_joint[idx] = str(j.canonical_joint_id)
    if sorted(source_indices) != list(range(weights.shape[1])):
        raise RuntimeError(
            "V9_TEACHER_ORACLE_TARGET_PERMUTATION_DRIFT:" + repr(source_indices)
        )

    influences = []
    for r, sid in enumerate(surface_sids):
        for target_index in range(weights.shape[1]):
            w = float(weights[r, target_index])
            if w <= 1e-15:
                continue
            jid = source_to_joint[target_index]
            influences.append(SkinInfluenceProposal(sid, jid, w))

    proposal = SkinProposalIR(
        tuple(influences),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        model_provenance="DIAGNOSTIC_EXACT_ARTIST_TEACHER_REPROJECTION_V9",
        metadata={
            "teacher_only": True,
            "product_authority_minted": False,
            "all_projected_rows_used_for_oracle": True,
        },
    )
    teacher_skin = qualify_skin(
        surface,
        skeleton,
        proposal,
        max_simplex_repair_l1=1e-7,
        max_total_correction_l1=1e-3,
    )
    return teacher_skin, weights, valid, dist, tri, bary


def error_stats(err, mask):
    x = np.asarray(err, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return {
        "row_count": int(len(x)),
        "mean_l1": float(np.mean(x)) if len(x) else None,
        "p50_l1": q(x, 0.50),
        "p95_l1": q(x, 0.95),
        "p99_l1": q(x, 0.99),
        "max_l1": float(np.max(x)) if len(x) else None,
        "gt_0p1": int(np.count_nonzero(x > 0.1)),
        "gt_1": int(np.count_nonzero(x > 1.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--surface-json", type=Path, required=True)
    ap.add_argument("--candidate-json", type=Path, required=True)
    ap.add_argument("--fresh-inference-dir", type=Path, required=True)
    ap.add_argument("--teacher-bank", type=Path, required=True)
    ap.add_argument("--projection-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ctx = _ctx(a.authority_root, a.run_id)
    surface = rigging_surface_from_dict(read(a.surface_json))
    candidate = canonical_mesh_candidate_from_dict(read(a.candidate_json))
    skeleton = qualified_skeleton_from_dict(
        read(a.fresh_inference_dir / "fresh_qualified_skeleton.json")
    )
    predicted_skin = qualified_skin_from_dict(
        read(a.fresh_inference_dir / "fresh_qualified_skin.json")
    )
    cameraset = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
        )
    )
    policy = mesh_policy_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.MeshQualificationPolicyIR.v1",
        )
    )
    cameras = tuple(sorted(cameraset.cameras, key=lambda c: int(c.view_index)))
    projection = read(a.projection_report)

    teacher_skin, bankW_target, valid, dist, tri, bary = teacher_to_skin(
        surface, skeleton, a.teacher_bank
    )
    sids, jids, predW = matrix_from_skin(surface, skeleton, predicted_skin)
    _, _, teacherW_canonical = matrix_from_skin(surface, skeleton, teacher_skin)

    # Reorder target-index bank into the current canonical-joint order using
    # source_proposal_id. This independently checks teacher_to_skin().
    canonical_from_bank = np.zeros_like(teacherW_canonical)
    for jcol, joint in enumerate(skeleton.joints):
        idx = int(str(joint.source_proposal_id).rsplit(":", 1)[1])
        canonical_from_bank[:, jcol] = bankW_target[:, idx]
    canonical_binding_l1 = np.abs(
        canonical_from_bank - teacherW_canonical
    ).sum(axis=1)
    canonical_binding_l1_max = float(np.max(canonical_binding_l1))
    # The historical projector serializes weights as float32 and itself admits
    # simplex residual up to 1e-6. qualify_skin() deterministically renormalizes
    # those rows to exact simplex. Treat only a discrepancy beyond the frozen
    # projector numeric contract as a semantic column-binding drift.
    if canonical_binding_l1_max > 1e-6:
        raise RuntimeError(
            "V9_TEACHER_ORACLE_CANONICAL_COLUMN_BINDING_DRIFT:"
            f"{canonical_binding_l1_max}"
        )

    row_err = np.abs(predW - teacherW_canonical).sum(axis=1)
    all_mask = np.ones(len(valid), dtype=bool)

    rest_t, Wt, faces = _candidate_skin_matrix(
        candidate, surface=surface, skeleton=skeleton, skin=teacher_skin
    )
    rest_p, Wp, faces_p = _candidate_skin_matrix(
        candidate, surface=surface, skeleton=skeleton, skin=predicted_skin
    )
    if not np.allclose(rest_t, rest_p, atol=0.0, rtol=0.0) or not np.array_equal(
        faces, faces_p
    ):
        raise RuntimeError("V9_TEACHER_ORACLE_CARRIER_BINDING_DRIFT")

    edges, face_edges = unique_edges(faces)
    joint_ids, probes = probe_bank(skeleton, cameras)
    teacher_g3, _ = evaluate_g3_and_collect_constraints(
        rest_t,
        Wt,
        faces,
        edges,
        face_edges,
        probes,
        policy,
        collect_constraints=False,
    )
    predicted_g3, _ = evaluate_g3_and_collect_constraints(
        rest_p,
        Wp,
        faces,
        edges,
        face_edges,
        probes,
        policy,
        collect_constraints=False,
    )
    teacher_motion = exact_motion_court(
        ctx, rest_t, Wt, faces, joint_ids, skeleton, cameras, policy
    )
    predicted_motion = exact_motion_court(
        ctx, rest_p, Wp, faces, joint_ids, skeleton, cameras, policy
    )

    report = {
        "schema": "RealSaS.V9ExactTeacherProjectionOracleAudit.v1",
        "status": "DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
        "product_authority_minted": False,
        "training_used": False,
        "teacher_used_by_product_inference": False,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "surface_node_count": len(surface.surface_nodes),
        "candidate_vertex_count": len(candidate.vertices),
        "teacher_projection": projection,
        "teacher_bank": {
            "row_count": int(len(valid)),
            "qualified_rebind_l1_max": canonical_binding_l1_max,
            "qualified_rebind_numeric_contract_l1_max": 1e-6,
            "clean_row_count": int(np.count_nonzero(valid)),
            "coverage": float(np.mean(valid)),
            "invalid_row_count": int(np.count_nonzero(~valid)),
            "source_surface_distance_p50": q(dist, 0.50),
            "source_surface_distance_p95": q(dist, 0.95),
            "source_surface_distance_p99": q(dist, 0.99),
            "source_surface_distance_max": float(np.max(dist)),
            "source_triangle_count_referenced": int(len(np.unique(tri))),
            "barycentric_max_residual": float(
                np.max(np.abs(np.asarray(bary).sum(axis=1) - 1.0))
            ),
        },
        "fresh_prediction_vs_exact_reprojected_teacher": {
            "ALL": error_stats(row_err, all_mask),
            "TEACHER_VALID": error_stats(row_err, valid),
            "TEACHER_INVALID": error_stats(row_err, ~valid),
        },
        "fresh_prediction_mechanics": {
            "g3_passed": bool(predicted_g3["passed"]),
            "g3_failure_invariants": predicted_g3["failure_invariants"],
            "g3_max_condition": float(predicted_g3["maximum_condition_number"]),
            "g3_max_edge_ratio": float(predicted_g3["maximum_edge_ratio"]),
            "failed_motion_frames": int(
                predicted_motion["failed_motion_frame_count"]
            ),
            "max_motion_edge_ratio": float(
                predicted_motion["maximum_motion_edge_ratio"]
            ),
            "max_motion_condition_number": float(
                predicted_motion["maximum_motion_condition_number"]
            ),
        },
        "all_projected_teacher_oracle_mechanics": {
            "ORACLE_NOT_PRODUCT": True,
            "g3_passed": bool(teacher_g3["passed"]),
            "g3_failure_invariants": teacher_g3["failure_invariants"],
            "g3_min_area_ratio": float(teacher_g3["minimum_area_ratio"]),
            "g3_max_area_ratio": float(teacher_g3["maximum_area_ratio"]),
            "g3_max_condition": float(teacher_g3["maximum_condition_number"]),
            "g3_min_edge_ratio": float(teacher_g3["minimum_edge_ratio"]),
            "g3_max_edge_ratio": float(teacher_g3["maximum_edge_ratio"]),
            "failed_motion_frames": int(teacher_motion["failed_motion_frame_count"]),
            "max_motion_edge_ratio": float(
                teacher_motion["maximum_motion_edge_ratio"]
            ),
            "max_motion_condition_number": float(
                teacher_motion["maximum_motion_condition_number"]
            ),
        },
        "decision_inputs": {
            "projection_rebinds_exact_current_surface_ids": True,
            "target_axis_rebinds_by_source_proposal_index": True,
            "teacher_oracle_is_training_target_diagnostic_only": True,
            "a100_not_used_by_this_court": True,
        },
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "V9_TEACHER_ORACLE_RESULT="
        + json.dumps(
            {
                "coverage": report["teacher_bank"]["coverage"],
                "invalid_rows": report["teacher_bank"]["invalid_row_count"],
                "pred_valid_p95": report[
                    "fresh_prediction_vs_exact_reprojected_teacher"
                ]["TEACHER_VALID"]["p95_l1"],
                "pred_invalid_p95": report[
                    "fresh_prediction_vs_exact_reprojected_teacher"
                ]["TEACHER_INVALID"]["p95_l1"],
                "teacher_g3_passed": report[
                    "all_projected_teacher_oracle_mechanics"
                ]["g3_passed"],
                "teacher_g3_max_condition": report[
                    "all_projected_teacher_oracle_mechanics"
                ]["g3_max_condition"],
                "teacher_failed_motion_frames": report[
                    "all_projected_teacher_oracle_mechanics"
                ]["failed_motion_frames"],
                "teacher_max_motion_edge_ratio": report[
                    "all_projected_teacher_oracle_mechanics"
                ]["max_motion_edge_ratio"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
