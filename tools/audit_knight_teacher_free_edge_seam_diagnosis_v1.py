from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import _pose_skin_matrices
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import _stress_angle
from tools.audit_knight_james_twigg_unsafe_patch_completion_court_v1 import unsafe_components
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask,
    exact,
    face_indices,
    load,
    stress_arbitrary_weights,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.audit_knight_topology_region_pair_decomposition_court_v1 import source_components
from tools.demo.render_knight_motion_preview_v1 import _ctx


PREREG = Path("canonical/KNIGHT_TEACHER_FREE_EDGE_SEAM_DIAG_PREREG_V1_20260929.json")


def _edge_table(P, F):
    edge_faces = defaultdict(list)
    for fi, row in enumerate(F.tolist()):
        a, b, c = map(int, row)
        for x, y in ((a, b), (b, c), (c, a)):
            if x > y:
                x, y = y, x
            edge_faces[(x, y)].append(int(fi))
    edges = np.asarray(sorted(edge_faces), dtype=np.int64)
    lookup = {tuple(map(int, e)): i for i, e in enumerate(edges.tolist())}
    face_edge_index = np.empty((len(F), 3), dtype=np.int64)
    for fi, row in enumerate(F.tolist()):
        a, b, c = map(int, row)
        for k, (x, y) in enumerate(((a, b), (b, c), (c, a))):
            if x > y:
                x, y = y, x
            face_edge_index[fi, k] = lookup[(x, y)]
    return edges, edge_faces, face_edge_index


def _edge_stretch(P, W, edges, jids, sk, cams, env):
    stress_angle = _stress_angle(env)
    frames = derive_joint_frames_from_skeleton(sk, cameras=cams)
    hom = np.concatenate(
        [P, np.ones((len(P), 1), dtype=np.float64)], axis=1
    )
    rest_len = np.linalg.norm(P[edges[:, 1]] - P[edges[:, 0]], axis=1)
    max_ratio = np.ones(len(edges), dtype=np.float64)
    for jid in sorted(jids):
        for axis in range(3):
            for sign in (-1.0, 1.0):
                skin_by_id = _pose_skin_matrices(
                    sk,
                    frames,
                    joint_id=jid,
                    local_axis_index=axis,
                    degrees=sign * float(stress_angle),
                )
                matrices = np.stack([skin_by_id[x] for x in jids], axis=0)
                per = np.stack(
                    [(hom @ matrices[j].T)[:, :3] for j in range(len(jids))],
                    axis=1,
                )
                posed = np.sum(per * W[:, :, None], axis=1)
                plen = np.linalg.norm(
                    posed[edges[:, 1]] - posed[edges[:, 0]], axis=1
                )
                ratio = plen / np.maximum(rest_len, 1e-12)
                max_ratio = np.maximum(max_ratio, ratio)
    return max_ratio


def _truth_region(teacher_bank, teacher_source, sk, cand):
    _, _, tri, sf = teacher_weights(teacher_bank, teacher_source, sk, cand)
    src_comp = source_components(int(sf.max()) + 1, sf)
    tri_comp = np.asarray([src_comp[int(row[0])] for row in sf], dtype=np.int64)
    return np.asarray([tri_comp[int(t)] for t in tri], dtype=np.int64)


def _rule_metrics(name, seam_edge, face_edge_index, truth, unsafe):
    pred_mixed = np.any(seam_edge[face_edge_index], axis=1)
    truth_mixed = np.asarray(
        [len(set(int(truth[v]) for v in row)) > 1 for row in F_GLOBAL],
        dtype=bool,
    )
    tp = int(np.count_nonzero(pred_mixed & truth_mixed))
    fp = int(np.count_nonzero(pred_mixed & (~truth_mixed)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, int(np.count_nonzero(truth_mixed)))
    unsafe_truth = unsafe & truth_mixed
    unsafe_tp = int(np.count_nonzero(pred_mixed & unsafe_truth))
    unsafe_recall = unsafe_tp / max(1, int(np.count_nonzero(unsafe_truth)))
    return {
        "name": name,
        "predicted_mixed_face_count": int(np.count_nonzero(pred_mixed)),
        "true_positive_mixed_face_count": tp,
        "false_positive_mixed_face_count": fp,
        "mixed_face_precision": float(precision),
        "mixed_face_recall": float(recall),
        "unsafe_truth_cross_region_face_count": int(np.count_nonzero(unsafe_truth)),
        "unsafe_truth_cross_region_recovered": unsafe_tp,
        "unsafe_truth_cross_region_recall": float(unsafe_recall),
    }


F_GLOBAL = None


def main():
    global F_GLOBAL
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--weights-npz", type=Path, required=True)
    p.add_argument("--teacher-bank", type=Path, required=True)
    p.add_argument("--teacher-source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    prereg = json.loads(PREREG.read_text())
    if prereg.get("status") != "FROZEN_BEFORE_EDGE_SEAM_DIAGNOSIS_RESULT":
        raise RuntimeError("EDGE_SEAM_DIAG_PREREG_DRIFT")

    rr = _ctx(a.authority_root, a.run_id)["run_root"]
    cand = exact(rr, "candidate", canonical_mesh_candidate_from_dict)
    sk = exact(rr, "skeleton", qualified_skeleton_from_dict)
    cams = tuple(
        sorted(
            exact(rr, "cameras", qualified_camera_set_from_dict).cameras,
            key=lambda x: int(x.view_index),
        )
    )
    env = load(
        rr / "artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",
        deformation_envelope_from_dict,
    )
    policy = load(
        rr / "artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",
        mesh_policy_from_dict,
    )
    surface = load(
        rr / "artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",
        rigging_surface_from_dict,
    )

    P = np.asarray([v.P for v in cand.vertices], dtype=np.float64)
    F = face_indices(cand)
    F_GLOBAL = F
    dense = dense_supported_face_mask(rr, cand, surface)

    with np.load(a.weights_npz, allow_pickle=False) as z:
        jids = tuple(map(str, z["joint_ids"].tolist()))
        W = np.asarray(z["arachne"], dtype=np.float64)
    if W.shape != (len(P), len(jids)):
        raise RuntimeError("EDGE_SEAM_WEIGHT_SHAPE_DRIFT")

    stress = stress_arbitrary_weights(P, W, F, jids, sk, cams, env, policy)
    unsafe = np.zeros(len(F), dtype=bool)
    unsafe[np.asarray(stress["unsafe_face_indices"], dtype=np.int64)] = True

    edges, edge_faces, face_edge_index = _edge_table(P, F)
    max_stretch = _edge_stretch(P, W, edges, jids, sk, cams, env)
    weight_l1 = np.sum(np.abs(W[edges[:, 0]] - W[edges[:, 1]]), axis=1)

    # Scope candidate seam decisions to edges touching an unsafe dense face.
    in_scope = np.zeros(len(edges), dtype=bool)
    for ei, edge in enumerate(map(tuple, edges.tolist())):
        in_scope[ei] = any(bool(unsafe[fi] and dense[fi]) for fi in edge_faces[edge])

    # Teacher begins only here.
    truth = _truth_region(a.teacher_bank, a.teacher_source, sk, cand)
    truth_seam_edge = truth[edges[:, 0]] != truth[edges[:, 1]]

    rules = {
        "STRETCH_GT4": in_scope & (max_stretch > 4.0),
        "WEIGHT_L1_GT1": in_scope & (weight_l1 > 1.0),
        "STRETCH_GT4_AND_WEIGHT_L1_GT1": in_scope
        & (max_stretch > 4.0)
        & (weight_l1 > 1.0),
        "STRETCH_GT4_OR_WEIGHT_L1_GT1": in_scope
        & ((max_stretch > 4.0) | (weight_l1 > 1.0)),
    }

    edge_metrics = {}
    for name, pred in rules.items():
        tp = int(np.count_nonzero(pred & truth_seam_edge))
        fp = int(np.count_nonzero(pred & (~truth_seam_edge)))
        fn = int(np.count_nonzero((~pred) & truth_seam_edge & in_scope))
        edge_metrics[name] = {
            "predicted_seam_edge_count": int(np.count_nonzero(pred)),
            "true_positive_edge_count": tp,
            "false_positive_edge_count": fp,
            "false_negative_in_scope_edge_count": fn,
            "edge_precision": float(tp / max(1, tp + fp)),
            "edge_recall_in_scope": float(tp / max(1, tp + fn)),
        }

    face_metrics = {
        name: _rule_metrics(name, pred, face_edge_index, truth, unsafe)
        for name, pred in rules.items()
    }

    comps = unsafe_components(F, unsafe, dense)
    giant = comps[:2]
    giant_mask = np.zeros(len(F), dtype=bool)
    for comp in giant:
        giant_mask[np.asarray(comp, dtype=np.int64)] = True
    truth_mixed = np.asarray(
        [len(set(int(truth[v]) for v in row)) > 1 for row in F],
        dtype=bool,
    )
    giant_metrics = {}
    for name, pred in rules.items():
        pm = np.any(pred[face_edge_index], axis=1)
        denom = int(np.count_nonzero(giant_mask & truth_mixed))
        hit = int(np.count_nonzero(giant_mask & truth_mixed & pm))
        predicted = int(np.count_nonzero(giant_mask & pm))
        tp = hit
        fp = int(np.count_nonzero(giant_mask & pm & (~truth_mixed)))
        giant_metrics[name] = {
            "giant_patch_truth_cross_region_face_count": denom,
            "giant_patch_recovered_cross_region_face_count": hit,
            "giant_patch_cross_region_recall": float(hit / max(1, denom)),
            "giant_patch_predicted_mixed_face_count": predicted,
            "giant_patch_precision": float(tp / max(1, tp + fp)),
        }

    report = {
        "schema": "RealSaS.KnightTeacherFreeEdgeSeamDiagnosis.v1",
        "status": "DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
        "preregistration": str(PREREG),
        "teacher_used_by_edge_scores": False,
        "scope": {
            "face_count": int(len(F)),
            "edge_count": int(len(edges)),
            "unsafe_face_count": int(np.count_nonzero(unsafe)),
            "in_scope_edge_count": int(np.count_nonzero(in_scope)),
            "largest_unsafe_patch_face_counts": [int(len(x)) for x in comps[:10]],
        },
        "score_distribution": {
            "max_stretch_p50_in_scope": float(np.quantile(max_stretch[in_scope], 0.5)),
            "max_stretch_p95_in_scope": float(np.quantile(max_stretch[in_scope], 0.95)),
            "max_stretch_p99_in_scope": float(np.quantile(max_stretch[in_scope], 0.99)),
            "max_stretch_max_in_scope": float(np.max(max_stretch[in_scope])),
            "weight_l1_p50_in_scope": float(np.quantile(weight_l1[in_scope], 0.5)),
            "weight_l1_p95_in_scope": float(np.quantile(weight_l1[in_scope], 0.95)),
            "weight_l1_p99_in_scope": float(np.quantile(weight_l1[in_scope], 0.99)),
            "weight_l1_max_in_scope": float(np.max(weight_l1[in_scope])),
        },
        "edge_metrics_teacher_eval_only": edge_metrics,
        "face_metrics_teacher_eval_only": face_metrics,
        "giant_patch_metrics_teacher_eval_only": giant_metrics,
        "claim_boundary": (
            "Synthetic stretch and Arachne weight-L1 scores are computed before teacher "
            "source topology is loaded. Teacher labels only evaluate the four frozen rules; "
            "no threshold search or product mutation occurs."
        ),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_TEACHER_FREE_EDGE_SEAM_DIAG_PASS",
        json.dumps(
            {
                "scope": report["scope"],
                "edge_metrics": edge_metrics,
                "face_metrics": face_metrics,
                "giant_patch_metrics": giant_metrics,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
