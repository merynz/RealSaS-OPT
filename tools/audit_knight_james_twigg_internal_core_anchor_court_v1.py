from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from tools.audit_knight_james_twigg_region_inference_v1 import (
    EPSILON,
    MAX_MEANSHIFT_SEEDS,
    PCA_COMPONENTS,
    PROBE_DEGREES,
    _deterministic_seed_indices,
    _face_areas,
    _rotation_signatures,
    _split_holeless,
    _teacher_region_eval,
    _vertex_labels_from_faces,
)
from tools.audit_knight_james_twigg_unsafe_patch_completion_court_v1 import (
    complete_unsafe_patches,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask,
    exact,
    face_indices,
    load,
    motion_metrics,
    stress_arbitrary_weights,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.demo.render_knight_motion_preview_v1 import _ctx


PREREG = Path("canonical/KNIGHT_JAMES_TWIGG_INTERNAL_CORE_ANCHOR_PREREG_V1_20260929.json")
CLOSURE_PREREG = Path("canonical/KNIGHT_TEACHER_FREE_TOPOLOGY_CLOSURE_PREREG_V1_20260929.json")
MIN_SUPPORT_FACES = 2
MIN_VOTE_CONFIDENCE = 0.8


def _fit_safe_modes_and_project_all(Z, fit_mask, valid):
    idx = np.where(np.asarray(fit_mask, dtype=bool))[0]
    if len(idx) < 2:
        raise RuntimeError("JT_INTERNAL_TOO_FEW_SAFE_FACES")
    X = Z[idx].astype(np.float64)
    nc = min(PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=nc, svd_solver="randomized", random_state=0)
    Xp = pca.fit_transform(X)
    seeds = _deterministic_seed_indices(Xp, MAX_MEANSHIFT_SEEDS)
    ms = MeanShift(
        bandwidth=float(EPSILON),
        bin_seeding=False,
        cluster_all=True,
        max_iter=200,
        n_jobs=-1,
    )
    ms.fit(Xp[seeds])
    centers = np.asarray(ms.cluster_centers_, dtype=np.float64)
    if not len(centers):
        raise RuntimeError("JT_INTERNAL_NO_MODES")
    tree = cKDTree(centers)

    face_label = np.full(len(Z), -1, dtype=np.int64)
    mode_dist = np.full(len(Z), np.inf, dtype=np.float64)
    vidx = np.where(np.asarray(valid, dtype=bool))[0]
    Vall = pca.transform(Z[vidx].astype(np.float64))
    dist, lab = tree.query(Vall, k=1)
    face_label[vidx] = np.asarray(lab, dtype=np.int64)
    mode_dist[vidx] = np.asarray(dist, dtype=np.float64)

    safe_dist = mode_dist[idx]
    safe_core = np.zeros(len(Z), dtype=bool)
    safe_core[idx] = safe_dist <= float(EPSILON) / 4.0

    return face_label, mode_dist, safe_core, {
        "cluster_count": int(len(centers)),
        "fit_face_count": int(len(idx)),
        "seed_face_count": int(len(seeds)),
        "pca_components": int(nc),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "safe_core_face_count": int(np.count_nonzero(safe_core)),
        "safe_core_fraction": float(np.count_nonzero(safe_core) / max(1, len(idx))),
        "safe_mode_distance_p50": float(np.quantile(safe_dist, 0.5)),
        "safe_mode_distance_p95": float(np.quantile(safe_dist, 0.95)),
        "safe_mode_distance_max": float(np.max(safe_dist)),
    }


def _augment_unlabeled_vertices_from_unsafe_core(
    P,
    F,
    unsafe,
    projected_label,
    mode_dist,
    initial_label,
):
    labels = np.asarray(initial_label, dtype=np.int64).copy()
    area = _face_areas(P, F)
    projected_core = (
        np.asarray(unsafe, dtype=bool)
        & (np.asarray(projected_label) >= 0)
        & np.isfinite(mode_dist)
        & (np.asarray(mode_dist) <= float(EPSILON) / 4.0)
    )

    votes = [defaultdict(float) for _ in range(len(P))]
    counts = [defaultdict(int) for _ in range(len(P))]
    existing_disagreement_face_incidence = 0
    for fi in np.where(projected_core)[0]:
        lab = int(projected_label[fi])
        w = float(area[fi])
        for v0 in F[fi]:
            v = int(v0)
            votes[v][lab] += w
            counts[v][lab] += 1
            if labels[v] >= 0 and labels[v] != lab:
                existing_disagreement_face_incidence += 1

    added = 0
    rejected_support = 0
    rejected_confidence = 0
    added_by_label = defaultdict(int)
    for v, d in enumerate(votes):
        if labels[v] >= 0 or not d:
            continue
        rows = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
        top_lab, top_weight = rows[0]
        total_weight = sum(x[1] for x in rows)
        conf = float(top_weight / max(total_weight, 1e-15))
        support = int(counts[v][int(top_lab)])
        if support < MIN_SUPPORT_FACES:
            rejected_support += 1
            continue
        if conf < MIN_VOTE_CONFIDENCE:
            rejected_confidence += 1
            continue
        labels[v] = int(top_lab)
        added += 1
        added_by_label[int(top_lab)] += 1

    return labels, projected_core, {
        "projected_core_unsafe_face_count": int(np.count_nonzero(projected_core)),
        "projected_core_unsafe_fraction": float(
            np.count_nonzero(projected_core) / max(1, np.count_nonzero(unsafe))
        ),
        "internal_anchor_vertex_count": int(added),
        "rejected_vertex_count_min_support": int(rejected_support),
        "rejected_vertex_count_vote_confidence": int(rejected_confidence),
        "existing_label_disagreement_face_incidence": int(
            existing_disagreement_face_incidence
        ),
        "added_anchor_vertices_by_label": {
            str(k): int(v) for k, v in sorted(added_by_label.items())
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--weights-npz", type=Path, required=True)
    p.add_argument("--teacher-bank", type=Path, required=True)
    p.add_argument("--teacher-source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    prereg = json.loads(PREREG.read_text())
    if prereg.get("status") != "FROZEN_BEFORE_INTERNAL_CORE_ANCHOR_RESULT":
        raise RuntimeError("JT_INTERNAL_PREREG_DRIFT")
    closure_prereg = json.loads(CLOSURE_PREREG.read_text())
    th = closure_prereg["closure_thresholds"]

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
    dense = dense_supported_face_mask(rr, cand, surface)

    with np.load(a.weights_npz, allow_pickle=False) as z:
        jids = tuple(map(str, z["joint_ids"].tolist()))
        Wa = np.asarray(z["arachne"], dtype=np.float64)
    if Wa.shape != (len(P), len(jids)):
        raise RuntimeError("JT_INTERNAL_WEIGHT_SHAPE_DRIFT")

    infer_stress = stress_arbitrary_weights(P, Wa, F, jids, sk, cams, env, policy)
    unsafe = np.zeros(len(F), dtype=bool)
    unsafe[np.asarray(infer_stress["unsafe_face_indices"], dtype=np.int64)] = True
    safe = (~unsafe) & dense

    Z, valid, probes = _rotation_signatures(P, F, Wa, jids, sk, cams, PROBE_DEGREES)
    projected_label, mode_dist, safe_core, cluster_meta = (
        _fit_safe_modes_and_project_all(Z, safe & valid, valid)
    )

    # Reconstruct the exact existing safe-label path using the frozen safe modes.
    safe_face_label = np.full(len(F), -1, dtype=np.int64)
    safe_face_label[safe & valid] = projected_label[safe & valid]
    initial_label, confidence, prop_meta = _vertex_labels_from_faces(
        P, F, safe_face_label, safe_core, dense, safe
    )

    baseline_label, baseline_patch_reports, baseline_patch_meta = complete_unsafe_patches(
        P, F, dense, unsafe, initial_label
    )
    baseline_eval = _teacher_region_eval(
        a.teacher_bank, a.teacher_source, sk, cand, baseline_label, F, unsafe
    )

    anchored_label, projected_core, anchor_meta = (
        _augment_unlabeled_vertices_from_unsafe_core(
            P, F, unsafe, projected_label, mode_dist, initial_label
        )
    )
    inferred_label, patch_reports, patch_meta = complete_unsafe_patches(
        P, F, dense, unsafe, anchored_label
    )
    teacher_eval = _teacher_region_eval(
        a.teacher_bank, a.teacher_source, sk, cand, inferred_label, F, unsafe
    )

    # Teacher weights are introduced only after inference is frozen.
    Wt, tjids, _, _ = teacher_weights(a.teacher_bank, a.teacher_source, sk, cand)
    tix = {str(j): i for i, j in enumerate(tjids)}
    missing = [j for j in jids if str(j) not in tix]
    if missing:
        raise RuntimeError("JT_INTERNAL_TEACHER_JOINT_DRIFT:" + json.dumps(missing))
    Wt = np.stack([Wt[:, tix[str(j)]] for j in jids], axis=1)
    Wt = np.maximum(Wt, 0.0)
    Wt /= np.maximum(Wt.sum(axis=1, keepdims=True), 1e-15)

    P2, W2, F2, split_meta = _split_holeless(P, Wt, F, inferred_label)
    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    before_motion = motion_metrics(P, Wt, F, jids, sk, cams, rr, source_report)
    after_motion = motion_metrics(P2, W2, F2, jids, sk, cams, rr, source_report)
    before_stress = stress_arbitrary_weights(P, Wt, F, jids, sk, cams, env, policy)
    after_stress = stress_arbitrary_weights(P2, W2, F2, jids, sk, cams, env, policy)

    checks = {
        "no_face_deletion": int(split_meta["face_deletion_count"])
        <= int(th["face_deletion_count_max"]),
        "rest_area_preserved": float(split_meta["rest_area_error_max"])
        <= float(th["rest_area_error_max"]),
        "actual_gt10": int(after_motion["max_edge_gt_10"])
        <= int(th["actual_motion_max_edge_gt_10_max"]),
        "actual_gt4": int(after_motion["max_edge_gt_4"])
        <= int(th["actual_motion_max_edge_gt_4_max"]),
        "actual_worst_edge": float(after_motion["worst_edge_max"])
        <= float(th["actual_motion_worst_edge_max"]),
        "synthetic_unsafe": int(after_stress["unsafe_face_count"])
        <= int(th["synthetic_unsafe_face_count_max"]),
        "mixed_precision": float(teacher_eval["mixed_face_precision"])
        >= float(th["mixed_face_precision_min"]),
        "unsafe_cross_region_recall": float(
            teacher_eval["unsafe_truth_cross_region_recall"]
        )
        >= float(th["unsafe_cross_region_recall_min"]),
    }
    closure = all(checks.values())

    report = {
        "schema": "RealSaS.KnightJamesTwiggInternalCoreAnchorCourt.v1",
        "status": "PASS_TOPOLOGY_CLOSED_FOR_KNIGHT" if closure else "FAIL_TOPOLOGY_REMAINS_OPEN",
        "preregistration": str(PREREG),
        "closure_preregistration": str(CLOSURE_PREREG),
        "teacher_used_by_inference": False,
        "teacher_topology_used_for_evaluation_only": True,
        "teacher_weights_used_for_post_inference_mechanical_isolation_only": True,
        "operator": {
            "mode_fit": "SAFE_DENSE_VALID_ONLY",
            "unsafe_projection": "FROZEN_PCA_AND_MEANSHIFT_MODES",
            "unsafe_core_threshold": float(EPSILON) / 4.0,
            "minimum_incident_projected_core_faces": MIN_SUPPORT_FACES,
            "minimum_area_weighted_label_confidence": MIN_VOTE_CONFIDENCE,
            "holeless_repair": "VERTEX_DUPLICATION_PLUS_REGION_PURE_SUBDIVISION",
        },
        "probe": {"degrees": PROBE_DEGREES, "count": len(probes)},
        "clustering": cluster_meta,
        "initial_propagation": prop_meta,
        "internal_anchor": anchor_meta,
        "baseline_teacher_evaluation_only": baseline_eval,
        "teacher_evaluation_only": teacher_eval,
        "patch_completion": {
            **patch_meta,
            "completed_patch_count": sum(r["status"] == "COMPLETED" for r in patch_reports),
            "partial_patch_count": sum(r["status"] == "PARTIAL" for r in patch_reports),
            "abstained_patch_count": sum(
                str(r["status"]).startswith("ABSTAIN") for r in patch_reports
            ),
        },
        "holeless_split": split_meta,
        "topology_isolated_mechanics": {
            "before_motion": {k: v for k, v in before_motion.items() if k != "frames"},
            "after_motion": {k: v for k, v in after_motion.items() if k != "frames"},
            "before_stress": {
                k: v for k, v in before_stress.items() if k != "unsafe_face_indices"
            },
            "after_stress": {
                k: v for k, v in after_stress.items() if k != "unsafe_face_indices"
            },
        },
        "frozen_thresholds": th,
        "checks": checks,
        "closure_pass": closure,
        "finding": {
            "internal_anchor_improves_unsafe_cross_region_recall": bool(
                teacher_eval["unsafe_truth_cross_region_recall"]
                > baseline_eval["unsafe_truth_cross_region_recall"]
            ),
            "internal_anchor_preserves_or_improves_mixed_precision": bool(
                teacher_eval["mixed_face_precision"]
                >= baseline_eval["mixed_face_precision"]
            ),
        },
        "claim_boundary": (
            "Unsafe faces do not learn PCA or mean-shift modes. Only strict h/4 projections "
            "onto frozen safe modes may seed previously unlabeled vertices. Teacher topology "
            "is evaluation-only; teacher weights are introduced only after inference is frozen."
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_JT_INTERNAL_CORE_ANCHOR_" + ("PASS" if closure else "FAIL"),
        json.dumps(
            {
                "internal_anchor": anchor_meta,
                "baseline_teacher_eval": baseline_eval,
                "teacher_eval": teacher_eval,
                "patch_completion": report["patch_completion"],
                "after_motion": report["topology_isolated_mechanics"]["after_motion"],
                "after_stress": report["topology_isolated_mechanics"]["after_stress"],
                "checks": checks,
                "closure_pass": closure,
                "finding": report["finding"],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
