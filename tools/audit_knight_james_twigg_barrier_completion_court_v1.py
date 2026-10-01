from __future__ import annotations

import argparse
import heapq
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
from tools.audit_knight_james_twigg_region_inference_v1 import (
    EPSILON,
    PROBE_DEGREES,
    _cluster_signatures,
    _rotation_signatures,
    _split_holeless,
    _teacher_region_eval,
    _vertex_labels_from_faces,
)
from tools.audit_knight_james_twigg_unsafe_patch_completion_court_v1 import (
    unsafe_components,
)
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import (
    _edge_stretch,
    _edge_table,
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


PREREG = Path("canonical/KNIGHT_JAMES_TWIGG_BARRIER_COMPLETION_PREREG_V1_20260929.json")
CLOSURE_PREREG = Path("canonical/KNIGHT_TEACHER_FREE_TOPOLOGY_CLOSURE_PREREG_V1_20260929.json")


def _filtered_dense_graph(P, F, dense, barrier_keys):
    edges = {}
    for fi in np.where(dense)[0]:
        a, b, c = map(int, F[int(fi)])
        for x, y in ((a, b), (b, c), (c, a)):
            if x > y:
                x, y = y, x
            if (x, y) in barrier_keys:
                continue
            w = float(np.linalg.norm(P[x] - P[y]))
            if not np.isfinite(w) or w <= 1e-15:
                continue
            edges[(x, y)] = min(edges.get((x, y), np.inf), w)
    nbr = defaultdict(list)
    for (a, b), w in edges.items():
        nbr[a].append((b, w))
        nbr[b].append((a, w))
    return nbr, edges


def _complete_with_barriers(P, F, dense, unsafe, initial_label, barrier_keys):
    labels = np.asarray(initial_label, dtype=np.int64).copy()
    nbr, kept_edges = _filtered_dense_graph(P, F, dense, barrier_keys)

    vfaces = defaultdict(list)
    for fi, face in enumerate(F.tolist()):
        for v in face:
            vfaces[int(v)].append(int(fi))

    comps = unsafe_components(F, unsafe, dense)
    reports = []
    total_new = 0
    total_unreachable = 0

    for ci, comp in enumerate(comps):
        patch_vertices = sorted(set(int(v) for fi in comp for v in F[int(fi)]))
        patch_set = set(patch_vertices)

        seed_map = {}
        # Existing teacher-free labels inside the patch are immutable anchors.
        for v in patch_vertices:
            if labels[v] >= 0:
                seed_map[int(v)] = int(labels[v])

        # Safe exterior labels may enter only across non-barrier dense edges.
        for u in patch_vertices:
            for v, _ in nbr.get(u, ()):
                if v in patch_set or labels[v] < 0:
                    continue
                safe_neighbor = any(
                    (not unsafe[fi]) and dense[fi] for fi in vfaces[int(v)]
                )
                if safe_neighbor:
                    seed_map.setdefault(int(v), int(labels[v]))

        unique_labels = sorted(set(seed_map.values()))
        if len(unique_labels) < 2:
            reports.append(
                {
                    "component_index": int(ci),
                    "unsafe_face_count": int(len(comp)),
                    "vertex_count": int(len(patch_vertices)),
                    "seed_count": int(len(seed_map)),
                    "seed_label_count": int(len(unique_labels)),
                    "completed_vertex_count": 0,
                    "unreachable_patch_vertex_count": int(
                        sum(labels[v] < 0 for v in patch_vertices)
                    ),
                    "status": "ABSTAIN_LT2_BARRIER_SEPARATED_LABELS",
                }
            )
            continue

        allowed = patch_set | set(seed_map)
        best = {v: np.inf for v in allowed}
        best_lab = {v: -1 for v in allowed}
        heap = []
        for v, lab in sorted(seed_map.items()):
            best[v] = 0.0
            best_lab[v] = int(lab)
            heapq.heappush(heap, (0.0, int(lab), int(v)))

        while heap:
            d, lab, u = heapq.heappop(heap)
            if d != best[u] or lab != best_lab[u]:
                continue
            for v, w in nbr.get(u, ()):
                if v not in allowed:
                    continue
                nd = d + w
                if nd < best[v] - 1e-15 or (
                    abs(nd - best[v]) <= 1e-15
                    and (best_lab[v] < 0 or lab < best_lab[v])
                ):
                    best[v] = nd
                    best_lab[v] = lab
                    heapq.heappush(heap, (nd, lab, v))

        changed = 0
        unreachable = 0
        for v in patch_vertices:
            if labels[v] >= 0:
                continue
            if np.isfinite(best.get(v, np.inf)) and best_lab.get(v, -1) >= 0:
                labels[v] = int(best_lab[v])
                changed += 1
            else:
                unreachable += 1

        total_new += changed
        total_unreachable += unreachable
        reports.append(
            {
                "component_index": int(ci),
                "unsafe_face_count": int(len(comp)),
                "vertex_count": int(len(patch_vertices)),
                "seed_count": int(len(seed_map)),
                "seed_label_count": int(len(unique_labels)),
                "completed_vertex_count": int(changed),
                "unreachable_patch_vertex_count": int(unreachable),
                "status": "COMPLETED" if unreachable == 0 else "PARTIAL",
            }
        )

    return labels, reports, {
        "unsafe_patch_count": int(len(comps)),
        "newly_labeled_vertex_count": int(total_new),
        "unreachable_patch_vertex_count": int(total_unreachable),
        "kept_dense_edge_count": int(len(kept_edges)),
        "barrier_edge_count": int(len(barrier_keys)),
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
    if prereg.get("status") != "FROZEN_BEFORE_BARRIER_COMPLETION_RESULT":
        raise RuntimeError("JT_BARRIER_PREREG_DRIFT")
    th = json.loads(CLOSURE_PREREG.read_text())["closure_thresholds"]

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
        raise RuntimeError("JT_BARRIER_WEIGHT_SHAPE_DRIFT")

    infer_stress = stress_arbitrary_weights(P, Wa, F, jids, sk, cams, env, policy)
    unsafe = np.zeros(len(F), dtype=bool)
    unsafe[np.asarray(infer_stress["unsafe_face_indices"], dtype=np.int64)] = True
    safe = (~unsafe) & dense

    Z, valid, probes = _rotation_signatures(P, F, Wa, jids, sk, cams, PROBE_DEGREES)
    face_label, core, cluster_meta = _cluster_signatures(Z, safe & valid, EPSILON)
    initial_label, confidence, prop_meta = _vertex_labels_from_faces(
        P, F, face_label, core, dense, safe
    )

    edges, edge_faces, face_edge_index = _edge_table(P, F)
    edge_stretch = _edge_stretch(P, Wa, edges, jids, sk, cams, env)
    in_scope = np.zeros(len(edges), dtype=bool)
    for ei, edge in enumerate(map(tuple, edges.tolist())):
        in_scope[ei] = any(bool(unsafe[fi] and dense[fi]) for fi in edge_faces[edge])
    barrier_mask = in_scope & (edge_stretch > 4.0)
    barrier_keys = {
        tuple(map(int, edges[i])) for i in np.where(barrier_mask)[0].tolist()
    }

    inferred_label, patch_reports, patch_meta = _complete_with_barriers(
        P, F, dense, unsafe, initial_label, barrier_keys
    )

    # Teacher begins only after topology labels are frozen.
    teacher_eval = _teacher_region_eval(
        a.teacher_bank, a.teacher_source, sk, cand, inferred_label, F, unsafe
    )

    Wt, tjids, _, _ = teacher_weights(a.teacher_bank, a.teacher_source, sk, cand)
    tix = {str(j): i for i, j in enumerate(tjids)}
    missing = [j for j in jids if str(j) not in tix]
    if missing:
        raise RuntimeError("JT_BARRIER_TEACHER_JOINT_DRIFT:" + json.dumps(missing))
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
        "schema": "RealSaS.KnightJamesTwiggBarrierCompletionCourt.v1",
        "status": "PASS_TOPOLOGY_CLOSED_FOR_KNIGHT" if closure else "FAIL_TOPOLOGY_REMAINS_OPEN",
        "preregistration": str(PREREG),
        "closure_preregistration": str(CLOSURE_PREREG),
        "teacher_used_by_inference": False,
        "operator": {
            "label_authority": "JAMES_TWIGG_SAFE_ROTATION_SIGNATURES",
            "barrier_source": "STAGE35_CONTROLLED_EDGE_STRETCH",
            "barrier_threshold": 4.0,
            "barrier_semantics": "CANNOT_CROSS_ONLY",
            "completion": "PATCH_LOCAL_MULTI_SOURCE_GEODESIC_ON_BARRIER_FILTERED_DENSE_GRAPH",
            "holeless_repair": "VERTEX_DUPLICATION_PLUS_REGION_PURE_SUBDIVISION",
        },
        "probe": {"degrees": PROBE_DEGREES, "count": len(probes)},
        "clustering": cluster_meta,
        "initial_propagation": prop_meta,
        "barrier": {
            "candidate_edge_count": int(len(edges)),
            "in_scope_edge_count": int(np.count_nonzero(in_scope)),
            "barrier_edge_count": int(np.count_nonzero(barrier_mask)),
            "stretch_p95_barrier": float(np.quantile(edge_stretch[barrier_mask], 0.95))
            if np.any(barrier_mask)
            else None,
        },
        "patch_completion": {
            **patch_meta,
            "completed_patch_count": sum(r["status"] == "COMPLETED" for r in patch_reports),
            "partial_patch_count": sum(r["status"] == "PARTIAL" for r in patch_reports),
            "abstained_patch_count": sum(
                str(r["status"]).startswith("ABSTAIN") for r in patch_reports
            ),
        },
        "patches": patch_reports,
        "teacher_evaluation_only": teacher_eval,
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
        "claim_boundary": (
            "Stage35 stretch creates no region labels and directly authorizes no split. "
            "It only removes cannot-cross edges from James/Twigg patch propagation. "
            "Teacher topology evaluates frozen labels; teacher weights are introduced "
            "after inference solely for topology-isolated mechanics."
        ),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_JT_BARRIER_COMPLETION_" + ("PASS" if closure else "FAIL"),
        json.dumps(
            {
                "barrier": report["barrier"],
                "patch_completion": report["patch_completion"],
                "teacher_eval": teacher_eval,
                "after_motion": report["topology_isolated_mechanics"]["after_motion"],
                "after_stress": report["topology_isolated_mechanics"]["after_stress"],
                "checks": checks,
                "closure_pass": closure,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
