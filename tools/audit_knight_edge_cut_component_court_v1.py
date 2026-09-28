from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
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
from tools.audit_knight_edge_seam_two_core_court_v1 import _two_core
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import (
    _edge_stretch,
    _edge_table,
    _truth_region,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask,
    exact,
    face_indices,
    load,
    stress_arbitrary_weights,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx


PREREG = Path("canonical/KNIGHT_EDGE_CUT_COMPONENT_PREREG_V1_20260929.json")


def _components(n, edges, keep):
    nbr = [[] for _ in range(n)]
    for ei in np.where(keep)[0]:
        a, b = map(int, edges[int(ei)])
        nbr[a].append(b)
        nbr[b].append(a)
    label = np.full(n, -1, dtype=np.int64)
    sizes = []
    cid = 0
    for s in range(n):
        if label[s] >= 0:
            continue
        q = [s]
        label[s] = cid
        count = 0
        while q:
            u = q.pop()
            count += 1
            for v in nbr[u]:
                if label[v] < 0:
                    label[v] = cid
                    q.append(v)
        sizes.append(count)
        cid += 1
    return label, sizes


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
    if prereg.get("status") != "FROZEN_BEFORE_EDGE_CUT_COMPONENT_RESULT":
        raise RuntimeError("EDGE_CUT_COMPONENT_PREREG_DRIFT")

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
        W = np.asarray(z["arachne"], dtype=np.float64)

    stress = stress_arbitrary_weights(P, W, F, jids, sk, cams, env, policy)
    unsafe = np.zeros(len(F), dtype=bool)
    unsafe[np.asarray(stress["unsafe_face_indices"], dtype=np.int64)] = True

    edges, edge_faces, face_edge_index = _edge_table(P, F)
    stretch = _edge_stretch(P, W, edges, jids, sk, cams, env)
    l1 = np.sum(np.abs(W[edges[:, 0]] - W[edges[:, 1]]), axis=1)

    dense_edge = np.zeros(len(edges), dtype=bool)
    in_scope = np.zeros(len(edges), dtype=bool)
    boundary_vertex = np.zeros(len(P), dtype=bool)
    dense_face_set = set(map(int, np.where(dense)[0]))
    for ei, edge in enumerate(map(tuple, edges.tolist())):
        fs = edge_faces[edge]
        dense_edge[ei] = any(int(fi) in dense_face_set for fi in fs)
        in_scope[ei] = any(bool(unsafe[fi] and dense[fi]) for fi in fs)
        if len(fs) == 1:
            boundary_vertex[int(edge[0])] = True
            boundary_vertex[int(edge[1])] = True

    base = in_scope & (stretch > 4.0) & (l1 > 1.0)
    seam = _two_core(edges, base, boundary_vertex)
    keep = dense_edge & (~seam)
    region, sizes = _components(len(P), edges, keep)

    # Teacher begins only here.
    truth = _truth_region(a.teacher_bank, a.teacher_source, sk, cand)
    truth_mixed = np.asarray(
        [len(set(int(truth[v]) for v in row)) > 1 for row in F], dtype=bool
    )
    pred_mixed = np.asarray(
        [len(set(int(region[v]) for v in row)) > 1 for row in F], dtype=bool
    )
    tp = int(np.count_nonzero(pred_mixed & truth_mixed))
    fp = int(np.count_nonzero(pred_mixed & (~truth_mixed)))
    unsafe_truth = unsafe & truth_mixed
    uhit = int(np.count_nonzero(pred_mixed & unsafe_truth))

    metrics = {
        "region_count": int(len(sizes)),
        "largest_region_sizes": list(map(int, sorted(sizes, reverse=True)[:20])),
        "seam_edge_count": int(np.count_nonzero(seam)),
        "predicted_mixed_face_count": int(np.count_nonzero(pred_mixed)),
        "mixed_face_precision": float(tp / max(1, tp + fp)),
        "mixed_face_recall": float(tp / max(1, np.count_nonzero(truth_mixed))),
        "unsafe_truth_cross_region_face_count": int(np.count_nonzero(unsafe_truth)),
        "unsafe_truth_cross_region_recovered": uhit,
        "unsafe_truth_cross_region_recall": float(
            uhit / max(1, np.count_nonzero(unsafe_truth))
        ),
    }
    report = {
        "schema": "RealSaS.KnightEdgeCutComponentCourt.v1",
        "status": "DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
        "preregistration": str(PREREG),
        "teacher_used_by_operator": False,
        "metrics_teacher_eval_only": metrics,
        "passes_frozen_label_gates": bool(
            metrics["mixed_face_precision"] >= 0.90
            and metrics["unsafe_truth_cross_region_recall"] >= 0.70
        ),
        "claim_boundary": (
            "Edge scoring, 2-core pruning, dense-graph cutting, and connected components "
            "are all frozen before teacher source topology is loaded."
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_EDGE_CUT_COMPONENT_PASS", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
