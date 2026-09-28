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


PREREG = Path("canonical/KNIGHT_EDGE_SEAM_TWO_CORE_PREREG_V1_20260929.json")


def _two_core(edges, predicted, boundary_vertex):
    active = np.asarray(predicted, dtype=bool).copy()
    incident = defaultdict(list)
    for ei in np.where(active)[0]:
        a, b = map(int, edges[int(ei)])
        incident[a].append(int(ei))
        incident[b].append(int(ei))

    degree = defaultdict(int)
    for v, eis in incident.items():
        degree[v] = sum(active[ei] for ei in eis)

    q = deque(
        sorted(
            v
            for v, d in degree.items()
            if d < 2 and not bool(boundary_vertex[v])
        )
    )
    queued = set(q)
    while q:
        v = q.popleft()
        queued.discard(v)
        if bool(boundary_vertex[v]) or degree.get(v, 0) >= 2:
            continue
        for ei in incident.get(v, ()):
            if not active[ei]:
                continue
            active[ei] = False
            a, b = map(int, edges[ei])
            for u in (a, b):
                degree[u] = max(0, degree.get(u, 0) - 1)
                if (
                    degree[u] < 2
                    and not bool(boundary_vertex[u])
                    and u not in queued
                ):
                    q.append(u)
                    queued.add(u)
    return active


def _metrics(pred, truth_seam_edge, face_edge_index, truth, unsafe, F):
    tp = int(np.count_nonzero(pred & truth_seam_edge))
    fp = int(np.count_nonzero(pred & (~truth_seam_edge)))
    pred_face = np.any(pred[face_edge_index], axis=1)
    truth_mixed = np.asarray(
        [len(set(int(truth[v]) for v in row)) > 1 for row in F], dtype=bool
    )
    ftp = int(np.count_nonzero(pred_face & truth_mixed))
    ffp = int(np.count_nonzero(pred_face & (~truth_mixed)))
    unsafe_truth = unsafe & truth_mixed
    uhit = int(np.count_nonzero(pred_face & unsafe_truth))
    return {
        "edge_count": int(np.count_nonzero(pred)),
        "edge_precision": float(tp / max(1, tp + fp)),
        "predicted_mixed_face_count": int(np.count_nonzero(pred_face)),
        "mixed_face_precision": float(ftp / max(1, ftp + ffp)),
        "unsafe_truth_cross_region_face_count": int(np.count_nonzero(unsafe_truth)),
        "unsafe_truth_cross_region_recovered": uhit,
        "unsafe_truth_cross_region_recall": float(
            uhit / max(1, np.count_nonzero(unsafe_truth))
        ),
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
    if prereg.get("status") != "FROZEN_BEFORE_EDGE_SEAM_TWO_CORE_RESULT":
        raise RuntimeError("EDGE_TWO_CORE_PREREG_DRIFT")

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

    in_scope = np.zeros(len(edges), dtype=bool)
    boundary_vertex = np.zeros(len(P), dtype=bool)
    for ei, edge in enumerate(map(tuple, edges.tolist())):
        fs = edge_faces[edge]
        in_scope[ei] = any(bool(unsafe[fi] and dense[fi]) for fi in fs)
        if len(fs) == 1:
            boundary_vertex[int(edge[0])] = True
            boundary_vertex[int(edge[1])] = True

    raw = {
        "STRETCH_GT4": in_scope & (stretch > 4.0),
        "STRETCH_GT4_AND_WEIGHT_L1_GT1": in_scope
        & (stretch > 4.0)
        & (l1 > 1.0),
    }
    core = {name: _two_core(edges, pred, boundary_vertex) for name, pred in raw.items()}

    # Teacher begins only here.
    truth = _truth_region(a.teacher_bank, a.teacher_source, sk, cand)
    truth_seam_edge = truth[edges[:, 0]] != truth[edges[:, 1]]

    result = {}
    for name in raw:
        result[name] = {
            "raw": _metrics(
                raw[name], truth_seam_edge, face_edge_index, truth, unsafe, F
            ),
            "two_core": _metrics(
                core[name], truth_seam_edge, face_edge_index, truth, unsafe, F
            ),
            "removed_edge_count": int(
                np.count_nonzero(raw[name]) - np.count_nonzero(core[name])
            ),
            "retained_fraction": float(
                np.count_nonzero(core[name]) / max(1, np.count_nonzero(raw[name]))
            ),
        }

    report = {
        "schema": "RealSaS.KnightEdgeSeamTwoCoreCourt.v1",
        "status": "DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
        "preregistration": str(PREREG),
        "teacher_used_by_operator": False,
        "mesh_boundary_vertex_count": int(np.count_nonzero(boundary_vertex)),
        "results_teacher_eval_only": result,
        "claim_boundary": (
            "Raw seam rules and graph 2-core pruning are frozen before teacher topology "
            "is loaded. Teacher labels only measure retained precision and unsafe cross-region recall."
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_EDGE_SEAM_TWO_CORE_PASS", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
