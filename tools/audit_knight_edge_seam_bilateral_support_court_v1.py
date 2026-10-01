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
from tools.audit_knight_edge_seam_two_core_court_v1 import _metrics, _two_core
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


PREREG = Path("canonical/KNIGHT_EDGE_SEAM_BILATERAL_SUPPORT_PREREG_V1_20260929.json")


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
    if prereg.get("status") != "FROZEN_BEFORE_BILATERAL_SUPPORT_RESULT":
        raise RuntimeError("EDGE_BILATERAL_PREREG_DRIFT")

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
    dom = np.argmax(W, axis=1)

    in_scope = np.zeros(len(edges), dtype=bool)
    boundary_vertex = np.zeros(len(P), dtype=bool)
    dense_nbr = defaultdict(set)
    for fi in np.where(dense)[0]:
        a0, b0, c0 = map(int, F[int(fi)])
        for x, y in ((a0, b0), (b0, c0), (c0, a0)):
            dense_nbr[x].add(y)
            dense_nbr[y].add(x)
    for ei, edge in enumerate(map(tuple, edges.tolist())):
        fs = edge_faces[edge]
        in_scope[ei] = any(bool(unsafe[fi] and dense[fi]) for fi in fs)
        if len(fs) == 1:
            boundary_vertex[int(edge[0])] = True
            boundary_vertex[int(edge[1])] = True

    base = in_scope & (stretch > 4.0) & (l1 > 1.0)
    bilateral = np.zeros(len(edges), dtype=bool)
    for ei in np.where(base)[0]:
        u, v = map(int, edges[int(ei)])
        us = any(int(w) != v and int(dom[int(w)]) == int(dom[u]) for w in dense_nbr[u])
        vs = any(int(w) != u and int(dom[int(w)]) == int(dom[v]) for w in dense_nbr[v])
        bilateral[int(ei)] = bool(us and vs)

    bilateral_core = _two_core(edges, bilateral, boundary_vertex)

    # Teacher begins only here.
    truth = _truth_region(a.teacher_bank, a.teacher_source, sk, cand)
    truth_seam_edge = truth[edges[:, 0]] != truth[edges[:, 1]]

    result = {
        "base": _metrics(base, truth_seam_edge, face_edge_index, truth, unsafe, F),
        "bilateral_support": _metrics(
            bilateral, truth_seam_edge, face_edge_index, truth, unsafe, F
        ),
        "bilateral_support_two_core": _metrics(
            bilateral_core, truth_seam_edge, face_edge_index, truth, unsafe, F
        ),
        "counts": {
            "base_edge_count": int(np.count_nonzero(base)),
            "bilateral_edge_count": int(np.count_nonzero(bilateral)),
            "bilateral_two_core_edge_count": int(np.count_nonzero(bilateral_core)),
        },
    }

    report = {
        "schema": "RealSaS.KnightEdgeSeamBilateralSupportCourt.v1",
        "status": "DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
        "preregistration": str(PREREG),
        "teacher_used_by_operator": False,
        "results_teacher_eval_only": result,
        "claim_boundary": (
            "Arachne dominant-joint bilateral neighborhood support and graph 2-core "
            "are fixed before teacher source topology is loaded."
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_EDGE_SEAM_BILATERAL_SUPPORT_PASS", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
