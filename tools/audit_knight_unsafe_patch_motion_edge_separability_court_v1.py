from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from tools.audit_knight_james_twigg_unsafe_patch_completion_court_v1 import (
    unsafe_components,
)
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


PREREG = Path("canonical/KNIGHT_UNSAFE_PATCH_MOTION_EDGE_SEPARABILITY_PREREG_V1_20260929.json")


def _edge_set(F, faces):
    out = set()
    for fi in faces:
        a, b, c = map(int, F[int(fi)])
        for x, y in ((a, b), (b, c), (c, a)):
            if x > y:
                x, y = y, x
            out.add((x, y))
    return sorted(out)


def _jsd(p, q):
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    q = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    p /= max(float(p.sum()), 1e-15)
    q /= max(float(q.sum()), 1e-15)
    m = 0.5 * (p + q)
    maskp = p > 0.0
    maskq = q > 0.0
    klp = float(np.sum(p[maskp] * np.log(p[maskp] / np.maximum(m[maskp], 1e-15))))
    klq = float(np.sum(q[maskq] * np.log(q[maskq] / np.maximum(m[maskq], 1e-15))))
    return 0.5 * (klp + klq)


def _summary(scores, truth_cross):
    truth_cross = np.asarray(truth_cross, dtype=bool)
    out = {
        "edge_count": int(len(truth_cross)),
        "cross_edge_count": int(np.count_nonzero(truth_cross)),
        "same_edge_count": int(np.count_nonzero(~truth_cross)),
        "cross_fraction": float(np.mean(truth_cross)) if len(truth_cross) else 0.0,
    }
    for name, vals0 in scores.items():
        vals = np.asarray(vals0, dtype=np.float64)
        same = vals[~truth_cross]
        cross = vals[truth_cross]
        row = {
            "same_p50": float(np.quantile(same, 0.50)) if len(same) else None,
            "same_p95": float(np.quantile(same, 0.95)) if len(same) else None,
            "same_p99": float(np.quantile(same, 0.99)) if len(same) else None,
            "cross_p01": float(np.quantile(cross, 0.01)) if len(cross) else None,
            "cross_p05": float(np.quantile(cross, 0.05)) if len(cross) else None,
            "cross_p50": float(np.quantile(cross, 0.50)) if len(cross) else None,
        }
        if len(same) and len(cross):
            row["roc_auc"] = float(roc_auc_score(truth_cross.astype(np.int64), vals))
        else:
            row["roc_auc"] = None
        out[name] = row
    return out


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
    if prereg.get("status") != "FROZEN_BEFORE_MOTION_EDGE_SEPARABILITY_RESULT":
        raise RuntimeError("MOTION_EDGE_SEPARABILITY_PREREG_DRIFT")

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
    W = np.maximum(W, 0.0)
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-15)

    stress = stress_arbitrary_weights(P, W, F, jids, sk, cams, env, policy)
    unsafe = np.zeros(len(F), dtype=bool)
    unsafe[np.asarray(stress["unsafe_face_indices"], dtype=np.int64)] = True
    comps = unsafe_components(F, unsafe, dense)

    # Freeze every teacher-free edge score first.
    patch_rows = []
    all_edges = []
    all_l1 = []
    all_jsd = []
    all_dom = []
    for ci, comp in enumerate(comps):
        edges = _edge_set(F, comp)
        l1 = []
        jsd = []
        dom = []
        for u, v in edges:
            l1.append(float(0.5 * np.sum(np.abs(W[u] - W[v]))))
            jsd.append(float(_jsd(W[u], W[v])))
            dom.append(float(int(np.argmax(W[u]) != np.argmax(W[v]))))
        patch_rows.append(
            {
                "component_index": int(ci),
                "unsafe_face_count": int(len(comp)),
                "edge_count": int(len(edges)),
                "edges": edges,
                "scores": {
                    "half_l1": l1,
                    "jsd": jsd,
                    "dominant_joint_disagreement": dom,
                },
            }
        )
        all_edges.extend(edges)
        all_l1.extend(l1)
        all_jsd.extend(jsd)
        all_dom.extend(dom)

    # Teacher truth begins here, after all scores are frozen.
    _, _, tri, sf = teacher_weights(a.teacher_bank, a.teacher_source, sk, cand)
    src_comp = source_components(int(sf.max()) + 1, sf)
    tri_comp = np.asarray([src_comp[int(row[0])] for row in sf], dtype=np.int64)
    truth = np.asarray([tri_comp[int(t)] for t in tri], dtype=np.int64)

    evaluated = []
    for row in patch_rows:
        cross = [bool(truth[u] != truth[v]) for u, v in row["edges"]]
        evaluated.append(
            {
                "component_index": row["component_index"],
                "unsafe_face_count": row["unsafe_face_count"],
                **_summary(row["scores"], cross),
            }
        )

    all_cross = [bool(truth[u] != truth[v]) for u, v in all_edges]
    global_summary = _summary(
        {
            "half_l1": all_l1,
            "jsd": all_jsd,
            "dominant_joint_disagreement": all_dom,
        },
        all_cross,
    )

    continuous = ("half_l1", "jsd")
    global_best = max(
        float(global_summary[k]["roc_auc"] or 0.0) for k in continuous
    )
    top2 = evaluated[:2]
    strong = False
    for k in continuous:
        ga = float(global_summary[k]["roc_auc"] or 0.0)
        patch_aucs = [
            float(r[k]["roc_auc"])
            for r in top2
            if r[k]["roc_auc"] is not None
        ]
        if ga >= 0.90 and patch_aucs and min(patch_aucs) >= 0.85:
            strong = True
    if strong:
        diagnosis = "MOTION_EDGE_SIGNAL_STRONG"
    elif global_best >= 0.75:
        diagnosis = "MOTION_EDGE_SIGNAL_PARTIAL"
    else:
        diagnosis = "MOTION_EDGE_SIGNAL_WEAK"

    report = {
        "schema": "RealSaS.KnightUnsafePatchMotionEdgeSeparabilityCourt.v1",
        "status": "DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
        "preregistration": str(PREREG),
        "teacher_used_by_score_construction": False,
        "unsafe_patch_count": int(len(comps)),
        "global": global_summary,
        "largest_patches": evaluated[:10],
        "diagnosis": diagnosis,
        "claim_boundary": (
            "Arachne-derived edge scores are frozen before teacher source-component "
            "identity is loaded. Teacher is used only to evaluate whether the scores "
            "separate SAME_REGION from CROSS_REGION edges. No threshold is selected "
            "and no mesh or weight artifact is mutated."
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_UNSAFE_PATCH_MOTION_EDGE_SEPARABILITY_PASS",
        json.dumps(
            {
                "diagnosis": diagnosis,
                "global": global_summary,
                "largest_two": evaluated[:2],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
