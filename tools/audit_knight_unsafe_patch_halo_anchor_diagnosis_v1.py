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
from tools.audit_knight_james_twigg_region_inference_v1 import (
    EPSILON,
    PROBE_DEGREES,
    _cluster_signatures,
    _rotation_signatures,
    _vertex_labels_from_faces,
)
from tools.audit_knight_james_twigg_unsafe_patch_completion_court_v1 import (
    complete_unsafe_patches,
    edge_graph_from_faces,
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


PREREG = Path("canonical/KNIGHT_UNSAFE_PATCH_HALO_ANCHOR_DIAG_PREREG_V1_20260929.json")
HALO_HOPS = (1, 2, 3, 4, 8)


def _vertex_face_index(F):
    out = defaultdict(list)
    for fi, row in enumerate(F.tolist()):
        for v in row:
            out[int(v)].append(int(fi))
    return out


def _hop_halo_labels(
    patch_vertices,
    dense_nbr,
    initial_label,
    safe_vertex,
    max_hops,
):
    patch_set = set(map(int, patch_vertices))
    dist = {v: 0 for v in patch_set}
    q = deque(sorted(patch_set))
    by_hop = {}
    seen_candidates = set()

    while q:
        u = q.popleft()
        du = int(dist[u])
        if du >= max_hops:
            continue
        for v, _ in dense_nbr.get(u, ()):
            v = int(v)
            nd = du + 1
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                q.append(v)

    for h in HALO_HOPS:
        labels = defaultdict(int)
        candidates = []
        for v, d in dist.items():
            if v in patch_set or d < 1 or d > h:
                continue
            if not bool(safe_vertex[v]):
                continue
            lab = int(initial_label[v])
            if lab < 0:
                continue
            labels[lab] += 1
            candidates.append(v)
            seen_candidates.add(v)
        by_hop[str(h)] = {
            "distinct_label_count": int(len(labels)),
            "labeled_safe_vertex_count": int(len(candidates)),
            "label_support": {str(k): int(v) for k, v in sorted(labels.items())},
        }
    return by_hop


def _teacher_truth(
    teacher_bank,
    teacher_source,
    skeleton,
    candidate,
):
    _, _, tri, sf = teacher_weights(
        teacher_bank, teacher_source, skeleton, candidate
    )
    src_comp = source_components(int(sf.max()) + 1, sf)
    tri_comp = np.asarray([src_comp[int(row[0])] for row in sf], dtype=np.int64)
    return np.asarray([tri_comp[int(t)] for t in tri], dtype=np.int64)


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
    if prereg.get("status") != "FROZEN_BEFORE_HALO_ANCHOR_DIAGNOSIS_RESULT":
        raise RuntimeError("HALO_ANCHOR_DIAG_PREREG_DRIFT")

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
    if W.shape != (len(P), len(jids)):
        raise RuntimeError("HALO_ANCHOR_WEIGHT_SHAPE_DRIFT")

    base_stress = stress_arbitrary_weights(P, W, F, jids, sk, cams, env, policy)
    unsafe = np.zeros(len(F), dtype=bool)
    unsafe[np.asarray(base_stress["unsafe_face_indices"], dtype=np.int64)] = True
    safe = (~unsafe) & dense

    Z, valid, probes = _rotation_signatures(P, F, W, jids, sk, cams, PROBE_DEGREES)
    face_label, core, cluster_meta = _cluster_signatures(Z, safe & valid, EPSILON)
    initial_label, confidence, prop_meta = _vertex_labels_from_faces(
        P, F, face_label, core, dense, safe
    )

    # Re-run current operator only to freeze exactly which components abstain.
    _, current_reports, current_meta = complete_unsafe_patches(
        P, F, dense, unsafe, initial_label
    )
    current_by_index = {int(r["component_index"]): r for r in current_reports}

    dense_nbr, _ = edge_graph_from_faces(P, F, np.where(dense)[0])
    vfaces = _vertex_face_index(F)
    safe_vertex = np.zeros(len(P), dtype=bool)
    for v, fs in vfaces.items():
        safe_vertex[v] = any(bool(safe[fi]) for fi in fs)

    comps = unsafe_components(F, unsafe, dense)

    # Teacher is loaded only after teacher-free inference and all halo traversal
    # rules are fixed. It is used solely to classify the observed failure.
    truth = _teacher_truth(a.teacher_bank, a.teacher_source, sk, cand)

    patches = []
    for ci, comp in enumerate(comps):
        current = current_by_index[ci]
        if not str(current["status"]).startswith("ABSTAIN"):
            continue
        patch_vertices = sorted(set(int(v) for fi in comp for v in F[int(fi)]))
        halo = _hop_halo_labels(
            patch_vertices,
            dense_nbr,
            initial_label,
            safe_vertex,
            max(HALO_HOPS),
        )
        truth_labels = sorted(set(int(truth[v]) for v in patch_vertices))
        truth_cross_faces = sum(
            len(set(int(truth[v]) for v in F[int(fi)])) > 1 for fi in comp
        )
        first_two = next(
            (h for h in HALO_HOPS if halo[str(h)]["distinct_label_count"] >= 2),
            None,
        )
        patches.append(
            {
                "component_index": int(ci),
                "unsafe_face_count": int(len(comp)),
                "vertex_count": int(len(patch_vertices)),
                "current_boundary_label_count": int(current["boundary_label_count"]),
                "current_boundary_seed_count": int(current["boundary_seed_count"]),
                "initial_labeled_patch_vertex_count": int(
                    np.count_nonzero(initial_label[np.asarray(patch_vertices)] >= 0)
                ),
                "halo": halo,
                "first_hop_with_two_labels": None if first_two is None else int(first_two),
                "teacher_eval_only": {
                    "truth_region_count_on_patch_vertices": int(len(truth_labels)),
                    "truth_cross_region_face_count": int(truth_cross_faces),
                    "truth_patch_is_cross_region": bool(
                        len(truth_labels) >= 2 or truth_cross_faces > 0
                    ),
                },
            }
        )

    summary = {}
    for boundary_labels in (0, 1):
        rows = [r for r in patches if r["current_boundary_label_count"] == boundary_labels]
        cross = [r for r in rows if r["teacher_eval_only"]["truth_patch_is_cross_region"]]
        summary[f"boundary_labels_{boundary_labels}"] = {
            "patch_count": int(len(rows)),
            "unsafe_face_count": int(sum(r["unsafe_face_count"] for r in rows)),
            "teacher_eval_cross_region_patch_count": int(len(cross)),
            "teacher_eval_cross_region_face_count": int(
                sum(r["unsafe_face_count"] for r in cross)
            ),
            "cross_region_recovered_by_hop": {
                str(h): int(
                    sum(
                        r["halo"][str(h)]["distinct_label_count"] >= 2
                        for r in cross
                    )
                )
                for h in HALO_HOPS
            },
        }

    all_cross = [r for r in patches if r["teacher_eval_only"]["truth_patch_is_cross_region"]]
    recovered = {
        str(h): int(
            sum(r["halo"][str(h)]["distinct_label_count"] >= 2 for r in all_cross)
        )
        for h in HALO_HOPS
    }
    cross_total = len(all_cross)
    recovered_fraction_4 = recovered["4"] / max(1, cross_total)
    recovered_fraction_8 = recovered["8"] / max(1, cross_total)

    if recovered_fraction_4 >= 0.5:
        diagnosis = "LOCAL_ANCHOR_RULE_TOO_STRICT"
    elif recovered_fraction_8 < 0.5:
        diagnosis = "REPRESENTATION_RECALL_LIMIT"
    else:
        diagnosis = "MIXED__HALO_HELPS_BUT_NOT_WITHIN_4_HOPS"

    report = {
        "schema": "RealSaS.KnightUnsafePatchHaloAnchorDiagnosis.v1",
        "status": "DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
        "preregistration": str(PREREG),
        "teacher_used_by_inference": False,
        "probe": {"degrees": PROBE_DEGREES, "count": len(probes)},
        "clustering": cluster_meta,
        "initial_propagation": prop_meta,
        "current_patch_completion": current_meta,
        "abstained_patch_count": int(len(patches)),
        "teacher_eval_cross_region_abstained_patch_count": int(cross_total),
        "cross_region_recovered_by_hop": recovered,
        "cross_region_recovered_fraction_by_hop": {
            str(h): float(recovered[str(h)] / max(1, cross_total)) for h in HALO_HOPS
        },
        "summary": summary,
        "diagnosis": diagnosis,
        "patches": patches,
        "claim_boundary": (
            "Teacher topology is evaluation-only and is loaded after the teacher-free "
            "labels, unsafe components, traversal graph, halo radii, and decision rule "
            "are fixed. This run diagnoses missing-anchor locality; it does not promote "
            "a repair operator."
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_UNSAFE_PATCH_HALO_ANCHOR_DIAG_PASS",
        json.dumps(
            {
                "abstained_patch_count": report["abstained_patch_count"],
                "teacher_eval_cross_region_abstained_patch_count": cross_total,
                "cross_region_recovered_by_hop": recovered,
                "cross_region_recovered_fraction_by_hop": report[
                    "cross_region_recovered_fraction_by_hop"
                ],
                "summary": summary,
                "diagnosis": diagnosis,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
