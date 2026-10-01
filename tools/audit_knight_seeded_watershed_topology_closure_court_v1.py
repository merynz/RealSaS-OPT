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
    edge_graph_from_faces,
    unsafe_components,
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
from tools.audit_knight_unsafe_patch_motion_edge_separability_court_v1 import _jsd
from tools.demo.render_knight_motion_preview_v1 import _ctx


PREREG = Path("canonical/KNIGHT_SEEDED_WATERSHED_TOPOLOGY_CLOSURE_PREREG_V1_20260929.json")


class DSU:
    def __init__(self, nodes, seed_label):
        self.parent = {int(v): int(v) for v in nodes}
        self.size = {int(v): 1 for v in nodes}
        self.label = {int(v): int(seed_label.get(int(v), -1)) for v in nodes}

    def find(self, x):
        x = int(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union_if_compatible(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return True, False
        la = self.label[ra]
        lb = self.label[rb]
        if la >= 0 and lb >= 0 and la != lb:
            return False, True
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
            la, lb = lb, la
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.label[ra] = la if la >= 0 else lb
        return True, False


def _direct_core_seed_mask(F, face_label, core, dense, safe, initial_label, confidence, n):
    touched = np.zeros(n, dtype=bool)
    usable = (
        np.asarray(core, dtype=bool)
        & np.asarray(dense, dtype=bool)
        & np.asarray(safe, dtype=bool)
        & (np.asarray(face_label) >= 0)
    )
    for fi in np.where(usable)[0]:
        touched[np.asarray(F[int(fi)], dtype=np.int64)] = True
    return (
        touched
        & (np.asarray(initial_label) >= 0)
        & (np.asarray(confidence) > 0.5)
    )


def _watershed_patch(
    P,
    F,
    W,
    comp,
    dense_nbr,
    direct_seed,
    initial_label,
):
    patch_vertices = sorted(set(int(v) for fi in comp for v in F[int(fi)]))
    patch_set = set(patch_vertices)

    boundary_seed_nodes = set()
    for u in patch_vertices:
        for v, _ in dense_nbr.get(u, ()):
            v = int(v)
            if v not in patch_set and bool(direct_seed[v]) and int(initial_label[v]) >= 0:
                boundary_seed_nodes.add(v)

    nodes = sorted(patch_set | boundary_seed_nodes)
    seed_label = {}
    for v in nodes:
        if bool(direct_seed[v]) and int(initial_label[v]) >= 0:
            seed_label[v] = int(initial_label[v])

    distinct = sorted(set(seed_label.values()))
    if len(distinct) < 2:
        return None, {
            "status": "ABSTAIN_LT2_DIRECT_CORE_LABELS",
            "unsafe_face_count": int(len(comp)),
            "vertex_count": int(len(patch_vertices)),
            "hard_seed_count": int(len(seed_label)),
            "hard_seed_label_count": int(len(distinct)),
            "boundary_hard_seed_count": int(len(boundary_seed_nodes)),
        }

    edge_rows = []
    seen = set()
    for u in nodes:
        for v, geom_len in dense_nbr.get(u, ()):
            v = int(v)
            if v not in patch_set and v not in boundary_seed_nodes:
                continue
            if u not in patch_set and v not in patch_set:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edge_rows.append(
                (
                    float(_jsd(W[a], W[b])),
                    float(geom_len),
                    int(a),
                    int(b),
                )
            )
    edge_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    dsu = DSU(nodes, seed_label)
    rejected = []
    for jsd, geom_len, a, b in edge_rows:
        merged, conflict = dsu.union_if_compatible(a, b)
        if conflict:
            rejected.append((a, b, jsd, geom_len))

    out = {}
    unlabeled = 0
    for v in patch_vertices:
        r = dsu.find(v)
        lab = int(dsu.label[r])
        if lab < 0:
            unlabeled += 1
        out[int(v)] = lab

    region_sizes = defaultdict(int)
    for v in patch_vertices:
        region_sizes[int(out[v])] += 1

    return out, {
        "status": "COMPLETED" if unlabeled == 0 else "PARTIAL",
        "unsafe_face_count": int(len(comp)),
        "vertex_count": int(len(patch_vertices)),
        "hard_seed_count": int(len(seed_label)),
        "hard_seed_label_count": int(len(distinct)),
        "boundary_hard_seed_count": int(len(boundary_seed_nodes)),
        "candidate_edge_count": int(len(edge_rows)),
        "watershed_conflict_edge_count": int(len(rejected)),
        "unlabeled_vertex_count": int(unlabeled),
        "local_region_count": int(len([k for k in region_sizes if k >= 0])),
        "largest_region_sizes": sorted(
            [int(v) for k, v in region_sizes.items() if k >= 0], reverse=True
        )[:20],
        "conflict_jsd_p50": float(np.quantile([x[2] for x in rejected], 0.5))
        if rejected
        else None,
        "conflict_jsd_p05": float(np.quantile([x[2] for x in rejected], 0.05))
        if rejected
        else None,
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
    if prereg.get("status") != "FROZEN_BEFORE_SEEDED_WATERSHED_RESULT":
        raise RuntimeError("SEEDED_WATERSHED_PREREG_DRIFT")
    th = prereg["closure_thresholds"]

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
    Wa = np.maximum(Wa, 0.0)
    Wa /= np.maximum(Wa.sum(axis=1, keepdims=True), 1e-15)

    infer_stress = stress_arbitrary_weights(P, Wa, F, jids, sk, cams, env, policy)
    unsafe = np.zeros(len(F), dtype=bool)
    unsafe[np.asarray(infer_stress["unsafe_face_indices"], dtype=np.int64)] = True
    safe = (~unsafe) & dense

    Z, valid, probes = _rotation_signatures(P, F, Wa, jids, sk, cams, PROBE_DEGREES)
    face_label, core, cluster_meta = _cluster_signatures(Z, safe & valid, EPSILON)
    initial_label, confidence, prop_meta = _vertex_labels_from_faces(
        P, F, face_label, core, dense, safe
    )
    direct_seed = _direct_core_seed_mask(
        F,
        face_label,
        core,
        dense,
        safe,
        initial_label,
        confidence,
        len(P),
    )

    dense_nbr, _ = edge_graph_from_faces(P, F, np.where(dense)[0])
    comps = unsafe_components(F, unsafe, dense)

    inferred = np.asarray(initial_label, dtype=np.int64).copy()
    patch_reports = []
    changed = 0
    for ci, comp in enumerate(comps):
        assigned, meta = _watershed_patch(
            P, F, Wa, comp, dense_nbr, direct_seed, initial_label
        )
        meta["component_index"] = int(ci)
        if assigned is not None:
            for v, lab in assigned.items():
                if lab >= 0:
                    if inferred[int(v)] != int(lab):
                        changed += 1
                    inferred[int(v)] = int(lab)
        patch_reports.append(meta)

    # Teacher topology begins only after inference is frozen.
    teacher_eval = _teacher_region_eval(
        a.teacher_bank, a.teacher_source, sk, cand, inferred, F, unsafe
    )

    # Teacher weights begin only after inference is frozen.
    Wt, tjids, _, _ = teacher_weights(a.teacher_bank, a.teacher_source, sk, cand)
    tix = {str(j): i for i, j in enumerate(tjids)}
    missing = [j for j in jids if str(j) not in tix]
    if missing:
        raise RuntimeError("SEEDED_WATERSHED_TEACHER_JOINT_DRIFT:" + json.dumps(missing))
    Wt = np.stack([Wt[:, tix[str(j)]] for j in jids], axis=1)
    Wt = np.maximum(Wt, 0.0)
    Wt /= np.maximum(Wt.sum(axis=1, keepdims=True), 1e-15)

    P2, W2, F2, split_meta = _split_holeless(P, Wt, F, inferred)
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
        "schema": "RealSaS.KnightSeededWatershedTopologyClosureCourt.v1",
        "status": "PASS_TOPOLOGY_CLOSED_FOR_KNIGHT" if closure else "FAIL_TOPOLOGY_REMAINS_OPEN",
        "preregistration": str(PREREG),
        "teacher_used_by_inference": False,
        "teacher_topology_used_for_evaluation_only": True,
        "teacher_weights_used_for_post_inference_mechanical_isolation_only": True,
        "inference": {
            "probe_degrees": PROBE_DEGREES,
            "probe_count": len(probes),
            "clustering": cluster_meta,
            "initial_propagation": prop_meta,
            "direct_core_seed_count": int(np.count_nonzero(direct_seed)),
            "changed_patch_vertex_label_count": int(changed),
            "completed_patch_count": int(
                sum(r["status"] == "COMPLETED" for r in patch_reports)
            ),
            "partial_patch_count": int(
                sum(r["status"] == "PARTIAL" for r in patch_reports)
            ),
            "abstained_patch_count": int(
                sum(str(r["status"]).startswith("ABSTAIN") for r in patch_reports)
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
            "Patch labels are produced by a teacher-free seeded minimum-spanning-forest "
            "ordered only by Arachne skin-row JSD. Different direct James/Twigg core seed "
            "labels may not merge. Teacher topology and weights enter only after inference "
            "is frozen."
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_SEEDED_WATERSHED_TOPOLOGY_CLOSURE_" + ("PASS" if closure else "FAIL"),
        json.dumps(
            {
                "inference": report["inference"],
                "teacher_eval": teacher_eval,
                "top_patches": patch_reports[:5],
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
