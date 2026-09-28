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
    _split_holeless,
    _teacher_region_eval,
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

PREREG = Path("canonical/KNIGHT_MECHANICAL_MUTEX_WATERSHED_CLOSURE_PREREG_V1_20260929.json")


class MutexDSU:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.mutex = [set() for _ in range(n)]

    def find(self, x):
        x = int(x)
        p = int(self.parent[x])
        if p != x:
            self.parent[x] = self.find(p)
        return int(self.parent[x])

    def _canonical_mutex(self, r):
        r = self.find(r)
        out = set()
        for x in tuple(self.mutex[r]):
            rx = self.find(x)
            if rx != r:
                out.add(rx)
        self.mutex[r] = out
        return out

    def are_mutex(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        ma = self._canonical_mutex(ra)
        mb = self._canonical_mutex(rb)
        return rb in ma or ra in mb

    def add_mutex(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.mutex[ra].add(rb)
        self.mutex[rb].add(ra)
        return True

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.are_mutex(ra, rb):
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        ma = self._canonical_mutex(ra)
        mb = self._canonical_mutex(rb)
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        merged_mutex = {self.find(x) for x in (ma | mb)}
        merged_mutex.discard(ra)
        merged_mutex.discard(rb)
        self.mutex[ra] = set(merged_mutex)
        self.mutex[rb] = set()
        for x in tuple(merged_mutex):
            rx = self.find(x)
            mx = self._canonical_mutex(rx)
            mx.discard(rb)
            mx.add(ra)
            self.mutex[rx] = mx
        return True


def _partition_labels(n, dense_edge, edges, stretch, l1, in_scope):
    events = []
    repulsive = np.zeros(len(edges), dtype=bool)
    for ei, (a0, b0) in enumerate(edges.tolist()):
        if not bool(dense_edge[ei]):
            continue
        a, b = int(a0), int(b0)
        s = float(stretch[ei])
        d = float(l1[ei])
        strong_rep = bool(in_scope[ei] and s > 4.0 and d > 1.0)
        repulsive[ei] = strong_rep
        if strong_rep:
            cs = (s - 4.0) / max(s + 4.0, 1e-15)
            cd = (d - 1.0) / max(d + 1.0, 1e-15)
            conf = float(max(0.0, min(cs, cd)))
            kind = 0  # repulsive wins exact ties
        else:
            cs = max((4.0 - s) / max(4.0 + s, 1e-15), 0.0)
            cd = max((1.0 - d) / max(1.0 + d, 1e-15), 0.0)
            conf = float(min(cs, cd))
            kind = 1
        events.append((-conf, kind, min(a, b), max(a, b), int(ei)))

    events.sort()
    dsu = MutexDSU(n)
    mutex_added = 0
    union_added = 0
    union_blocked = 0
    repulsive_inside_already_merged = 0
    for neg_conf, kind, a, b, ei in events:
        if kind == 0:
            if dsu.find(a) == dsu.find(b):
                repulsive_inside_already_merged += 1
            elif dsu.add_mutex(a, b):
                mutex_added += 1
        else:
            before_same = dsu.find(a) == dsu.find(b)
            ok = dsu.union(a, b)
            if not before_same and ok:
                union_added += 1
            elif not ok:
                union_blocked += 1

    roots = [dsu.find(i) for i in range(n)]
    remap = {}
    label = np.empty(n, dtype=np.int64)
    for i, r in enumerate(roots):
        if r not in remap:
            remap[r] = len(remap)
        label[i] = remap[r]
    sizes = np.bincount(label)
    return label, repulsive, {
        "component_count": int(len(sizes)),
        "largest_component_sizes": list(map(int, sorted(sizes.tolist(), reverse=True)[:30])),
        "repulsive_edge_count": int(np.count_nonzero(repulsive)),
        "mutex_constraint_count": int(mutex_added),
        "attractive_union_count": int(union_added),
        "attractive_union_blocked_by_mutex": int(union_blocked),
        "repulsive_edge_arrived_after_endpoint_merge": int(repulsive_inside_already_merged),
        "event_count": int(len(events)),
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
    if prereg.get("status") != "FROZEN_BEFORE_MUTEX_WATERSHED_RESULT":
        raise RuntimeError("MUTEX_WATERSHED_PREREG_DRIFT")
    th = prereg["closure_thresholds"]

    rr = _ctx(a.authority_root, a.run_id)["run_root"]
    cand = exact(rr, "candidate", canonical_mesh_candidate_from_dict)
    sk = exact(rr, "skeleton", qualified_skeleton_from_dict)
    cams = tuple(sorted(
        exact(rr, "cameras", qualified_camera_set_from_dict).cameras,
        key=lambda x: int(x.view_index),
    ))
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

    base_stress = stress_arbitrary_weights(P, Wa, F, jids, sk, cams, env, policy)
    unsafe = np.zeros(len(F), dtype=bool)
    unsafe[np.asarray(base_stress["unsafe_face_indices"], dtype=np.int64)] = True

    edges, edge_faces, face_edge_index = _edge_table(P, F)
    stretch = _edge_stretch(P, Wa, edges, jids, sk, cams, env)
    l1 = np.sum(np.abs(Wa[edges[:, 0]] - Wa[edges[:, 1]]), axis=1)

    dense_edge = np.zeros(len(edges), dtype=bool)
    in_scope = np.zeros(len(edges), dtype=bool)
    for ei, edge in enumerate(map(tuple, edges.tolist())):
        fs = edge_faces[edge]
        dense_edge[ei] = any(bool(dense[fi]) for fi in fs)
        in_scope[ei] = any(bool(dense[fi] and unsafe[fi]) for fi in fs)

    inferred, repulsive, solver_meta = _partition_labels(
        len(P), dense_edge, edges, stretch, l1, in_scope
    )

    # Teacher topology starts only after labels are frozen.
    teacher_eval = _teacher_region_eval(
        a.teacher_bank, a.teacher_source, sk, cand, inferred, F, unsafe
    )

    # Teacher weights start only after labels are frozen, to isolate topology mechanics.
    Wt, tjids, _, _ = teacher_weights(a.teacher_bank, a.teacher_source, sk, cand)
    tix = {str(j): i for i, j in enumerate(tjids)}
    missing = [j for j in jids if str(j) not in tix]
    if missing:
        raise RuntimeError("MUTEX_WATERSHED_TEACHER_JOINT_DRIFT:" + json.dumps(missing))
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
        "no_face_deletion": int(split_meta["face_deletion_count"]) <= int(th["face_deletion_count_max"]),
        "rest_area_preserved": float(split_meta["rest_area_error_max"]) <= float(th["rest_area_error_max"]),
        "actual_gt10": int(after_motion["max_edge_gt_10"]) <= int(th["actual_motion_max_edge_gt_10_max"]),
        "actual_gt4": int(after_motion["max_edge_gt_4"]) <= int(th["actual_motion_max_edge_gt_4_max"]),
        "actual_worst_edge": float(after_motion["worst_edge_max"]) <= float(th["actual_motion_worst_edge_max"]),
        "synthetic_unsafe": int(after_stress["unsafe_face_count"]) <= int(th["synthetic_unsafe_face_count_max"]),
        "mixed_precision": float(teacher_eval["mixed_face_precision"]) >= float(th["mixed_face_precision_min"]),
        "unsafe_cross_region_recall": float(teacher_eval["unsafe_truth_cross_region_recall"]) >= float(th["unsafe_cross_region_recall_min"]),
    }
    closure = all(checks.values())

    report = {
        "schema": "RealSaS.KnightMechanicalMutexWatershedClosureCourt.v1",
        "status": "PASS_TOPOLOGY_CLOSED_FOR_KNIGHT" if closure else "FAIL_TOPOLOGY_REMAINS_OPEN",
        "preregistration": str(PREREG),
        "teacher_used_by_partition_solver": False,
        "teacher_topology_used_for_evaluation_only": True,
        "teacher_weights_used_for_post_inference_mechanical_isolation_only": True,
        "solver": solver_meta,
        "teacher_evaluation_only": teacher_eval,
        "holeless_split": split_meta,
        "topology_isolated_mechanics": {
            "before_motion": {k:v for k,v in before_motion.items() if k!="frames"},
            "after_motion": {k:v for k,v in after_motion.items() if k!="frames"},
            "before_stress": {k:v for k,v in before_stress.items() if k!="unsafe_face_indices"},
            "after_stress": {k:v for k,v in after_stress.items() if k!="unsafe_face_indices"},
        },
        "checks": checks,
        "closure_pass": closure,
        "claim_boundary": (
            "The partition is solved entirely from dense adjacency plus teacher-free Arachne/Stage35 mechanical evidence. "
            "Teacher data enters only after labels are frozen."
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(
        "KNIGHT_MECHANICAL_MUTEX_WATERSHED_CLOSURE_"+("PASS" if closure else "FAIL"),
        json.dumps({
            "solver":solver_meta,
            "teacher_eval":teacher_eval,
            "after_motion":report["topology_isolated_mechanics"]["after_motion"],
            "after_stress":report["topology_isolated_mechanics"]["after_stress"],
            "checks":checks,
            "closure_pass":closure,
        }, sort_keys=True)
    )

if __name__ == "__main__":
    main()
