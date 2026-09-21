from __future__ import annotations

"""Bounded adaptive EMPTY-or-GRAPH refinement for VF11 C2 V3.

Research-only. Starting from the frozen 12-cell corrected C1-primary cohort:
  * already-certified C2 V2 graph roots terminate,
  * unresolved leaves are classified by corrected C0 V3,
  * non-empty/unknown leaves are tested for a C2 V2 axis graph patch,
  * unresolved leaves alone are split into octants.

A terminal leaf is therefore exactly one of:
  EMPTY_POSITIVE, EMPTY_NEGATIVE, GRAPH, UNRESOLVED.

No global Knight topology claim is made.
"""

from collections import Counter, defaultdict
import time
from typing import Callable

import numpy as np
import torch

from axis_graph_c2_v2 import (
    C1_REGULAR,
    prepare_field,
    prepare_planes,
    certify_axis_graph_c2_v2_prepared,
)
from paired_profile_c0_v3 import cell_bounds_from_index
from range_engine_c0_v3 import certify_cell_c0_ladder_prepared


EMPTY_STATES = {"PROVEN_EMPTY_POSITIVE", "PROVEN_EMPTY_NEGATIVE"}
GRAPH_STATE = "PROVEN_AXIS_GRAPH_PATCH"


def _split_octants(lo: np.ndarray, hi: np.ndarray):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    mid = (lo + hi) * 0.5
    out = []
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                bits = np.array([bx, by, bz], dtype=np.int64)
                a = np.where(bits == 0, lo, mid)
                b = np.where(bits == 0, mid, hi)
                out.append((bits, a, b))
    return out


def _sample_box(lo: np.ndarray, hi: np.ndarray, per_axis: int = 3) -> np.ndarray:
    axes = [
        np.linspace(float(lo[i]), float(hi[i]), per_axis, dtype=np.float64)
        for i in range(3)
    ]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T


def _direct_values(field, planes: torch.Tensor, pts: np.ndarray) -> np.ndarray:
    q = torch.tensor(
        np.asarray(pts, dtype=np.float64).reshape(1, -1, 3),
        dtype=torch.float64,
    )
    with torch.no_grad():
        return field(planes, q)["sdf"].reshape(-1).detach().cpu().numpy()


def _direct_axis_derivative(
    field,
    planes: torch.Tensor,
    pts: np.ndarray,
    axis: int,
) -> np.ndarray:
    q = torch.tensor(
        np.asarray(pts, dtype=np.float64).reshape(1, -1, 3),
        dtype=torch.float64,
        requires_grad=True,
    )
    out = field(planes, q)["sdf"]
    grad = torch.autograd.grad(out.sum(), q, create_graph=False)[0]
    return grad[..., axis].reshape(-1).detach().cpu().numpy()


def _sample_face(
    lo: np.ndarray,
    hi: np.ndarray,
    axis: int,
    coordinate: float,
    per_axis: int = 3,
) -> np.ndarray:
    varying = [a for a in range(3) if a != axis]
    u = np.linspace(float(lo[varying[0]]), float(hi[varying[0]]), per_axis)
    v = np.linspace(float(lo[varying[1]]), float(hi[varying[1]]), per_axis)
    pts = []
    for x in u:
        for y in v:
            p = np.empty(3, dtype=np.float64)
            p[axis] = float(coordinate)
            p[varying[0]] = x
            p[varying[1]] = y
            pts.append(p)
    return np.asarray(pts, dtype=np.float64)


def _check_c0_empty_empirically(
    field,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    state: str,
) -> list[dict]:
    pts = _sample_box(lo, hi, per_axis=3)
    vals = _direct_values(field, planes, pts)
    bad = False
    if state == "PROVEN_EMPTY_POSITIVE":
        bad = bool(np.any(vals <= 0.0))
    elif state == "PROVEN_EMPTY_NEGATIVE":
        bad = bool(np.any(vals >= 0.0))
    else:
        return []
    if not bad:
        return []
    return [{
        "kind": "C0_EMPTY_SIGN",
        "state": state,
        "sample_min": float(vals.min()),
        "sample_max": float(vals.max()),
    }]


def _check_axis_graph_empirically(
    field,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    cert,
) -> list[dict]:
    out = []
    for a in cert.attempts:
        deriv = a.derivative_result
        if deriv.state in C1_REGULAR:
            pts = _sample_box(lo, hi, per_axis=3)
            vals = _direct_axis_derivative(field, planes, pts, a.axis)
            bad = (
                bool(np.any(vals <= 0.0))
                if deriv.state == "PROVEN_DIRECTIONAL_REGULAR_POSITIVE"
                else bool(np.any(vals >= 0.0))
            )
            if bad:
                out.append({
                    "kind": "AXIS_DERIVATIVE_SIGN",
                    "axis": int(a.axis),
                    "state": deriv.state,
                    "sample_min": float(vals.min()),
                    "sample_max": float(vals.max()),
                })

        for role, fc in (("ENTRY", a.entry_face), ("EXIT", a.exit_face)):
            if fc is None or fc.state != "PROVEN_FACE_SIGN":
                continue
            pts = _sample_face(lo, hi, fc.axis, fc.coordinate, per_axis=3)
            vals = _direct_values(field, planes, pts)
            bad = (
                bool(np.any(vals <= 0.0))
                if fc.desired_sign > 0
                else bool(np.any(vals >= 0.0))
            )
            if bad:
                out.append({
                    "kind": f"{role}_FACE_SIGN",
                    "axis": int(fc.axis),
                    "desired_sign": int(fc.desired_sign),
                    "sample_min": float(vals.min()),
                    "sample_max": float(vals.max()),
                })
    return out


def _node_volume_weight(relative_depth: int) -> float:
    return 1.0 / float(8 ** int(relative_depth))


def run_adaptive_empty_or_graph(
    field,
    planes: torch.Tensor,
    c1_fixed_profile: dict,
    c2_v2_profile: dict,
    *,
    domain_lo: float,
    domain_hi: float,
    max_relative_depth: int = 2,
    c0_micro_depth: int = 2,
    c1_micro_depth: int = 2,
    face_micro_depth: int = 2,
    empirical_checks: bool = True,
    logger: Callable[[str], None] | None = print,
    record_callback: Callable[[dict], None] | None = None,
) -> dict:
    if max_relative_depth < 0:
        raise ValueError("NEGATIVE_RELATIVE_DEPTH")

    roots = [
        r for r in c1_fixed_profile["records"]
        if bool(r["primary_handoff"]) and r["c1_v2_state"] != "UNKNOWN"
    ]
    if len(roots) != 12:
        raise ValueError(f"EXPECTED_12_ROOTS:{len(roots)}")
    c2map = {int(r["anchor_id"]): r for r in c2_v2_profile["records"]}
    if set(c2map) != {int(r["anchor_id"]) for r in roots}:
        raise ValueError("C2V2_ROOT_SET_MISMATCH")

    p = prepare_field(field)
    pp = prepare_planes(planes)

    # Existing parent graph proofs are terminals at relative depth 0.
    terminals: list[dict] = []
    active: list[dict] = []
    root_meta = {}

    for root in roots:
        anchor = int(root["anchor_id"])
        idx = tuple(int(v) for v in root["cell_index_xyz"])
        abs_depth = int(root["depth"])
        lo, hi = cell_bounds_from_index(idx, abs_depth, domain_lo, domain_hi)
        prior = c2map[anchor]
        root_meta[anchor] = {
            "source_c0_state": root["c0_state_m2"],
            "source_c2v1_existence": bool(
                prior.get("c2_v1_state") == "PROVEN_CENTER_CHORD_CROSSING"
            ),
        }
        node = {
            "root_anchor_id": anchor,
            "path": "",
            "relative_depth": 0,
            "absolute_depth": abs_depth,
            "lo": lo,
            "hi": hi,
        }
        if prior["axis_graph_state"] == GRAPH_STATE:
            rec = {
                "root_anchor_id": anchor,
                "path": "",
                "relative_depth": 0,
                "absolute_depth": abs_depth,
                "state": "GRAPH",
                "c0_state": root["c0_state_m2"],
                "axis_graph_state": GRAPH_STATE,
                "selected_axis": prior["selected_axis_name"],
                "bottleneck_margin": prior["bottleneck_margin"],
                "volume_weight_in_root": 1.0,
                "empirical_contradictions": [],
                "elapsed_seconds": 0.0,
                "inherited_parent_graph": True,
            }
            terminals.append(rec)
        else:
            active.append(node)

    records = list(terminals)
    level_summaries = []

    def summarize(level: int, active_nodes: list[dict]):
        by_root_term = defaultdict(list)
        for r in records:
            by_root_term[int(r["root_anchor_id"])].append(r)
        active_roots = Counter(int(n["root_anchor_id"]) for n in active_nodes)

        fully_closed_roots = []
        existence_all_empty_alarms = []
        for root in roots:
            a = int(root["anchor_id"])
            if active_roots[a] == 0:
                fully_closed_roots.append(a)
                if bool(root["existence_proven"] if "existence_proven" in root else False):
                    pass
                # C2 V1 source existence is not stored in C1 fixed profile; caller
                # can audit it separately. The traversal reports graph descendants.

        terminal_counts = Counter(r["state"] for r in records)
        active_volume = sum(
            _node_volume_weight(int(n["relative_depth"]))
            for n in active_nodes
        )
        # Normalize by 12 equal-volume root boxes.
        unresolved_root_volume_fraction = active_volume / 12.0

        graph_roots = {
            int(r["root_anchor_id"]) for r in records if r["state"] == "GRAPH"
        }
        zero_exists_unresolved = sum(
            1 for n in active_nodes if n.get("last_c0_state") == "PROVEN_ZERO_EXISTS"
        )

        return {
            "relative_depth": int(level),
            "terminal_counts_cumulative": dict(sorted(terminal_counts.items())),
            "active_unresolved_leaf_count": int(len(active_nodes)),
            "unresolved_root_volume_fraction": float(unresolved_root_volume_fraction),
            "fully_closed_root_count": int(len(fully_closed_roots)),
            "fully_closed_root_ids": sorted(fully_closed_roots),
            "roots_with_graph_descendant_count": int(len(graph_roots)),
            "roots_with_graph_descendant_ids": sorted(graph_roots),
            "zero_exists_unresolved_leaf_count": int(zero_exists_unresolved),
        }

    level_summaries.append(summarize(0, active))

    started = time.perf_counter()
    processed = 0

    for rel_depth in range(1, max_relative_depth + 1):
        next_active = []
        children = []
        for node in active:
            for child_i, (_, clo, chi) in enumerate(
                _split_octants(node["lo"], node["hi"])
            ):
                children.append({
                    "root_anchor_id": int(node["root_anchor_id"]),
                    "path": f'{node["path"]}{child_i}',
                    "relative_depth": rel_depth,
                    "absolute_depth": int(node["absolute_depth"]) + 1,
                    "lo": clo,
                    "hi": chi,
                })

        if logger:
            logger(
                f"C2V3_LEVEL_BEGIN rel={rel_depth} "
                f"children={len(children)} active_parents={len(active)}"
            )

        for node in children:
            t0 = time.perf_counter()
            c0cert = certify_cell_c0_ladder_prepared(
                p,
                pp,
                node["lo"],
                node["hi"],
                max_micro_depth=c0_micro_depth,
            )
            c0 = c0cert.by_micro_depth[c0_micro_depth]
            contradictions = []

            if c0.state in EMPTY_STATES:
                state = (
                    "EMPTY_POSITIVE"
                    if c0.state == "PROVEN_EMPTY_POSITIVE"
                    else "EMPTY_NEGATIVE"
                )
                graph_cert = None
                if empirical_checks:
                    contradictions.extend(
                        _check_c0_empty_empirically(
                            field, pp, node["lo"], node["hi"], c0.state
                        )
                    )
            else:
                graph_cert = certify_axis_graph_c2_v2_prepared(
                    p,
                    pp,
                    node["lo"],
                    node["hi"],
                    c1_max_micro_depth=c1_micro_depth,
                    face_micro_depth=face_micro_depth,
                )
                if empirical_checks:
                    contradictions.extend(
                        _check_axis_graph_empirically(
                            field, pp, node["lo"], node["hi"], graph_cert
                        )
                    )
                if graph_cert.state == GRAPH_STATE:
                    state = "GRAPH"
                else:
                    state = "UNRESOLVED"

            elapsed = time.perf_counter() - t0
            rec = {
                "root_anchor_id": int(node["root_anchor_id"]),
                "path": node["path"],
                "relative_depth": int(node["relative_depth"]),
                "absolute_depth": int(node["absolute_depth"]),
                "state": state,
                "c0_state": c0.state,
                "c0_lower": float(c0.lower),
                "c0_upper": float(c0.upper),
                "c0_unresolved_leaf_count": int(c0.unresolved_leaf_count),
                "c0_actual_box_evals": int(c0cert.actual_evaluated_box_count),
                "axis_graph_state": None if graph_cert is None else graph_cert.state,
                "selected_axis": (
                    None if graph_cert is None else graph_cert.selected_axis_name
                ),
                "bottleneck_margin": (
                    None if graph_cert is None else graph_cert.bottleneck_margin
                ),
                "axis_graph_box_evals": (
                    0 if graph_cert is None else int(graph_cert.evaluated_box_count)
                ),
                "volume_weight_in_root": _node_volume_weight(rel_depth),
                "empirical_contradictions": contradictions,
                "elapsed_seconds": float(elapsed),
                "inherited_parent_graph": False,
            }
            records.append(rec)
            processed += 1

            if record_callback is not None:
                record_callback(rec)

            if state == "UNRESOLVED":
                node["last_c0_state"] = c0.state
                next_active.append(node)

            if logger and (
                processed <= 16 or processed % 32 == 0 or contradictions
            ):
                logger(
                    f"C2V3_PROGRESS processed={processed} rel={rel_depth} "
                    f"root={node['root_anchor_id']} path={node['path']} "
                    f"state={state} c0={c0.state} "
                    f"axis={rec['selected_axis']} "
                    f"elapsed={elapsed:.3f}s "
                    f"contradictions={len(contradictions)}"
                )

        active = next_active
        level_summaries.append(summarize(rel_depth, active))
        if logger:
            logger(
                f"C2V3_LEVEL_DONE rel={rel_depth} "
                f"remaining={len(active)} summary={level_summaries[-1]}"
            )
        if not active:
            break

    # Remaining active leaves are explicit final abstentions.
    final_unresolved_keys = {
        (int(n["root_anchor_id"]), n["path"]) for n in active
    }
    contradiction_count = sum(bool(r["empirical_contradictions"]) for r in records)

    graph_roots = defaultdict(int)
    for r in records:
        if r["state"] == "GRAPH":
            graph_roots[int(r["root_anchor_id"])] += 1

    final_active_roots = Counter(int(n["root_anchor_id"]) for n in active)
    fully_closed_roots = [
        int(r["anchor_id"]) for r in roots
        if final_active_roots[int(r["anchor_id"])] == 0
    ]

    # Source existence roots come from the C2 V2 profile transition field when
    # present. Fall back to C0 ZERO_EXISTS for roots without it.
    existence_roots = []
    for root in roots:
        a = int(root["anchor_id"])
        prior = c2map[a]
        source_exists = bool(
            prior.get("c2_v1_state") == "PROVEN_CENTER_CHORD_CROSSING"
            or root["c0_state_m2"] == "PROVEN_ZERO_EXISTS"
        )
        if source_exists:
            existence_roots.append(a)

    existence_partition_alarms = [
        a for a in existence_roots
        if a in fully_closed_roots and graph_roots[a] == 0
    ]

    return {
        "schema": "RealSaS.VF11C2V3AdaptiveEmptyOrGraphProfile.v1",
        "status": "RESEARCH_MEASUREMENT_ONLY__NO_PRODUCT_AUTHORITY",
        "numeric_rigor": "ORDINARY_FLOAT64__NOT_DIRECTED_ROUNDING",
        "root_count": 12,
        "max_relative_depth": int(max_relative_depth),
        "c0_micro_depth": int(c0_micro_depth),
        "c1_micro_depth": int(c1_micro_depth),
        "face_micro_depth": int(face_micro_depth),
        "level_summaries": level_summaries,
        "fully_closed_root_count": int(len(fully_closed_roots)),
        "fully_closed_root_ids": sorted(fully_closed_roots),
        "existence_root_ids": sorted(existence_roots),
        "existence_partition_alarm_count": int(len(existence_partition_alarms)),
        "existence_partition_alarm_root_ids": sorted(existence_partition_alarms),
        "final_unresolved_leaf_count": int(len(active)),
        "final_unresolved_root_ids": sorted(final_active_roots),
        "empirical_contradiction_count": int(contradiction_count),
        "processed_child_count": int(processed),
        "elapsed_seconds_total": float(time.perf_counter() - started),
        "records": records,
    }
