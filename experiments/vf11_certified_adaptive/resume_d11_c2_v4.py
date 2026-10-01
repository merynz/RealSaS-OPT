from __future__ import annotations

"""C2 V4: resume the frozen C2 V3 adaptive partition by exactly one level.

Consumes the d10 C2 V3 profile and evaluates only its 72 final unresolved leaves.
Each unresolved leaf is split into eight d11 children and classified as:
EMPTY, GRAPH, or UNRESOLVED.

Research-only. No prior level recomputation, no product authority, no global topology.
"""

from collections import Counter, defaultdict
import time
from typing import Callable

import numpy as np
import torch

from adaptive_empty_graph_c2_v3 import (
    EMPTY_STATES,
    GRAPH_STATE,
    _check_axis_graph_empirically,
    _check_c0_empty_empirically,
)
from axis_graph_c2_v2 import (
    prepare_field,
    prepare_planes,
    certify_axis_graph_c2_v2_prepared,
)
from paired_profile_c0_v3 import cell_bounds_from_index
from range_engine_c0_v3 import certify_cell_c0_ladder_prepared


def child_bits(child_index: int) -> np.ndarray:
    i = int(child_index)
    if i < 0 or i > 7:
        raise ValueError("CHILD_INDEX_OUT_OF_RANGE")
    return np.asarray([(i >> 2) & 1, (i >> 1) & 1, i & 1], dtype=np.int64)


def descend_box(
    lo: np.ndarray,
    hi: np.ndarray,
    path: str,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(lo, dtype=np.float64).copy()
    b = np.asarray(hi, dtype=np.float64).copy()
    for ch in str(path):
        i = int(ch)
        bits = child_bits(i)
        mid = (a + b) * 0.5
        na = np.where(bits == 0, a, mid)
        nb = np.where(bits == 0, mid, b)
        a, b = na, nb
    return a, b


def _source_roots(c1_fixed_profile: dict) -> dict[int, dict]:
    roots = [
        r for r in c1_fixed_profile["records"]
        if bool(r["primary_handoff"]) and r["c1_v2_state"] != "UNKNOWN"
    ]
    if len(roots) != 12:
        raise ValueError(f"EXPECTED_12_PRIMARY_ROOTS:{len(roots)}")
    return {int(r["anchor_id"]): r for r in roots}


def _source_final_terminals_and_active(
    prior_profile: dict,
) -> tuple[list[dict], list[dict]]:
    max_rel = int(prior_profile["executed_max_relative_depth"])
    if max_rel != 2:
        raise ValueError(f"EXPECTED_SOURCE_RELATIVE_DEPTH_2:{max_rel}")

    records = prior_profile["records"]
    active = [
        r for r in records
        if int(r["relative_depth"]) == max_rel and r["state"] == "UNRESOLVED"
    ]
    if len(active) != int(prior_profile["final_unresolved_leaf_count"]):
        raise ValueError(
            "SOURCE_FINAL_UNRESOLVED_COUNT_MISMATCH:"
            f"{len(active)}!={prior_profile['final_unresolved_leaf_count']}"
        )

    terminals = []
    for r in records:
        rel = int(r["relative_depth"])
        state = r["state"]
        if state == "UNRESOLVED":
            continue
        # A certified record remains terminal iff no later record descends from it.
        root = int(r["root_anchor_id"])
        path = str(r["path"])
        has_descendant = any(
            int(q["root_anchor_id"]) == root
            and int(q["relative_depth"]) > rel
            and str(q["path"]).startswith(path)
            for q in records
        )
        if not has_descendant:
            terminals.append(r)

    return terminals, active


def _root_bounds(
    root: dict,
    domain_lo: float,
    domain_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    return cell_bounds_from_index(
        tuple(int(v) for v in root["cell_index_xyz"]),
        int(root["depth"]),
        domain_lo,
        domain_hi,
    )


def run_d11_resume(
    field,
    planes: torch.Tensor,
    c1_fixed_profile: dict,
    c2_v1_profile: dict,
    prior_c2_v3_profile: dict,
    *,
    domain_lo: float,
    domain_hi: float,
    c0_micro_depth: int = 2,
    c1_micro_depth: int = 2,
    face_micro_depth: int = 2,
    empirical_checks: bool = True,
    logger: Callable[[str], None] | None = print,
    record_callback: Callable[[dict], None] | None = None,
) -> dict:
    roots = _source_roots(c1_fixed_profile)
    c2v1 = {int(r["anchor_id"]): r for r in c2_v1_profile["records"]}
    if set(c2v1) != set(roots):
        raise ValueError("C2V1_ROOT_SET_MISMATCH")

    source_terminals, source_active = _source_final_terminals_and_active(
        prior_c2_v3_profile
    )
    if len(source_active) != 72:
        raise ValueError(f"EXPECTED_72_SOURCE_UNRESOLVED:{len(source_active)}")

    p = prepare_field(field)
    pp = prepare_planes(planes)

    new_records = []
    started = time.perf_counter()

    for parent_ord, parent in enumerate(source_active, 1):
        anchor = int(parent["root_anchor_id"])
        root_lo, root_hi = _root_bounds(
            roots[anchor], domain_lo, domain_hi
        )
        parent_lo, parent_hi = descend_box(
            root_lo, root_hi, str(parent["path"])
        )
        mid = (parent_lo + parent_hi) * 0.5

        for child_i in range(8):
            bits = child_bits(child_i)
            lo = np.where(bits == 0, parent_lo, mid)
            hi = np.where(bits == 0, mid, parent_hi)
            path = f"{parent['path']}{child_i}"

            t0 = time.perf_counter()
            c0cert = certify_cell_c0_ladder_prepared(
                p,
                pp,
                lo,
                hi,
                max_micro_depth=c0_micro_depth,
            )
            c0 = c0cert.by_micro_depth[c0_micro_depth]
            contradictions = []
            graph_cert = None

            if c0.state in EMPTY_STATES:
                state = (
                    "EMPTY_POSITIVE"
                    if c0.state == "PROVEN_EMPTY_POSITIVE"
                    else "EMPTY_NEGATIVE"
                )
                if empirical_checks:
                    contradictions.extend(
                        _check_c0_empty_empirically(
                            field, pp, lo, hi, c0.state
                        )
                    )
            else:
                graph_cert = certify_axis_graph_c2_v2_prepared(
                    p,
                    pp,
                    lo,
                    hi,
                    c1_max_micro_depth=c1_micro_depth,
                    face_micro_depth=face_micro_depth,
                )
                if empirical_checks:
                    contradictions.extend(
                        _check_axis_graph_empirically(
                            field, pp, lo, hi, graph_cert
                        )
                    )
                state = (
                    "GRAPH"
                    if graph_cert.state == GRAPH_STATE
                    else "UNRESOLVED"
                )

            elapsed = time.perf_counter() - t0
            rec = {
                "root_anchor_id": anchor,
                "parent_path": str(parent["path"]),
                "path": path,
                "relative_depth": 3,
                "absolute_depth": int(roots[anchor]["depth"]) + 3,
                "state": state,
                "c0_state": c0.state,
                "c0_lower": float(c0.lower),
                "c0_upper": float(c0.upper),
                "c0_unresolved_leaf_count": int(c0.unresolved_leaf_count),
                "c0_actual_box_evals": int(c0cert.actual_evaluated_box_count),
                "axis_graph_state": (
                    None if graph_cert is None else graph_cert.state
                ),
                "selected_axis": (
                    None if graph_cert is None else graph_cert.selected_axis_name
                ),
                "bottleneck_margin": (
                    None if graph_cert is None else graph_cert.bottleneck_margin
                ),
                "axis_graph_box_evals": (
                    0 if graph_cert is None
                    else int(graph_cert.evaluated_box_count)
                ),
                "volume_weight_in_root": 1.0 / 512.0,
                "empirical_contradictions": contradictions,
                "elapsed_seconds": float(elapsed),
            }
            new_records.append(rec)
            if record_callback is not None:
                record_callback(rec)

        if logger and (
            parent_ord <= 4 or parent_ord % 8 == 0 or parent_ord == len(source_active)
        ):
            done = parent_ord * 8
            states = Counter(r["state"] for r in new_records)
            logger(
                f"C2V4_D11_PROGRESS parents={parent_ord}/{len(source_active)} "
                f"children={done}/576 states={dict(states)}"
            )

    final_unresolved = [
        r for r in new_records if r["state"] == "UNRESOLVED"
    ]
    final_terminals = source_terminals + [
        r for r in new_records if r["state"] != "UNRESOLVED"
    ]

    active_roots = Counter(
        int(r["root_anchor_id"]) for r in final_unresolved
    )
    fully_closed_root_ids = sorted(
        a for a in roots if active_roots[a] == 0
    )
    graph_roots = sorted({
        int(r["root_anchor_id"])
        for r in final_terminals
        if r["state"] == "GRAPH"
    })

    existence_roots = sorted(
        a for a, r in c2v1.items()
        if r["c2"]["state"] == "PROVEN_CENTER_CHORD_CROSSING"
    )
    existence_partition_alarms = sorted(
        a for a in existence_roots
        if a in fully_closed_root_ids and a not in graph_roots
    )

    empirical_contradictions = [
        r for r in new_records if r["empirical_contradictions"]
    ]

    source_fraction = float(
        prior_c2_v3_profile["level_summaries"][-1][
            "unresolved_root_volume_fraction"
        ]
    )
    target_fraction = (
        len(final_unresolved) * (1.0 / 512.0) / 12.0
    )
    relative_reduction = (
        0.0 if source_fraction <= 0.0
        else 1.0 - target_fraction / source_fraction
    )

    new_state_counts = Counter(r["state"] for r in new_records)
    new_c0_counts = Counter(r["c0_state"] for r in new_records)
    new_graph_axes = Counter(
        r["selected_axis"]
        for r in new_records
        if r["state"] == "GRAPH"
    )

    unresolved_c0 = Counter(
        r["c0_state"] for r in final_unresolved
    )

    return {
        "schema": "RealSaS.VF11C2V4D11ResumeProfile.v1",
        "status": "RESEARCH_MEASUREMENT_ONLY__NO_PRODUCT_AUTHORITY",
        "numeric_rigor": "ORDINARY_FLOAT64__NOT_DIRECTED_ROUNDING",
        "source_max_relative_depth": 2,
        "target_relative_depth": 3,
        "source_unresolved_leaf_count": int(len(source_active)),
        "processed_child_count": int(len(new_records)),
        "new_child_state_counts": dict(sorted(new_state_counts.items())),
        "new_child_c0_state_counts": dict(sorted(new_c0_counts.items())),
        "new_graph_axis_counts": dict(sorted(new_graph_axes.items())),
        "final_unresolved_leaf_count": int(len(final_unresolved)),
        "final_unresolved_c0_state_counts": dict(sorted(unresolved_c0.items())),
        "final_unresolved_root_ids": sorted(active_roots),
        "fully_closed_root_count": int(len(fully_closed_root_ids)),
        "fully_closed_root_ids": fully_closed_root_ids,
        "roots_with_graph_descendant_count": int(len(graph_roots)),
        "roots_with_graph_descendant_ids": graph_roots,
        "existence_root_ids": existence_roots,
        "existence_partition_alarm_count": int(
            len(existence_partition_alarms)
        ),
        "existence_partition_alarm_root_ids": existence_partition_alarms,
        "empirical_contradiction_count": int(
            len(empirical_contradictions)
        ),
        "source_unresolved_root_volume_fraction": source_fraction,
        "target_unresolved_root_volume_fraction": float(target_fraction),
        "relative_unresolved_volume_reduction": float(relative_reduction),
        "source_fully_closed_root_count": int(
            prior_c2_v3_profile["fully_closed_root_count"]
        ),
        "additional_root_closure_count": int(
            len(fully_closed_root_ids)
            - int(prior_c2_v3_profile["fully_closed_root_count"])
        ),
        "source_graph_descendant_root_count": int(
            prior_c2_v3_profile["level_summaries"][-1][
                "roots_with_graph_descendant_count"
            ]
        ),
        "graph_descendant_root_gain": int(
            len(graph_roots)
            - int(
                prior_c2_v3_profile["level_summaries"][-1][
                    "roots_with_graph_descendant_count"
                ]
            )
        ),
        "elapsed_seconds_total": float(time.perf_counter() - started),
        "new_records": new_records,
    }
