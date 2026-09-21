from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import adaptive_empty_graph_c2_v3 as a3  # noqa: E402


class DummyField:
    pass


def _root_records():
    rows = []
    for i in range(12):
        rows.append({
            "anchor_id": i,
            "cell_index_xyz": [100 + i, 120 + i, 140 + i],
            "depth": 8,
            "primary_handoff": True,
            "c0_state_m2": "PROVEN_ZERO_EXISTS" if i < 3 else "UNKNOWN",
            "c1_v2_state": "PROVEN_DIRECTIONAL_REGULAR_POSITIVE",
        })
    return {"records": rows}


def _c2v2():
    rows = []
    for i in range(12):
        graph = i < 2
        rows.append({
            "anchor_id": i,
            "axis_graph_state": (
                "PROVEN_AXIS_GRAPH_PATCH" if graph else "UNKNOWN_AXIS_GRAPH"
            ),
            "selected_axis_name": "Z" if graph else None,
            "bottleneck_margin": 0.1 if graph else None,
        })
    return {"records": rows}


def _c2v1():
    rows = []
    for i in range(12):
        rows.append({
            "anchor_id": i,
            "c2": {
                "state": (
                    "PROVEN_CENTER_CHORD_CROSSING" if i < 7 else "REGULAR_ONLY"
                )
            },
        })
    return {"records": rows}


def _empty_cert():
    summary = SimpleNamespace(
        state="PROVEN_EMPTY_POSITIVE",
        lower=0.1,
        upper=0.2,
        unresolved_leaf_count=0,
    )
    return SimpleNamespace(
        by_micro_depth={2: summary},
        actual_evaluated_box_count=1,
    )


def test_c2v3_detects_all_empty_partition_alarm_for_existence_roots(monkeypatch):
    monkeypatch.setattr(a3, "prepare_field", lambda field: None)
    monkeypatch.setattr(a3, "prepare_planes", lambda planes: planes)
    monkeypatch.setattr(
        a3,
        "certify_cell_c0_ladder_prepared",
        lambda *args, **kwargs: _empty_cert(),
    )

    def should_not_graph(*args, **kwargs):
        raise AssertionError("axis graph should not run for C0-proven empty child")

    monkeypatch.setattr(
        a3, "certify_axis_graph_c2_v2_prepared", should_not_graph
    )

    out = a3.run_adaptive_empty_or_graph(
        DummyField(),
        torch.zeros(1, 3, 1, 8, 8, dtype=torch.float64),
        _root_records(),
        _c2v2(),
        _c2v1(),
        domain_lo=-0.875,
        domain_hi=0.875,
        max_relative_depth=1,
        empirical_checks=False,
        logger=None,
    )

    assert out["root_count"] == 12
    assert out["fully_closed_root_count"] == 12
    assert out["final_unresolved_leaf_count"] == 0
    # Existence roots 0..6; roots 0 and 1 already have inherited GRAPH leaves.
    assert out["existence_root_ids"] == list(range(7))
    assert out["existence_partition_alarm_root_ids"] == [2, 3, 4, 5, 6]
    assert out["existence_partition_alarm_count"] == 5


def test_octant_split_conserves_parent_box():
    lo = np.array([-1.0, -2.0, -3.0])
    hi = np.array([ 1.0,  2.0,  3.0])
    children = a3._split_octants(lo, hi)
    assert len(children) == 8

    parent_volume = float(np.prod(hi - lo))
    child_volume = sum(float(np.prod(b - a)) for _, a, b in children)
    assert abs(parent_volume - child_volume) < 1e-12

    for bits, a, b in children:
        assert np.all(a >= lo)
        assert np.all(b <= hi)
        assert np.all(b > a)
