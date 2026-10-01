from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import resume_d11_c2_v4 as v4  # noqa: E402


class DummyField:
    pass


def _roots():
    records = []
    for a in range(12):
        records.append({
            "anchor_id": a,
            "cell_index_xyz": [100 + a, 120 + a, 140 + a],
            "depth": 8,
            "primary_handoff": True,
            "c1_v2_state": "PROVEN_DIRECTIONAL_REGULAR_POSITIVE",
        })
    return {"records": records}


def _c2v1():
    rows = []
    for a in range(12):
        rows.append({
            "anchor_id": a,
            "c2": {
                "state": (
                    "PROVEN_CENTER_CHORD_CROSSING"
                    if a in {2, 3}
                    else "REGULAR_ONLY"
                )
            },
        })
    return {"records": rows}


def _prior():
    records = [
        {
            "root_anchor_id": 0,
            "path": "",
            "relative_depth": 0,
            "state": "GRAPH",
        },
        {
            "root_anchor_id": 1,
            "path": "",
            "relative_depth": 0,
            "state": "GRAPH",
        },
    ]

    # Exactly 72 final unresolved d10 leaves distributed across roots 2..11.
    counts = {2: 8, 3: 8}
    counts.update({a: 7 for a in range(4, 12)})
    assert sum(counts.values()) == 72
    for a, n in counts.items():
        for j in range(n):
            # Unique two-digit octant paths per root.
            path = f"{j // 8}{j % 8}"
            records.append({
                "root_anchor_id": a,
                "path": path,
                "relative_depth": 2,
                "state": "UNRESOLVED",
            })

    return {
        "executed_max_relative_depth": 2,
        "final_unresolved_leaf_count": 72,
        "fully_closed_root_count": 2,
        "level_summaries": [{
            "unresolved_root_volume_fraction": 0.09375,
            "roots_with_graph_descendant_count": 2,
        }],
        "records": records,
    }


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


def test_descend_box_path_matches_repeated_octants():
    lo = np.array([-1.0, -2.0, -3.0])
    hi = np.array([ 1.0,  2.0,  3.0])

    a, b = v4.descend_box(lo, hi, "07")

    # child 0 then child 7.
    mid0 = (lo + hi) * 0.5
    lo0 = lo
    hi0 = mid0
    mid1 = (lo0 + hi0) * 0.5
    expected_a = mid1
    expected_b = hi0

    assert np.allclose(a, expected_a)
    assert np.allclose(b, expected_b)


def test_resume_only_expands_72_source_unresolved_and_audits_existence(monkeypatch):
    monkeypatch.setattr(v4, "prepare_field", lambda field: None)
    monkeypatch.setattr(v4, "prepare_planes", lambda planes: planes)
    monkeypatch.setattr(
        v4,
        "certify_cell_c0_ladder_prepared",
        lambda *args, **kwargs: _empty_cert(),
    )

    def should_not_graph(*args, **kwargs):
        raise AssertionError("graph certifier should not run for C0-empty child")

    monkeypatch.setattr(
        v4,
        "certify_axis_graph_c2_v2_prepared",
        should_not_graph,
    )

    out = v4.run_d11_resume(
        DummyField(),
        torch.zeros(1, 3, 1, 8, 8, dtype=torch.float64),
        _roots(),
        _c2v1(),
        _prior(),
        domain_lo=-0.875,
        domain_hi=0.875,
        empirical_checks=False,
        logger=None,
    )

    assert out["source_unresolved_leaf_count"] == 72
    assert out["processed_child_count"] == 576
    assert out["new_child_state_counts"] == {"EMPTY_POSITIVE": 576}
    assert out["final_unresolved_leaf_count"] == 0
    assert out["fully_closed_root_count"] == 12
    assert out["additional_root_closure_count"] == 10
    assert out["existence_root_ids"] == [2, 3]
    assert out["existence_partition_alarm_root_ids"] == [2, 3]
    assert out["existence_partition_alarm_count"] == 2
    assert out["empirical_contradiction_count"] == 0
