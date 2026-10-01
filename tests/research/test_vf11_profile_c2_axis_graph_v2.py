from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import profile_c2_axis_graph_v2 as p2  # noqa: E402
from axis_graph_c2_v2 import AxisGraphAttempt, AxisGraphCertificate  # noqa: E402
from directional_regular_c1_v1 import C1CandidateResult  # noqa: E402


class DummyField:
    pass


def _deriv(axis, state="PROVEN_DIRECTIONAL_REGULAR_POSITIVE", margin=0.2):
    return C1CandidateResult(
        direction=tuple(float(v) for v in np.eye(3)[axis]),
        state=state,
        lower_margin=margin if "POSITIVE" in state else -1.0,
        upper_margin=1.0 if "POSITIVE" in state else -margin,
        evaluated_box_count=1,
        positive_leaf_count=1 if "POSITIVE" in state else 0,
        negative_leaf_count=1 if "NEGATIVE" in state else 0,
        unresolved_leaf_count=0,
        max_micro_depth=0,
        max_depth_reached=0,
        coverage_complete=True,
        certified_terminal_lower=margin if "POSITIVE" in state else -1.0,
        certified_terminal_upper=1.0 if "POSITIVE" in state else -margin,
        certificate_margin=margin,
    )


def test_axis_graph_profiler_preserves_12_cell_primary_cohort(monkeypatch):
    records = []
    for i in range(12):
        records.append({
            "anchor_id": 200 + i,
            "cell_index_xyz": [30+i, 40+i, 50+i],
            "depth": 8,
            "primary_handoff": True,
            "c0_state_m2": "UNKNOWN",
            "c1_v2_state": "PROVEN_DIRECTIONAL_REGULAR_POSITIVE",
            "c1_v2_direction": [1.0, 0.0, 0.0],
            "c1_v2_certificate_margin": 0.25,
        })
    source = {"records": records}

    monkeypatch.setattr(p2, "prepare_field", lambda field: None)
    monkeypatch.setattr(p2, "prepare_planes", lambda planes: planes)

    def fake_cert(*args, **kwargs):
        attempt = AxisGraphAttempt(
            axis=0,
            axis_name="X",
            derivative_result=_deriv(0),
            oriented_sign=1,
            derivative_margin=0.2,
            entry_face=None,
            exit_face=None,
            graph_certified=True,
            bottleneck_margin=0.1,
            evaluated_box_count=3,
        )
        unknown_y = AxisGraphAttempt(
            axis=1,
            axis_name="Y",
            derivative_result=_deriv(1, state="UNKNOWN", margin=0.2),
            oriented_sign=None,
            derivative_margin=None,
            entry_face=None,
            exit_face=None,
            graph_certified=False,
            bottleneck_margin=None,
            evaluated_box_count=1,
        )
        unknown_z = AxisGraphAttempt(
            axis=2,
            axis_name="Z",
            derivative_result=_deriv(2, state="UNKNOWN", margin=0.2),
            oriented_sign=None,
            derivative_margin=None,
            entry_face=None,
            exit_face=None,
            graph_certified=False,
            bottleneck_margin=None,
            evaluated_box_count=1,
        )
        return AxisGraphCertificate(
            state="PROVEN_AXIS_GRAPH_PATCH",
            selected_axis=0,
            selected_axis_name="X",
            selected_oriented_sign=1,
            bottleneck_margin=0.1,
            attempts=(attempt, unknown_y, unknown_z),
            evaluated_box_count=5,
        )

    monkeypatch.setattr(p2, "certify_axis_graph_c2_v2_prepared", fake_cert)
    monkeypatch.setattr(
        p2,
        "_direct_directional_samples",
        lambda *a, **k: np.ones(125, dtype=np.float64),
    )

    out = p2.profile_c2_axis_graph_v2(
        DummyField(),
        torch.zeros(1, 3, 1, 8, 8, dtype=torch.float64),
        source,
        domain_lo=-0.875,
        domain_hi=0.875,
        c1_max_micro_depth=0,
        face_micro_depth=0,
        logger=None,
    )

    assert out["primary_c1_regular_count"] == 12
    assert out["graph_patch_count"] == 12
    assert out["state_counts"] == {"PROVEN_AXIS_GRAPH_PATCH": 12}
    assert out["selected_axis_counts"] == {"X": 12}
    assert out["empirical_contradiction_count"] == 0
