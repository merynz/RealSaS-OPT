from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import profile_c2_primary_v1 as profiler  # noqa: E402
from oriented_crossing_c2_v1 import C2CrossingCertificate, PointValueBound  # noqa: E402


class DummyField:
    pass


def test_c2_primary_profiler_preserves_12_cell_cohort(monkeypatch):
    records = []
    for i in range(19):
        regular = i < 12
        records.append({
            "anchor_id": 100 + i,
            "cell_index_xyz": [50 + i, 60 + i, 70 + i],
            "depth": 8,
            "primary_handoff": True,
            "c0_state_m2": "UNKNOWN",
            "c1_v2_state": (
                "PROVEN_DIRECTIONAL_REGULAR_POSITIVE"
                if regular else "UNKNOWN"
            ),
            "c1_v2_direction": [1.0, 0.0, 0.0] if regular else None,
            "c1_v2_certificate_margin": 0.1 if regular else None,
        })
    source = {"records": records}

    monkeypatch.setattr(profiler, "prepare_field", lambda field: None)
    monkeypatch.setattr(profiler, "prepare_planes", lambda planes: planes)

    def fake_cert(*args, **kwargs):
        return C2CrossingCertificate(
            state="PROVEN_CENTER_CHORD_CROSSING",
            oriented_direction=(1.0, 0.0, 0.0),
            center_entry=PointValueBound(-2.0, -1.0, (-1.0, 0.0, 0.0)),
            center_exit=PointValueBound(1.0, 2.0, (1.0, 0.0, 0.0)),
            center_chord_certified=True,
            center_entry_margin=1.0,
            center_exit_margin=1.0,
            full_oriented_crossing_certified=False,
            face_certificates=(),
            c0_zero_exists=False,
            evaluated_box_count=2,
        )

    monkeypatch.setattr(
        profiler, "certify_oriented_crossing_c2_prepared", fake_cert
    )
    monkeypatch.setattr(
        profiler,
        "_direct_values",
        lambda field, planes, pts: np.asarray([-1.5, 1.5], dtype=np.float64),
    )

    out = profiler.profile_c2_primary(
        DummyField(),
        torch.zeros(1, 3, 1, 8, 8, dtype=torch.float64),
        source,
        domain_lo=-0.875,
        domain_hi=0.875,
        face_micro_depth=0,
        logger=None,
    )

    assert out["primary_c1_regular_count"] == 12
    assert out["existence_proven_count"] == 12
    assert out["center_chord_crossing_count"] == 12
    assert out["full_oriented_crossing_count"] == 0
    assert out["state_counts"] == {"PROVEN_CENTER_CHORD_CROSSING": 12}
    assert out["empirical_contradiction_count"] == 0
