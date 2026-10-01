from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import axis_graph_c2_v2 as g2  # noqa: E402
from directional_regular_c1_v1 import C1CandidateResult  # noqa: E402
from oriented_crossing_c2_v1 import FaceSignCertificate  # noqa: E402


class TinyField(nn.Module):
    def __init__(self, channels=2, hidden=8):
        super().__init__()
        width = 3 * channels
        self.body = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.sdf_head = nn.Linear(hidden, 1)

    @staticmethod
    def sample(planes, points):
        b, _, c, h, w = planes.shape
        x, y, z = points.unbind(-1)
        grids = torch.stack(
            [
                torch.stack([x, y], -1),
                torch.stack([x, z], -1),
                torch.stack([y, z], -1),
            ],
            dim=1,
        )
        sampled = F.grid_sample(
            planes.reshape(b * 3, c, h, w),
            grids.reshape(b * 3, points.shape[1], 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return (
            sampled.squeeze(-1)
            .reshape(b, 3, c, points.shape[1])
            .permute(0, 3, 1, 2)
            .reshape(b, points.shape[1], 3 * c)
        )

    def forward(self, planes, points):
        h = self.body(self.sample(planes, points))
        return {"sdf": self.sdf_head(h).squeeze(-1)}


def _deriv(axis, state, margin=None):
    if state == g2.C1_POS:
        lo, hi = margin, margin + 1.0
        cmin, cmax, cm = margin, margin + 1.0, margin
    elif state == g2.C1_NEG:
        lo, hi = -(margin + 1.0), -margin
        cmin, cmax, cm = -(margin + 1.0), -margin, margin
    else:
        lo, hi = -1.0, 1.0
        cmin = cmax = cm = None
    return C1CandidateResult(
        direction=tuple(float(v) for v in np.eye(3)[axis]),
        state=state,
        lower_margin=float(lo),
        upper_margin=float(hi),
        evaluated_box_count=1,
        positive_leaf_count=1 if state == g2.C1_POS else 0,
        negative_leaf_count=1 if state == g2.C1_NEG else 0,
        unresolved_leaf_count=1 if state == "UNKNOWN" else 0,
        max_micro_depth=0,
        max_depth_reached=0,
        coverage_complete=True,
        certified_terminal_lower=cmin,
        certified_terminal_upper=cmax,
        certificate_margin=cm,
    )


def _face(axis, role, desired_sign, proven, margin=0.2):
    return FaceSignCertificate(
        axis=axis,
        coordinate=0.0,
        role=role,
        desired_sign=desired_sign,
        state="PROVEN_FACE_SIGN" if proven else "UNKNOWN",
        margin=margin if proven else None,
        evaluated_box_count=1,
        certified_leaf_count=1 if proven else 0,
        unresolved_leaf_count=0 if proven else 1,
    )


def test_axis_graph_selects_largest_bottleneck_margin(monkeypatch):
    calls = {"deriv": 0, "face": 0}
    derivs = [
        _deriv(0, g2.C1_POS, 0.5),
        _deriv(1, g2.C1_NEG, 0.4),
        _deriv(2, "UNKNOWN"),
    ]

    def fake_deriv(*args, **kwargs):
        i = calls["deriv"]
        calls["deriv"] += 1
        return derivs[i]

    faces = iter([
        _face(0, "AXIS_ENTRY", -1, True, 0.3),
        _face(0, "AXIS_EXIT", 1, True, 0.2),   # X bottleneck 0.2
        _face(1, "AXIS_ENTRY", -1, True, 0.35),
        _face(1, "AXIS_EXIT", 1, True, 0.30), # Y bottleneck 0.3
    ])

    monkeypatch.setattr(g2, "_evaluate_candidate_v2", fake_deriv)
    monkeypatch.setattr(g2, "certify_face_sign_prepared", lambda *a, **k: next(faces))

    cert = g2.certify_axis_graph_c2_v2_prepared(
        None,
        torch.zeros(1, 3, 1, 8, 8, dtype=torch.float64),
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        c1_max_micro_depth=0,
        face_micro_depth=0,
    )
    assert cert.state == "PROVEN_AXIS_GRAPH_PATCH"
    assert cert.selected_axis == 1
    assert cert.selected_axis_name == "Y"
    assert cert.selected_oriented_sign == -1
    assert abs(cert.bottleneck_margin - 0.30) < 1e-12


def test_axis_graph_fails_closed_when_one_required_face_is_unknown(monkeypatch):
    derivs = iter([
        _deriv(0, g2.C1_POS, 0.5),
        _deriv(1, "UNKNOWN"),
        _deriv(2, "UNKNOWN"),
    ])
    faces = iter([
        _face(0, "AXIS_ENTRY", -1, True, 0.2),
        _face(0, "AXIS_EXIT", 1, False),
    ])
    monkeypatch.setattr(g2, "_evaluate_candidate_v2", lambda *a, **k: next(derivs))
    monkeypatch.setattr(g2, "certify_face_sign_prepared", lambda *a, **k: next(faces))

    cert = g2.certify_axis_graph_c2_v2_prepared(
        None,
        torch.zeros(1, 3, 1, 8, 8, dtype=torch.float64),
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        c1_max_micro_depth=0,
        face_micro_depth=0,
    )
    assert cert.state == "UNKNOWN_AXIS_GRAPH"
    assert cert.selected_axis is None


def test_axis_graph_constant_zero_field_is_unknown():
    torch.manual_seed(601)
    field = TinyField().double().eval()
    for p in field.parameters():
        p.data.zero_()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)

    cert = g2.certify_axis_graph_c2_v2(
        field,
        planes,
        np.array([-0.2, -0.2, -0.2], dtype=np.float64),
        np.array([ 0.2,  0.2,  0.2], dtype=np.float64),
        c1_max_micro_depth=0,
        face_micro_depth=0,
    )
    assert cert.state == "UNKNOWN_AXIS_GRAPH"
    assert all(a.derivative_result.state == "UNKNOWN" for a in cert.attempts)
