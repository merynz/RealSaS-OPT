from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import oriented_crossing_c2_v1 as c2  # noqa: E402


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


def test_increasing_direction_orientation_and_center_chord_geometry():
    d = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    pos = c2.orient_increasing_direction(c2.C1_POS, d)
    neg = c2.orient_increasing_direction(c2.C1_NEG, d)
    assert np.allclose(pos, -neg)

    lo = np.array([-2.0, -1.0, 0.0])
    hi = np.array([ 2.0,  3.0, 4.0])
    entry, exit = c2.center_chord_endpoints(lo, hi, pos)
    assert np.all(entry >= lo - 1e-12)
    assert np.all(entry <= hi + 1e-12)
    assert np.all(exit >= lo - 1e-12)
    assert np.all(exit <= hi + 1e-12)
    delta = exit - entry
    assert np.linalg.norm(np.cross(delta, pos)) < 1e-10
    assert np.any(np.isclose(entry, lo) | np.isclose(entry, hi))
    assert np.any(np.isclose(exit, lo) | np.isclose(exit, hi))


def test_face_sign_certifier_handles_constant_positive_and_negative_fields():
    torch.manual_seed(501)
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)
    lo = np.array([-0.30, -0.25, -0.20], dtype=np.float64)
    hi = np.array([ 0.30,  0.25,  0.20], dtype=np.float64)

    field = TinyField().double().eval()
    for parameter in field.parameters():
        parameter.data.zero_()
    field.sdf_head.bias.data.fill_(1.0)
    p = c2.prepare_field(field)
    pp = c2.prepare_planes(planes)

    flo = lo.copy()
    fhi = hi.copy()
    flo[0] = fhi[0] = lo[0]
    cert = c2.certify_face_sign_prepared(
        p, pp, flo, fhi,
        fixed_axis=0, role="TEST", desired_sign=1, max_micro_depth=1
    )
    assert cert.state == "PROVEN_FACE_SIGN"
    assert cert.margin is not None and cert.margin > 0.9
    assert cert.unresolved_leaf_count == 0

    field.sdf_head.bias.data.fill_(-1.0)
    p = c2.prepare_field(field)
    cert = c2.certify_face_sign_prepared(
        p, pp, flo, fhi,
        fixed_axis=0, role="TEST", desired_sign=-1, max_micro_depth=1
    )
    assert cert.state == "PROVEN_FACE_SIGN"
    assert cert.margin is not None and cert.margin > 0.9
    assert cert.unresolved_leaf_count == 0


def test_point_value_bound_contains_direct_evaluation():
    torch.manual_seed(502)
    field = TinyField().double().eval()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)
    p = c2.prepare_field(field)
    pp = c2.prepare_planes(planes)
    point = np.array([0.13, -0.22, 0.31], dtype=np.float64)
    bound = c2.point_value_bound_prepared(p, pp, point)
    q = torch.tensor(point.reshape(1, 1, 3), dtype=torch.float64)
    value = float(field(pp, q)["sdf"].item())
    assert value >= bound.lower - 1e-9
    assert value <= bound.upper + 1e-9


def _fake_face(axis, role, desired_sign, proven=True):
    return c2.FaceSignCertificate(
        axis=axis,
        coordinate=0.0,
        role=role,
        desired_sign=desired_sign,
        state="PROVEN_FACE_SIGN" if proven else "UNKNOWN",
        margin=0.25 if proven else None,
        evaluated_box_count=1,
        certified_leaf_count=1 if proven else 0,
        unresolved_leaf_count=0 if proven else 1,
    )


def test_c2_state_hierarchy_with_precomputed_semantics(monkeypatch):
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([ 1.0,  1.0,  1.0])
    d = np.array([1.0, 0.0, 0.0])
    planes = torch.zeros(1, 3, 1, 8, 8, dtype=torch.float64)

    point_bounds = iter([
        c2.PointValueBound(-2.0, -1.0, (-1.0, 0.0, 0.0)),
        c2.PointValueBound( 1.0,  2.0, ( 1.0, 0.0, 0.0)),
    ])
    monkeypatch.setattr(c2, "point_value_bound_prepared", lambda *a, **k: next(point_bounds))
    monkeypatch.setattr(
        c2,
        "certify_face_sign_prepared",
        lambda *a, role, desired_sign, fixed_axis, **k:
            _fake_face(fixed_axis, role, desired_sign, proven=True),
    )
    full = c2.certify_oriented_crossing_c2_prepared(
        None, planes, lo, hi,
        c1_state=c2.C1_POS, c1_direction=d,
        c0_state="UNKNOWN", face_micro_depth=0,
    )
    assert full.state == "PROVEN_FULL_ORIENTED_CROSSING"
    assert full.center_chord_certified
    assert full.full_oriented_crossing_certified

    point_bounds = iter([
        c2.PointValueBound(-2.0, -1.0, (-1.0, 0.0, 0.0)),
        c2.PointValueBound( 1.0,  2.0, ( 1.0, 0.0, 0.0)),
    ])
    monkeypatch.setattr(c2, "point_value_bound_prepared", lambda *a, **k: next(point_bounds))
    monkeypatch.setattr(
        c2,
        "certify_face_sign_prepared",
        lambda *a, role, desired_sign, fixed_axis, **k:
            _fake_face(fixed_axis, role, desired_sign, proven=False),
    )
    center = c2.certify_oriented_crossing_c2_prepared(
        None, planes, lo, hi,
        c1_state=c2.C1_POS, c1_direction=d,
        c0_state="UNKNOWN", face_micro_depth=0,
    )
    assert center.state == "PROVEN_CENTER_CHORD_CROSSING"
    assert center.center_chord_certified
    assert not center.full_oriented_crossing_certified

    point_bounds = iter([
        c2.PointValueBound(0.5, 1.0, (-1.0, 0.0, 0.0)),
        c2.PointValueBound(1.5, 2.0, ( 1.0, 0.0, 0.0)),
    ])
    monkeypatch.setattr(c2, "point_value_bound_prepared", lambda *a, **k: next(point_bounds))
    bridge = c2.certify_oriented_crossing_c2_prepared(
        None, planes, lo, hi,
        c1_state=c2.C1_POS, c1_direction=d,
        c0_state="PROVEN_ZERO_EXISTS", face_micro_depth=0,
    )
    assert bridge.state == "PROVEN_C0_EXISTENCE_PLUS_LINE_UNIQUENESS"

    point_bounds = iter([
        c2.PointValueBound(0.5, 1.0, (-1.0, 0.0, 0.0)),
        c2.PointValueBound(1.5, 2.0, ( 1.0, 0.0, 0.0)),
    ])
    monkeypatch.setattr(c2, "point_value_bound_prepared", lambda *a, **k: next(point_bounds))
    regular = c2.certify_oriented_crossing_c2_prepared(
        None, planes, lo, hi,
        c1_state=c2.C1_POS, c1_direction=d,
        c0_state="UNKNOWN", face_micro_depth=0,
    )
    assert regular.state == "REGULAR_ONLY"
