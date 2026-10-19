import json
from pathlib import Path

import torch

from models.iris.vnext.h_rep_probe_v1 import (
    PROBE_ARMS_V1,
    HRepresentationProbeV1,
    build_probe_input_v1,
    h_rep_probe_objective_v1,
)


def test_all_probe_arms_have_exact_equal_capacity_and_input_envelope():
    h = torch.randn(11, 384)
    xyz = torch.randn(11, 3).clamp(-1, 1)
    field = torch.randn(11)
    shuffled = h.flip(0)
    counts = []
    shapes = []
    for arm in PROBE_ARMS_V1:
        model = HRepresentationProbeV1(arm)
        x = build_probe_input_v1(
            arm,
            h=h,
            xyz_iris=xyz,
            raw_field=field,
            shuffled_h=shuffled if arm == "SHUFFLED_H_XYZ" else None,
        )
        out = model(x)
        assert out["distance_iris"].shape == (11,)
        assert out["offset_to_reference_iris"].shape == (11, 3)
        assert out["normal_unit"].shape == (11, 3)
        assert out["near_logit"].shape == (11,)
        counts.append(model.trainable_parameter_count())
        shapes.append(tuple(x.shape))
    assert len(set(counts)) == 1
    assert set(shapes) == {(11, 387)}


def test_raw_field_and_xyz_controls_zero_unused_evidence_channels():
    h = torch.randn(7, 384)
    xyz = torch.randn(7, 3).clamp(-1, 1)
    field = torch.randn(7)
    raw = build_probe_input_v1("RAW_FIELD_XYZ", h=h, xyz_iris=xyz, raw_field=field)
    xyz_only = build_probe_input_v1("XYZ_ONLY", h=h, xyz_iris=xyz, raw_field=field)
    assert torch.equal(raw[:, 0], field)
    assert torch.count_nonzero(raw[:, 1:384]) == 0
    assert torch.count_nonzero(xyz_only[:, :384]) == 0
    assert torch.equal(raw[:, 384:], xyz)
    assert torch.equal(xyz_only[:, 384:], xyz)


def test_probe_objective_is_finite_and_backpropagates():
    model = HRepresentationProbeV1("H_XYZ")
    n = 17
    h = torch.randn(n, 384)
    xyz = torch.randn(n, 3).clamp(-1, 1)
    field = torch.randn(n)
    pred = model(build_probe_input_v1("H_XYZ", h=h, xyz_iris=xyz, raw_field=field))
    target_distance = torch.rand(n) * 0.1
    target_offset = torch.randn(n, 3) * 0.02
    normal = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
    loss = h_rep_probe_objective_v1(
        pred,
        distance_iris=target_distance,
        offset_to_reference_iris=target_offset,
        normal_unit=normal,
        normal_valid=torch.ones(n, dtype=torch.bool),
        near_label=target_distance <= 0.03,
    )
    assert all(torch.isfinite(v).all() for v in loss.values())
    loss["total"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_preregistration_keeps_controls_and_tessa_blocked():
    path = Path("canonical/H_REP_P0_PREREGISTRATION_V1.json")
    value = json.loads(path.read_text())
    assert value["schema"] == "RealSaS.HRepP0Preregistration.v1"
    assert tuple(value["probe_arms"]) == PROBE_ARMS_V1
    assert value["shared_probe_architecture"]["equal_parameter_count_across_arms"] is True
    assert value["interpretation"]["pass_authorizes_tessa"] is False
    assert value["interpretation"]["pass_authorizes_surf_a2"] is False
    assert value["heldout_court"]["thresholds_may_change_after_results"] is False
