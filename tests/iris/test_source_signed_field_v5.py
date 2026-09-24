from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from models.iris.v5.source_signed_field_v5 import (
    SourceSignedFieldPolicyV5,
    exterior_sign_audit_v5,
    source_signed_distance_fields_v5,
    source_signed_field_numpy_v5,
    source_signed_field_torch_v5,
)


def _camera_arrays(views: int = 8):
    origins = np.zeros((views, 3), dtype=np.float32)
    right = np.tile(np.asarray([[1.0, 0.0, 0.0]], np.float32), (views, 1))
    up = np.tile(np.asarray([[0.0, 1.0, 0.0]], np.float32), (views, 1))
    half = np.ones((views,), dtype=np.float32)
    return origins, right, up, half


def test_signed_2d_convention_is_negative_inside_positive_outside():
    policy = SourceSignedFieldPolicyV5(required_resolution=5)
    masks = np.zeros((8, 5, 5), dtype=bool)
    masks[:, 2, 2] = True
    signed = source_signed_distance_fields_v5(masks, policy=policy)
    assert signed.shape == (8, 5, 5)
    assert np.all(signed[:, 2, 2] < 0.0)
    assert np.all(signed[:, 0, 0] > 0.0)


def test_numpy_and_torch_match_f4_manual_bilinear_contract():
    policy = SourceSignedFieldPolicyV5(required_resolution=4)
    yy, xx = np.meshgrid(
        np.arange(4, dtype=np.float32),
        np.arange(4, dtype=np.float32),
        indexing="ij",
    )
    signed = np.stack(
        [0.10 * xx + 0.03 * yy + 0.01 * view - 0.20 for view in range(8)],
        axis=0,
    ).astype(np.float32)
    origins, right, up, cam_half = _camera_arrays()
    points = np.asarray(
        [
            [-0.75, 0.50, 0.0],
            [-0.10, -0.20, 0.2],
            [0.25, 0.25, -0.4],
            [0.90, -0.90, 0.7],
            [1.00, -1.00, 0.0],
        ],
        dtype=np.float32,
    )
    kwargs = dict(
        signed_2d=signed,
        center_xyz=np.zeros(3, np.float32),
        normalization_half_extent=1.0,
        camera_origins=origins,
        camera_right=right,
        camera_screen_up=up,
        camera_half_extent=cam_half,
        policy=policy,
    )
    cpu = source_signed_field_numpy_v5(points, **kwargs)
    gpu_semantics = source_signed_field_torch_v5(points, **kwargs).cpu().numpy()
    np.testing.assert_allclose(cpu, gpu_semantics, rtol=0.0, atol=2e-7)


def test_closed_raster_support_and_nearest_edge_extension_are_exact():
    policy = SourceSignedFieldPolicyV5(required_resolution=4)
    yy, xx = np.meshgrid(
        np.arange(4, dtype=np.float32),
        np.arange(4, dtype=np.float32),
        indexing="ij",
    )
    signed = np.stack(
        [0.10 * xx + 0.03 * yy + 0.01 * view - 0.20 for view in range(8)],
        axis=0,
    ).astype(np.float32)
    origins, right, up, cam_half = _camera_arrays()
    points = np.asarray(
        [
            [1.0, -1.0, 0.0],
            [1.001, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    values = source_signed_field_numpy_v5(
        points,
        signed_2d=signed,
        center_xyz=np.zeros(3, np.float32),
        normalization_half_extent=1.0,
        camera_origins=origins,
        camera_right=right,
        camera_screen_up=up,
        camera_half_extent=cam_half,
        policy=policy,
    )
    # q=(1,-1) projects to sx=RES, sy=RES and therefore remains valid.
    # Nearest extension selects the last raster cell. View 7 wins the MAX.
    expected_last = (0.10 * 3 + 0.03 * 3 + 0.01 * 7 - 0.20) * 0.5
    assert values[0] == pytest.approx(expected_last, abs=2e-7)
    # sx>RES is outside every admitted view and therefore receives unseen +0.25.
    assert values[1] == pytest.approx(0.25, abs=0.0)


def test_view_max_scale_unseen_and_clamp():
    policy = SourceSignedFieldPolicyV5(required_resolution=4)
    origins, right, up, cam_half = _camera_arrays()
    point = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    signed = np.zeros((8, 4, 4), dtype=np.float32)
    for view in range(8):
        signed[view] = float(view) * 0.02
    values = source_signed_field_numpy_v5(
        point,
        signed_2d=signed,
        center_xyz=np.zeros(3, np.float32),
        normalization_half_extent=1.0,
        camera_origins=origins,
        camera_right=right,
        camera_screen_up=up,
        camera_half_extent=cam_half,
        policy=policy,
    )
    assert values[0] == pytest.approx(7 * 0.02 * 0.5, abs=2e-7)

    high = np.full((8, 4, 4), 100.0, np.float32)
    low = np.full((8, 4, 4), -100.0, np.float32)
    common = dict(
        center_xyz=np.zeros(3, np.float32),
        normalization_half_extent=1.0,
        camera_origins=origins,
        camera_right=right,
        camera_screen_up=up,
        camera_half_extent=cam_half,
        policy=policy,
    )
    assert source_signed_field_numpy_v5(point, signed_2d=high, **common)[0] == 0.25
    assert source_signed_field_numpy_v5(point, signed_2d=low, **common)[0] == -0.25

    unseen = source_signed_field_numpy_v5(
        np.asarray([[2.0, 2.0, 0.0]], np.float32),
        signed_2d=signed,
        **common,
    )
    assert unseen[0] == 0.25


def test_camera_basis_is_consumed_without_silent_normalization():
    policy = SourceSignedFieldPolicyV5(required_resolution=4)
    signed = np.zeros((8, 4, 4), dtype=np.float32)
    signed[:, :, 3] = 0.2
    origins, right, up, cam_half = _camera_arrays()
    right_scaled = right.copy()
    right_scaled[:, 0] *= 2.0
    q = np.asarray([[0.5, 0.0, 0.0]], np.float32)
    regular = source_signed_field_numpy_v5(
        q,
        signed_2d=signed,
        center_xyz=np.zeros(3, np.float32),
        normalization_half_extent=1.0,
        camera_origins=origins,
        camera_right=right,
        camera_screen_up=up,
        camera_half_extent=cam_half,
        policy=policy,
    )
    scaled = source_signed_field_numpy_v5(
        q,
        signed_2d=signed,
        center_xyz=np.zeros(3, np.float32),
        normalization_half_extent=1.0,
        camera_origins=origins,
        camera_right=right_scaled,
        camera_screen_up=up,
        camera_half_extent=cam_half,
        policy=policy,
    )
    assert scaled[0] != regular[0]


def test_independent_exterior_sign_audit_ignores_metric_magnitude():
    values = np.asarray([1e-8, 0.01, 0.25, -7.0], dtype=np.float32)
    certified = np.asarray([True, True, True, False])
    report = exterior_sign_audit_v5(values, certified)
    assert report == {
        "certified_exterior_count": 3,
        "nonpositive_certified_exterior_count": 0,
        "nonpositive_fraction": 0.0,
        "passed": True,
    }
    failed = exterior_sign_audit_v5(
        np.asarray([1e-8, 0.0], np.float32),
        np.asarray([True, True]),
    )
    assert failed["passed"] is False
    assert failed["nonpositive_certified_exterior_count"] == 1


def test_canonical_policy_matches_f4_contract_and_frozen_training_authorization():
    policy = json.loads(
        Path("canonical/IRIS_V5_CONTINUOUS_SOURCE_FIELD_POLICY_20260923.json").read_text()
    )
    ontology = json.loads(
        Path("canonical/IRIS_V5_FIELD_ONTOLOGY_RECLOSURE_20260923.json").read_text()
    )
    contract = policy["exact_target_contract"]
    assert contract["signed_2d"] == "EDT(background)-EDT(foreground)"
    assert contract["raster_cell_support"] if "raster_cell_support" in contract else True
    assert contract["sample_coordinates"] == "(sy-0.5,sx-0.5)"
    assert contract["interpolation"] == "BILINEAR"
    assert contract["edge_extension"] == "NEAREST"
    assert contract["multi_view_aggregation"] == "MAX_OVER_VALID_VIEWS"
    assert contract["unseen_value"] == 0.25
    assert contract["field_clamp"] == {"min": -0.25, "max": 0.25}
    assert ontology["evidence"]["exact_r512_exterior_sign_audit"] == {
        "certified_exterior_count": 128851336,
        "nonpositive_certified_exterior_count": 0,
        "pass": True,
        "active_interpretation": "ALL_SOURCE_CERTIFIED_EXTERIOR_R512_POINTS_ARE_STRICTLY_POSITIVE",
    }
    auth = ontology["training_authorization"]
    assert auth["authorized"] is False
    assert auth["blocker"] == "CLOSE_ZERO_UPDATE_H1_H3_BRIDGE_DIAGNOSTIC_BEFORE_FURTHER_TP64_TRAINING_OR_OBJECTIVE_CHANGE"
    assert auth["prior_field_failure_closure"] == "canonical/IRIS_V5_FIELD_FAILURE_DIAGNOSTIC_CLOSURE_20260924.json"
    assert auth["replay_refresh_authorized"] is False
    assert auth["target_aggregation_change_authorized"] is False
    assert auth["architecture_change_authorized"] is False
    assert auth["spatial_resolution_change_authorized"] is False
    assert auth["regularity_loss_change_authorized"] is False
    assert auth["exact_r512_and_stage13_checkpoint_selection_blind"] is True
    diag = ontology["diagnostic_authorization"]
    assert diag["authorized"] is False
    assert diag["completed_scope"] == "KNIGHT_FIT1__FIVE_FROZEN_STATES__EXTERIOR_DECODER_SHELL_COMPATIBILITY_ONLY"
    assert diag["preregistration"] == "canonical/IRIS_V5_EXTERIOR_DECODER_SHELL_COMPAT_PREREG_20260924.json"
    assert diag["closure"] == "canonical/IRIS_V5_EXTERIOR_DECODER_SHELL_COMPAT_CLOSURE_20260924.json"
    assert diag["optimizer_steps_executed"] == 0
    assert diag["parameter_updates_executed"] == 0
    assert diag["stage13_qualification_executed"] is False
    requal = ontology["requalification_authorization"]
    assert requal["authorized"] is False
    assert requal["completed_scope"] == "KNIGHT_FIT1__FIVE_FROZEN_STATES__CORRECTED_EXACT_SHELL_CONTAINMENT__FROZEN_STAGE13_P999"
    assert requal["closure"] == "canonical/IRIS_V5_CORRECTED_DECODER_STAGE13_REQUALIFICATION_CLOSURE_20260924.json"
    assert requal["optimizer_steps_executed"] == 0
    assert requal["parameter_updates_executed"] == 0
    morph = ontology["morphology_diagnostic_authorization"]
    assert morph["authorized"] is False
    assert morph["closure"] == "canonical/IRIS_V5_STAGE13_FROZEN_MASK_MORPHOLOGY_CLOSURE_20260924.json"
    assert morph["optimizer_steps_executed"] == 0
    assert morph["parameter_updates_executed"] == 0
    owner = ontology["zero_surface_owner_diagnostic_authorization"]
    assert owner["authorized"] is False
    assert owner["closure"] == "canonical/IRIS_V5_ZERO_SURFACE_OWNER_DIAGNOSTIC_CLOSURE_20260924.json"
    assert owner["optimizer_steps_executed"] == 0
    assert owner["parameter_updates_executed"] == 0
    longh = ontology["long_horizon_training_authorization"]
    assert longh["authorized"] is False
    assert longh["closure"] == "canonical/IRIS_V5_TP64_LONG_HORIZON_LOW_LR_CLOSURE_20260924.json"
    assert longh["run_id"] == "20260924T115953Z"
    bc = ontology["bc_endpoint_training_authorization"]
    assert bc["authorized"] is False
    assert bc["preregistration"] == "canonical/IRIS_V5_TP64_BC_ENDPOINT_CLOSURE_PREREG_20260924.json"
    diag2 = ontology["field_failure_diagnostic_authorization"]
    assert diag2["authorized"] is False
    assert diag2["closure"] == "canonical/IRIS_V5_FIELD_FAILURE_DIAGNOSTIC_CLOSURE_20260924.json"
    assert diag2["run_id"] == "20260924T143047Z"
    assert diag2["optimizer_steps_executed"] == 0
    assert diag2["parameter_updates_executed"] == 0
    bridge = ontology["h1_h3_bridge_diagnostic_authorization"]
    assert bridge["authorized"] is False
    assert bridge["status"] == "REQUIRES_SEPARATE_PREREGISTRATION"
    assert bridge["optimizer_steps_permitted"] == 0
    assert bridge["parameter_updates_permitted"] == 0
