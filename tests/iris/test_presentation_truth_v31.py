from __future__ import annotations

import numpy as np
import torch

from models.iris.truth.presentation_response_v1 import (
    build_measured_compositing_arrays_v1,
    build_surface_response_arrays_v1,
    control_balanced_action_weights_v1,
)
from models.iris.v31.lofo_trainer_v1 import presentation_partition_v1
from models.iris.v31.presentation_objective_v1 import (
    PAIR_CHANNELS_V31_PRESENTATION,
    appearance_rgba_loss_v1,
    build_presentation_targets_v1,
    heteroscedastic_mechanical_loss_v1,
    pair_probe_loss_v1,
)


def _mesh():
    vertices = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [4, 5, 6]], dtype=np.int64)
    return vertices, faces


def _actions():
    return [
        {"control_index_privileged": 0, "amplitude": 0.1, "sign": -1},
        {"control_index_privileged": 0, "amplitude": 0.1, "sign": 1},
        {"control_index_privileged": 1, "amplitude": 0.1, "sign": -1},
        {"control_index_privileged": 1, "amplitude": 0.1, "sign": 1},
    ]


def _isolated(vertices):
    out = np.repeat(vertices[None, ...], 4, axis=0)
    out[0, :3, 0] -= 0.1
    out[1, :3, 0] += 0.1
    out[2, 5, 0] -= 0.2
    out[2, 6, 0] -= 0.2
    out[3, 5, 0] += 0.2
    out[3, 6, 0] += 0.2
    return out


def _packed():
    return {
        "schema": "RealSaS.SourceRayAppearanceMeasurementCandidate.v1",
        "rays": [
            {
                "ray_index": 0,
                "view_index": 0,
                "pixel_x": 10,
                "pixel_y": 10,
                "canonical_hits": [
                    {"face_index": 0, "barycentric": [1 / 3, 1 / 3, 1 / 3], "depth": 4.0},
                    {"face_index": 1, "barycentric": [1 / 3, 1 / 3, 1 / 3], "depth": 5.0},
                ],
                "reconstructed_full_small_rgba": [0.5, 0.25, 0.0, 1.0],
                "interventions": [
                    {"removal_small_rgba": [0.0, 0.5, 0.0, 1.0]},
                    {"removal_small_rgba": [0.5, 0.0, 0.0, 0.5]},
                ],
                "measurement": {
                    "status": "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION",
                    "valid": True,
                    "effective_alpha": [0.5, 1.0],
                    "straight_linear_rgb": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    "incoming_transmittance": [1.0, 0.5],
                    "contribution": [0.5, 0.5],
                },
            }
        ],
    }


def _normalization(scale=1.0):
    return {
        "schema": "RealSaS.NormalizationDomainIR.v1",
        "center_xyz": [0.0, 0.0, 0.5 * scale],
        "half_extent": 2.0 * scale,
    }


def test_control_balanced_measure_is_explicit_and_normalized():
    weights = control_balanced_action_weights_v1(_actions())
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.25, 0.25])
    assert np.isclose(weights.sum(), 1.0)


def test_surface_response_detects_nonrigid_without_semantic_class():
    vertices, faces = _mesh()
    arrays = build_surface_response_arrays_v1(vertices, faces, _isolated(vertices), _actions())
    assert arrays["surface_face_valid"].tolist() == [True, True]
    assert arrays["surface_nonrigid_response"][1] > arrays["surface_nonrigid_response"][0]
    assert np.isclose(arrays["action_weight"].sum(), 1.0)


def test_measured_compositing_produces_complementary_negative():
    arrays = build_measured_compositing_arrays_v1(_packed())
    assert arrays["composite_pair_face_a"].tolist() == [0, 1]
    assert arrays["composite_pair_face_b"].tolist() == [1, 0]
    np.testing.assert_allclose(arrays["composite_pair_front_probability"], [1.0, 0.0])
    np.testing.assert_allclose(arrays["composite_pair_occlusion_probability"], [0.5, 0.0])
    assert arrays["composite_pair_student_observable"].tolist() == [True, True]


def test_v31_projection_learns_attenuation_not_semantic_owner_or_order():
    vertices, faces = _mesh()
    response = build_surface_response_arrays_v1(vertices, faces, _isolated(vertices), _actions())
    targets = build_presentation_targets_v1(
        rest_vertices=vertices,
        faces=faces,
        normalization=_normalization(),
        presentation_arrays=response,
        packed_measurement=_packed(),
    )
    assert targets.mechanical_target.shape == (2, 2)
    assert targets.appearance_rgba.shape == (2, 4)
    # Two translucent contributors make each standalone colour underdetermined
    # from the observed composite; both teacher-factorized RGBA rows stay masked.
    assert targets.appearance_valid.tolist() == [False, False]
    assert targets.pair_target.shape == (2, 1)
    assert targets.pair_valid.tolist() == [True, True]
    np.testing.assert_allclose(targets.pair_target[:, 0], [0.5, 0.0])
    assert PAIR_CHANNELS_V31_PRESENTATION == ("occlusion_attenuation_logit",)


def test_v31_mechanical_targets_are_invariant_to_uniform_source_scale():
    vertices, faces = _mesh()
    response = build_surface_response_arrays_v1(vertices, faces, _isolated(vertices), _actions())
    base = build_presentation_targets_v1(
        rest_vertices=vertices,
        faces=faces,
        normalization=_normalization(1.0),
        presentation_arrays=response,
        packed_measurement=_packed(),
    )

    scale = 3.0
    scaled_vertices = vertices * scale
    scaled_response = build_surface_response_arrays_v1(
        scaled_vertices,
        faces,
        _isolated(vertices) * scale,
        _actions(),
    )
    scaled = build_presentation_targets_v1(
        rest_vertices=scaled_vertices,
        faces=faces,
        normalization=_normalization(scale),
        presentation_arrays=scaled_response,
        packed_measurement=_packed(),
    )
    np.testing.assert_allclose(scaled.mechanical_xyz, base.mechanical_xyz, atol=1e-6, rtol=0)
    np.testing.assert_allclose(scaled.mechanical_target, base.mechanical_target, atol=1e-6, rtol=0)


def test_v31_losses_are_finite_and_backpropagate_only_declared_quantities():
    raw_mech = torch.tensor([[[0.2, 0.0, 0.1, 0.0]]], requires_grad=True)
    target_mech = torch.tensor([[[0.3, 0.2]]])
    mech = heteroscedastic_mechanical_loss_v1(raw_mech, target_mech, torch.ones(1, 1, dtype=torch.bool))
    assert torch.isfinite(mech)
    mech.backward()
    assert raw_mech.grad is not None

    rgba = torch.tensor([[[0.3, 0.4, 0.5, 0.6]]], requires_grad=True)
    rgba_loss = appearance_rgba_loss_v1(rgba, torch.tensor([[[0.2, 0.4, 0.5, 0.7]]]), torch.ones(1, 1, dtype=torch.bool))
    rgba_loss.backward()
    assert rgba.grad is not None

    pair = torch.tensor([[[0.0]]], requires_grad=True)
    pair_loss = pair_probe_loss_v1(pair, torch.tensor([[[0.5]]]), torch.ones(1, 1, dtype=torch.bool))
    pair_loss.backward()
    assert pair.grad is not None


def test_presentation_partition_is_deterministic_and_nonempty_when_possible():
    a = presentation_partition_v1("fit8_snake_v1", "appearance", 20, 26102831, 0.8)
    b = presentation_partition_v1("fit8_snake_v1", "appearance", 20, 26102831, 0.8)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert len(a[0]) > 0 and len(a[1]) > 0
    assert not set(a[0].tolist()).intersection(a[1].tolist())
