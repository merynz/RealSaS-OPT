from __future__ import annotations

import numpy as np
import torch

from models.iris.truth.presentation_response_v1 import (
    build_measured_compositing_arrays_v1,
    build_surface_response_arrays_v1,
    control_balanced_action_weights_v1,
)
from models.iris.vnext.presentation_truth_v31_adapter_v1 import build_v31_training_targets_v1
from models.iris.vnext.training_v3 import mechanical_demand_loss_v3


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
    # First face remains rigidly translated under control 0.
    out[0, :3, 0] -= 0.1
    out[1, :3, 0] += 0.1
    # Second face stretches anisotropically under control 1.
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


def test_v31_adapter_abstains_semantic_and_unmeasured_mechanical_heads():
    vertices, faces = _mesh()
    response = build_surface_response_arrays_v1(vertices, faces, _isolated(vertices), _actions())
    targets = build_v31_training_targets_v1(
        rest_vertices=vertices,
        faces=faces,
        normalization={
            "schema": "RealSaS.NormalizationDomainIR.v1",
            "center_xyz": [0.0, 0.0, 0.5],
            "half_extent": 2.0,
        },
        presentation_arrays=response,
        packed_measurement=_packed(),
    )
    assert targets["front_probability_valid"].tolist() == [True, True]
    assert targets["same_drawable_affinity_valid"].tolist() == [False, False]
    assert targets["rigidity_valid"].tolist() == [False, False]
    assert targets["contact_valid"].tolist() == [False, False]
    np.testing.assert_allclose(targets["ray_hit_visibility_target"], [0.5, 0.5])


def test_mechanical_loss_does_not_train_undefined_tangent_or_rigidity():
    predicted = {
        "stretch_major": torch.tensor([[1.2, 1.1]], requires_grad=True),
        "stretch_minor": torch.tensor([[0.9, 1.0]], requires_grad=True),
        "tangent_direction_2theta": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True),
        "bend_demand": torch.tensor([[0.2, 0.2]], requires_grad=True),
        "rigidity_probability": torch.tensor([[0.5, 0.5]], requires_grad=True),
        "contact_persistence_probability": torch.tensor([[0.5, 0.5]], requires_grad=True),
    }
    truth = {
        "stretch_major": torch.tensor([[1.0, 1.0]]),
        "stretch_minor": torch.tensor([[1.0, 1.0]]),
        "tangent_direction_2theta": torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
        "bend_demand": torch.zeros(1, 2),
        "rigidity_probability": torch.zeros(1, 2),
        "contact_persistence_probability": torch.zeros(1, 2),
    }
    no = torch.zeros(1, 2, dtype=torch.bool)
    yes = torch.ones(1, 2, dtype=torch.bool)
    loss = mechanical_demand_loss_v3(
        predicted,
        truth,
        mechanical_valid_mask=yes,
        stretch_valid_mask=yes,
        tangent_valid_mask=no,
        bend_valid_mask=no,
        rigidity_valid_mask=no,
        contact_valid_mask=no,
    )
    assert loss["stretch_supervision_present"].item() == 1.0
    assert loss["tangent_supervision_present"].item() == 0.0
    assert loss["rigidity_supervision_present"].item() == 0.0
    loss["total"].backward()
    assert predicted["stretch_major"].grad is not None
    assert predicted["tangent_direction_2theta"].grad is None
    assert predicted["rigidity_probability"].grad is None
