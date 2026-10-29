from __future__ import annotations

import numpy as np
import torch

from models.iris.v31.evidence_adapter_model_v2 import IRISEvidenceAdapterConfigV2
from models.iris.v31.geometry_supervision_v1 import (
    build_front_surface_normal_truth_v1,
    surface_normal_cosine_loss_v1,
    truth139_geometry_objective_v1,
)
from models.iris.v31.presentation_objective_v1 import (
    MECHANICAL_CHANNELS_V31_PRESENTATION,
    PAIR_CHANNELS_V31_PRESENTATION,
)
from models.iris.v31.truth139_model_v1 import IRISV31Truth139V1


def _normalization():
    return {
        "schema": "RealSaS.NormalizationDomainIR.v1",
        "center_xyz": [0.0, 0.0, 0.0],
        "half_extent": 2.0,
    }


def test_front_surface_normal_truth_uses_depth_sorted_first_hit():
    vertices = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    packed = {
        "schema": "RealSaS.SourceRayAppearanceMeasurementCandidate.v1",
        "rays": [
            {
                "view_index": 0,
                "complete": True,
                "has_depth_tie": False,
                "canonical_hits": [
                    {"face_index": 0, "barycentric": [0.25, 0.25, 0.5], "depth": 1.0},
                    {"face_index": 1, "barycentric": [0.25, 0.25, 0.5], "depth": 2.0},
                ],
                "measurement": {
                    "status": "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION",
                    "valid": True,
                },
            }
        ],
    }
    truth = build_front_surface_normal_truth_v1(
        rest_vertices=vertices,
        faces=faces,
        normalization=_normalization(),
        packed_measurement=packed,
    )
    assert truth.face_index.tolist() == [0]
    assert truth.view_index.tolist() == [0]
    assert np.allclose(truth.xyz[0], [0.0, 0.0, 0.0])
    assert np.allclose(truth.normal_world[0], [0.0, 0.0, 1.0])
    assert truth.valid.tolist() == [True]


def test_geometry_objective_penalizes_wrong_sign_and_wrong_normal():
    predicted_field = torch.tensor([[-0.01, 0.02]], dtype=torch.float32)
    target_fstar = torch.tensor([[0.01, 0.02]], dtype=torch.float32)
    predicted_normal = torch.tensor([[[0.0, 0.0, -1.0]]], dtype=torch.float32)
    target_normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    result = truth139_geometry_objective_v1(
        predicted_field=predicted_field,
        target_fstar=target_fstar,
        predicted_normal=predicted_normal,
        target_normal=target_normal,
        normal_valid=valid,
        field_clamp=0.25,
        huber_beta=0.01,
        near_zero_band=0.03,
        sign_weight=1.0,
        normal_weight=0.5,
    )
    assert result["sign_wrong_count"] == 1
    assert float(result["near_zero_wrong_sign"]) > 0.0
    assert torch.allclose(result["normal_cosine"], torch.tensor(2.0))
    assert float(result["total"]) > float(result["direct_fstar"])


def test_normal_cosine_is_zero_for_exact_alignment():
    normal = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    loss = surface_normal_cosine_loss_v1(normal, normal, valid)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_truth139_model_adds_normal_head_without_replacing_dense_carrier():
    config = IRISEvidenceAdapterConfigV2(
        carrier_resolution=4,
        carrier_channels=8,
        mechanical_channels=MECHANICAL_CHANNELS_V31_PRESENTATION,
        pair_channels=PAIR_CHANNELS_V31_PRESENTATION,
        query_width=32,
        query_layers=3,
        fourier_bands=4,
        private_bottleneck=8,
        evidence_adapter_bottleneck=8,
        lift_chunk=16,
        delta_bound=0.03,
    )
    model = IRISV31Truth139V1(config)
    carrier = torch.randn(1, 8, 4, 4, 4)
    xyz = torch.tensor([[[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    normal = model.query_surface_normal(carrier, xyz)
    assert normal.shape == (1, 2, 3)
    assert torch.allclose(torch.linalg.vector_norm(normal, dim=-1), torch.ones((1, 2)), atol=1e-5)
    assert model.encoder.__class__.__name__ == "DenseObservationEncoderV31"
