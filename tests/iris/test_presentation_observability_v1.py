from __future__ import annotations

import numpy as np

from models.iris.v31.presentation_objective_v1 import build_presentation_targets_v1


def test_occluded_teacher_hit_is_not_valid_student_appearance_target():
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
    presentation = {
        "surface_face_valid": np.asarray([True, True]),
        "surface_nonrigid_response": np.asarray([0.0, 0.0]),
        "surface_response_energy": np.asarray([0.0, 0.0]),
    }
    packed = {
        "schema": "RealSaS.SourceRayAppearanceMeasurementCandidate.v1",
        "rays": [
            {
                "view_index": 0,
                "canonical_hits": [
                    {"face_index": 0, "barycentric": [1 / 3, 1 / 3, 1 / 3]},
                    {"face_index": 1, "barycentric": [1 / 3, 1 / 3, 1 / 3]},
                ],
                "measurement": {
                    "status": "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION",
                    "valid": True,
                    "effective_alpha": [1.0, 1.0],
                    "straight_linear_rgb": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    "contribution": [1.0, 0.0],
                },
            }
        ],
    }
    targets = build_presentation_targets_v1(
        rest_vertices=vertices,
        faces=faces,
        normalization={
            "schema": "RealSaS.NormalizationDomainIR.v1",
            "center_xyz": [0.0, 0.0, 0.5],
            "half_extent": 2.0,
        },
        presentation_arrays=presentation,
        packed_measurement=packed,
        contribution_floor=1.0e-6,
    )
    assert targets.appearance_valid.tolist() == [True, False]
    assert targets.pair_valid.tolist() == [False, False]
