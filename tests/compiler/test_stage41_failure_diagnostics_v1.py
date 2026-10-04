import numpy as np
import pytest

from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import (
    DynamicMotionProofFailure,
    _frame_metrics,
)
from compiler.realsas_compiler_services.proof.failure_signatures import (
    derive_failure_signatures,
)


class Vertex:
    def __init__(self, vid):
        self.canonical_mesh_vertex_id = vid


class Mesh:
    vertices = (Vertex("a"), Vertex("b"), Vertex("c"))
    faces = (("a", "b", "c"),)


class Policy:
    g3_min_dynamic_area_ratio = 0.05
    g3_max_dynamic_area_ratio = 20.0
    g3_max_dynamic_condition_number = 2.0


REST = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def test_stage41_condition_failure_carries_measured_context():
    posed = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(DynamicMotionProofFailure) as exc_info:
        _frame_metrics(
            Mesh(),
            REST,
            posed,
            Policy(),
            clip_id="demo_run_v1",
            time_seconds=0.25,
        )

    exc = exc_info.value
    assert exc.failure_code == "MOTION_V2_DYNAMIC_TRIANGLE_CONDITION_FAIL"
    assert exc.measurements["clip_id"] == "demo_run_v1"
    assert exc.measurements["time_seconds"] == 0.25
    assert exc.measurements["max_triangle_condition_number"] > 2.0
    assert exc.measurements["min_triangle_area_ratio"] >= 0.05
    assert exc.measurements["failure_localization"]["max_condition_face_index"] == 0


def test_stage41_area_failure_precedes_condition_and_carries_thresholds():
    posed = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.001, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(DynamicMotionProofFailure) as exc_info:
        _frame_metrics(
            Mesh(),
            REST,
            posed,
            Policy(),
            clip_id="demo_slash_v1",
            time_seconds=0.5,
        )

    exc = exc_info.value
    assert exc.failure_code == "MOTION_V2_DYNAMIC_TRIANGLE_AREA_RATIO_FAIL"
    assert exc.measurements["min_triangle_area_ratio"] < 0.05
    assert (
        exc.measurements["frozen_policy_thresholds"][
            "g3_min_dynamic_area_ratio"
        ]
        == 0.05
    )


def test_exact_motion_signature_localizes_condition_without_spurious_missing_clip():
    signatures = derive_failure_signatures(
        "MOTION",
        {
            "clip_id": "demo_run_v1",
            "time_seconds": 0.25,
            "min_triangle_area_ratio": 0.2,
            "max_triangle_area_ratio": 1.2,
            "max_triangle_condition_number": 9.0,
            "frozen_policy_thresholds": {
                "g3_min_dynamic_area_ratio": 0.05,
                "g3_max_dynamic_area_ratio": 20.0,
                "g3_max_dynamic_condition_number": 4.0,
            },
            "failure_localization": {"max_condition_face_index": 7},
        },
        status="FAIL",
    )
    families = [row["failure_family"] for row in signatures]
    assert families == ["triangle_condition_exceeded"]
    assert signatures[0]["localization"]["max_condition_face_index"] == 7
    assert signatures[0]["metadata"]["causal_owner_not_inferred_from_failure"] is True
