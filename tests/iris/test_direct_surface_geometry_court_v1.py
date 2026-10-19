from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tools.iris_vnext.audit_fit8_direct_surface_geometry_v1 import (
    SURFACE_SAMPLE_COUNT,
    deterministic_surface_samples,
    measure_surface_against_shard,
    topology_diagnostics,
)


def _tetra():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def test_closed_tetra_topology_diagnostics_are_independent_of_loss():
    vertices, faces = _tetra()
    metrics = topology_diagnostics(vertices, faces)
    assert metrics["component_count"] == 1
    assert metrics["boundary_edge_count"] == 0
    assert metrics["nonmanifold_edge_count"] == 0
    assert metrics["degenerate_face_count"] == 0
    assert metrics["face_count"] == 4


def test_surface_sampling_is_deterministic_for_same_seed():
    vertices, faces = _tetra()
    a_points, a_normals = deterministic_surface_samples(vertices, faces, count=1024, seed=7)
    b_points, b_normals = deterministic_surface_samples(vertices, faces, count=1024, seed=7)
    np.testing.assert_array_equal(a_points, b_points)
    np.testing.assert_array_equal(a_normals, b_normals)
    assert np.isfinite(a_points).all()
    assert np.isfinite(a_normals).all()


def test_measurement_court_reports_available_authority_without_inventing_mask_depth(tmp_path: Path):
    vertices, faces = _tetra()
    stage12 = tmp_path / "candidate.npz"
    np.savez_compressed(
        stage12,
        vertices_normalized=vertices,
        faces=faces,
        implicit_normals=np.zeros_like(vertices),
    )

    # Source-defensible teacher fields available in the V4 shard contract.
    shard = tmp_path / "subject.pt"
    teacher_surface = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1 / 3, 1 / 3, 1 / 3],
        ],
        dtype=torch.float32,
    )
    mechanical_points = teacher_surface[:4].clone()
    mechanical_normals = torch.nn.functional.normalize(
        torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    torch.save(
        {
            "subject_id": "Snake",
            "teacher_surface_points_iris": teacher_surface,
            "mechanical_points": mechanical_points,
            "mechanical_normals": mechanical_normals,
        },
        shard,
    )

    result = measure_surface_against_shard(
        subject_id="Snake",
        stage12_path=stage12,
        prepared_shard_path=shard,
        teacher_component_count=1,
        seed=5,
    )
    assert result["measurement_only"] is True
    assert result["scientific_pass_claimed"] is False
    assert result["candidate_topology"]["component_count"] == 1
    assert result["component_count_error"] == 0
    assert result["surface_sample_count"] == SURFACE_SAMPLE_COUNT
    assert result["surface_distance"]["candidate_to_teacher"]["p95"] >= 0.0
    assert result["normal_alignment"]["absolute_dot_mean"] >= 0.0
    assert result["authority"]["mask_depth_truth"] == "NOT_PRESENT_IN_V4_SHARD__NOT_MEASURED_BY_THIS_COURT"
