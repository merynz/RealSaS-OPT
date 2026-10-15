import numpy as np
import pytest

from tools.corpus.fit8_dynamic_fixed_frame_presentation_v1 import (
    compare_dynamic_presentation_to_rest_v1,
    evaluate_fixed_frame_presentation_v1,
    solve_subject_dynamic_frame_v1,
)
from tools.corpus.fit8_static_ray_census_v1 import StaticRayCensusPolicyV1, solve_canonical_orbit_frame_v1


def tetra():
    vertices = np.asarray(
        [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int64,
    )
    return vertices, faces


def test_dynamic_frame_is_solved_once_over_rest_and_all_samples():
    rest, _ = tetra()
    posed = rest + np.asarray([10.0, 0.0, 0.0])
    frame = solve_subject_dynamic_frame_v1(
        rest_vertices=rest,
        posed_vertices=[posed],
        frame_scale=1.2,
    )
    assert frame.center_xyz[0] == pytest.approx(5.0)
    rest_only = solve_canonical_orbit_frame_v1(rest, frame_scale=1.2)
    pose_only = solve_canonical_orbit_frame_v1(posed, frame_scale=1.2)
    assert frame.sha256() != rest_only.sha256()
    assert frame.sha256() != pose_only.sha256()


def test_fixed_frame_snapshots_share_frame_and_rigid_motion_stays_trivial():
    rest, faces = tetra()
    posed = rest @ np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    ).T + np.asarray([3.0, 0.0, 0.0])
    frame = solve_subject_dynamic_frame_v1(
        rest_vertices=rest,
        posed_vertices=[posed],
        frame_scale=1.2,
    )
    policy = StaticRayCensusPolicyV1(
        resolution=32,
        frame_scale=1.2,
        provenance="FIXED_FRAME_SYNTHETIC",
    )
    a = evaluate_fixed_frame_presentation_v1(
        vertices=rest,
        faces=faces,
        frame=frame,
        policy=policy,
    )
    b = evaluate_fixed_frame_presentation_v1(
        vertices=posed,
        faces=faces,
        frame=frame,
        policy=policy,
    )
    assert a.frame_sha256 == b.frame_sha256 == frame.sha256()
    assert a.component_count == b.component_count == 1
    assert a.mixed_pair_count_sum == b.mixed_pair_count_sum == 0
    delta = compare_dynamic_presentation_to_rest_v1(a, b)
    assert delta.new_directed_edges_vs_rest == 0
    assert delta.reversed_directed_edges_vs_rest == 0
    assert delta.strict_pair_flips_vs_rest == 0


def test_delta_rejects_frame_drift():
    rest, faces = tetra()
    policy = StaticRayCensusPolicyV1(
        resolution=32,
        frame_scale=1.2,
        provenance="FIXED_FRAME_DRIFT_GUARD",
    )
    a = evaluate_fixed_frame_presentation_v1(
        vertices=rest,
        faces=faces,
        frame=solve_canonical_orbit_frame_v1(rest, frame_scale=1.2),
        policy=policy,
    )
    moved = rest + np.asarray([4.0, 0.0, 0.0])
    b = evaluate_fixed_frame_presentation_v1(
        vertices=moved,
        faces=faces,
        frame=solve_canonical_orbit_frame_v1(moved, frame_scale=1.2),
        policy=policy,
    )
    with pytest.raises(ValueError, match="CONTEXT_DRIFT"):
        compare_dynamic_presentation_to_rest_v1(a, b)
