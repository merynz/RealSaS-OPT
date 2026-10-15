import numpy as np

from tools.corpus.fit8_static_ray_census_v1 import (
    StaticRayCensusPolicyV1,
    build_static_ray_census_v1,
    component_surface_classes_v1,
    face_edge_components_v1,
    solve_canonical_orbit_frame_v1,
)


def _crossing_sheets():
    # Two open rectangular sheets spanning X/Z. Their Y depth slopes in opposite
    # directions, so canonical view S sees A before B on one image side and B
    # before A on the other. Each sheet is two triangles and one edge component.
    x0, x1 = -0.8, 0.8
    z0, z1 = -0.6, 0.6
    slope = 0.35
    vertices = np.asarray(
        [
            [x0, slope * x0, z0], [x1, slope * x1, z0],
            [x1, slope * x1, z1], [x0, slope * x0, z1],
            [x0, -slope * x0, z0], [x1, -slope * x1, z0],
            [x1, -slope * x1, z1], [x0, -slope * x0, z1],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
        dtype=np.int64,
    )
    component = np.asarray([0, 0, 1, 1], dtype=np.int32)
    material = np.asarray([3, 3, 7, 7], dtype=np.int32)
    owner = np.asarray([11, 11, 19, 19], dtype=np.int32)
    return vertices, faces, component, material, owner


def test_component_and_surface_class_are_edge_connected_and_open():
    vertices, faces, expected, *_ = _crossing_sheets()
    del vertices
    actual = face_edge_components_v1(faces)
    assert np.array_equal(actual, expected)
    classes = component_surface_classes_v1(faces, actual)
    assert classes == {0: "OPEN", 1: "OPEN"}


def test_frame_matches_single_global_eight_view_contract():
    vertices, *_ = _crossing_sheets()
    frame = solve_canonical_orbit_frame_v1(vertices, frame_scale=1.12)
    frame.validate()
    assert np.allclose(frame.center_xyz, (0.0, 0.0, 0.0))
    assert frame.half_extent > 0.8
    assert len(frame.sha256()) == 64


def test_same_view_mixed_depth_order_is_preserved_in_all_hit_evidence():
    vertices, faces, component, material, owner = _crossing_sheets()
    summary, arrays = build_static_ray_census_v1(
        subject_key="UNIT_CROSSING_SHEETS",
        vertices=vertices,
        faces=faces,
        face_component_id=component,
        face_material_id=material,
        face_owner_id=owner,
        policy=StaticRayCensusPolicyV1(
            resolution=96,
            frame_scale=1.12,
            provenance="synthetic crossing-sheet guard",
        ),
    )
    # View 0 = S, forward +Y. Both ordered relations must occur in one view.
    mixed = np.asarray(arrays["v0_mixed_component_pair_counts"], dtype=np.int64)
    assert mixed.shape == (1, 4)
    assert tuple(mixed[0, :2]) == (0, 1)
    assert mixed[0, 2] > 0
    assert mixed[0, 3] > 0
    assert summary["pair_evidence"]["0"]["mixed_pair_count"] == 1
    assert summary["material_region_count"] == 2
    assert summary["owner_region_count"] == 2
    assert summary["surface_class_visibility"]["visible_fraction_by_surface_class"]["OPEN"] == 1.0

    offsets = arrays["v0_hit_offsets"]
    assert offsets.shape == (96 * 96 + 1,)
    assert offsets[-1] == len(arrays["v0_hit_triangle_id"])
    assert np.count_nonzero(arrays["v0_front_component_id"] >= 0) > 0


def test_summary_has_exact_eight_canonical_views():
    vertices, faces, component, material, owner = _crossing_sheets()
    summary, _ = build_static_ray_census_v1(
        subject_key="UNIT_EIGHT_VIEW",
        vertices=vertices,
        faces=faces,
        face_component_id=component,
        face_material_id=material,
        face_owner_id=owner,
        policy=StaticRayCensusPolicyV1(resolution=64, provenance="synthetic eight-view guard"),
    )
    assert [row["view_index"] for row in summary["static_views"]] == list(range(8))
    assert summary["camera_contract"]["yaw_deg"] == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
