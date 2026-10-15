import numpy as np
import pytest

from tools.corpus.fit8_dynamic_mechanics_metrics_v1 import (
    measure_dynamic_mechanical_consequence_v1,
)


def fixture():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def test_uniform_scale_has_exact_similarity_metrics():
    rest, faces = fixture()
    posed = rest * 2.0
    result = measure_dynamic_mechanical_consequence_v1(
        rest_vertices=rest,
        posed_vertices=posed,
        faces=faces,
    )
    assert result.unique_edge_count == 5
    assert result.face_count == 2
    assert result.edge_stretch_p99 == pytest.approx(2.0)
    assert result.edge_stretch_max_diagnostic == pytest.approx(2.0)
    assert result.face_condition_p99 == pytest.approx(1.0)
    assert result.face_condition_max_diagnostic == pytest.approx(1.0)
    assert result.face_area_ratio_p01 == pytest.approx(4.0)
    assert result.face_area_ratio_p99 == pytest.approx(4.0)
    assert result.face_area_ratio_min == pytest.approx(4.0)
    assert result.face_area_ratio_max == pytest.approx(4.0)


def test_rigid_transform_preserves_all_mechanical_metrics():
    rest, faces = fixture()
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    posed = rest @ rotation.T + np.asarray([3.0, -2.0, 5.0])
    result = measure_dynamic_mechanical_consequence_v1(
        rest_vertices=rest,
        posed_vertices=posed,
        faces=faces,
    )
    assert result.edge_stretch_p99 == pytest.approx(1.0)
    assert result.face_condition_p99 == pytest.approx(1.0)
    assert result.face_area_ratio_p01 == pytest.approx(1.0)
    assert result.face_area_ratio_p99 == pytest.approx(1.0)


def test_degenerate_rest_geometry_is_fail_closed():
    rest = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="ZERO_REST_FACE_AREA"):
        measure_dynamic_mechanical_consequence_v1(
            rest_vertices=rest,
            posed_vertices=rest,
            faces=faces,
        )
