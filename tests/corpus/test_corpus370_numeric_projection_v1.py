import numpy as np
import pytest

from tools.corpus.project_corpus370_numeric_targets_v1 import iris_surface_targets, tessa_teacher_targets


def test_component_sign_is_never_promoted_and_invalid_udf_rows_are_omitted():
    distance = {'query_xyz_canonical': np.array([[0., 0., 1.], [2., 2., 2.]]),
                'unsigned_distance': np.array([1., 9.]), 'unsigned_distance_valid': np.array([True, False]),
                'signed_distance_value': np.array([-900.]), 'occupancy_value': np.array([True])}
    surface = {'surface_xyz_canonical': np.array([[0., 0., 0.]]), 'normal_canonical': np.array([[0., 0., 1.]]),
               'normal_valid': np.array([True])}
    result = iris_surface_targets(distance, surface, normals_status='PASS')
    assert set(result) == {'surface_query_xyz_canonical', 'unsigned_surface_distance', 'surface_normal_xyz', 'surface_normal_valid'}
    np.testing.assert_array_equal(result['unsigned_surface_distance'], [1., 0.])
    np.testing.assert_array_equal(result['surface_normal_valid'], [False, True])
    assert result['surface_query_xyz_canonical'].shape == (2, 3)
    without = iris_surface_targets(distance, surface, normals_status='FAIL')
    assert not any('normal' in key for key in without)
    with pytest.raises(ValueError, match='NORMALS_UNRESOLVED'):
        iris_surface_targets(distance, surface, normals_status='UNKNOWN')


def teacher_fixture():
    return {'vertices_source_world': np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
            'faces': np.array([[0, 1, 2]], dtype=np.int32), 'source_triangle_id': np.array([4, 4, 4], dtype=np.int32),
            'source_barycentric': np.eye(3)}


def test_tessa_exact_canonical_transform_preserves_source_binding():
    teacher = teacher_fixture()
    matrix = np.array([[0., -2., 0., 5.], [2., 0., 0., 6.], [0., 0., 2., 7.], [0., 0., 0., 1.]])
    result = tessa_teacher_targets(teacher, {'source_world_to_canonical': matrix, 'canonical_to_source_world': np.linalg.inv(matrix)})
    np.testing.assert_array_equal(result['vertices_canonical'], [[5, 6, 7], [5, 8, 7], [3, 6, 7]])
    np.testing.assert_array_equal(result['source_triangle_id'], teacher['source_triangle_id'])
    np.testing.assert_array_equal(result['source_barycentric'], teacher['source_barycentric'])
    assert not any(k in result for k in ('dense3d_z', 'predicted_geometry_g'))


@pytest.mark.parametrize('linear', [np.diag([-1., 1., 1.]), np.diag([1., 2., 1.])])
def test_tessa_rejects_orientation_reversal_and_anisotropic_normalization(linear):
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    with pytest.raises(ValueError, match='POSITIVE_UNIFORM_SIMILARITY'):
        tessa_teacher_targets(teacher_fixture(), {'source_world_to_canonical': matrix, 'canonical_to_source_world': np.linalg.inv(matrix)})
