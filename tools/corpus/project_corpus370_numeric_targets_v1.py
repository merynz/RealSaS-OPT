"""Loss-preserving numeric projection for sealed materialization inputs.

These functions project only the explicitly passed family. Component-local sign
arrays are intentionally ignored: no path here can produce a global SDF target.
The separate global-solid consumer requires an independently qualified payload.
"""
from __future__ import annotations

import numpy as np

from models.materialization_contract_v1 import require, validate_boolean_mask


def xyz(value, code):
    array = np.asarray(value)
    require(array.ndim == 2 and array.shape[1] == 3 and len(array) > 0
            and np.issubdtype(array.dtype, np.floating) and np.isfinite(array).all(), code)
    return array


def iris_surface_targets(distance, surface, *, normals_status):
    query = xyz(distance['query_xyz_canonical'], 'PROJECT_QUERY_XYZ')
    udf = np.asarray(distance['unsigned_distance'])
    require(udf.shape == (len(query),) and np.isfinite(udf).all() and np.all(udf >= 0), 'PROJECT_UDF')
    valid = validate_boolean_mask(np.asarray(distance['unsigned_distance_valid']), len(query), 'PROJECT_UDF_VALID')
    require(np.any(valid), 'PROJECT_UDF_VALID_EMPTY')
    # Compact out invalid distance rows. The family does not gain zero targets.
    query, udf = query[valid], udf[valid]
    hit = xyz(surface['surface_xyz_canonical'], 'PROJECT_SOURCE_SURFACE_XYZ')
    result = {'surface_query_xyz_canonical': np.concatenate((query, hit)),
              'unsigned_surface_distance': np.concatenate((udf, np.zeros(len(hit), dtype=udf.dtype)))}
    # Each appended hit is a proven point on the exact surface, so its distance
    # is mathematically zero. This is not a placeholder for absent supervision.
    require(normals_status in {'PASS', 'MASKED', 'FAIL', 'QUARANTINED'}, 'PROJECT_NORMALS_UNRESOLVED')
    if normals_status == 'PASS':
        normal = xyz(surface['normal_canonical'], 'PROJECT_NORMAL_XYZ')
        require(normal.shape == hit.shape, 'PROJECT_NORMAL_CARDINALITY')
        normal_valid = validate_boolean_mask(np.asarray(surface['normal_valid']), len(hit), 'PROJECT_NORMAL_VALID')
        require(np.any(normal_valid) and np.all(np.abs(np.linalg.norm(normal[normal_valid], axis=1) - 1) <= 5e-3),
                'PROJECT_NORMAL_NOT_UNIT')
        result['surface_normal_xyz'] = np.concatenate((np.zeros_like(query), normal))
        result['surface_normal_valid'] = np.concatenate((np.zeros(len(query), dtype=bool), normal_valid))
    return result


def tessa_teacher_targets(teacher, snapshot):
    vertices = xyz(teacher['vertices_source_world'], 'PROJECT_TEACHER_VERTICES')
    matrix = np.asarray(snapshot['source_world_to_canonical'], dtype=np.float64)
    inverse = np.asarray(snapshot['canonical_to_source_world'], dtype=np.float64)
    require(matrix.shape == inverse.shape == (4, 4) and np.isfinite(matrix).all() and np.isfinite(inverse).all()
            and np.allclose(matrix @ inverse, np.eye(4), atol=1e-9, rtol=1e-9)
            and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12), 'PROJECT_CANONICAL_MATRIX')
    linear = matrix[:3, :3]
    scale2 = np.trace(linear.T @ linear) / 3
    require(scale2 > 0 and np.linalg.det(linear) > 0
            and np.allclose(linear.T @ linear, scale2 * np.eye(3), atol=max(scale2, 1e-12) * 1e-9, rtol=1e-9),
            'PROJECT_CANONICAL_POSITIVE_UNIFORM_SIMILARITY')
    faces = np.asarray(teacher['faces'])
    triangle = np.asarray(teacher['source_triangle_id'])
    bary = np.asarray(teacher['source_barycentric'])
    require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
            and np.issubdtype(faces.dtype, np.integer) and np.all(faces >= 0) and np.all(faces < len(vertices)),
            'PROJECT_TEACHER_FACES')
    require(triangle.shape == (len(vertices),) and np.issubdtype(triangle.dtype, np.integer) and np.all(triangle >= 0),
            'PROJECT_TEACHER_SOURCE_TRIANGLE')
    require(bary.shape == (len(vertices), 3) and np.isfinite(bary).all() and np.all(bary >= -1e-6)
            and np.all(np.abs(bary.sum(axis=1) - 1) <= 1e-5), 'PROJECT_TEACHER_BARYCENTRIC')
    return {'vertices_canonical': (vertices @ linear.T + matrix[:3, 3]).astype(np.float32),
            'faces': faces.copy(), 'source_triangle_id': triangle.copy(), 'source_barycentric': bary.copy()}
