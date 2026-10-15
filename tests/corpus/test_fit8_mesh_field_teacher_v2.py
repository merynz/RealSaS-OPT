import numpy as np

from compiler.realsas_compiler_core.rigid_attachment_surface_anchor_v1 import _closest_point_barycentric
from tools.corpus.fit8_mesh_field_teacher_v1 import (
    MeshFieldTeacherPolicyV1,
    qualify_single_component_global_sign_v1,
)
from tools.corpus.fit8_mesh_field_teacher_v2 import build_mesh_field_teacher_v2
from tools.corpus.fit8_surface_topology_invariants_v1 import compute_surface_topology_invariants_v1


VERTICES = np.asarray([
    [-1.0, -1.0, -1.0],
    [ 1.0, -1.0, -1.0],
    [ 1.0,  1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0,  1.0,  1.0],
    [-1.0,  1.0,  1.0],
], dtype=np.float64)

# Two triangles per cube side. Orientation is not used as sign authority by the
# multi-ray parity court, but the carrier is a closed orientable manifold.
FACES = np.asarray([
    [0, 2, 1], [0, 3, 2],
    [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4],
    [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6],
    [3, 0, 4], [3, 4, 7],
], dtype=np.int64)


def _policy():
    return MeshFieldTeacherPolicyV1(
        spatial_tolerance=1.0e-9,
        minimum_triangle_area2=1.0e-12,
        distance_tie_tolerance=1.0e-10,
        query_batch_size=16,
        provenance="synthetic_cube_guard",
    )


def _admission(faces=FACES, *, self_intersection_free=True, containment=True):
    topology = compute_surface_topology_invariants_v1(faces)
    return qualify_single_component_global_sign_v1(
        topology,
        self_intersection_free=self_intersection_free,
        containment_semantics_qualified=containment,
        provenance="synthetic_cube_guard",
    )


def _brute_distance(point):
    rows = []
    for face_id, face in enumerate(FACES):
        closest, bary = _closest_point_barycentric(point, VERTICES[face])
        rows.append((float(np.linalg.norm(closest - point)), face_id, bary))
    minimum = min(row[0] for row in rows)
    tied = [row for row in rows if abs(row[0] - minimum) <= 1.0e-10]
    canonical = min(tied, key=lambda row: row[1])
    return minimum, canonical[1], canonical[2], len(tied)


def test_closed_cube_inside_outside_and_surface_are_signed_correctly():
    queries = np.asarray([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.25, 0.10],
    ])
    result = build_mesh_field_teacher_v2(
        vertices=VERTICES,
        faces=FACES,
        query_xyz=queries,
        sign_admission=_admission(),
        policy=_policy(),
        provenance="synthetic_cube_guard",
    )
    assert np.all(result.orientation_valid)
    assert result.inside.tolist() == [True, False, False]
    assert np.allclose(result.unsigned_distance, [1.0, 1.0, 0.0], atol=1.0e-10)
    assert np.allclose(result.oriented_level_set, [-1.0, 1.0, 0.0], atol=1.0e-10)
    assert np.all(result.nonambiguous_vote_count >= 3)


def test_bvh_distance_and_surface_address_match_bruteforce_ericson_oracle():
    rng = np.random.default_rng(20261015)
    queries = rng.uniform(-2.5, 2.5, size=(32, 3))
    result = build_mesh_field_teacher_v2(
        vertices=VERTICES,
        faces=FACES,
        query_xyz=queries,
        sign_admission=_admission(),
        policy=_policy(),
        provenance="synthetic_cube_guard",
    )
    for index, point in enumerate(queries):
        distance, face_id, bary, tie_count = _brute_distance(point)
        assert abs(result.unsigned_distance[index] - distance) <= 1.0e-12
        assert int(result.nearest_triangle_id[index]) == face_id
        assert np.allclose(result.nearest_barycentric[index], bary, atol=1.0e-12)
        assert int(result.nearest_address_tie_count[index]) == tie_count


def test_shared_face_diagonal_preserves_nearest_address_ambiguity():
    query = np.asarray([[1.0, 0.0, 0.0]])
    result = build_mesh_field_teacher_v2(
        vertices=VERTICES,
        faces=FACES,
        query_xyz=query,
        sign_admission=_admission(),
        policy=_policy(),
        provenance="synthetic_cube_guard",
    )
    distance, face_id, bary, tie_count = _brute_distance(query[0])
    assert distance == 0.0
    assert tie_count >= 2
    assert int(result.nearest_address_tie_count[0]) == tie_count
    assert int(result.nearest_triangle_id[0]) == face_id
    assert np.allclose(result.nearest_barycentric[0], bary, atol=1.0e-12)


def test_open_surface_keeps_exact_udf_but_global_sign_abstains():
    open_faces = FACES[:-2]
    admission = _admission(open_faces)
    assert not admission.admitted
    assert "OPEN_BOUNDARY" in admission.reasons
    queries = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = build_mesh_field_teacher_v2(
        vertices=VERTICES,
        faces=open_faces,
        query_xyz=queries,
        sign_admission=admission,
        policy=_policy(),
        provenance="synthetic_open_cube_guard",
    )
    assert np.isfinite(result.unsigned_distance).all()
    assert not np.any(result.orientation_valid)
    assert np.isnan(result.oriented_level_set).all()


def test_sign_admission_requires_geometric_and_containment_preconditions():
    no_self_intersection_proof = _admission(self_intersection_free=False)
    no_containment = _admission(containment=False)
    assert not no_self_intersection_proof.admitted
    assert "SELF_INTERSECTION_UNQUALIFIED" in no_self_intersection_proof.reasons
    assert not no_containment.admitted
    assert "CONTAINMENT_SEMANTICS_UNQUALIFIED" in no_containment.reasons
