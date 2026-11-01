import numpy as np

from tools.corpus.revalidate_existing_truth139_v1 import oriented_triangle_multiset


def test_triangle_geometry_survives_vertex_and_face_readdressing():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 1, 2], [0, 3, 1]])
    permutation = np.array([3, 1, 0, 2])
    reverse = np.argsort(permutation)
    changed_faces = np.roll(reverse[faces[::-1]], 1, axis=1)
    assert oriented_triangle_multiset(vertices, faces) == oriented_triangle_multiset(vertices[permutation], changed_faces)


def test_face_winding_and_duplicate_multiplicity_are_preserved():
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    f = np.array([[0, 1, 2]])
    original = oriented_triangle_multiset(v, f)
    assert original != oriented_triangle_multiset(v, f[:, ::-1])
    assert original != oriented_triangle_multiset(v, np.repeat(f, 2, axis=0))
    changed = v.copy()
    changed[0, 0] = 1e-12
    assert original != oriented_triangle_multiset(changed, f)
