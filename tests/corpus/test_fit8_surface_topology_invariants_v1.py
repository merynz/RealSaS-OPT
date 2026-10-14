from tools.corpus.fit8_surface_topology_invariants_v1 import compute_surface_topology_invariants_v1


def test_single_triangle_open_disk():
    report = compute_surface_topology_invariants_v1([(0, 1, 2)])
    component = report.components[0]
    assert (component.vertex_count, component.edge_count, component.face_count) == (3, 3, 1)
    assert component.euler_characteristic == 1
    assert component.boundary_edge_count == 3
    assert component.boundary_loop_count == 1
    assert component.orientable is True
    assert component.genus == 0


def test_tetrahedron_closed_sphere_topology():
    faces = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    report = compute_surface_topology_invariants_v1(faces)
    component = report.components[0]
    assert (component.vertex_count, component.edge_count, component.face_count) == (4, 6, 4)
    assert component.euler_characteristic == 2
    assert component.boundary_edge_count == 0
    assert component.boundary_loop_count == 0
    assert component.orientable is True
    assert component.genus == 0


def test_two_components_remain_separate():
    report = compute_surface_topology_invariants_v1([(0, 1, 2), (3, 4, 5)])
    assert report.component_count == 2


def test_nonmanifold_edge_disables_orientability_and_genus_authority():
    report = compute_surface_topology_invariants_v1(
        [(0, 1, 2), (1, 0, 3), (0, 1, 4)]
    )
    component = report.components[0]
    assert component.nonmanifold_edge_count == 1
    assert not component.orientability_valid
    assert component.orientable is None
    assert not component.genus_valid
    assert component.genus is None


def test_inconsistent_input_winding_can_still_be_orientable_after_face_flip():
    report = compute_surface_topology_invariants_v1([(0, 1, 2), (0, 1, 3)])
    component = report.components[0]
    assert component.orientable is True
    assert component.genus == 0
