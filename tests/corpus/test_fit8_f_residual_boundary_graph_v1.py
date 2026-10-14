from tools.corpus.fit8_face_strand_boundary_incidence_v1 import (
    FaceBoundaryIncidencePolicyV1,
    qualify_face_strand_boundary_incidence_v1,
)
from tools.corpus.fit8_face_strand_teacher_v1 import (
    FaceSegmentV1,
    FaceStrandTeacherV1,
    FaceStrandV1,
)
from tools.corpus.fit8_f_residual_boundary_graph_v1 import (
    ABSTAIN,
    CANONICAL,
    NO_RESIDUAL,
    ResidualBoundaryGraphCanonicalizationPolicyV1,
    canonicalize_residual_boundary_graph_v1,
)
from tools.corpus.fit8_local_patch_teacher_v1 import (
    CrossingCanonicalizationPolicyV1,
    EdgeCrossingV1,
    ExtractionCellTeacherV1,
    ExtractionCellV1,
    LocalPatchLowerBoundV1,
)

CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)
CROSS_POLICY = CrossingCanonicalizationPolicyV1(1e-8, "SYNTHETIC_BOUNDARY_GRAPH_TEST_V1")
FACE_POLICY = FaceBoundaryIncidencePolicyV1(1e-8, "SYNTHETIC_BOUNDARY_GRAPH_TEST_V1")
GRAPH_POLICY = ResidualBoundaryGraphCanonicalizationPolicyV1(6, "SYNTHETIC_BOUNDARY_GRAPH_TEST_V1")
FACE_SCHEMA = "RealSaS.FIT8FaceStrandTeacher.v1"


def _crossing(edge, t, triangle):
    return EdgeCrossingV1(
        edge_index=edge,
        t=t,
        xyz=(-1.0 + 2.0 * t, -1.0, -1.0),
        source_triangle_ids=(triangle,),
        geometric_component_ids=(0,),
        owner_region_ids=(),
        material_region_ids=(),
        mechanical_owner_ids=(),
        normal_xyz=(1.0, 0.0, 0.0),
        normal_valid=True,
    )


def _cell(patches, crossings=()):
    patch_rows = tuple(
        LocalPatchLowerBoundV1(index, tuple(sorted(triangle_ids)), (index,))
        for index, triangle_ids in enumerate(patches)
    )
    triangle_ids = tuple(sorted({tid for row in patch_rows for tid in row.source_triangle_ids}))
    by_edge = {}
    for crossing in crossings:
        by_edge[crossing.edge_index] = by_edge.get(crossing.edge_index, 0) + 1
    result = ExtractionCellTeacherV1(
        schema_version="RealSaS.FIT8LocalPatchTeacher.v1",
        cell=CELL,
        active=bool(triangle_ids),
        intersecting_triangle_ids=triangle_ids,
        local_patch_lower_bound=patch_rows,
        edge_crossings=tuple(crossings),
        coplanar_edge_support=(),
        max_distinct_crossings_per_edge=max(by_edge.values(), default=0),
        crossing_canonicalization=CROSS_POLICY,
    )
    result.validate()
    return result


def _empty_face(index):
    return FaceStrandTeacherV1(FACE_SCHEMA, index, (), (), ())


def _strand(local_id, triangle_id, p0, p1):
    return FaceStrandV1(
        strand_local_id=local_id,
        source_triangle_ids=(triangle_id,),
        segments=(FaceSegmentV1(triangle_id, p0, p1),),
        kind="BOUNDARY_TO_BOUNDARY_ARC",
        boundary_endpoint_count=2,
        interior_endpoint_count=0,
        max_graph_degree=1,
    )


def _faces_and_qualifications(face_override=None):
    faces = [_empty_face(index) for index in range(6)]
    if face_override is not None:
        faces[face_override.face_index] = face_override
    qualifications = [
        qualify_face_strand_boundary_incidence_v1(CELL, face, policy=FACE_POLICY)
        for face in faces
    ]
    return tuple(faces), tuple(qualifications)


def test_same_edge_residual_under_cube_rotation_has_same_canonical_code():
    # Any cube edge can rotate to any other cube edge. Both cells carry one
    # same-patch two-crossing residual and should canonicalize identically.
    a = _cell([(0, 1)], (_crossing(0, 0.25, 0), _crossing(0, 0.75, 1)))
    b = _cell([(0, 1)], (_crossing(4, 0.25, 0), _crossing(4, 0.75, 1)))
    faces, quals = _faces_and_qualifications()
    ra = canonicalize_residual_boundary_graph_v1(a, faces, quals, policy=GRAPH_POLICY)
    rb = canonicalize_residual_boundary_graph_v1(b, faces, quals, policy=GRAPH_POLICY)
    assert ra.status == rb.status == CANONICAL
    assert ra.canonical_code == rb.canonical_code
    assert ra.rotation_count_considered == 24


def test_patch_local_id_swap_does_not_change_signature():
    a = _cell(
        [(0, 1), (2, 3, 4)],
        (
            _crossing(0, 0.2, 0), _crossing(0, 0.8, 1),
            _crossing(1, 0.2, 2), _crossing(1, 0.5, 3), _crossing(1, 0.8, 4),
        ),
    )
    b = _cell(
        [(2, 3, 4), (0, 1)],
        (
            _crossing(0, 0.2, 0), _crossing(0, 0.8, 1),
            _crossing(1, 0.2, 2), _crossing(1, 0.5, 3), _crossing(1, 0.8, 4),
        ),
    )
    faces, quals = _faces_and_qualifications()
    ra = canonicalize_residual_boundary_graph_v1(a, faces, quals, policy=GRAPH_POLICY)
    rb = canonicalize_residual_boundary_graph_v1(b, faces, quals, policy=GRAPH_POLICY)
    assert ra.canonical_code == rb.canonical_code
    assert ra.patch_permutation_count_considered == 2


def test_no_residual_cell_is_explicit_not_a_template():
    cell = _cell([(0,)], (_crossing(0, 0.5, 0),))
    faces, quals = _faces_and_qualifications()
    result = canonicalize_residual_boundary_graph_v1(cell, faces, quals, policy=GRAPH_POLICY)
    assert result.status == NO_RESIDUAL
    assert result.canonical_code is None


def test_exact_patch_canonicalization_cap_abstains_instead_of_approximating():
    cell = _cell(
        [(0, 1), (2,), (3,)],
        (_crossing(0, 0.25, 0), _crossing(0, 0.75, 1)),
    )
    faces, quals = _faces_and_qualifications()
    policy = ResidualBoundaryGraphCanonicalizationPolicyV1(2, "SYNTHETIC_CAP_TEST")
    result = canonicalize_residual_boundary_graph_v1(cell, faces, quals, policy=policy)
    assert result.status == ABSTAIN
    assert "CAP_EXCEEDED" in result.abstention_reason


def test_boundary_incidence_distinguishes_opposite_edge_arcs_from_adjacent_edge_arc_composition():
    cell = _cell([(0, 1, 2)])
    opposite_face = FaceStrandTeacherV1(
        FACE_SCHEMA,
        1,
        (
            _strand(0, 0, (-1.0, -0.5), (1.0, -0.5)),
            _strand(1, 1, (-1.0, 0.0), (1.0, 0.0)),
            _strand(2, 2, (-1.0, 0.5), (1.0, 0.5)),
        ),
        (),
        (),
    )
    adjacent_face = FaceStrandTeacherV1(
        FACE_SCHEMA,
        1,
        (
            _strand(0, 0, (-1.0, -0.5), (1.0, -0.5)),
            _strand(1, 1, (-1.0, 0.0), (1.0, 0.0)),
            _strand(2, 2, (-1.0, 0.5), (0.0, -1.0)),
        ),
        (),
        (),
    )
    faces_a, quals_a = _faces_and_qualifications(opposite_face)
    faces_b, quals_b = _faces_and_qualifications(adjacent_face)
    ra = canonicalize_residual_boundary_graph_v1(cell, faces_a, quals_a, policy=GRAPH_POLICY)
    rb = canonicalize_residual_boundary_graph_v1(cell, faces_b, quals_b, policy=GRAPH_POLICY)
    assert ra.status == rb.status == CANONICAL
    assert ra.canonical_code != rb.canonical_code
