import pytest

from tools.corpus.fit8_face_strand_edge_crossing_binding_v1 import (
    EXACT,
    FaceStrandCrossingBindingPolicyV1,
    qualify_face_strand_edge_crossing_binding_v1,
)
from tools.corpus.fit8_face_strand_teacher_v1 import (
    FaceSegmentV1,
    FaceStrandTeacherV1,
    FaceStrandV1,
    SCHEMA_VERSION as FACE_SCHEMA_VERSION,
    derive_face_strands_v1,
)
from tools.corpus.fit8_local_patch_teacher_v1 import (
    CrossingCanonicalizationPolicyV1,
    ExtractionCellV1,
    SourceTriangleV1,
    extract_cell_teacher_v1,
)

CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)
CROSSING_POLICY = CrossingCanonicalizationPolicyV1(1e-8, "SYNTHETIC_STRAND_CROSSING_BINDING")
BINDING_POLICY = FaceStrandCrossingBindingPolicyV1(1e-8, "SYNTHETIC_STRAND_CROSSING_BINDING")


def _tri(triangle_id, a, b, c):
    return SourceTriangleV1(triangle_id, (a, b, c), 0)


def test_boundary_arc_endpoints_bind_to_specific_edge_crossings():
    triangle = _tri(0, (0.0, 0.0, 0.0), (2.0, -2.0, 0.0), (2.0, 2.0, 0.0))
    teacher = extract_cell_teacher_v1(
        CELL,
        (triangle,),
        {},
        crossing_canonicalization=CROSSING_POLICY,
    )
    face = derive_face_strands_v1(CELL, 1, (triangle,), {})
    assert len(face.strands) == 1
    assert face.strands[0].boundary_endpoint_count == 2
    result = qualify_face_strand_edge_crossing_binding_v1(
        teacher,
        face,
        policy=BINDING_POLICY,
    )
    endpoints = result.strands[0].boundary_endpoints
    assert len(endpoints) == 2
    assert all(endpoint.status == EXACT for endpoint in endpoints)
    assert all(len(endpoint.edge_crossing_indices) == 1 for endpoint in endpoints)
    assert {edge for endpoint in endpoints for edge in endpoint.cube_edge_indices} == {9, 11}


def test_closed_loop_has_no_boundary_crossing_binding():
    points = [(1.0, 0.0, 0.5), (1.0, 0.5, 0.0), (1.0, 0.0, -0.5), (1.0, -0.5, 0.0)]
    triangles = tuple(
        _tri(index, points[index], points[(index + 1) % 4], (0.0, 0.0, 0.0))
        for index in range(4)
    )
    adjacency = {index: ((index - 1) % 4, (index + 1) % 4) for index in range(4)}
    teacher = extract_cell_teacher_v1(
        CELL,
        triangles,
        adjacency,
        crossing_canonicalization=CROSSING_POLICY,
    )
    face = derive_face_strands_v1(CELL, 1, triangles, adjacency)
    assert face.strands[0].kind == "CLOSED_LOOP"
    result = qualify_face_strand_edge_crossing_binding_v1(
        teacher,
        face,
        policy=BINDING_POLICY,
    )
    assert result.strands[0].boundary_endpoints == ()


def test_unbound_boundary_endpoint_fails_closed():
    triangle = _tri(0, (0.0, 0.0, 0.0), (2.0, -2.0, 0.0), (2.0, 2.0, 0.0))
    teacher = extract_cell_teacher_v1(
        CELL,
        (triangle,),
        {},
        crossing_canonicalization=CROSSING_POLICY,
    )
    fake = FaceStrandTeacherV1(
        FACE_SCHEMA_VERSION,
        1,
        (
            FaceStrandV1(
                0,
                (0,),
                (FaceSegmentV1(0, (-1.0, 0.25), (1.0, 0.25)),),
                "BOUNDARY_TO_BOUNDARY_ARC",
                2,
                0,
                1,
            ),
        ),
        (),
        (),
    )
    fake.validate()
    with pytest.raises(ValueError, match="UNBOUND_ENDPOINT"):
        qualify_face_strand_edge_crossing_binding_v1(
            teacher,
            fake,
            policy=BINDING_POLICY,
        )
