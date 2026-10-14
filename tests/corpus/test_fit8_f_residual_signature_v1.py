from tools.corpus.fit8_f_residual_signature_v1 import (
    build_residual_signature_histogram_v1,
    derive_edge_residual_signature_v1,
    derive_face_residual_signature_v1,
)
from tools.corpus.fit8_face_strand_teacher_v1 import FaceSegmentV1, FaceStrandTeacherV1, FaceStrandV1
from tools.corpus.fit8_local_patch_teacher_v1 import (
    CrossingCanonicalizationPolicyV1,
    EdgeCrossingV1,
    ExtractionCellTeacherV1,
    ExtractionCellV1,
    LocalPatchLowerBoundV1,
)

POLICY = CrossingCanonicalizationPolicyV1(1e-7, "SYNTHETIC_TEST")
CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)


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


def _cell(crossings=(), patches=None):
    patches = patches or (LocalPatchLowerBoundV1(0, (0, 1), (0,)),)
    triangles = tuple(sorted({tid for patch in patches for tid in patch.source_triangle_ids}))
    per_edge = {}
    for crossing in crossings:
        per_edge[crossing.edge_index] = per_edge.get(crossing.edge_index, 0) + 1
    return ExtractionCellTeacherV1(
        schema_version="RealSaS.FIT8LocalPatchTeacher.v1",
        cell=CELL,
        active=bool(triangles),
        intersecting_triangle_ids=triangles,
        local_patch_lower_bound=tuple(patches),
        edge_crossings=tuple(crossings),
        coplanar_edge_support=(),
        max_distinct_crossings_per_edge=max(per_edge.values(), default=0),
        crossing_canonicalization=POLICY,
    )


def _segment(tid, a=(0.0, 0.0), b=(1.0, 1.0)):
    return FaceSegmentV1(tid, a, b)


def _strand(local_id, tids, kind, boundary, interior, degree):
    return FaceStrandV1(
        strand_local_id=local_id,
        source_triangle_ids=tuple(tids),
        segments=tuple(_segment(tid) for tid in tids),
        kind=kind,
        boundary_endpoint_count=boundary,
        interior_endpoint_count=interior,
        max_graph_degree=degree,
    )


def test_edge_signature_detects_same_patch_multi_crossing():
    cell = _cell((_crossing(0, 0.25, 0), _crossing(0, 0.75, 1)))
    signature = derive_edge_residual_signature_v1(cell, 0)
    assert signature is not None
    assert signature.crossing_multiplicity_multiset == (2,)


def test_edge_signature_ordinary_f_b_case_returns_none():
    patches = (
        LocalPatchLowerBoundV1(0, (0,), (0,)),
        LocalPatchLowerBoundV1(1, (1,), (1,)),
    )
    cell = _cell((_crossing(0, 0.25, 0), _crossing(0, 0.75, 1)), patches=patches)
    assert derive_edge_residual_signature_v1(cell, 0) is None


def test_face_signature_ignores_orientation_triangle_ids_and_segment_geometry():
    cell = _cell(())
    face_a = FaceStrandTeacherV1(
        schema_version="RealSaS.FIT8FaceStrandTeacher.v1",
        face_index=0,
        strands=(
            _strand(0, (0,), "CLOSED_LOOP", 0, 0, 2),
            _strand(1, (1,), "BOUNDARY_TO_BOUNDARY_ARC", 2, 0, 2),
        ),
        coplanar_triangle_ids=(),
        point_degeneracy_triangle_ids=(),
    )
    face_b = FaceStrandTeacherV1(
        schema_version="RealSaS.FIT8FaceStrandTeacher.v1",
        face_index=5,
        strands=(
            FaceStrandV1(0, (1,), (FaceSegmentV1(1, (-5.0, 2.0), (7.0, 9.0)),), "BOUNDARY_TO_BOUNDARY_ARC", 2, 0, 2),
            FaceStrandV1(1, (0,), (FaceSegmentV1(0, (4.0, 4.0), (4.5, 4.5)),), "CLOSED_LOOP", 0, 0, 2),
        ),
        coplanar_triangle_ids=(),
        point_degeneracy_triangle_ids=(),
    )
    assert derive_face_residual_signature_v1(cell, face_a) == derive_face_residual_signature_v1(cell, face_b)


def test_canonicalization_only_face_does_not_become_model_residual_template():
    cell = _cell(())
    face = FaceStrandTeacherV1(
        schema_version="RealSaS.FIT8FaceStrandTeacher.v1",
        face_index=0,
        strands=(_strand(0, (0,), "BOUNDARY_TO_BOUNDARY_ARC", 2, 0, 2),),
        coplanar_triangle_ids=(1,),
        point_degeneracy_triangle_ids=(),
    )
    assert derive_face_residual_signature_v1(cell, face) is None


def test_histogram_ignores_non_residual_none_and_orders_by_frequency():
    cell = _cell((_crossing(0, 0.25, 0), _crossing(0, 0.75, 1)))
    signature = derive_edge_residual_signature_v1(cell, 0)
    report = build_residual_signature_histogram_v1([signature, None, signature])
    assert report.event_count == 2
    assert report.unique_signature_count == 1
    assert report.counts[0][1] == 2
