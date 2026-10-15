from dataclasses import replace

from tools.corpus.fit8_f_geometry_measurement_protocol_v1 import (
    FGeometryMeasurementProtocolV1,
)


H = "a" * 64


def _protocol():
    return FGeometryMeasurementProtocolV1(
        protocol_id="FIT8_F_MATCHED_GEOMETRY_V1",
        canonical_camera_contract_sha256=H,
        extraction_scale_manifest_sha256=H,
        thin_structure_manifest_sha256=H,
        source_surface_sampling="DETERMINISTIC_AREA_WEIGHTED_SOURCE_SURFACE",
        candidate_surface_sampling="DETERMINISTIC_AREA_WEIGHTED_CANDIDATE_SURFACE",
        source_sample_count=10000,
        candidate_sample_count=10000,
        deterministic_seed=20261015,
        symmetric_surface_distance_evaluator="EXACT_TRIANGLE_CLOSEST_POINT_BIDIRECTIONAL",
        normal_error_evaluator="NEAREST_QUALIFIED_SOURCE_NORMAL",
        normal_orientation_policy="ORIENTED_WHERE_VALID_ELSE_ABSOLUTE",
        canonical_depth_evaluator="EXACT_EIGHT_CANONICAL_CAMERA_ZBUFFER",
        silhouette_evaluator="EIGHT_VIEW_FOREGROUND_IOU",
        false_bridge_missing_sheet_evaluator="QUALIFIED_LOCAL_SHEET_SOURCE_ADDRESS_COURT",
        self_intersection_evaluator="NONADJACENT_TRIANGLE_INTERSECTION",
        thin_structure_evaluator="PREREGISTERED_THIN_STRUCTURE_MANIFEST_RECALL",
        distance_normalization="EXTRACTION_PITCH",
        spatial_tolerance=1e-7,
        provenance="synthetic_guard",
    )


def test_protocol_hash_is_deterministic_and_content_addressed():
    a = _protocol()
    b = _protocol()
    assert a.sha256() == b.sha256()
    changed = replace(b, candidate_sample_count=20000)
    assert changed.sha256() != a.sha256()


def test_global_orientation_is_not_required_for_all_surface_regions():
    protocol = _protocol()
    protocol.validate()
    assert protocol.normal_orientation_policy == "ORIENTED_WHERE_VALID_ELSE_ABSOLUTE"


def test_unknown_normal_or_distance_policy_is_rejected():
    try:
        replace(_protocol(), normal_orientation_policy="GLOBAL_SIGN_ALWAYS").validate()
    except ValueError as exc:
        assert "NORMAL_POLICY_INVALID" in str(exc)
    else:
        raise AssertionError("expected normal policy rejection")

    try:
        replace(_protocol(), distance_normalization="SUBJECT_BBOX_DIAGONAL").validate()
    except ValueError as exc:
        assert "DISTANCE_NORMALIZATION_INVALID" in str(exc)
    else:
        raise AssertionError("expected distance normalization rejection")
