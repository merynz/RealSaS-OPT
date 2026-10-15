from dataclasses import replace

from tools.corpus.fit8_f_geometry_teacher_slice_v1 import (
    FGeometryTeacherSliceV1,
    PARENT_BOUND,
    PARENT_DEFERRED,
    ZERO_SHA256,
)


def _slice(**kwargs):
    values = dict(
        subject_key="HORSE",
        source_exact_bytes_sha256="1" * 64,
        source_carrier_sha256="2" * 64,
        source_geometry_frame_sha256="3" * 64,
        topology_qualification_sha256="4" * 64,
        self_intersection_qualification_sha256="5" * 64,
        extraction_truth_sha256="6" * 64,
        extraction_scale_manifest_sha256="7" * 64,
        field_policy_sha256="8" * 64,
        parent_state=PARENT_DEFERRED,
        parent_full_truth_bundle_sha256=ZERO_SHA256,
        provenance="synthetic geometry-slice guard",
    )
    values.update(kwargs)
    return FGeometryTeacherSliceV1(**values)


def test_deferred_parent_is_geometry_eligible_but_not_full_truth_closed():
    value = _slice()
    value.validate()
    assert value.geometry_court_eligible
    assert not value.full_truth_lineage_closed
    assert len(value.sha256()) == 64


def test_bound_parent_closes_full_truth_lineage():
    value = _slice(
        parent_state=PARENT_BOUND,
        parent_full_truth_bundle_sha256="9" * 64,
    )
    value.validate()
    assert value.geometry_court_eligible
    assert value.full_truth_lineage_closed


def test_deferred_parent_cannot_smuggle_nonzero_parent_sha():
    bad = _slice(parent_full_truth_bundle_sha256="9" * 64)
    try:
        bad.validate()
    except ValueError as exc:
        assert "DEFERRED_PARENT_MUST_BE_ZERO" in str(exc)
    else:
        raise AssertionError("expected deferred-parent hash rejection")


def test_geometry_slice_rejects_missing_required_authority_hash():
    bad = replace(_slice(), topology_qualification_sha256=ZERO_SHA256)
    try:
        bad.validate()
    except ValueError as exc:
        assert "ZERO_SHA_FORBIDDEN:topology_qualification_sha256" in str(exc)
    else:
        raise AssertionError("expected required authority hash rejection")
