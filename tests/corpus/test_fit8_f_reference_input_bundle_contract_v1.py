from dataclasses import replace

from tools.corpus.fit8_f_reference_input_bundle_contract_v1 import (
    DIAGNOSTIC_FIT8,
    FReferenceInputBundleManifestV1,
    QUALIFIED_FIT8,
    SYNTHETIC_SMOKE,
    ZERO_SHA256,
)

H = "a" * 64
H2 = "b" * 64


def _manifest(**kwargs):
    values = dict(
        subject_key="Horse",
        arm="F_A",
        authority_state=QUALIFIED_FIT8,
        res=(64, 64, 64),
        global_sign_authority=True,
        support_clipping_required=False,
        source_sha256=H,
        teacher_bundle_sha256=H2,
        extraction_truth_sha256=H,
        canonical_camera_contract_sha256=H2,
        extraction_scale_manifest_sha256=H,
        measurement_protocol_sha256=H2,
        bundle_npz_sha256=H,
        provenance="synthetic_guard",
        metadata={
            "source_exact_bytes_verified": True,
            "teacher_bundle_validated": True,
            "extraction_truth_validated": True,
            "measurement_protocol_preregistered": True,
        },
    )
    values.update(kwargs)
    return FReferenceInputBundleManifestV1(**values)


def test_qualified_bundle_is_matched_court_eligible_only_with_full_lineage():
    value = _manifest()
    value.validate()
    assert value.matched_court_eligible
    assert len(value.manifest_sha256()) == 64


def test_qualified_bundle_rejects_placeholder_hash():
    value = replace(_manifest(), extraction_truth_sha256=ZERO_SHA256)
    try:
        value.validate()
    except ValueError as exc:
        assert "PLACEHOLDER_SHA_FORBIDDEN:extraction_truth_sha256" in str(exc)
    else:
        raise AssertionError("expected placeholder hash rejection")


def test_qualified_bundle_rejects_unvalidated_teacher_or_protocol():
    value = replace(
        _manifest(),
        metadata={
            "source_exact_bytes_verified": True,
            "teacher_bundle_validated": False,
            "extraction_truth_validated": True,
            "measurement_protocol_preregistered": True,
        },
    )
    try:
        value.validate()
    except ValueError as exc:
        assert "QUALIFIED_TEACHER_NOT_VALIDATED" in str(exc)
    else:
        raise AssertionError("expected teacher validation rejection")


def test_diagnostic_fit8_requires_real_source_hash_but_is_not_matched_eligible():
    value = replace(
        _manifest(),
        authority_state=DIAGNOSTIC_FIT8,
        teacher_bundle_sha256=ZERO_SHA256,
        extraction_truth_sha256=ZERO_SHA256,
        canonical_camera_contract_sha256=ZERO_SHA256,
        extraction_scale_manifest_sha256=ZERO_SHA256,
        measurement_protocol_sha256=ZERO_SHA256,
        metadata={},
    )
    value.validate()
    assert not value.matched_court_eligible

    try:
        replace(value, source_sha256=ZERO_SHA256).validate()
    except ValueError as exc:
        assert "DIAGNOSTIC_SOURCE_SHA_REQUIRED" in str(exc)
    else:
        raise AssertionError("expected diagnostic source hash rejection")


def test_synthetic_smoke_cannot_masquerade_as_fit8_subject():
    smoke = replace(
        _manifest(),
        subject_key="SYNTHETIC_SPHERE_SMOKE",
        authority_state=SYNTHETIC_SMOKE,
        source_sha256=ZERO_SHA256,
        teacher_bundle_sha256=ZERO_SHA256,
        extraction_truth_sha256=ZERO_SHA256,
        canonical_camera_contract_sha256=ZERO_SHA256,
        extraction_scale_manifest_sha256=ZERO_SHA256,
        measurement_protocol_sha256=ZERO_SHA256,
        metadata={},
    )
    smoke.validate()
    assert not smoke.matched_court_eligible

    try:
        replace(smoke, subject_key="Horse").validate()
    except ValueError as exc:
        assert "SMOKE_SUBJECT_MUST_BE_SYNTHETIC" in str(exc)
    else:
        raise AssertionError("expected smoke subject rejection")
