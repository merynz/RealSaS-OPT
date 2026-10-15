from dataclasses import replace

from tools.corpus.fit8_f_reference_input_bundle_contract_v1 import QUALIFIED_FIT8, ZERO_SHA256
from tools.corpus.fit8_f_reference_input_bundle_contract_v2 import FReferenceInputBundleManifestV2


def _qualified(**kwargs):
    values = dict(
        subject_key="HORSE",
        arm="F_A",
        authority_state=QUALIFIED_FIT8,
        res=(64, 64, 64),
        global_sign_authority=True,
        support_clipping_required=False,
        source_sha256="1" * 64,
        source_carrier_sha256="2" * 64,
        geometry_teacher_slice_sha256="3" * 64,
        canonical_camera_contract_sha256="4" * 64,
        measurement_protocol_sha256="5" * 64,
        dense_grid_authority_sha256="6" * 64,
        array_content_sha256="7" * 64,
        bundle_npz_sha256="8" * 64,
        provenance="synthetic v2 manifest guard",
        metadata={
            "source_exact_bytes_verified": True,
            "source_carrier_verified": True,
            "geometry_teacher_slice_validated": True,
            "canonical_camera_contract_validated": True,
            "measurement_protocol_preregistered": True,
            "dense_grid_authority_validated": True,
            "array_content_validated": True,
            "outside_shell_qualified": True,
        },
    )
    values.update(kwargs)
    return FReferenceInputBundleManifestV2(**values)


def test_qualified_manifest_uses_geometry_slice_without_full_teacher_bundle_field():
    value = _qualified()
    value.validate()
    payload = value.canonical_payload()
    assert "geometry_teacher_slice_sha256" in payload
    assert "teacher_bundle_sha256" not in payload
    assert "extraction_truth_sha256" not in payload
    assert value.matched_geometry_court_eligible
    assert len(value.sha256()) == 64


def test_qualified_manifest_rejects_zero_geometry_slice():
    bad = replace(_qualified(), geometry_teacher_slice_sha256=ZERO_SHA256)
    try:
        bad.validate()
    except ValueError as exc:
        assert "PLACEHOLDER_SHA_FORBIDDEN:geometry_teacher_slice_sha256" in str(exc)
    else:
        raise AssertionError("expected geometry-slice SHA rejection")


def test_qualified_manifest_rejects_unqualified_outside_shell():
    metadata = dict(_qualified().metadata)
    metadata["outside_shell_qualified"] = False
    bad = replace(_qualified(), metadata=metadata)
    try:
        bad.validate()
    except ValueError as exc:
        assert "QUALIFICATION_FLAG_FALSE:outside_shell_qualified" in str(exc)
    else:
        raise AssertionError("expected outside-shell qualification rejection")
