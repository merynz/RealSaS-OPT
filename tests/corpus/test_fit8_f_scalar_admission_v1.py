from tools.corpus.fit8_f_scalar_admission_v1 import (
    ABSTAIN,
    F_A,
    F_B,
    F_C,
    F_R,
    GLOBAL_METRIC_SIGN,
    LOCAL_ORIENTED_SHEET_SIGN,
    UNSIGNED_ONLY,
    FScalarEvidenceV1,
    admit_scalar_evidence_for_arm_v1,
)


H = "a" * 64


def _evidence(sign_class, *, sheet=None, global_authority=False):
    return FScalarEvidenceV1(
        sign_class=sign_class,
        cell_xyz=(1, 2, 3),
        local_sheet_id=sheet,
        orientation_provenance="synthetic_guard",
        validity_artifact_sha256=H,
        source_component_sign_authority=global_authority,
    )


def test_f_a_rejects_unsigned_open_surface_evidence():
    decision = admit_scalar_evidence_for_arm_v1(
        F_A,
        _evidence(UNSIGNED_ONLY, global_authority=False),
    )
    assert not decision.admitted
    assert decision.reason == "F_A_REQUIRES_GLOBAL_METRIC_SIGN"


def test_f_a_accepts_only_qualified_global_sign():
    evidence = _evidence(GLOBAL_METRIC_SIGN, global_authority=True)
    decision = admit_scalar_evidence_for_arm_v1(F_A, evidence)
    assert decision.admitted


def test_f_b_and_f_r_accept_local_sheet_oriented_sign():
    evidence = _evidence(
        LOCAL_ORIENTED_SHEET_SIGN,
        sheet=7,
        global_authority=False,
    )
    assert admit_scalar_evidence_for_arm_v1(F_B, evidence).admitted
    assert admit_scalar_evidence_for_arm_v1(F_R, evidence).admitted


def test_local_oriented_sign_without_sheet_identity_is_invalid():
    evidence = _evidence(LOCAL_ORIENTED_SHEET_SIGN, sheet=None)
    try:
        evidence.validate()
    except ValueError as exc:
        assert "LOCAL_SIGN_REQUIRES_SHEET" in str(exc)
    else:
        raise AssertionError("expected local sheet requirement")


def test_f_c_can_use_unsigned_explicit_topology_reference_but_not_abstention():
    unsigned = admit_scalar_evidence_for_arm_v1(F_C, _evidence(UNSIGNED_ONLY))
    assert unsigned.admitted
    abstain = admit_scalar_evidence_for_arm_v1(F_C, _evidence(ABSTAIN))
    assert not abstain.admitted
