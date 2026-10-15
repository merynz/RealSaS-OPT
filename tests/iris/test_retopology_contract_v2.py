from dataclasses import replace

from models.iris.vnext.retopology_contract_v2 import (
    FSurfaceVertexBindingV2,
    TCompilerPromotionInputV2,
    TRetopologyProposalV2,
    validate_retopology_batch_v2,
)


H = "a" * 64
H2 = "b" * 64


def _binding(face=0):
    return FSurfaceVertexBindingV2(
        f_face_id=face,
        barycentric=(0.2, 0.3, 0.5),
        residual_xyz=(0.0, 0.0, 0.0),
        residual_limit=0.01,
        confidence=1.0,
        provenance="synthetic_guard",
    )


def _proposal(**kwargs):
    values = dict(
        source_f_surface_sha256=H,
        vertex_bindings=(_binding(0), _binding(1), _binding(2)),
        faces=((0, 1, 2),),
        decoder_family="TEST",
        proposal_provenance="synthetic_guard",
    )
    values.update(kwargs)
    return TRetopologyProposalV2(**values)


def test_free_xyz_authority_is_forbidden():
    proposal = _proposal(free_xyz_authority=True)
    try:
        proposal.validate()
    except ValueError as exc:
        assert "FREE_XYZ_AUTHORITY_FORBIDDEN" in str(exc)
    else:
        raise AssertionError("expected free XYZ rejection")


def test_barycentric_binding_must_be_valid_and_residual_bounded():
    bad_sum = replace(_binding(), barycentric=(0.2, 0.3, 0.6))
    try:
        bad_sum.validate()
    except ValueError as exc:
        assert "BARYCENTRIC_SUM_INVALID" in str(exc)
    else:
        raise AssertionError("expected barycentric sum rejection")

    bad_residual = replace(_binding(), residual_xyz=(0.02, 0.0, 0.0))
    try:
        bad_residual.validate()
    except ValueError as exc:
        assert "RESIDUAL_EXCEEDS_LIMIT" in str(exc)
    else:
        raise AssertionError("expected residual bound rejection")


def test_compiler_promotion_checks_binding_face_domain():
    value = TCompilerPromotionInputV2(
        proposal=_proposal(),
        f_surface_vertex_count=10,
        f_surface_face_count=2,
        mechanical_probe_set_sha256=H,
    )
    try:
        value.validate()
    except ValueError as exc:
        assert "BINDING_FACE_OOB" in str(exc)
    else:
        raise AssertionError("expected F face range rejection")


def test_matched_retopology_batch_rejects_f_surface_or_probe_drift():
    a = TCompilerPromotionInputV2(
        proposal=_proposal(),
        f_surface_vertex_count=10,
        f_surface_face_count=3,
        mechanical_probe_set_sha256=H,
    )
    b = TCompilerPromotionInputV2(
        proposal=replace(_proposal(), source_f_surface_sha256=H2),
        f_surface_vertex_count=10,
        f_surface_face_count=3,
        mechanical_probe_set_sha256=H,
    )
    try:
        validate_retopology_batch_v2([a, b])
    except ValueError as exc:
        assert "F_SURFACE_DRIFT" in str(exc)
    else:
        raise AssertionError("expected F surface drift rejection")
