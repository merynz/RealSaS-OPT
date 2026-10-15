from tools.corpus.fit8_f_geometry_backend_contract_v1 import (
    CANDIDATE,
    WITNESS,
    FGeometryBackendRunBindingV1,
    FGeometryBackendV1,
    FGeometryWitnessComparisonV1,
)


H = "a" * 64


def _backend(role):
    return FGeometryBackendV1(
        backend_id=f"backend:{role}",
        role=role,
        extraction_family="SYNTHETIC",
        implementation_ref="tests.synthetic",
        implementation_sha256=H,
        deterministic_given_inputs=True,
        supports_open_surface=True,
        supports_nonmanifold_surface=False,
        provenance="synthetic_guard",
    )


def test_witness_backend_cannot_enter_candidate_arm():
    binding = FGeometryBackendRunBindingV1(
        arm="F_R",
        backend=_backend(WITNESS),
        teacher_bundle_sha256=H,
        extraction_truth_sha256=H,
        measurement_protocol_sha256=H,
    )
    try:
        binding.validate()
    except ValueError as exc:
        assert "CANDIDATE_ARM_REQUIRES_CANDIDATE_ROLE" in str(exc)
    else:
        raise AssertionError("expected witness/candidate role rejection")


def test_tetra_witness_role_is_explicitly_non_candidate():
    binding = FGeometryBackendRunBindingV1(
        arm="WITNESS_TETRA",
        backend=_backend(WITNESS),
        teacher_bundle_sha256=H,
        extraction_truth_sha256=H,
        measurement_protocol_sha256=H,
    )
    binding.validate()


def test_material_witness_disagreement_forces_abstention():
    comparison = FGeometryWitnessComparisonV1(
        candidate_arm="F_R",
        witness_arm="WITNESS_TETRA",
        candidate_measurement_protocol_sha256=H,
        witness_measurement_protocol_sha256=H,
        containment_disagrees=False,
        topology_disagrees=True,
        geometry_disagrees=False,
    )
    comparison.validate()
    assert comparison.has_material_disagreement
    assert not comparison.candidate_may_be_selected


def test_candidate_backend_can_bind_candidate_arm():
    binding = FGeometryBackendRunBindingV1(
        arm="F_B",
        backend=_backend(CANDIDATE),
        teacher_bundle_sha256=H,
        extraction_truth_sha256=H,
        measurement_protocol_sha256=H,
    )
    binding.validate()
