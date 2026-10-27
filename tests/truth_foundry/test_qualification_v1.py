import unittest

from models.iris.truth.qualification_v1 import (
    ConsumerOracleRequirementV1,
    ConsumerOracleResultV1,
    FamilyQualificationInputV1,
    evaluate_teacher_oracle_gate,
    observability_audit,
    qualify_families,
    qualify_family,
)


H = "a" * 64
R = "b" * 64


def candidate(
    family="RELATIVE_AFFINE_TRANSPORT",
    *,
    measurement_status="MEASURED",
    observability="OBSERVATION_CONDITIONED",
    witness_sha256=H,
    valid_count=8,
    total_count=10,
    source_equivalent=True,
    coverage_qualified=True,
    replay_deterministic=True,
    leakage_free=True,
    conditioning_key=None,
):
    return FamilyQualificationInputV1(
        family=family,
        measurement_status=measurement_status,
        observability=observability,
        witness_sha256=witness_sha256,
        valid_count=valid_count,
        total_count=total_count,
        source_equivalent=source_equivalent,
        coverage_qualified=coverage_qualified,
        replay_deterministic=replay_deterministic,
        leakage_free=leakage_free,
        conditioning_key=conditioning_key,
    )


class TruthQualificationTests(unittest.TestCase):
    def test_observation_conditioned_measurement_can_supervise_only_after_all_local_courts(self):
        decision = qualify_family(candidate())
        self.assertTrue(decision.oracle_eligible)
        self.assertTrue(decision.deterministic_student_supervision_eligible)
        self.assertFalse(decision.conditional_student_supervision_eligible)
        self.assertFalse(decision.unconditional_abstention_required)
        self.assertAlmostEqual(decision.valid_fraction, 0.8)

        leaking = qualify_family(candidate(leakage_free=False))
        self.assertTrue(leaking.oracle_eligible)
        self.assertFalse(leaking.deterministic_student_supervision_eligible)
        self.assertIn("STUDENT_INPUT_LEAKAGE_NOT_CLEARED", leaking.reasons)

    def test_ambiguous_and_privileged_truth_remain_oracle_only(self):
        ambiguous = qualify_family(candidate(observability="AMBIGUOUS", witness_sha256=None))
        privileged = qualify_family(candidate(observability="PRIVILEGED_ONLY", witness_sha256=None))
        for decision in (ambiguous, privileged):
            self.assertTrue(decision.oracle_eligible)
            self.assertFalse(decision.deterministic_student_supervision_eligible)
            self.assertFalse(decision.conditional_student_supervision_eligible)
            self.assertTrue(decision.unconditional_abstention_required)

    def test_mechanism_conditional_never_becomes_unconditional_label(self):
        decision = qualify_family(candidate(
            observability="MECHANISM_CONDITIONAL",
            conditioning_key="student_available_mechanism_context_v1",
        ))
        self.assertTrue(decision.oracle_eligible)
        self.assertFalse(decision.deterministic_student_supervision_eligible)
        self.assertTrue(decision.conditional_student_supervision_eligible)
        self.assertTrue(decision.unconditional_abstention_required)
        self.assertIn("UNCONDITIONAL_MECHANISM_NOT_IDENTIFIABLE", decision.reasons)

        no_schema = qualify_family(candidate(observability="MECHANISM_CONDITIONAL"))
        self.assertFalse(no_schema.conditional_student_supervision_eligible)
        self.assertIn("CONDITIONING_SCHEMA_NOT_DECLARED", no_schema.reasons)

    def test_conditioned_families_require_content_addressed_witness(self):
        with self.assertRaisesRegex(ValueError, "CONDITIONING_WITNESS_REQUIRED"):
            candidate(witness_sha256=None)
        with self.assertRaisesRegex(ValueError, "CONDITIONING_WITNESS_REQUIRED"):
            candidate(observability="MECHANISM_CONDITIONAL", witness_sha256=None)

    def test_unknown_or_unavailable_never_turns_into_negative_supervision(self):
        unknown = qualify_family(candidate(observability="UNKNOWN", witness_sha256=None))
        self.assertTrue(unknown.oracle_eligible)
        self.assertFalse(unknown.deterministic_student_supervision_eligible)
        self.assertIn("UNKNOWN_OBSERVABILITY_REQUIRES_ABSTENTION", unknown.reasons)

        unavailable = qualify_family(candidate(
            measurement_status="UNAVAILABLE",
            observability="UNKNOWN",
            witness_sha256=None,
            valid_count=0,
            total_count=10,
        ))
        self.assertFalse(unavailable.oracle_eligible)
        self.assertFalse(unavailable.deterministic_student_supervision_eligible)
        self.assertIn("MEASUREMENT_NOT_MEASURED", unavailable.reasons)

    def test_oracle_requires_source_equivalence_coverage_replay_and_valid_measurements(self):
        cases = (
            (candidate(source_equivalent=False), "SOURCE_EQUIVALENCE_NOT_PROVEN"),
            (candidate(coverage_qualified=False), "COVERAGE_NOT_QUALIFIED"),
            (candidate(replay_deterministic=False), "REPLAY_NOT_DETERMINISTIC"),
            (candidate(valid_count=0), "NO_VALID_MEASUREMENTS"),
        )
        for value, reason in cases:
            with self.subTest(reason=reason):
                decision = qualify_family(value)
                self.assertFalse(decision.oracle_eligible)
                self.assertFalse(decision.deterministic_student_supervision_eligible)
                self.assertIn(reason, decision.reasons)

    def test_teacher_oracle_gate_requires_runnable_families_and_real_behavioral_result(self):
        decisions = qualify_families((
            candidate("RELATIVE_AFFINE_TRANSPORT"),
            candidate("ORIENTED_GAP_AND_VELOCITY", observability="AMBIGUOUS", witness_sha256=None),
        ))
        requirements = (ConsumerOracleRequirementV1(
            consumer="anchor_grip",
            families=("RELATIVE_AFFINE_TRANSPORT", "ORIENTED_GAP_AND_VELOCITY"),
        ),)

        missing = evaluate_teacher_oracle_gate(decisions, requirements, ())
        self.assertFalse(missing.passed)
        self.assertIn("anchor_grip:BEHAVIORAL_RESULT_MISSING", missing.failed_requirements)

        failed = evaluate_teacher_oracle_gate(
            decisions,
            requirements,
            (ConsumerOracleResultV1("anchor_grip", False, R),),
        )
        self.assertFalse(failed.passed)
        self.assertIn("anchor_grip:BEHAVIORAL_RESULT_FAILED", failed.failed_requirements)

        passed = evaluate_teacher_oracle_gate(
            decisions,
            requirements,
            (ConsumerOracleResultV1("anchor_grip", True, R),),
        )
        self.assertTrue(passed.passed)
        self.assertEqual(passed.failed_requirements, ())
        self.assertEqual(passed.consumers[0].result_witness_sha256, R)

    def test_oracle_gate_fails_closed_when_required_family_is_not_qualified(self):
        decisions = qualify_families((candidate(
            "RELATIVE_AFFINE_TRANSPORT",
            coverage_qualified=False,
        ),))
        gate = evaluate_teacher_oracle_gate(
            decisions,
            (ConsumerOracleRequirementV1("anchor_grip", ("RELATIVE_AFFINE_TRANSPORT",)),),
            (ConsumerOracleResultV1("anchor_grip", True, R),),
        )
        self.assertFalse(gate.passed)
        self.assertEqual(gate.consumers[0].ineligible_families, ("RELATIVE_AFFINE_TRANSPORT",))
        self.assertIn(
            "anchor_grip:RELATIVE_AFFINE_TRANSPORT:ORACLE_INELIGIBLE",
            gate.failed_requirements,
        )

    def test_observability_audit_is_serializable_and_keeps_abstention_explicit(self):
        decisions = qualify_families((
            candidate("RELATIVE_AFFINE_TRANSPORT"),
            candidate("BOUNDARY_RESPONSE", observability="AMBIGUOUS", witness_sha256=None),
        ))
        audit = observability_audit(decisions)
        self.assertTrue(audit["RELATIVE_AFFINE_TRANSPORT"]["deterministic_student_supervision_eligible"])
        self.assertTrue(audit["BOUNDARY_RESPONSE"]["unconditional_abstention_required"])
        self.assertIn(
            "AMBIGUOUS_REQUIRES_ABSTENTION_OR_DISTRIBUTION",
            audit["BOUNDARY_RESPONSE"]["reasons"],
        )

    def test_duplicate_or_empty_requirements_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "FAMILIES_REQUIRED_UNIQUE"):
            ConsumerOracleRequirementV1("graphics", ())
        with self.assertRaisesRegex(ValueError, "DUPLICATE_FAMILY"):
            qualify_families((candidate(), candidate()))


if __name__ == "__main__":
    unittest.main()
