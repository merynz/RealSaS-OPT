"""Fail-closed Truth Foundry V1 observability and oracle admission.

This module deliberately separates two questions that are easy to conflate:

1. May a measured teacher family be used by a teacher-oracle Compiler court?
2. May the same family supervise the student deterministically from its declared
   observation domain?

A rich teacher can distinguish mechanisms that the student cannot identify from
its inputs.  Such measurements may remain useful for oracle consumer courts while
being forbidden as deterministic student labels.  Local code here produces only
candidate decisions; it cannot mint Registry qualification, ModelRelease
promotion, ProductRevision authority or a genericity claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from models.iris.truth.contracts_v1 import FAMILIES, OBSERVABILITY, sha256


MEASUREMENT_STATUS = frozenset({"MEASURED", "UNAVAILABLE", "UNSUPPORTED"})


@dataclass(frozen=True)
class FamilyQualificationInputV1:
    family: str
    measurement_status: str
    observability: str
    witness_sha256: str | None
    valid_count: int
    total_count: int
    source_equivalent: bool
    coverage_qualified: bool
    replay_deterministic: bool
    leakage_free: bool
    conditioning_key: str | None = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError("TRUTH_QUALIFICATION_UNKNOWN_FAMILY")
        if self.measurement_status not in MEASUREMENT_STATUS:
            raise ValueError("TRUTH_QUALIFICATION_MEASUREMENT_STATUS")
        if self.observability not in OBSERVABILITY:
            raise ValueError("TRUTH_QUALIFICATION_OBSERVABILITY_STATUS")
        if type(self.valid_count) is not int or type(self.total_count) is not int:
            raise ValueError("TRUTH_QUALIFICATION_COUNTS_MUST_BE_INTEGERS")
        if self.valid_count < 0 or self.total_count < 0 or self.valid_count > self.total_count:
            raise ValueError("TRUTH_QUALIFICATION_INVALID_COUNTS")
        if self.witness_sha256 is not None:
            sha256(self.witness_sha256)
        if self.observability in {"OBSERVATION_CONDITIONED", "MECHANISM_CONDITIONAL"} and self.witness_sha256 is None:
            raise ValueError("TRUTH_QUALIFICATION_CONDITIONING_WITNESS_REQUIRED")
        if self.measurement_status != "MEASURED":
            if self.valid_count != 0:
                raise ValueError("TRUTH_QUALIFICATION_UNMEASURED_VALID_COUNT")
            if self.observability not in {"UNKNOWN", "PRIVILEGED_ONLY"}:
                raise ValueError("TRUTH_QUALIFICATION_UNMEASURED_SUPERVISION_STATE")
        if self.conditioning_key is not None and (
            not isinstance(self.conditioning_key, str) or not self.conditioning_key.strip()
        ):
            raise ValueError("TRUTH_QUALIFICATION_EMPTY_CONDITIONING_KEY")


@dataclass(frozen=True)
class FamilyQualificationDecisionV1:
    family: str
    oracle_eligible: bool
    deterministic_student_supervision_eligible: bool
    conditional_student_supervision_eligible: bool
    unconditional_abstention_required: bool
    valid_fraction: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ConsumerOracleRequirementV1:
    consumer: str
    families: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.consumer, str) or not self.consumer.strip():
            raise ValueError("TRUTH_ORACLE_CONSUMER_NAME_REQUIRED")
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("TRUTH_ORACLE_CONSUMER_FAMILIES_REQUIRED_UNIQUE")
        if any(family not in FAMILIES for family in self.families):
            raise ValueError("TRUTH_ORACLE_CONSUMER_UNKNOWN_FAMILY")


@dataclass(frozen=True)
class ConsumerOracleResultV1:
    consumer: str
    passed: bool
    witness_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.consumer, str) or not self.consumer.strip():
            raise ValueError("TRUTH_ORACLE_RESULT_CONSUMER_REQUIRED")
        if type(self.passed) is not bool:
            raise ValueError("TRUTH_ORACLE_RESULT_PASS_MUST_BE_BOOL")
        sha256(self.witness_sha256)


@dataclass(frozen=True)
class ConsumerOracleDecisionV1:
    consumer: str
    passed: bool
    ineligible_families: tuple[str, ...]
    behavioral_result_present: bool
    behavioral_result_passed: bool
    result_witness_sha256: str | None


@dataclass(frozen=True)
class TeacherOracleGateV1:
    passed: bool
    consumers: tuple[ConsumerOracleDecisionV1, ...]
    failed_requirements: tuple[str, ...]


def qualify_family(candidate: FamilyQualificationInputV1) -> FamilyQualificationDecisionV1:
    """Return candidate eligibility without upgrading source authority.

    Oracle eligibility means the measured family is sufficiently qualified to be
    supplied to a fixed teacher-oracle consumer court.  Student eligibility is
    stricter: privileged, ambiguous and unknown mechanisms never become
    deterministic labels merely because the teacher can measure them.
    """
    reasons: list[str] = []
    valid_fraction = candidate.valid_count / candidate.total_count if candidate.total_count else 0.0

    if candidate.measurement_status != "MEASURED":
        reasons.append("MEASUREMENT_NOT_MEASURED")
    if candidate.valid_count == 0:
        reasons.append("NO_VALID_MEASUREMENTS")
    if not candidate.source_equivalent:
        reasons.append("SOURCE_EQUIVALENCE_NOT_PROVEN")
    if not candidate.coverage_qualified:
        reasons.append("COVERAGE_NOT_QUALIFIED")
    if not candidate.replay_deterministic:
        reasons.append("REPLAY_NOT_DETERMINISTIC")

    oracle_eligible = not reasons

    student_common_ok = oracle_eligible
    if not candidate.leakage_free:
        reasons.append("STUDENT_INPUT_LEAKAGE_NOT_CLEARED")
        student_common_ok = False

    deterministic = False
    conditional = False
    if candidate.observability == "OBSERVATION_CONDITIONED":
        deterministic = student_common_ok
    elif candidate.observability == "MECHANISM_CONDITIONAL":
        if candidate.conditioning_key is None:
            reasons.append("CONDITIONING_SCHEMA_NOT_DECLARED")
        else:
            conditional = student_common_ok
        reasons.append("UNCONDITIONAL_MECHANISM_NOT_IDENTIFIABLE")
    elif candidate.observability == "PRIVILEGED_ONLY":
        reasons.append("PRIVILEGED_ONLY_NOT_STUDENT_SUPERVISION")
    elif candidate.observability == "AMBIGUOUS":
        reasons.append("AMBIGUOUS_REQUIRES_ABSTENTION_OR_DISTRIBUTION")
    elif candidate.observability == "UNKNOWN":
        reasons.append("UNKNOWN_OBSERVABILITY_REQUIRES_ABSTENTION")

    return FamilyQualificationDecisionV1(
        family=candidate.family,
        oracle_eligible=oracle_eligible,
        deterministic_student_supervision_eligible=deterministic,
        conditional_student_supervision_eligible=conditional,
        unconditional_abstention_required=not deterministic,
        valid_fraction=valid_fraction,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def qualify_families(candidates: Sequence[FamilyQualificationInputV1]) -> tuple[FamilyQualificationDecisionV1, ...]:
    if not candidates:
        raise ValueError("TRUTH_QUALIFICATION_EMPTY_FAMILY_SET")
    if len({candidate.family for candidate in candidates}) != len(candidates):
        raise ValueError("TRUTH_QUALIFICATION_DUPLICATE_FAMILY")
    return tuple(qualify_family(candidate) for candidate in candidates)


def evaluate_teacher_oracle_gate(
    decisions: Sequence[FamilyQualificationDecisionV1],
    requirements: Sequence[ConsumerOracleRequirementV1],
    results: Sequence[ConsumerOracleResultV1],
) -> TeacherOracleGateV1:
    """Fail closed unless every declared oracle consumer is runnable *and* passes.

    Passing this gate is necessary before learned training is worth launching, but
    it is intentionally not sufficient for training admission or product
    activation.  Missing behavioral results never default to PASS.
    """
    if not decisions:
        raise ValueError("TRUTH_ORACLE_EMPTY_DECISION_SET")
    if not requirements:
        raise ValueError("TRUTH_ORACLE_EMPTY_REQUIREMENTS")

    decision_by_family = {decision.family: decision for decision in decisions}
    if len(decision_by_family) != len(decisions):
        raise ValueError("TRUTH_ORACLE_DUPLICATE_FAMILY_DECISION")

    requirement_by_consumer = {requirement.consumer: requirement for requirement in requirements}
    if len(requirement_by_consumer) != len(requirements):
        raise ValueError("TRUTH_ORACLE_DUPLICATE_CONSUMER_REQUIREMENT")

    result_by_consumer = {result.consumer: result for result in results}
    if len(result_by_consumer) != len(results):
        raise ValueError("TRUTH_ORACLE_DUPLICATE_CONSUMER_RESULT")
    unexpected = set(result_by_consumer) - set(requirement_by_consumer)
    if unexpected:
        raise ValueError("TRUTH_ORACLE_UNDECLARED_CONSUMER_RESULT")

    consumer_decisions: list[ConsumerOracleDecisionV1] = []
    failures: list[str] = []
    for requirement in requirements:
        ineligible = tuple(
            family
            for family in requirement.families
            if family not in decision_by_family or not decision_by_family[family].oracle_eligible
        )
        for family in ineligible:
            failures.append(f"{requirement.consumer}:{family}:ORACLE_INELIGIBLE")

        result = result_by_consumer.get(requirement.consumer)
        if result is None:
            failures.append(f"{requirement.consumer}:BEHAVIORAL_RESULT_MISSING")
            behavioral_passed = False
            witness = None
        else:
            behavioral_passed = result.passed
            witness = result.witness_sha256
            if not result.passed:
                failures.append(f"{requirement.consumer}:BEHAVIORAL_RESULT_FAILED")

        passed = not ineligible and result is not None and behavioral_passed
        consumer_decisions.append(
            ConsumerOracleDecisionV1(
                consumer=requirement.consumer,
                passed=passed,
                ineligible_families=ineligible,
                behavioral_result_present=result is not None,
                behavioral_result_passed=behavioral_passed,
                result_witness_sha256=witness,
            )
        )

    return TeacherOracleGateV1(
        passed=not failures,
        consumers=tuple(consumer_decisions),
        failed_requirements=tuple(failures),
    )


def observability_audit(decisions: Sequence[FamilyQualificationDecisionV1]) -> Mapping[str, Mapping[str, object]]:
    """Stable serializable candidate audit for TruthQualificationReportV1 wiring."""
    if len({decision.family for decision in decisions}) != len(decisions):
        raise ValueError("TRUTH_QUALIFICATION_DUPLICATE_FAMILY")
    return {
        decision.family: {
            "oracle_eligible": decision.oracle_eligible,
            "deterministic_student_supervision_eligible": decision.deterministic_student_supervision_eligible,
            "conditional_student_supervision_eligible": decision.conditional_student_supervision_eligible,
            "unconditional_abstention_required": decision.unconditional_abstention_required,
            "valid_fraction": decision.valid_fraction,
            "reasons": list(decision.reasons),
        }
        for decision in decisions
    }
