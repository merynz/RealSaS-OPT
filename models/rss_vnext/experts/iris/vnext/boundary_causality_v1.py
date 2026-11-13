from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class BoundaryCause(str, Enum):
    GEOMETRIC_TERMINATION = "GEOMETRIC_TERMINATION"
    MATERIAL_BOUNDARY = "MATERIAL_BOUNDARY"
    ALPHA_BOUNDARY = "ALPHA_BOUNDARY"
    OCCLUSION_VISIBILITY_BOUNDARY = "OCCLUSION_VISIBILITY_BOUNDARY"
    SOURCE_PRIMITIVE_COMPONENT_SEAM = "SOURCE_PRIMITIVE_COMPONENT_SEAM"
    RESIDUAL_UNKNOWN = "RESIDUAL_UNKNOWN"


class EvidenceAuthority(str, Enum):
    RAW_AUTHORED = "RAW_AUTHORED"
    DETERMINISTIC_DERIVED = "DETERMINISTIC_DERIVED"
    QUALIFIED_DERIVED = "QUALIFIED_DERIVED"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"


_AUTHORITATIVE_AUTHORITIES = frozenset(
    {
        EvidenceAuthority.RAW_AUTHORED,
        EvidenceAuthority.DETERMINISTIC_DERIVED,
        EvidenceAuthority.QUALIFIED_DERIVED,
    }
)
_KNOWN_CAUSES = frozenset(cause for cause in BoundaryCause if cause is not BoundaryCause.RESIDUAL_UNKNOWN)


@dataclass(frozen=True, order=True)
class BoundaryEvidenceV1:
    cause: BoundaryCause
    authority: EvidenceAuthority
    provenance: str

    def __post_init__(self) -> None:
        if not self.provenance or not self.provenance.strip():
            raise ValueError("BOUNDARY_EVIDENCE_PROVENANCE_REQUIRED")
        if self.cause is BoundaryCause.RESIDUAL_UNKNOWN:
            raise ValueError("RESIDUAL_UNKNOWN_IS_ABSTENTION_NOT_POSITIVE_EVIDENCE")

    @property
    def is_authoritative_training_evidence(self) -> bool:
        return self.authority in _AUTHORITATIVE_AUTHORITIES


@dataclass(frozen=True)
class BoundaryCausalityRecordV1:
    positive_causes: tuple[BoundaryCause, ...]
    evidence: tuple[BoundaryEvidenceV1, ...]
    abstained: bool

    def __post_init__(self) -> None:
        unique_causes = tuple(sorted(set(self.positive_causes), key=lambda x: x.value))
        if unique_causes != self.positive_causes:
            raise ValueError("BOUNDARY_CAUSES_MUST_BE_SORTED_UNIQUE")

        unique_evidence = tuple(
            sorted(
                set(self.evidence),
                key=lambda e: (e.cause.value, e.authority.value, e.provenance),
            )
        )
        if unique_evidence != self.evidence:
            raise ValueError("BOUNDARY_EVIDENCE_MUST_BE_SORTED_UNIQUE")

        if self.abstained:
            if self.positive_causes != (BoundaryCause.RESIDUAL_UNKNOWN,):
                raise ValueError("ABSTENTION_MUST_BE_RESIDUAL_UNKNOWN_ONLY")
            if any(e.is_authoritative_training_evidence for e in self.evidence):
                raise ValueError("ABSTENTION_CANNOT_CONTAIN_AUTHORITATIVE_POSITIVE_EVIDENCE")
            return

        if BoundaryCause.RESIDUAL_UNKNOWN in self.positive_causes:
            raise ValueError("RESIDUAL_UNKNOWN_CANNOT_MIX_WITH_KNOWN_POSITIVE_CAUSES")
        if not self.positive_causes:
            raise ValueError("NON_ABSTAINED_BOUNDARY_REQUIRES_POSITIVE_CAUSE")
        if not set(self.positive_causes).issubset(_KNOWN_CAUSES):
            raise ValueError("UNRECOGNIZED_BOUNDARY_CAUSE")

        authoritative_causes = {
            e.cause for e in self.evidence if e.is_authoritative_training_evidence
        }
        if set(self.positive_causes) != authoritative_causes:
            raise ValueError("POSITIVE_CAUSES_MUST_MATCH_AUTHORITATIVE_EVIDENCE")

    @property
    def training_target_available(self) -> bool:
        return not self.abstained


@dataclass(frozen=True)
class BoundaryCausalityTeacherV1:
    schema: str = "RealSaS.IRISBoundaryCausalityTeacher.v1"

    def qualify(self, evidence: Iterable[BoundaryEvidenceV1]) -> BoundaryCausalityRecordV1:
        ordered_evidence = tuple(
            sorted(
                set(evidence),
                key=lambda e: (e.cause.value, e.authority.value, e.provenance),
            )
        )
        authoritative_causes = tuple(
            sorted(
                {
                    e.cause
                    for e in ordered_evidence
                    if e.is_authoritative_training_evidence
                },
                key=lambda x: x.value,
            )
        )

        if not authoritative_causes:
            return BoundaryCausalityRecordV1(
                positive_causes=(BoundaryCause.RESIDUAL_UNKNOWN,),
                evidence=ordered_evidence,
                abstained=True,
            )

        return BoundaryCausalityRecordV1(
            positive_causes=authoritative_causes,
            evidence=ordered_evidence,
            abstained=False,
        )


def boundary_causality_teacher_v1() -> BoundaryCausalityTeacherV1:
    return BoundaryCausalityTeacherV1()
